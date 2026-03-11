# Transolver: Physics-Attention Neural Operator
# Author: Shengning Wang
#
# Faithful reimplementation of: Wu et al., "Transolver: A Fast Transformer
# Solver for PDEs on General Geometries", ICML 2024.
# Source: https://github.com/thuml/Transolver
#
# This file provides the ORIGINAL Transolver architecture as a baseline
# for comparison with STMFormer. Uses the irregular mesh variant.

import torch
from torch import nn, Tensor
from tqdm.auto import tqdm
from typing import List, Optional


def _trunc_normal_(tensor: Tensor, std: float = 0.02) -> Tensor:
    """Truncated normal initialization (compatible replacement for timm).

    Fills tensor in-place from a truncated normal distribution
    with mean=0 and the given std. Values beyond 2*std are resampled.

    Args:
        tensor: Tensor to initialize in-place.
        std: Standard deviation of the normal distribution.

    Returns:
        The initialized tensor (same object, modified in-place).
    """
    with torch.no_grad():
        tensor.normal_(0, std)
        # Clamp to [-2*std, 2*std] and resample outliers
        while True:
            mask = tensor.abs() > 2 * std
            if not mask.any():
                break
            tensor[mask] = torch.empty_like(tensor[mask]).normal_(0, std)
    return tensor


# ============================================================
# MLP (with optional residual connections)
# ============================================================

class MLP(nn.Module):
    """Multi-layer perceptron with optional residual connections.

    Architecture:
        Linear(in) -> act -> [Linear(hidden) -> act (+ residual)] * n_layers -> Linear(out)

    When res=True, each hidden layer adds its output to its input (pre-activation
    residual). This matches the original Transolver MLP implementation.
    """

    def __init__(self, n_input: int, n_hidden: int, n_output: int,
                 n_layers: int = 1, res: bool = True):
        """
        Args:
            n_input: Input feature dimension.
            n_hidden: Hidden layer dimension.
            n_output: Output feature dimension.
            n_layers: Number of hidden residual layers (after the pre-projection).
            res: Whether to use residual connections in hidden layers.
        """
        super().__init__()
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), nn.GELU())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU())
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor. Shape: (..., n_input).

        Returns:
            Output tensor. Shape: (..., n_output).
        """
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


# ============================================================
# Physics Attention (Irregular Mesh)
# ============================================================

class PhysicsAttention(nn.Module):
    """Physics-aware slice attention for irregular meshes.

    The mechanism has three stages:
        1. **Slice**: Soft-cluster N mesh nodes into G slice tokens via learned
           coordinate-based weights. Features and coordinates are projected
           separately (in_project_fx, in_project_x), then the coordinate
           projection drives the slicing weights through a temperature-scaled
           softmax.
        2. **Attend**: Standard multi-head self-attention among the G slice
           tokens (Q/K/V projections, scaled dot-product).
        3. **Deslice**: Broadcast attended slice tokens back to N nodes using
           the same slicing weights.

    Complexity: O(N * G * D + G^2 * D) per head, where N = nodes, G = slices,
    D = dim_head. For N >> G this is much cheaper than O(N^2 * D).

    Args:
        dim: Model hidden dimension (input and output).
        num_heads: Number of attention heads.
        dim_head: Dimension per head. inner_dim = num_heads * dim_head.
        dropout: Dropout probability on attention weights and output.
        num_slices: Number of physics-informed slice tokens (G).
    """

    def __init__(self, dim: int, num_heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0, num_slices: int = 64):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Learnable temperature parameter, clamped to [0.1, 5] during forward
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)

        # Separate projections for features and coordinates
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)

        # Slice projection: maps each head's dim_head -> num_slices
        self.in_project_slice = nn.Linear(dim_head, num_slices)
        nn.init.orthogonal_(self.in_project_slice.weight)

        # Q / K / V projections on slice tokens
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply physics-attention to irregular mesh features.

        Args:
            x: Node features. Shape: (B, N, C) where C = dim.

        Returns:
            Attended features. Shape: (B, N, C).
        """
        B, N, C = x.shape
        H, D = self.num_heads, self.dim_head

        # --- (1) Slice: soft-cluster N nodes into G slice tokens ---
        # Feature projection -> (B, H, N, D)
        fx_mid = self.in_project_fx(x).reshape(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        # Coordinate projection -> (B, H, N, D)
        x_mid = self.in_project_x(x).reshape(B, N, H, D).permute(0, 2, 1, 3).contiguous()

        # Slice weights: (B, H, N, G), temperature-scaled softmax
        temp = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weights = self.softmax(self.in_project_slice(x_mid) / temp)  # (B, H, N, G)

        # Weighted aggregation: (B, H, G, D)
        slice_norm = slice_weights.sum(dim=2)  # (B, H, G)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1e-5)

        # --- (2) Attention among slice tokens ---
        q = self.to_q(slice_token)  # (B, H, G, D)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, G, G)
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v)  # (B, H, G, D)

        # --- (3) Deslice: broadcast back to N nodes ---
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)  # (B, H, N, D)
        out_x = out_x.permute(0, 2, 1, 3).reshape(B, N, H * D)  # (B, N, inner_dim)
        return self.to_out(out_x)


# ============================================================
# Transolver Block
# ============================================================

class TransolverBlock(nn.Module):
    """Single Transolver encoder block.

    Architecture:
        x -> LayerNorm -> PhysicsAttention -> (+residual)
          -> LayerNorm -> MLP              -> (+residual)
        [if last_layer: -> LayerNorm -> Linear(out_channels)]

    The MLP uses 0 hidden residual layers (just pre-linear + GELU + post-linear),
    matching the original implementation (n_layers=0, res=False).

    Args:
        num_heads: Number of attention heads.
        hidden_dim: Model width (hidden dimension).
        dropout: Dropout probability.
        mlp_ratio: MLP hidden dimension multiplier.
        last_layer: If True, append a final LayerNorm + Linear output head.
        out_channels: Output dimension for the last layer projection.
        num_slices: Number of physics-informed slice tokens.
    """

    def __init__(self, num_heads: int, hidden_dim: int, dropout: float,
                 mlp_ratio: int = 4, last_layer: bool = False,
                 out_channels: int = 1, num_slices: int = 32):
        super().__init__()
        self.last_layer = last_layer

        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            num_slices=num_slices,
        )

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False)

        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, fx: Tensor) -> Tensor:
        """
        Args:
            fx: Node features. Shape: (B, N, hidden_dim).

        Returns:
            If last_layer: output predictions. Shape: (B, N, out_channels).
            Otherwise: updated features. Shape: (B, N, hidden_dim).
        """
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.out_proj(self.ln_3(fx))
        return fx


# ============================================================
# Transolver (Irregular Mesh Variant)
# ============================================================

class Transolver(nn.Module):
    """Transolver: A Fast Transformer Solver for PDEs on General Geometries.

    This is the irregular mesh variant (no convolution layers). The pipeline is:
        1. Concatenate input_features and physical_coords -> (B, N, in_channels + spatial_dim)
        2. MLP preprocessing: project to hidden width
        3. Add learnable placeholder bias
        4. Pass through a stack of TransolverBlocks (last block produces output)

    Complexity per layer: O(N * G * C + G^2 * C), where N = nodes,
    G = num_slices, C = width. Total: O(depth * (N * G * C + G^2 * C)).

    Args:
        in_channels: Number of input feature channels per node.
        out_channels: Number of output channels per node.
        spatial_dim: Spatial dimensionality of the mesh (1, 2, or 3).
        width: Hidden dimension throughout the transformer.
        depth: Number of TransolverBlocks.
        num_slices: Number of physics-informed slice tokens per attention layer.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension multiplier within each block.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        width: int = 256,
        depth: int = 8,
        num_slices: int = 32,
        num_heads: int = 8,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.width = width
        self.depth = depth

        # Preprocessing MLP: raw (features || coords) -> hidden width
        # 2-layer MLP with residual: Linear -> GELU -> [Linear -> GELU + res] -> Linear
        self.preprocess = MLP(
            in_channels + spatial_dim,
            width * 2,
            width,
            n_layers=0,
            res=False,
        )

        # Learnable placeholder bias added to all node features after preprocessing
        self.placeholder = nn.Parameter(
            (1.0 / width) * torch.rand(width, dtype=torch.float)
        )

        # Stack of TransolverBlocks; the last block includes output projection
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=num_heads,
                hidden_dim=width,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                out_channels=out_channels,
                num_slices=num_slices,
                last_layer=(i == depth - 1),
            )
            for i in range(depth)
        ])

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply truncated normal init to Linear layers, constant init to LayerNorm."""
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """Per-module weight initialization callback.

        - Linear: trunc_normal_(weight, std=0.02), zeros_(bias).
        - LayerNorm / BatchNorm1d: ones_(weight), zeros_(bias).
        """
        if isinstance(m, nn.Linear):
            _trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        input_features: Tensor,
        physical_coords: Tensor,
        t_norm: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass for single-step prediction.

        Args:
            input_features: Per-node input features.
                Shape: (B, N, in_channels).
            physical_coords: Node spatial coordinates.
                Shape: (B, N, spatial_dim).
            t_norm: Normalized time step (unused in the original Transolver,
                accepted for interface compatibility). Shape: (B,).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Per-node output predictions. Shape: (B, N, out_channels).
        """
        # Concatenate features and coordinates, then project
        fx = torch.cat([physical_coords, input_features], dim=-1)  # (B, N, in_channels + spatial_dim)
        fx = self.preprocess(fx)  # (B, N, width)

        # Add learnable placeholder bias
        fx = fx + self.placeholder[None, None, :]  # broadcast (1, 1, width)

        # Pass through transformer blocks
        for block in self.blocks:
            fx = block(fx)

        return fx  # (B, N, out_channels)

    def predict(self, initial_state: Tensor, coords: Tensor, steps: int) -> Tensor:
        """Autoregressive inference for time-dependent PDE rollout.

        Iteratively applies the model for `steps` time steps, feeding each
        prediction as the input to the next step. The normalized time t_norm
        is passed for interface compatibility but is not used internally by
        the original Transolver architecture.

        Note: caller must call model.eval() before invoking this method.

        Args:
            initial_state: State at t=0. Shape: (B, N, in_channels).
            coords: Node coordinates in [-1, 1]. Shape: (B, N, spatial_dim).
            steps: Number of future steps to generate.

        Returns:
            Predicted sequence including initial state.
            Shape: (B, steps+1, N, out_channels).
        """
        device = next(self.parameters()).device
        B = initial_state.shape[0]
        current_state = initial_state.to(device)
        coords = coords.to(device)

        seq: List[Tensor] = [current_state.cpu()]

        with torch.no_grad():
            for t in tqdm(range(steps), desc='Predicting', leave=False, dynamic_ncols=True):
                t_norm = torch.full((B,), t / max(steps, 1), device=device)
                next_state = self.forward(current_state, coords, t_norm=t_norm)
                seq.append(next_state.cpu())
                current_state = next_state

        return torch.stack(seq, dim=1)

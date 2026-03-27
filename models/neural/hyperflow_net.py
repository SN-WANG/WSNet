# HyperFlowNet: A Spatio-Temporal Neural Operator for PDEs on Complex Geometries
# Author: Shengning Wang
#
# A grid-free neural operator combining three key components:
#   1. Physics Attention — soft-clustering N irregular mesh nodes into M
#      learnable slice tokens, performing multi-head attention in the compressed
#      slice space, then broadcasting back to N nodes. (Wu et al., ICML 2024)
#   2. Learnable Fourier Feature (LFF) spatial encoding — trainable frequency
#      projection that captures multi-scale spatial distances on irregular meshes.
#   3. Sinusoidal temporal encoding — frequency-based time embedding enabling
#      autoregressive rollout prediction for time-dependent PDEs.
#
# Physics Attention is based on: Wu et al., "Transolver: A Fast Transformer
# Solver for PDEs on General Geometries", ICML 2024.
# https://github.com/thuml/Transolver
#
# Complexity: O(N*M*C + M^2*C) per layer, where N = nodes, M = slices, C = width.

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import List, Optional


# ============================================================
# Spatial Encoder (Learnable Fourier Features)
# ============================================================

class SpatialEncoder(nn.Module):
    """Learnable Fourier Feature encoding for irregular mesh coordinates.

    Encodes spatial coordinates via a trainable frequency projection:
        gamma(x) = [sin(2 * pi * W * x); cos(2 * pi * W * x)]
    where W is a learnable projection matrix initialized from N(0, 1).

    Complexity:
        O(batch_size * num_nodes * coord_features) per forward pass.
    """

    def __init__(self, spatial_dim: int, coord_features: int = 8):
        """Initialize the spatial encoder.

        Args:
            spatial_dim: Spatial dimensionality of the mesh (2 or 3).
            coord_features: Half-dimension of the output encoding.
                Output dim: 2 * coord_features.
        """
        super().__init__()
        self.coord_features = coord_features
        self.freq_matrix = nn.Parameter(
            torch.randn(spatial_dim, coord_features)
        )  # (spatial_dim, coord_features)

    def forward(self, coords: Tensor) -> Tensor:
        """Encode mesh coordinates with learnable Fourier features.

        Args:
            coords: Node coordinates, normalized to [-1, 1].
                Shape: (batch_size, num_nodes, spatial_dim).

        Returns:
            LFF spatial encoding.
                Shape: (batch_size, num_nodes, 2 * coord_features).
        """
        proj = (2.0 * torch.pi) * (coords @ self.freq_matrix)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ============================================================
# Temporal Encoder (Sinusoidal Fourier Features)
# ============================================================

class TemporalEncoder(nn.Module):
    """Sinusoidal Fourier encoding for normalized time.

    Encodes normalized time t in [0, 1] via fixed sinusoidal frequencies:
        psi(t) = [sin(omega_i * t * freq_base); cos(omega_i * t * freq_base)]
    where omega_i decays exponentially with frequency index.

    The attribute name "time_encoder" on HyperFlowNet is the detection
    contract with RolloutTrainer.

    Complexity:
        O(batch_size * num_nodes * time_features) per forward pass.
    """

    def __init__(self, time_features: int = 4, freq_base: int = 1000):
        """Initialize the temporal encoder.

        Args:
            time_features: Half-dimension of the output embedding.
                Output dim: 2 * time_features.
            freq_base: Base for exponential frequency decay (analogous to
                the 10000 constant in Transformer sinusoidal PE).
        """
        super().__init__()
        self.time_features = time_features
        self.freq_base = freq_base

        indices = torch.arange(time_features, dtype=torch.float32)
        omega = freq_base ** (-indices / max(time_features, 1))
        self.register_buffer("omega", omega)  # (time_features,)

    def forward(self, t_norm: Tensor, num_nodes: int) -> Tensor:
        """Encode normalized time into sinusoidal embedding.

        Args:
            t_norm: Normalized frame times in [0, 1].
                Shape: (batch_size,).
            num_nodes: Number of mesh nodes for broadcasting.

        Returns:
            Temporal embedding broadcast to all nodes.
                Shape: (batch_size, num_nodes, 2 * time_features).
        """
        t_scaled = t_norm.float() * self.freq_base                # (batch_size,)
        angles = self.omega.unsqueeze(0) * t_scaled.unsqueeze(1)  # (batch_size, time_features)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embedding.unsqueeze(1).expand(-1, num_nodes, -1)


# ============================================================
# Physics Attention
# ============================================================

class PhysicsAttention(nn.Module):
    """Physics Attention: the core attention operator of HyperFlowNet.

    Maps N irregular mesh nodes to M learnable slice tokens via soft assignment,
    performs multi-head self-attention among slice tokens, then broadcasts
    results back to all mesh nodes. No regular grid required.

    Based on: Wu et al., "Transolver: A Fast Transformer Solver for PDEs
    on General Geometries", ICML 2024.

    Algorithm:
        1. w = Softmax(Linear(x))              (batch_size, num_nodes, num_slices)
        2. z = sum(w * x) / sum(w)              (batch_size, num_slices, width)
        3. z' = MultiHeadAttention(z, z, z)     (batch_size, num_slices, width)
        4. x' = w @ z'                          (batch_size, num_nodes, width)

    Complexity:
        O(batch_size * num_nodes * num_slices * width) for slice aggregation,
        plus O(batch_size * num_slices^2 * width) for inter-slice attention.
        Linear in num_nodes when num_slices << num_nodes.
    """

    def __init__(self, width: int, num_slices: int, num_heads: int):
        """Initialize Physics Attention.

        Args:
            width: Feature dimension. Must be divisible by num_heads.
            num_slices: Number of slice tokens (M). Typically 32-64.
            num_heads: Number of attention heads for inter-slice MHA.

        Raises:
            ValueError: If width is not divisible by num_heads.
        """
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(
                f"width={width} must be divisible by num_heads={num_heads}"
            )

        self.num_slices = num_slices
        self.slice_proj = nn.Linear(width, num_slices)
        self.attn = nn.MultiheadAttention(
            embed_dim=width, num_heads=num_heads, batch_first=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run Physics Attention.

        Args:
            x: Node features.
                Shape: (batch_size, num_nodes, width).

        Returns:
            Updated node features.
                Shape: (batch_size, num_nodes, width).
        """
        # 1. Soft slice assignment
        weights = F.softmax(self.slice_proj(x), dim=-1)  # (batch_size, num_nodes, num_slices)

        # 2. Aggregate into slice tokens via weighted mean
        weight_sum = (
            weights.sum(dim=1, keepdim=True)
            .transpose(1, 2)
            .clamp(min=1e-8)
        )  # (batch_size, num_slices, 1)
        slices = torch.bmm(weights.transpose(1, 2), x) / weight_sum  # (batch_size, num_slices, width)

        # 3. Inter-slice multi-head attention
        slices_out, _ = self.attn(slices, slices, slices)  # (batch_size, num_slices, width)

        # 4. Broadcast back to all mesh nodes
        return torch.bmm(weights, slices_out)  # (batch_size, num_nodes, width)


# ============================================================
# HyperFlowNet Block (Physics Attention + FFN, Pre-Norm)
# ============================================================

class HyperFlowNetBlock(nn.Module):
    """Single HyperFlowNet layer: pre-norm Physics Attention + pre-norm FFN.

    Residual connections wrap both sub-layers:
        y = x + PhysicsAttention(LayerNorm(x))
        z = y + FFN(LayerNorm(y))
    """

    def __init__(self, width: int, num_slices: int, num_heads: int, ffn_dim: int):
        """Initialize a HyperFlowNet block.

        Args:
            width: Feature dimension.
            num_slices: Number of slice tokens for Physics Attention.
            num_heads: Attention heads for inter-slice MHA.
            ffn_dim: Inner dimension of the feedforward network.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.physics_attn = PhysicsAttention(width, num_slices, num_heads)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run one HyperFlowNet block.

        Args:
            x: Node features.
                Shape: (batch_size, num_nodes, width).

        Returns:
            Updated node features.
                Shape: (batch_size, num_nodes, width).
        """
        x = x + self.physics_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# HyperFlowNet
# ============================================================

class HyperFlowNet(nn.Module):
    """HyperFlowNet: a spatio-temporal neural operator for PDEs on complex geometries.

    Spatio-Temporal Positional Encoding (togglable for ablation):
        - Spatial: LFF encoding gamma(x) = [sin(2*pi*W*x); cos(2*pi*W*x)]
          where W is a learnable frequency matrix.
        - Temporal: Sinusoidal encoding psi(t) = [sin(omega*t); cos(omega*t)]
          with fixed exponentially-decaying frequencies.

    Architecture:
        Input:  concat([features, spatial_enc?, temporal_enc?]) -> Linear -> width
        Layers: depth x HyperFlowNetBlock (PhysicsAttention + LayerNorm + FFN)
        Output: Linear(width -> out_channels)

    Complexity:
        O(num_nodes * num_slices * width + num_slices^2 * width) per layer,
        linear in num_nodes when num_slices << num_nodes.
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
        ffn_dim: Optional[int] = None,
        use_spatial_encoding: bool = True,
        use_temporal_encoding: bool = True,
        coord_features: int = 8,
        time_features: int = 4,
        freq_base: int = 1000,
    ):
        """Initialize HyperFlowNet.

        Args:
            in_channels: Number of input feature channels (e.g., 4 for [Vx, Vy, P, T]).
            out_channels: Number of output feature channels.
            spatial_dim: Spatial dimension of the mesh (2 or 3).
            width: Hidden channel dimension. Must be divisible by num_heads.
            depth: Number of HyperFlowNetBlock layers.
            num_slices: Number of slice tokens for Physics Attention. Default: 32.
            num_heads: Attention heads for inter-slice MHA. Default: 8.
            ffn_dim: Inner FFN dimension. Default: 4 * width.
            use_spatial_encoding: Enable LFF spatial encoding. Default: True.
                When False, raw coordinates are concatenated directly.
            use_temporal_encoding: Enable sinusoidal temporal encoding. Default: True.
                When False, time information is not injected.
            coord_features: LFF half-dimension (output: 2 * coord_features). Default: 8.
            time_features: Sinusoidal PE half-dimension. Default: 4.
            freq_base: Base for sinusoidal frequency decay. Default: 1000.

        Raises:
            ValueError: If width is not divisible by num_heads.
        """
        super().__init__()

        if width % num_heads != 0:
            raise ValueError(
                f"width={width} must be divisible by num_heads={num_heads}"
            )

        self.spatial_dim = spatial_dim
        self.use_spatial_encoding = use_spatial_encoding
        self.use_temporal_encoding = use_temporal_encoding

        if ffn_dim is None:
            ffn_dim = 4 * width

        # --- Temporal encoder (optional) ---
        # "time_encoder" attribute name is the detection contract with RolloutTrainer.
        if use_temporal_encoding:
            self.time_encoder = TemporalEncoder(
                time_features=time_features,
                freq_base=freq_base,
            )
            time_dim = 2 * time_features
        else:
            self.time_encoder = None
            time_dim = 0

        # --- Spatial encoder (optional) ---
        if use_spatial_encoding and coord_features > 0:
            self.spatial_encoder = SpatialEncoder(
                spatial_dim=spatial_dim,
                coord_features=coord_features,
            )
            coord_dim = 2 * coord_features
        else:
            self.spatial_encoder = None
            coord_dim = spatial_dim

        # --- Input embedding ---
        embed_in = in_channels + coord_dim + time_dim
        self.embed = nn.Linear(embed_in, width)

        # --- HyperFlowNet blocks ---
        self.blocks = nn.ModuleList([
            HyperFlowNetBlock(width, num_slices, num_heads, ffn_dim)
            for _ in range(depth)
        ])

        # --- Output projection ---
        self.proj = nn.Linear(width, out_channels)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_features: Tensor,
        physical_coords: Tensor,
        t_norm: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """HyperFlowNet forward pass operating directly on irregular mesh nodes.

        Args:
            input_features: Node feature fields at current time step.
                Shape: (batch_size, num_nodes, in_channels).
            physical_coords: Normalized node coordinates in [-1, 1].
                Shape: (batch_size, num_nodes, spatial_dim).
            t_norm: Normalized frame times in [0, 1].
                Shape: (batch_size,).
                Defaults to zeros if None. Ignored when use_temporal_encoding=False.
            **kwargs: Accepts (and ignores) extra keys for API compatibility.

        Returns:
            Predicted node features at the next time step.
                Shape: (batch_size, num_nodes, out_channels).
        """
        batch_size, num_nodes, _ = physical_coords.shape

        # --- Build input feature vector ---
        components = [input_features]

        # Spatial encoding
        if self.spatial_encoder is not None:
            components.append(self.spatial_encoder(physical_coords))
        else:
            components.append(physical_coords)

        # Temporal encoding
        if self.time_encoder is not None:
            if t_norm is None:
                t_norm = torch.zeros(batch_size, device=physical_coords.device)
            components.append(self.time_encoder(t_norm, num_nodes))

        # Input embedding
        x = self.embed(torch.cat(components, dim=-1))

        # HyperFlowNet blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        return self.proj(x)  # (batch_size, num_nodes, out_channels)

    # ------------------------------------------------------------------
    # Autoregressive Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        initial_state: Tensor,
        coords: Tensor,
        steps: int,
        boundary_condition=None,
    ) -> Tensor:
        """Autoregressive inference for time-dependent PDE rollout.

        Note: caller must call model.eval() before invoking this method.

        Args:
            initial_state: State at t=0.
                Shape: (batch_size, num_nodes, in_channels).
            coords: Node coordinates in [-1, 1].
                Shape: (batch_size, num_nodes, spatial_dim).
            steps: Number of future time steps to generate.
            boundary_condition: Optional BoundaryCondition instance.
                If provided, enforce() is called after each prediction step
                to hard-set wall node values to known boundary conditions,
                preventing error accumulation at no-slip boundaries.

        Returns:
            Predicted sequence including initial state.
                Shape: (batch_size, steps + 1, num_nodes, out_channels).
        """
        device = next(self.parameters()).device
        batch_size = initial_state.shape[0]
        current_state = initial_state.to(device)
        coords = coords.to(device)

        sequence: List[Tensor] = [current_state.cpu()]

        with torch.no_grad():
            for step in tqdm(range(steps), desc="Predicting", leave=False,
                             dynamic_ncols=True):
                t_norm = torch.full(
                    (batch_size,), step / max(steps, 1), device=device
                )
                next_state = self.forward(current_state, coords, t_norm=t_norm)

                if boundary_condition is not None:
                    next_state = boundary_condition.enforce(next_state)

                sequence.append(next_state.cpu())
                current_state = next_state

        return torch.stack(sequence, dim=1)

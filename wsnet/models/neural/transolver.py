# Transolver: Physics-Attention Neural Operator (grid-free)
# Author: Shengning Wang
#
# Based on: Wu et al., "Transolver: A Fast Transformer Solver for PDEs on General Geometries"
# ICML 2024. https://github.com/thuml/Transolver
#
# Key idea: replace the P2G -> grid convolution -> G2P pipeline entirely with
# Physics-Attention — a soft-clustering mechanism that maps N irregular mesh
# nodes to M learnable "physics slices", performs attention in the slice space,
# then broadcasts back to N nodes. No regular grid needed.
#
# Complexity: O(N*M*C + M^2*C) per layer, where N = nodes, M = slices, C = width.
# For N >> M, this is much cheaper than direct N^2 attention.

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import List, Optional


# ============================================================
# Sinusoidal Time Encoder  (identical to geofnot.py)
# ============================================================

class SinusoidalTimeEncoder(nn.Module):
    """
    Sinusoidal positional encoding for normalized time t in [0, 1].

    psi(t) = [sin(omega_i * t * max_steps); cos(omega_i * t * max_steps)]
    Output shape: (B, N, 2 * time_features).

    The 'time_encoder' attribute name is the detection contract with RolloutTrainer.
    """

    def __init__(self, time_features: int = 4, max_steps: int = 1000):
        """
        Args:
            time_features: Half-dimension of the output embedding.
            max_steps: Reference max time step for frequency scaling.
        """
        super().__init__()
        self.time_features = time_features
        self.max_steps = max_steps

        i = torch.arange(time_features, dtype=torch.float32)
        omega = max_steps ** (-i / max(time_features, 1))
        self.register_buffer('omega', omega)  # (time_features,)

    def encode_time(self, t_norm: Tensor, N: int) -> Tensor:
        """
        Args:
            t_norm: Normalized frame times in [0, 1]. Shape: (B,).
            N: Number of nodes for broadcasting.

        Returns:
            Temporal embedding. Shape: (B, N, 2 * time_features).
        """
        scaled_t = t_norm.float() * self.max_steps               # (B,)
        angles = self.omega.unsqueeze(0) * scaled_t.unsqueeze(1)  # (B, time_features)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 2*time_features)
        return emb.unsqueeze(1).expand(-1, N, -1)                 # (B, N, 2*time_features)


# ============================================================
# Physics-Attention
# ============================================================

class PhysicsAttention(nn.Module):
    """
    Physics-Attention: the core operator of Transolver.

    Maps N irregular mesh nodes to M physics slices via learned soft assignments,
    performs multi-head self-attention among the M slice tokens, then broadcasts
    back to N nodes. This replaces the P2G -> spectral conv -> G2P pipeline with
    a single differentiable operator.

    Algorithm (for one sample):
        w_{n,m}  = Softmax_m( Linear_slice(x_n) )        (B, N, M)   soft assignment
        z_m      = sum_n w_{n,m} * x_n / sum_n w_{n,m}   (B, M, C)   aggregate slices
        z'_m     = MHA(z, z, z)                           (B, M, C)   inter-slice attention
        x'_n     = sum_m w_{n,m} * z'_m                  (B, N, C)   broadcast back

    Complexity: O(B * N * M * C) for slice aggregation + O(B * M^2 * C) for attention.
    For N >> M, dominated by the O(N*M) terms — linear in N.
    """

    def __init__(self, width: int, num_slices: int, num_heads: int):
        """
        Args:
            width: Feature dimension (must be divisible by num_heads).
            num_slices: Number of physics slice tokens (M). Typically 32-64.
            num_heads: Number of attention heads for MHA among slice tokens.
        """
        super().__init__()
        assert width % num_heads == 0, \
            f'width={width} must be divisible by num_heads={num_heads}'

        # Projects node features to M soft slice weights
        self.slice_proj = nn.Linear(width, num_slices)

        # Multi-head self-attention among M slice tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=width, num_heads=num_heads, batch_first=True
        )

        self.num_slices = num_slices

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Node features. Shape: (B, N, width).

        Returns:
            Updated node features. Shape: (B, N, width).
        """
        B, N, C = x.shape

        # 1. Soft slice assignment: (B, N, M)
        w = F.softmax(self.slice_proj(x), dim=-1)  # (B, N, M)

        # 2. Aggregate physics tokens: z_m = sum_n w_{n,m} * x_n / sum_n w_{n,m}
        #    w.transpose(1,2): (B, M, N); x: (B, N, C) -> matmul -> (B, M, C)
        w_sum = w.sum(dim=1, keepdim=True).transpose(1, 2).clamp(min=1e-8)  # (B, M, 1)
        z = torch.bmm(w.transpose(1, 2), x) / w_sum  # (B, M, C)

        # 3. Inter-slice attention: (B, M, C) -> (B, M, C)
        z_prime, _ = self.attn(z, z, z)  # (B, M, C)

        # 4. Broadcast back to nodes: x'_n = sum_m w_{n,m} * z'_m
        #    w: (B, N, M); z_prime: (B, M, C) -> (B, N, C)
        x_prime = torch.bmm(w, z_prime)  # (B, N, C)

        return x_prime


# ============================================================
# Transolver Block  (Physics-Attention + FFN, pre-norm)
# ============================================================

class TransolverBlock(nn.Module):
    """
    Single Transolver layer: pre-norm Physics-Attention + pre-norm FFN.

    y = x + PhysicsAttention(LayerNorm(x))
    z = y + FFN(LayerNorm(y))
    """

    def __init__(self, width: int, num_slices: int, num_heads: int, ffn_dim: int):
        """
        Args:
            width: Feature dimension.
            num_slices: Number of physics slice tokens.
            num_heads: Attention heads for slice-space MHA.
            ffn_dim: Inner dimension of the feedforward network.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.phys_attn = PhysicsAttention(width, num_slices, num_heads)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Node features. Shape: (B, N, width).

        Returns:
            Updated node features. Shape: (B, N, width).
        """
        x = x + self.phys_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Transolver Network
# ============================================================

class Transolver(nn.Module):
    """
    Transolver: grid-free neural operator via Physics-Attention.

    Eliminates the P2G -> spectral conv -> G2P interpolation pipeline entirely.
    Instead, a soft-clustering attention mechanism aggregates N irregular mesh
    nodes into M learnable physics slice tokens, performs attention in that
    compressed space, then broadcasts back to N nodes. No regular grid needed
    -> no grid interpolation errors, no scatter-mean dissipation.

    Architecture:
        Embed:  cat([node_features, physical_coords, time_emb]) -> Linear -> width
        Layers: L x TransolverBlock (PhysicsAttention + LayerNorm + FFN)
        Output: Linear(width -> out_channels)

    Complexity per layer: O(N*M*C + M^2*C), linear in N for M << N.
    Reference: Wu et al., ICML 2024. https://github.com/thuml/Transolver
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
        time_features: int = 4,
        max_steps: int = 1000,
    ):
        """
        Args:
            in_channels: Number of input feature channels (e.g., 4 for [Vx, Vy, P, T]).
            out_channels: Number of output feature channels.
            spatial_dim: Spatial dimension of the mesh (2 or 3).
            width: Hidden channel dimension. Must be divisible by num_heads.
            depth: Number of TransolverBlock layers.
            num_slices: Number of physics slice tokens (M). Default 32.
            num_heads: Attention heads for slice-space MHA. Default 8.
            ffn_dim: Inner FFN dimension. Defaults to 4 * width.
            time_features: Sinusoidal PE half-dimension (output: 2*time_features dims).
            max_steps: Reference max time step for sinusoidal frequency scaling.

        Raises:
            AssertionError: If width is not divisible by num_heads.
        """
        super().__init__()

        assert width % num_heads == 0, \
            f'width={width} must be divisible by num_heads={num_heads}'

        self.spatial_dim = spatial_dim
        if ffn_dim is None:
            ffn_dim = 4 * width

        # Temporal encoder — 'time_encoder' attribute detected by RolloutTrainer
        self.time_encoder = SinusoidalTimeEncoder(
            time_features=time_features,
            max_steps=max_steps,
        )

        # Input embedding: [features + coords + time_emb] -> width
        # Physical coords are concatenated directly — no deformation net or grid needed
        embed_in = in_channels + spatial_dim + 2 * time_features
        self.embed = nn.Linear(embed_in, width)

        # Transolver blocks
        self.layers = nn.ModuleList([
            TransolverBlock(width, num_slices, num_heads, ffn_dim)
            for _ in range(depth)
        ])

        # Output projection
        self.proj = nn.Linear(width, out_channels)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(self, input_features: Tensor, physical_coords: Tensor,
                t_norm: Optional[Tensor] = None, **kwargs) -> Tensor:
        """
        Transolver forward pass. No P2G/G2P — operates directly on mesh nodes.

        Args:
            input_features: Node feature fields at time t.
                            Shape: (B, N, in_channels).
            physical_coords: Normalized node coordinates in [-1, 1].
                             Shape: (B, N, spatial_dim).
            t_norm: Normalized frame times in [0, 1]. Shape: (B,).
                    If None, defaults to zeros.
            **kwargs: Accepts (and ignores) latent_coords for API compatibility
                      with GeoFNO/GeoFNOT.

        Returns:
            Predicted node features at time t+1. Shape: (B, N, out_channels).
        """
        B, N, _ = physical_coords.shape

        if t_norm is None:
            t_norm = torch.zeros(B, device=physical_coords.device)

        # Temporal encoding: t_norm -> (B, N, 2*time_features)
        time_emb = self.time_encoder.encode_time(t_norm, N)

        # Embedding: cat([features, coords, time_emb]) -> hidden
        x = self.embed(torch.cat([input_features, physical_coords, time_emb], dim=-1))

        # Transolver blocks
        for layer in self.layers:
            x = layer(x)

        # Output projection
        return self.proj(x)  # (B, N, out_channels)

    # ------------------------------------------------------------------
    # Autoregressive Inference
    # ------------------------------------------------------------------

    def predict(self, initial_state: Tensor, coords: Tensor, steps: int) -> Tensor:
        """
        Autoregressive inference for time-dependent PDE rollout.

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

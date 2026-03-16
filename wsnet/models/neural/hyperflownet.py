# HyperFlowNet: Grid-Free Neural Operator for PDE Solving
# Author: Shengning Wang
#
# A grid-free neural operator that combines three key components:
#   1. Mesh Slice Attention — soft-clustering N irregular mesh nodes into M
#      learnable slice tokens, performing multi-head attention in the compressed
#      slice space, then broadcasting back to N nodes. No regular grid needed.
#   2. Random Fourier Feature (RFF) spatial encoding — fixed random-frequency
#      projection that captures multi-scale spatial distances on irregular meshes.
#   3. Sinusoidal temporal encoding — frequency-based time embedding enabling
#      autoregressive rollout prediction for time-dependent PDEs.
#
# Inspired by: Wu et al., "Transolver: A Fast Transformer Solver for PDEs on
# General Geometries", ICML 2024. https://github.com/thuml/Transolver
#
# Complexity: O(N*M*C + M^2*C) per layer, where N = nodes, M = slices, C = width.

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import List, Optional


# ============================================================
# Spatial Encoder (Random Fourier Features)
# ============================================================

class SpatialEncoder(nn.Module):
    """Random Fourier Feature encoding for irregular mesh coordinates.

    Encodes spatial coordinates via a fixed random projection:
        gamma(x) = [sin(2*pi*B*x); cos(2*pi*B*x)]
    where B ~ N(0, sigma^2) is a non-trainable projection matrix.

    This converts raw (x, y) coordinates into a 2*coord_features-dim
    representation that captures spatial distances at multiple frequencies,
    giving Mesh Slice Attention richer geometry context than raw coords.

    Output dim: 2 * coord_features.
    """

    def __init__(self, spatial_dim: int, coord_features: int = 8, sigma: float = 1.0):
        """
        Args:
            spatial_dim: Spatial dimensionality of the mesh (2 or 3).
            coord_features: Half-dimension of the output encoding.
                            Output shape: (..., 2 * coord_features).
            sigma: Standard deviation of the random projection matrix.
                   Controls spatial frequency bandwidth of the encoding.
        """
        super().__init__()
        self.coord_features = coord_features
        self.sigma = sigma

        B = torch.randn(spatial_dim, coord_features) * sigma
        self.register_buffer("B_matrix", B)  # (spatial_dim, coord_features)

    def forward(self, coords: Tensor) -> Tensor:
        """Encode coordinates with RFF.

        Args:
            coords: Node coordinates in [-1, 1]. Shape: (B, N, spatial_dim).

        Returns:
            RFF encoding. Shape: (B, N, 2 * coord_features).
        """
        proj = (2.0 * torch.pi) * (coords @ self.B_matrix)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ============================================================
# Sinusoidal Time Encoder
# ============================================================

class SinusoidalTimeEncoder(nn.Module):
    """Sinusoidal positional encoding for normalized time t in [0, 1].

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
        self.register_buffer("omega", omega)  # (time_features,)

    def encode_time(self, t_norm: Tensor, N: int) -> Tensor:
        """Encode normalized time into sinusoidal embedding.

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
# Mesh Slice Attention
# ============================================================

class MeshSliceAttention(nn.Module):
    """Mesh Slice Attention: the core operator of HyperFlowNet.

    Maps N irregular mesh nodes to M slice tokens via learned soft assignments,
    performs multi-head self-attention among the M slice tokens, then broadcasts
    back to N nodes.

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
            num_slices: Number of slice tokens (M). Typically 32-64.
            num_heads: Number of attention heads for MHA among slice tokens.
        """
        super().__init__()
        assert width % num_heads == 0, \
            f"width={width} must be divisible by num_heads={num_heads}"

        self.slice_proj = nn.Linear(width, num_slices)
        self.attn = nn.MultiheadAttention(
            embed_dim=width, num_heads=num_heads, batch_first=True
        )
        self.num_slices = num_slices

    def forward(self, x: Tensor) -> Tensor:
        """Run Mesh Slice Attention.

        Args:
            x: Node features. Shape: (B, N, width).

        Returns:
            Updated node features. Shape: (B, N, width).
        """
        B, N, C = x.shape

        # 1. Soft slice assignment: (B, N, M)
        w = F.softmax(self.slice_proj(x), dim=-1)

        # 2. Aggregate slice tokens: z_m = weighted mean of node features
        w_sum = w.sum(dim=1, keepdim=True).transpose(1, 2).clamp(min=1e-8)  # (B, M, 1)
        z = torch.bmm(w.transpose(1, 2), x) / w_sum  # (B, M, C)

        # 3. Inter-slice attention
        z_prime, _ = self.attn(z, z, z)  # (B, M, C)

        # 4. Broadcast back to nodes
        x_prime = torch.bmm(w, z_prime)  # (B, N, C)

        return x_prime


# ============================================================
# HyperFlowNet Block  (Mesh Slice Attention + FFN, pre-norm)
# ============================================================

class HyperFlowNetBlock(nn.Module):
    """Single HyperFlowNet layer: pre-norm Mesh Slice Attention + pre-norm FFN.

    y = x + MeshSliceAttention(LayerNorm(x))
    z = y + FFN(LayerNorm(y))
    """

    def __init__(self, width: int, num_slices: int, num_heads: int, ffn_dim: int):
        """
        Args:
            width: Feature dimension.
            num_slices: Number of slice tokens.
            num_heads: Attention heads for slice-space MHA.
            ffn_dim: Inner dimension of the feedforward network.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.mesh_slice_attn = MeshSliceAttention(width, num_slices, num_heads)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run one HyperFlowNet block.

        Args:
            x: Node features. Shape: (B, N, width).

        Returns:
            Updated node features. Shape: (B, N, width).
        """
        x = x + self.mesh_slice_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# HyperFlowNet Network
# ============================================================

class HyperFlowNet(nn.Module):
    """HyperFlowNet: grid-free neural operator for PDE solving.

    Eliminates the P2G -> spectral conv -> G2P interpolation pipeline entirely.
    Instead, a soft-clustering attention mechanism aggregates N irregular mesh
    nodes into M learnable slice tokens, performs attention in that compressed
    space, then broadcasts back to N nodes. No regular grid needed.

    Spatio-Temporal Positional Encoding (togglable for ablation):
        - Spatial: RFF encoding gamma(x) = [sin(2*pi*B*x); cos(2*pi*B*x)]
        - Temporal: Sinusoidal PE psi(t) = [sin(omega_i*t); cos(omega_i*t)]

    Architecture:
        Embed:  cat([features, spatial_enc?, time_enc?]) -> Linear -> width
        Layers: L x HyperFlowNetBlock (MeshSliceAttention + LayerNorm + FFN)
        Output: Linear(width -> out_channels)

    Complexity per layer: O(N*M*C + M^2*C), linear in N for M << N.

    Inspired by: Wu et al., "Transolver", ICML 2024.
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
        # Ablation switches
        use_spatial_encoding: bool = True,
        use_temporal_encoding: bool = True,
        # Spatial encoding params
        coord_features: int = 8,
        coord_sigma: float = 1.0,
        # Temporal encoding params
        time_features: int = 4,
        max_steps: int = 1000,
    ):
        """
        Args:
            in_channels: Number of input feature channels (e.g., 4 for [Vx, Vy, P, T]).
            out_channels: Number of output feature channels.
            spatial_dim: Spatial dimension of the mesh (2 or 3).
            width: Hidden channel dimension. Must be divisible by num_heads.
            depth: Number of HyperFlowNetBlock layers.
            num_slices: Number of slice tokens (M). Default: 32.
            num_heads: Attention heads for slice-space MHA. Default: 8.
            ffn_dim: Inner FFN dimension. Default: 4 * width.
            use_spatial_encoding: Enable RFF spatial encoding. Default: True.
                                  When False, raw coordinates are used directly.
            use_temporal_encoding: Enable sinusoidal temporal encoding. Default: True.
                                   When False, time information is not injected.
            coord_features: RFF half-dimension (output: 2 * coord_features). Default: 8.
            coord_sigma: RFF projection scale. Default: 1.0.
            time_features: Sinusoidal PE half-dimension. Default: 4.
            max_steps: Reference max time step for frequency scaling. Default: 1000.
        """
        super().__init__()

        assert width % num_heads == 0, \
            f"width={width} must be divisible by num_heads={num_heads}"

        self.spatial_dim = spatial_dim
        self.use_spatial_encoding = use_spatial_encoding
        self.use_temporal_encoding = use_temporal_encoding

        if ffn_dim is None:
            ffn_dim = 4 * width

        # --- Temporal encoder (optional) ---
        # 'time_encoder' attribute name is the detection contract with RolloutTrainer
        if use_temporal_encoding:
            self.time_encoder = SinusoidalTimeEncoder(
                time_features=time_features,
                max_steps=max_steps,
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
                sigma=coord_sigma,
            )
            coord_dim = 2 * coord_features
        else:
            self.spatial_encoder = None
            coord_dim = spatial_dim

        # --- Input embedding ---
        embed_in = in_channels + coord_dim + time_dim
        self.embed = nn.Linear(embed_in, width)

        # --- HyperFlowNet blocks ---
        self.layers = nn.ModuleList([
            HyperFlowNetBlock(width, num_slices, num_heads, ffn_dim)
            for _ in range(depth)
        ])

        # --- Output projection ---
        self.proj = nn.Linear(width, out_channels)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(self, input_features: Tensor, physical_coords: Tensor,
                t_norm: Optional[Tensor] = None, **kwargs) -> Tensor:
        """HyperFlowNet forward pass. Operates directly on mesh nodes.

        Args:
            input_features: Node feature fields at time t. Shape: (B, N, in_channels).
            physical_coords: Normalized node coordinates in [-1, 1].
                             Shape: (B, N, spatial_dim).
            t_norm: Normalized frame times in [0, 1]. Shape: (B,).
                    If None, defaults to zeros. Ignored when use_temporal_encoding=False.
            **kwargs: Accepts (and ignores) latent_coords for API compatibility.

        Returns:
            Predicted node features at time t+1. Shape: (B, N, out_channels).
        """
        B, N, _ = physical_coords.shape

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
                t_norm = torch.zeros(B, device=physical_coords.device)
            components.append(self.time_encoder.encode_time(t_norm, N))

        # Embedding
        x = self.embed(torch.cat(components, dim=-1))

        # --- HyperFlowNet blocks ---
        for layer in self.layers:
            x = layer(x)

        # --- Output projection ---
        return self.proj(x)  # (B, N, out_channels)

    # ------------------------------------------------------------------
    # Autoregressive Inference
    # ------------------------------------------------------------------

    def predict(self, initial_state: Tensor, coords: Tensor, steps: int,
                boundary_condition=None) -> Tensor:
        """Autoregressive inference for time-dependent PDE rollout.

        Note: caller must call model.eval() before invoking this method.

        Args:
            initial_state: State at t=0. Shape: (B, N, in_channels).
            coords: Node coordinates in [-1, 1]. Shape: (B, N, spatial_dim).
            steps: Number of future steps to generate.
            boundary_condition: Optional BoundaryCondition instance.
                If provided, enforce() is called after each prediction step
                to hard-set wall node values to known boundary conditions,
                preventing error accumulation at no-slip boundaries.

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
            for t in tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True):
                t_norm = torch.full((B,), t / max(steps, 1), device=device)
                next_state = self.forward(current_state, coords, t_norm=t_norm)

                # Hard BC enforcement: replace wall node predictions with known values
                if boundary_condition is not None:
                    next_state = boundary_condition.enforce(next_state)

                seq.append(next_state.cpu())
                current_state = next_state

        return torch.stack(seq, dim=1)

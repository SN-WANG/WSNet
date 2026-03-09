# Geometry-Aware Fourier Neural Operator with Temporal Encoding (Geo-FNOT)
# Author: Shengning Wang
#
# GeoFNOT = FNO Fourier spectral ops (optimal for smooth NS flows)
#           + KNN-IDW P2G from GeoWNO (better for irregular Fluent meshes)
#           + sinusoidal temporal encoding (no spatial RFF noise).
#
# Root-cause fix over GeoWNO:
#   - Replaces Haar WaveletConv with SpectralConv (Fourier basis wins on smooth flows)
#   - Drops RFF spatial encoding from lifting (reduces lifting dim, removes redundancy)
#   - Keeps KNN-IDW P2G (genuinely better than scatter-mean for irregular meshes)
#   - Keeps sinusoidal temporal PE (provides useful curriculum context)

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import List, Dict, Optional

from wsnet.models.neural.geofno import SpectralConv, FNOBlock, DeformationNet  # noqa: F401


# ============================================================
# Sinusoidal Time Encoder  (temporal-only, no spatial RFF)
# ============================================================

class SinusoidalTimeEncoder(nn.Module):
    """
    Sinusoidal positional encoding for normalized time t in [0, 1].

    Encoding formula:
        psi(t) = [sin(omega_i * t * max_steps); cos(omega_i * t * max_steps)]
        in R^{2 * time_features}

    where omega_i = max_steps^{-i / time_features} for i = 0, ..., time_features-1.

    The 'time_encoder' attribute name is the detection contract with RolloutTrainer:
    hasattr(model, 'time_encoder') routes t_norm through forward(). No spatial RFF
    encoding is included — the coordinate deformation net already handles geometry;
    RFF would inject redundant, fixed-scale spatial noise into the lifting layer.
    """

    def __init__(self, time_features: int = 4, max_steps: int = 1000):
        """
        Args:
            time_features: Half-dimension of the temporal embedding.
                           Output shape: (..., 2 * time_features).
            max_steps: Reference maximum time step used to scale sinusoidal
                       frequencies. Set close to the longest sequence length.
        """
        super().__init__()
        self.time_features = time_features
        self.max_steps = max_steps

        # omega_i = max_steps^(-i / time_features); shape: (time_features,)
        i = torch.arange(time_features, dtype=torch.float32)
        omega = max_steps ** (-i / max(time_features, 1))
        self.register_buffer('omega', omega)

    def encode_time(self, t_norm: Tensor, N: int) -> Tensor:
        """
        Encode normalized time t in [0, 1] with sinusoidal positional encoding.

        Args:
            t_norm: Normalized frame times in [0, 1]. Shape: (B,).
            N: Number of spatial nodes for broadcasting.

        Returns:
            Temporal embedding. Shape: (B, N, 2 * time_features).
        """
        scaled_t = t_norm.float() * self.max_steps               # (B,)
        angles = self.omega.unsqueeze(0) * scaled_t.unsqueeze(1)  # (B, time_features)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 2*time_features)
        return emb.unsqueeze(1).expand(-1, N, -1)                 # (B, N, 2*time_features)


# ============================================================
# Geo-FNOT Network
# ============================================================

class GeoFNOT(nn.Module):
    """
    Geometry-Aware Fourier Neural Operator with Temporal Encoding (Geo-FNOT).

    Best-of-both-worlds hybrid of GeoFNO and GeoWNO:
    - Retains GeoFNO's Fourier spectral convolutions (near-optimal for smooth NS flows)
    - Upgrades P2G to KNN-IDW (from GeoWNO, reduces scatter-mean dissipation)
    - Adds sinusoidal temporal encoding without spatial RFF (clean context signal)

    Pipeline:
        1. Temporal encoding: t_norm -> time_emb in R^{2*time_features}
        2. Lifting: cat([node_features, time_emb]) -> hidden (width)
        3. Deformation: physical_coords -> latent_coords via DeformationNet
        4. KNN-IDW P2G: irregular mesh nodes -> regular latent grid
        5. FNO blocks: Fourier spectral convolutions on latent grid
        6. G2P: latent grid -> original mesh nodes via bilinear interpolation
        7. Projection: hidden (width) -> out_channels via 2-layer MLP
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: List[int],
        latent_grid_size: List[int],
        depth: int,
        width: int,
        deformation_kwargs: Optional[Dict] = None,
        knn_k: int = 6,
        knn_chunk: int = 256,
        time_features: int = 4,
        max_steps: int = 1000,
    ):
        """
        Args:
            in_channels: Number of input feature channels (e.g., 4 for [Vx, Vy, P, T]).
            out_channels: Number of output feature channels.
            modes: Number of Fourier modes per spatial dimension.
                   Length determines spatial_dim (e.g., [12, 12] for 2D).
            latent_grid_size: Size of the regular latent grid, e.g. [64, 64].
                              Must have same length as modes.
            depth: Number of stacked FNO blocks.
            width: Hidden channel dimension throughout the network.
            deformation_kwargs: kwargs for DeformationNet (num_layers, hidden_dim).
                                Defaults to {'num_layers': 3, 'hidden_dim': 32}.
            knn_k: Number of nearest neighbors for KNN-IDW P2G. Default 6.
            knn_chunk: Grid cells processed per KNN chunk (bounds peak VRAM). Default 256.
            time_features: Sinusoidal PE half-dimension (output: 2*time_features dims).
            max_steps: Reference max time step for sinusoidal frequency scaling.
        """
        super().__init__()

        assert len(modes) == len(latent_grid_size), \
            f'modes (len={len(modes)}) and latent_grid_size (len={len(latent_grid_size)}) must match'

        self.spatial_dim = len(modes)
        self.modes = modes
        self.latent_grid_size = latent_grid_size
        self.width = width
        self.knn_k = knn_k
        self.knn_chunk = knn_chunk
        self.eps = 1e-8

        if deformation_kwargs is None:
            deformation_kwargs = {'num_layers': 3, 'hidden_dim': 32}

        # Coordinate deformation (same as GeoFNO; disabled for 1D)
        if self.spatial_dim > 1:
            self.deformation_net = DeformationNet(spatial_dim=self.spatial_dim, **deformation_kwargs)
        else:
            self.deformation_net = None

        # Temporal encoder — 'time_encoder' attribute is detected by RolloutTrainer via
        # hasattr(model, 'time_encoder'). This name must not change.
        self.time_encoder = SinusoidalTimeEncoder(
            time_features=time_features,
            max_steps=max_steps,
        )

        # Lifting: [in_channels + 2*time_features] -> width
        # No spatial RFF: deformation_net handles geometry; RFF adds redundant fixed-scale noise.
        self.fc_lift = nn.Linear(in_channels + 2 * time_features, width)

        # FNO blocks — Fourier spectral convolutions, reused from geofno.py
        self.fno_blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(depth)])

        # Projection: width -> 128 -> out_channels
        self.fc_proj1 = nn.Linear(width, 128)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_proj2 = nn.Linear(128, out_channels)

        # KNN-IDW grid centers buffer (precomputed, moves with .to(device))
        self._init_grid_centers()

    # ------------------------------------------------------------------
    # Initialization Helpers
    # ------------------------------------------------------------------

    def _init_grid_centers(self) -> None:
        """Precompute regular grid cell centers in [-1, 1]^d as a buffer.

        Registers:
            grid_centers: Shape (G, spatial_dim) where G = prod(latent_grid_size).
        """
        axes = [torch.linspace(-1.0, 1.0, g) for g in self.latent_grid_size]
        grids = torch.meshgrid(*axes, indexing='ij')
        centers = torch.stack([g.flatten() for g in grids], dim=-1)  # (G, spatial_dim)
        self.register_buffer('grid_centers', centers)

    # ------------------------------------------------------------------
    # Latent Coordinate Computation
    # ------------------------------------------------------------------

    def _compute_latent_coords(self, physical_coords: Tensor) -> Tensor:
        """Apply learned coordinate deformation to get latent grid coordinates.

        Args:
            physical_coords: Normalized node coordinates in [-1, 1].
                             Shape: (B, N, spatial_dim).

        Returns:
            Latent coordinates clamped to [-1, 1]. Shape: (B, N, spatial_dim).
        """
        if self.spatial_dim > 1 and self.deformation_net is not None:
            return torch.clamp(self.deformation_net(physical_coords), -1.0, 1.0)
        return physical_coords

    # ------------------------------------------------------------------
    # KNN-IDW Point-to-Grid  (from GeoWNO — better than scatter-mean)
    # ------------------------------------------------------------------

    def _p2g_knn(self, features: Tensor, latent_coords: Tensor) -> Tensor:
        """
        KNN Inverse-Distance-Weighted Point-to-Grid aggregation.

        For each grid cell center, finds the k nearest mesh nodes in latent
        space and aggregates their features with IDW weights (w_i ∝ 1/d_i).
        Uses chunked torch.cdist to keep peak GPU memory bounded.

        Args:
            features: Lifted node features. Shape: (B, N, width).
            latent_coords: Deformed node coordinates in [-1, 1].
                           Shape: (B, N, spatial_dim).

        Returns:
            Regular grid features. Shape: (B, width, G1, (G2), (G3)).
        """
        B, N, W = features.shape
        device = features.device
        G = self.grid_centers.shape[0]

        grid_out = torch.zeros(B, G, W, device=device, dtype=features.dtype)

        for chunk_start in range(0, G, self.knn_chunk):
            chunk_end = min(chunk_start + self.knn_chunk, G)
            centers_chunk = self.grid_centers[chunk_start:chunk_end]  # (G', spatial_dim)
            Gc = centers_chunk.shape[0]

            # Pairwise distances: (B, G', N)
            centers_exp = centers_chunk.unsqueeze(0).expand(B, -1, -1)
            dists = torch.cdist(centers_exp, latent_coords)  # (B, G', N)

            # k nearest neighbors
            topk_dists, topk_idx = dists.topk(self.knn_k, dim=-1, largest=False)  # (B, G', k)

            # IDW weights: 1/d, row-normalized
            weights = 1.0 / (topk_dists + self.eps)
            weights = weights / weights.sum(dim=-1, keepdim=True)  # (B, G', k)

            # Gather features via advanced indexing — no N-dim expansion
            b_idx = (torch.arange(B, device=device)
                     .view(B, 1, 1).expand(B, Gc, self.knn_k))  # (B, G', k)
            gathered = features[b_idx, topk_idx]                # (B, G', k, W)

            # Weighted aggregation
            grid_chunk = (gathered * weights.unsqueeze(-1)).sum(dim=-2)  # (B, G', W)
            grid_out[:, chunk_start:chunk_end] = grid_chunk

        # Reshape (B, G, W) -> (B, width, G1, G2, ...)
        target_shape = [B] + self.latent_grid_size + [W]
        grid_features = grid_out.view(*target_shape)
        perm = [0, self.spatial_dim + 1] + list(range(1, self.spatial_dim + 1))
        return grid_features.permute(*perm).contiguous()

    # ------------------------------------------------------------------
    # Grid-to-Point  (identical to GeoFNO / GeoWNO)
    # ------------------------------------------------------------------

    def _g2p_sample(self, grid_features: Tensor, latent_coords: Tensor) -> Tensor:
        """
        Grid-to-Point decoding via bilinear/trilinear interpolation (F.grid_sample).

        F.grid_sample expects (x, y[, z]) = (W[, H[, D]]) axis order, but
        latent_coords are stored as (dim0[, dim1[, dim2]]) = (H[, W[, D]]).
        The last dimension is flipped/reversed accordingly.

        Args:
            grid_features: Latent grid. Shape: (B, width, G1, (G2), (G3)).
            latent_coords: Latent node coordinates in [-1, 1].
                           Shape: (B, N, spatial_dim).

        Returns:
            Interpolated node features. Shape: (B, N, width).
        """
        B, N, _ = latent_coords.shape

        if self.spatial_dim == 1:
            grid_input = grid_features.unsqueeze(2)
            zeros = torch.zeros_like(latent_coords)
            coords_input = torch.cat([latent_coords, zeros], dim=-1).view(B, N, 1, 2)
            sampled = F.grid_sample(grid_input, coords_input, align_corners=True, padding_mode='border')
            return sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        elif self.spatial_dim == 2:
            # F.grid_sample: (x=W-axis, y=H-axis); latent_coords: (dim0=H, dim1=W) -> flip
            coords_input = latent_coords[..., [1, 0]].view(B, N, 1, 2)
            sampled = F.grid_sample(grid_features, coords_input, align_corners=True, padding_mode='border')
            return sampled.squeeze(-1).permute(0, 2, 1)

        elif self.spatial_dim == 3:
            # F.grid_sample: (x=W, y=H, z=D); latent_coords: (dim0=D, dim1=H, dim2=W) -> reverse
            coords_input = latent_coords[..., [2, 1, 0]].view(B, N, 1, 1, 3)
            sampled = F.grid_sample(grid_features, coords_input, align_corners=True, padding_mode='border')
            return sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(self, input_features: Tensor, physical_coords: Tensor,
                t_norm: Optional[Tensor] = None,
                latent_coords: Optional[Tensor] = None) -> Tensor:
        """
        Geo-FNOT forward pass.

        Args:
            input_features: Node feature fields at time t.
                            Shape: (B, N, in_channels).
            physical_coords: Normalized node coordinates in [-1, 1].
                             Shape: (B, N, spatial_dim).
            t_norm: Normalized frame times in [0, 1]. Shape: (B,).
                    If None, defaults to zeros.
            latent_coords: Pre-computed deformed coordinates in [-1, 1].
                           Shape: (B, N, spatial_dim). Pass a cached value from
                           _compute_latent_coords() to skip redundant deformation
                           network calls during rollout.

        Returns:
            Predicted node features at time t+1. Shape: (B, N, out_channels).
        """
        B, N, _ = physical_coords.shape

        if t_norm is None:
            t_norm = torch.zeros(B, device=physical_coords.device)

        # 1. Temporal encoding: t_norm -> (B, N, 2*time_features)
        time_emb = self.time_encoder.encode_time(t_norm, N)

        # 2. Lifting: cat([features, time_emb]) -> hidden (no spatial RFF)
        lifted = self.fc_lift(torch.cat([input_features, time_emb], dim=-1))  # (B, N, width)

        # 3. Coordinate deformation (use cached value if provided)
        if latent_coords is None:
            latent_coords = self._compute_latent_coords(physical_coords)

        # 4. KNN-IDW Point-to-Grid: irregular mesh -> regular latent grid
        grid_features = self._p2g_knn(lifted, latent_coords)  # (B, width, G1, ...)

        # 5. FNO blocks: Fourier spectral convolutions
        for block in self.fno_blocks:
            grid_features = block(grid_features)

        # 6. Grid-to-Point: latent grid -> original mesh nodes
        recovered = self._g2p_sample(grid_features, latent_coords)  # (B, N, width)

        # 7. Projection: width -> 128 -> out_channels
        output = F.gelu(self.fc_proj1(recovered))
        output = self.fc_proj2(self.dropout(output))

        return output

    # ------------------------------------------------------------------
    # Autoregressive Inference
    # ------------------------------------------------------------------

    def predict(self, initial_state: Tensor, coords: Tensor, steps: int) -> Tensor:
        """
        Autoregressive inference for time-dependent PDE rollout.

        Precomputes latent coordinates once (deformation_net is fixed at inference)
        and iterates the forward model for `steps` time steps.

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

        # Pre-compute latent coords once — physical coords are constant across steps
        latent_coords = self._compute_latent_coords(coords)

        seq: List[Tensor] = [current_state.cpu()]

        with torch.no_grad():
            for t in tqdm(range(steps), desc='Predicting', leave=False, dynamic_ncols=True):
                t_norm = torch.full((B,), t / max(steps, 1), device=device)
                next_state = self.forward(
                    current_state, coords, t_norm=t_norm, latent_coords=latent_coords
                )
                seq.append(next_state.cpu())
                current_state = next_state

        return torch.stack(seq, dim=1)

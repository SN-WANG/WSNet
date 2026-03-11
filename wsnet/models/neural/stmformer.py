# STMFormer: Spatio-Temporal Mesh Transformer for Grid-Free PDE Solving
# Author: Shengning Wang
#
# A grid-free neural operator that combines four key components:
#   1. Mesh Slice Attention — soft-clustering N irregular mesh nodes into M
#      learnable slice tokens, performing multi-head attention in the compressed
#      slice space, then broadcasting back to N nodes. No regular grid needed.
#   2. Random Fourier Feature (RFF) spatial encoding — fixed random-frequency
#      projection that captures multi-scale spatial distances on irregular meshes.
#   3. Sinusoidal temporal encoding — frequency-based time embedding enabling
#      autoregressive rollout prediction for time-dependent PDEs.
#   4. Boundary padding — extends the mesh with ghost nodes outside the boundary,
#      turning original boundary nodes into interior nodes for improved accuracy.
#
# Inspired by: Wu et al., "Transolver: A Fast Transformer Solver for PDEs on
# General Geometries", ICML 2024. https://github.com/thuml/Transolver
#
# Complexity: O(N*M*C + M^2*C) per layer, where N = nodes, M = slices, C = width.
# For N >> M, this is much cheaper than direct N^2 attention.

import torch
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import List, Optional, Tuple
from collections import Counter


# ============================================================
# Boundary Padding for Unstructured Meshes
# ============================================================

class BoundaryPadding(nn.Module):
    """Domain padding for unstructured meshes, analogous to FNO grid padding.

    Identifies boundary nodes via Delaunay triangulation, then creates ghost
    nodes outside the boundary by offsetting along estimated outward normals.
    Ghost node features are zero-padded. After model prediction, ghost nodes
    are stripped from the output.

    This converts original boundary nodes into interior nodes, improving
    prediction accuracy near boundaries where gradients are typically steep.

    The ghost node positions are cached after the first call and reused for
    subsequent forward passes with the same mesh topology.
    """

    def __init__(self, padding_ratio: float = 0.05):
        """
        Args:
            padding_ratio: Offset distance as a fraction of the domain extent.
                           E.g., 0.05 means ghost nodes are placed 5% of the
                           domain diagonal outside the boundary. Default: 0.05.
        """
        super().__init__()
        self.padding_ratio = padding_ratio
        self._n_original: Optional[int] = None
        self._cached_ghost_coords: Optional[Tensor] = None
        self._cached_coords_hash: Optional[int] = None

    def _coords_hash(self, coords: Tensor) -> int:
        """Compute a hash to detect when coordinates change."""
        return hash((coords.shape, float(coords[0, 0, 0]), float(coords[0, -1, -1])))

    def _find_boundary_nodes_2d(self, coords_np: np.ndarray) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Find boundary nodes and edges using Delaunay triangulation.

        Args:
            coords_np: Node coordinates. Shape: (N, 2).

        Returns:
            Tuple of (boundary_node_indices, boundary_edges).
        """
        from scipy.spatial import Delaunay

        tri = Delaunay(coords_np)

        # Count edge occurrences: boundary edges appear in exactly one simplex
        edge_count: Counter = Counter()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                edge_count[edge] += 1

        boundary_edges = [e for e, count in edge_count.items() if count == 1]
        boundary_nodes = sorted(set(n for e in boundary_edges for n in e))
        return boundary_nodes, boundary_edges

    def _compute_outward_normals(
        self, coords_np: np.ndarray,
        boundary_nodes: List[int],
        boundary_edges: List[Tuple[int, int]],
    ) -> np.ndarray:
        """Compute approximate outward normals at boundary nodes.

        Args:
            coords_np: All node coordinates. Shape: (N, 2).
            boundary_nodes: Indices of boundary nodes.
            boundary_edges: List of boundary edge index pairs.

        Returns:
            Outward normal vectors at boundary nodes. Shape: (len(boundary_nodes), 2).
        """
        centroid = coords_np.mean(axis=0)
        normals = np.zeros((len(boundary_nodes), 2))
        node_to_idx = {n: i for i, n in enumerate(boundary_nodes)}

        for edge in boundary_edges:
            n0, n1 = edge
            edge_vec = coords_np[n1] - coords_np[n0]
            # Perpendicular to edge
            normal = np.array([-edge_vec[1], edge_vec[0]])

            # Ensure outward direction (away from centroid)
            midpoint = (coords_np[n0] + coords_np[n1]) / 2
            if np.dot(normal, midpoint - centroid) < 0:
                normal = -normal

            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal = normal / norm_len

            # Accumulate to both endpoints
            if n0 in node_to_idx:
                normals[node_to_idx[n0]] += normal
            if n1 in node_to_idx:
                normals[node_to_idx[n1]] += normal

        # Normalize accumulated normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normals = normals / norms
        return normals

    def compute_ghost_nodes(self, coords: Tensor) -> Tensor:
        """Compute ghost node positions from mesh coordinates.

        Called once per unique mesh topology. Results are cached internally.

        Args:
            coords: Node coordinates. Shape: (B, N, spatial_dim).
                    Uses the first sample in the batch for computation.

        Returns:
            Ghost node coordinates. Shape: (N_ghost, spatial_dim).
        """
        coords_np = coords[0].detach().cpu().numpy()  # (N, D)
        spatial_dim = coords_np.shape[1]

        if spatial_dim != 2:
            raise NotImplementedError(
                f"BoundaryPadding currently supports 2D meshes only, got {spatial_dim}D"
            )

        boundary_nodes, boundary_edges = self._find_boundary_nodes_2d(coords_np)

        if len(boundary_nodes) == 0:
            return torch.zeros(0, spatial_dim, device=coords.device)

        normals = self._compute_outward_normals(coords_np, boundary_nodes, boundary_edges)

        # Compute offset distance from domain extent
        domain_extent = coords_np.max(axis=0) - coords_np.min(axis=0)
        offset_distance = np.linalg.norm(domain_extent) * self.padding_ratio

        # Ghost node positions: boundary node + offset * outward normal
        boundary_coords = coords_np[boundary_nodes]  # (N_boundary, 2)
        ghost_coords_np = boundary_coords + normals * offset_distance  # (N_boundary, 2)

        return torch.tensor(ghost_coords_np, dtype=coords.dtype, device=coords.device)

    def pad(self, features: Tensor, coords: Tensor) -> Tuple[Tensor, Tensor]:
        """Extend features and coordinates with zero-padded ghost nodes.

        Args:
            features: Node features. Shape: (B, N, C).
            coords: Node coordinates. Shape: (B, N, spatial_dim).

        Returns:
            Tuple of (padded_features, padded_coords):
                padded_features: (B, N + N_ghost, C)
                padded_coords: (B, N + N_ghost, spatial_dim)
        """
        B, N, C = features.shape
        self._n_original = N

        # Compute or retrieve cached ghost nodes
        coords_hash = self._coords_hash(coords)
        if self._cached_ghost_coords is None or self._cached_coords_hash != coords_hash:
            self._cached_ghost_coords = self.compute_ghost_nodes(coords)
            self._cached_coords_hash = coords_hash

        ghost_coords = self._cached_ghost_coords  # (N_ghost, D)
        N_ghost = ghost_coords.shape[0]

        if N_ghost == 0:
            return features, coords

        # Zero-padded ghost features: (B, N_ghost, C)
        ghost_features = torch.zeros(B, N_ghost, C, device=features.device, dtype=features.dtype)

        # Expand ghost coords to batch: (B, N_ghost, D)
        ghost_coords_batch = ghost_coords.unsqueeze(0).expand(B, -1, -1)

        padded_features = torch.cat([features, ghost_features], dim=1)
        padded_coords = torch.cat([coords, ghost_coords_batch], dim=1)

        return padded_features, padded_coords

    def strip(self, output: Tensor) -> Tensor:
        """Remove ghost nodes from model output.

        Args:
            output: Model output including ghost nodes. Shape: (B, N + N_ghost, C).

        Returns:
            Original nodes only. Shape: (B, N, C).
        """
        if self._n_original is None:
            return output
        return output[:, :self._n_original, :]


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
        self.register_buffer('B_matrix', B)  # (spatial_dim, coord_features)

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
# Mesh Slice Attention
# ============================================================

class MeshSliceAttention(nn.Module):
    """Mesh Slice Attention: the core operator of STMFormer.

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
            f'width={width} must be divisible by num_heads={num_heads}'

        self.slice_proj = nn.Linear(width, num_slices)
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
# STMFormer Block  (Mesh Slice Attention + FFN, pre-norm)
# ============================================================

class STMFormerBlock(nn.Module):
    """Single STMFormer layer: pre-norm Mesh Slice Attention + pre-norm FFN.

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
        """
        Args:
            x: Node features. Shape: (B, N, width).

        Returns:
            Updated node features. Shape: (B, N, width).
        """
        x = x + self.mesh_slice_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# STMFormer Network
# ============================================================

class STMFormer(nn.Module):
    """STMFormer: Spatio-Temporal Mesh Transformer for grid-free PDE solving.

    Eliminates the P2G -> spectral conv -> G2P interpolation pipeline entirely.
    Instead, a soft-clustering attention mechanism aggregates N irregular mesh
    nodes into M learnable slice tokens, performs attention in that compressed
    space, then broadcasts back to N nodes. No regular grid needed.

    Spatio-Temporal Positional Encoding (togglable for ablation):
        - Spatial: RFF encoding gamma(x) = [sin(2*pi*B*x); cos(2*pi*B*x)]
        - Temporal: Sinusoidal PE psi(t) = [sin(omega_i*t); cos(omega_i*t)]

    Boundary Padding (togglable for ablation):
        Extends the mesh with ghost nodes outside the boundary, converting
        original boundary nodes into interior nodes for improved accuracy.

    Architecture:
        Embed:  cat([features, spatial_enc?, time_enc?]) -> Linear -> width
        Layers: L x STMFormerBlock (MeshSliceAttention + LayerNorm + FFN)
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
        use_boundary_padding: bool = False,
        # Spatial encoding params
        coord_features: int = 8,
        coord_sigma: float = 1.0,
        # Temporal encoding params
        time_features: int = 4,
        max_steps: int = 1000,
        # Boundary padding params
        padding_ratio: float = 0.05,
    ):
        """
        Args:
            in_channels: Number of input feature channels (e.g., 4 for [Vx, Vy, P, T]).
            out_channels: Number of output feature channels.
            spatial_dim: Spatial dimension of the mesh (2 or 3).
            width: Hidden channel dimension. Must be divisible by num_heads.
            depth: Number of STMFormerBlock layers.
            num_slices: Number of slice tokens (M). Default: 32.
            num_heads: Attention heads for slice-space MHA. Default: 8.
            ffn_dim: Inner FFN dimension. Default: 4 * width.
            use_spatial_encoding: Enable RFF spatial encoding. Default: True.
                                  When False, raw coordinates are used directly.
            use_temporal_encoding: Enable sinusoidal temporal encoding. Default: True.
                                   When False, time information is not injected.
            use_boundary_padding: Enable boundary ghost-node padding. Default: False.
            coord_features: RFF half-dimension (output: 2 * coord_features). Default: 8.
            coord_sigma: RFF projection scale. Default: 1.0.
            time_features: Sinusoidal PE half-dimension. Default: 4.
            max_steps: Reference max time step for frequency scaling. Default: 1000.
            padding_ratio: Ghost node offset as fraction of domain extent. Default: 0.05.
        """
        super().__init__()

        assert width % num_heads == 0, \
            f'width={width} must be divisible by num_heads={num_heads}'

        self.spatial_dim = spatial_dim
        self.use_spatial_encoding = use_spatial_encoding
        self.use_temporal_encoding = use_temporal_encoding
        self.use_boundary_padding = use_boundary_padding

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

        # --- Boundary padding (optional) ---
        if use_boundary_padding:
            self.boundary_padding = BoundaryPadding(padding_ratio=padding_ratio)
        else:
            self.boundary_padding = None

        # --- Input embedding ---
        embed_in = in_channels + coord_dim + time_dim
        self.embed = nn.Linear(embed_in, width)

        # --- STMFormer blocks ---
        self.layers = nn.ModuleList([
            STMFormerBlock(width, num_slices, num_heads, ffn_dim)
            for _ in range(depth)
        ])

        # --- Output projection ---
        self.proj = nn.Linear(width, out_channels)

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(self, input_features: Tensor, physical_coords: Tensor,
                t_norm: Optional[Tensor] = None, **kwargs) -> Tensor:
        """STMFormer forward pass. Operates directly on mesh nodes.

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

        # --- Boundary padding: extend mesh with ghost nodes ---
        if self.boundary_padding is not None:
            input_features, physical_coords = self.boundary_padding.pad(
                input_features, physical_coords
            )
            _, N_padded, _ = physical_coords.shape
        else:
            N_padded = N

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
            components.append(self.time_encoder.encode_time(t_norm, N_padded))

        # Embedding
        x = self.embed(torch.cat(components, dim=-1))

        # --- STMFormer blocks ---
        for layer in self.layers:
            x = layer(x)

        # --- Output projection ---
        output = self.proj(x)

        # --- Strip ghost nodes ---
        if self.boundary_padding is not None:
            output = self.boundary_padding.strip(output)

        return output  # (B, N, out_channels)

    # ------------------------------------------------------------------
    # Autoregressive Inference
    # ------------------------------------------------------------------

    def predict(self, initial_state: Tensor, coords: Tensor, steps: int) -> Tensor:
        """Autoregressive inference for time-dependent PDE rollout.

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

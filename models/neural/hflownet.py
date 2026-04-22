# HyperFlowNet for spatio-temporal irregular-mesh flow prediction
# Author: Shengning Wang

from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm


# ============================================================
# Encoding Blocks
# ============================================================


class SpatialEncoder(nn.Module):
    """
    Learnable Fourier feature encoder for irregular mesh coordinates.
    """

    def __init__(self, spatial_dim: int, coord_features: int = 8) -> None:
        """
        Initialize the spatial encoder.

        Args:
            spatial_dim (int): Spatial dimensionality of the mesh.
            coord_features (int): Half-dimension of the encoded coordinates.
        """
        super().__init__()
        self.coord_features = coord_features
        self.freq_matrix = nn.Parameter(torch.randn(spatial_dim, coord_features))

    def forward(self, coords: Tensor) -> Tensor:
        """
        Encode physical coordinates with learnable Fourier features.

        Args:
            coords (Tensor): Node coordinates. (B, N, D).

        Returns:
            Tensor: Encoded coordinates. (B, N, 2 * C_COORD).
        """
        proj = (2.0 * torch.pi) * (coords @ self.freq_matrix)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TemporalEncoder(nn.Module):
    """
    Sinusoidal Fourier encoder for normalized rollout time.
    """

    def __init__(self, time_features: int = 4, freq_base: int = 1000) -> None:
        """
        Initialize the temporal encoder.

        Args:
            time_features (int): Half-dimension of the temporal embedding.
            freq_base (int): Base for exponentially decaying frequencies.
        """
        super().__init__()
        self.time_features = time_features
        self.freq_base = freq_base

        indices = torch.arange(time_features, dtype=torch.float32)
        omega = freq_base ** (-indices / max(time_features, 1))
        self.register_buffer("omega", omega, persistent=False)

    def forward(self, t_norm: Tensor, num_nodes: int) -> Tensor:
        """
        Encode normalized time and broadcast it to all nodes.

        Args:
            t_norm (Tensor): Normalized frame times. (B,).
            num_nodes (int): Number of mesh nodes.

        Returns:
            Tensor: Temporal embedding. (B, N, 2 * C_TIME).
        """
        t_scaled = t_norm.float() * self.freq_base
        angles = self.omega.unsqueeze(0) * t_scaled.unsqueeze(1)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embedding.unsqueeze(1).expand(-1, num_nodes, -1)


# ============================================================
# HyperFlow Blocks
# ============================================================


class PhysicsAttention(nn.Module):
    """
    Slice-space attention on irregular mesh nodes.
    """

    def __init__(self, width: int, num_slices: int, num_heads: int) -> None:
        """
        Initialize the physics attention module.

        Args:
            width (int): Node token width.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of attention heads in slice space.
        """
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")

        self.slice_proj = nn.Linear(width, num_slices)
        self.attn = nn.MultiheadAttention(embed_dim=width, num_heads=num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Run slice aggregation, slice attention, and node broadcast.

        Args:
            x (Tensor): Node tokens. (B, N, C).

        Returns:
            Tensor: Updated node tokens. (B, N, C).
        """
        weights = F.softmax(self.slice_proj(x), dim=-1)
        weight_sum = weights.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(1e-8)
        slices = torch.bmm(weights.transpose(1, 2), x) / weight_sum
        slices_out, _ = self.attn(slices, slices, slices)
        return torch.bmm(weights, slices_out)


class HyperFlowBlock(nn.Module):
    """
    One pre-norm HyperFlowNet block.
    """

    def __init__(self, width: int, num_slices: int, num_heads: int, ffn_dim: int) -> None:
        """
        Initialize one HyperFlowNet block.

        Args:
            width (int): Node token width.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of attention heads.
            ffn_dim (int): Hidden width of the feed-forward block.
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
        """
        Apply one residual HyperFlowNet update.

        Args:
            x (Tensor): Node tokens. (B, N, C).

        Returns:
            Tensor: Updated node tokens. (B, N, C).
        """
        x = x + self.physics_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# HyperFlowNet
# ============================================================


class HyperFlowNet(nn.Module):
    """
    Spatio-temporal neural operator on irregular meshes.
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
    ) -> None:
        """
        Initialize HyperFlowNet.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Hidden channel width.
            depth (int): Number of HyperFlowNet blocks.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of slice-space attention heads.
            ffn_dim (Optional[int]): Hidden width of the feed-forward block.
            use_spatial_encoding (bool): Whether to use learnable Fourier spatial encoding.
            use_temporal_encoding (bool): Whether to use sinusoidal temporal encoding.
            coord_features (int): Half-dimension of the Fourier spatial encoding.
            time_features (int): Half-dimension of the temporal encoding.
            freq_base (int): Base for temporal frequencies.
        """
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")

        if ffn_dim is None:
            ffn_dim = 4 * width

        self.use_spatial_encoding = use_spatial_encoding
        self.use_temporal_encoding = use_temporal_encoding

        if use_spatial_encoding and coord_features > 0:
            self.spatial_encoder = SpatialEncoder(spatial_dim=spatial_dim, coord_features=coord_features)
            coord_dim = 2 * coord_features
        else:
            self.spatial_encoder = None
            coord_dim = spatial_dim

        if use_temporal_encoding:
            self.time_encoder = TemporalEncoder(time_features=time_features, freq_base=freq_base)
            time_dim = 2 * time_features
        else:
            self.time_encoder = None
            time_dim = 0

        self.embed = nn.Linear(in_channels + coord_dim + time_dim, width)
        self.blocks = nn.ModuleList([
            HyperFlowBlock(width=width, num_slices=num_slices, num_heads=num_heads, ffn_dim=ffn_dim)
            for _ in range(depth)
        ])
        self.proj = nn.Linear(width, out_channels)

    def forward(self, inputs: Tensor, coords: Tensor, t_norm: Optional[Tensor] = None) -> Tensor:
        """
        Predict the next state on the mesh.

        Args:
            inputs (Tensor): Current node features. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            t_norm (Optional[Tensor]): Normalized rollout time. (B,).

        Returns:
            Tensor: Predicted next state. (B, N, C_OUT).
        """
        B, N, _ = coords.shape
        components = [inputs]

        if self.spatial_encoder is None:
            components.append(coords)
        else:
            components.append(self.spatial_encoder(coords))

        if self.time_encoder is not None:
            if t_norm is None:
                t_norm = torch.zeros(B, device=coords.device, dtype=coords.dtype)
            else:
                t_norm = t_norm.to(device=coords.device, dtype=coords.dtype)
            components.append(self.time_encoder(t_norm, N).to(dtype=coords.dtype))

        x = self.embed(torch.cat(components, dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.proj(x)

    def predict(self, inputs: Tensor, coords: Tensor, steps: int) -> Tensor:
        """
        Run autoregressive rollout prediction.

        Args:
            inputs (Tensor): Initial rollout state. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            steps (int): Number of rollout steps.

        Returns:
            Tensor: Predicted sequence with the initial state. (B, T + 1, N, C_OUT).
        """
        device = next(self.parameters()).device
        current_state = inputs.to(device)
        coords = coords.to(device)

        sequence: List[Tensor] = [current_state.cpu()]

        with torch.no_grad():
            iterator = tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True)
            for step_idx in iterator:
                if self.time_encoder is None:
                    step_t_norm = None
                else:
                    step_t_norm = torch.full(
                        (current_state.shape[0],), step_idx / max(steps, 1),
                        device=device, dtype=current_state.dtype,
                    )
                next_state = self.forward(current_state, coords, t_norm=step_t_norm)
                sequence.append(next_state.cpu())
                current_state = next_state

        return torch.stack(sequence, dim=1)

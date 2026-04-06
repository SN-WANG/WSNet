# HyperFlowNet for spatio-temporal irregular-mesh flow prediction
# Author: Shengning Wang

from math import pi, sqrt
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm


# ============================================================
# Encoding Blocks
# ============================================================

class TemporalEncoder(nn.Module):
    """
    Sinusoidal temporal encoder for normalized rollout time.
    """

    def __init__(self, time_features: int = 4, freq_base: int = 1000) -> None:
        """
        Initialize the temporal encoder.

        Args:
            time_features (int): Number of sinusoidal frequency pairs.
            freq_base (int): Reference time scale used to distribute frequencies.
        """
        super().__init__()
        self.time_features = time_features
        self.freq_base = freq_base

        indices = torch.arange(time_features, dtype=torch.float32)
        omega = freq_base ** (-indices / max(time_features, 1))
        self.register_buffer("omega", omega, persistent=False)
        self.out_dim = 2 * time_features

    def forward(self, t_norm: Tensor, num_nodes: int) -> Tensor:
        """
        Encode normalized time and broadcast it to all nodes.

        Args:
            t_norm (Tensor): Normalized time indices. (B,).
            num_nodes (int): Number of mesh nodes.

        Returns:
            Tensor: Time features. (B, N, C_TIME).
        """
        t_scaled = t_norm.float() * self.freq_base
        angles = self.omega.unsqueeze(0) * t_scaled.unsqueeze(1)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embedding.unsqueeze(1).expand(-1, num_nodes, -1)


class NodeStem(nn.Module):
    """
    Node embedding stem with compact coordinate, time, and boundary-aware geometry features.
    """

    def __init__(
        self,
        in_channels: int,
        spatial_dim: int,
        width: int,
        latent_dim: int,
        time_features: int = 4,
        freq_base: int = 1000,
    ) -> None:
        """
        Initialize the node stem.

        Args:
            in_channels (int): Number of node input channels.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Node token width.
            latent_dim (int): Geometry token width.
            time_features (int): Number of temporal sinusoidal frequency pairs.
            freq_base (int): Base for temporal frequencies.
        """
        super().__init__()
        self.in_channels = in_channels
        self.spatial_dim = spatial_dim
        self.width = width
        self.latent_dim = latent_dim

        unit_directions = self._build_unit_directions(spatial_dim)
        self.register_buffer("unit_directions", unit_directions, persistent=False)

        self.time_encoder = TemporalEncoder(
            time_features=time_features, freq_base=freq_base) if time_features > 0 else None
        self.geometry_dim = 3 * spatial_dim + unit_directions.shape[0] + 2
        self.time_dim = 0 if self.time_encoder is None else self.time_encoder.out_dim

        self.node_proj = nn.Linear(in_channels + self.geometry_dim + self.time_dim, width)
        self.node_norm = nn.LayerNorm(width)
        self.geometry_proj = nn.Linear(self.geometry_dim, latent_dim)
        self.geometry_norm = nn.LayerNorm(latent_dim)

    @staticmethod
    def _build_unit_directions(spatial_dim: int) -> Tensor:
        if spatial_dim == 2:
            angles = torch.linspace(0.0, 2.0 * pi, steps=9, dtype=torch.float32)[:-1]
            return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        phi = 0.5 * (1.0 + sqrt(5.0))
        directions = torch.tensor([
            [0.0, 1.0, phi],
            [0.0, 1.0, -phi],
            [0.0, -1.0, phi],
            [0.0, -1.0, -phi],
            [1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [-1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
            [-phi, 0.0, -1.0],
        ], dtype=torch.float32)
        return F.normalize(directions, dim=-1)

    def _boundary_geometry(self, coords: Tensor) -> Tensor:
        centered_coords = coords - coords.mean(dim=1, keepdim=True)
        radial_distance = torch.sqrt(centered_coords.square().sum(dim=-1, keepdim=True).clamp_min(1e-12))

        projections = torch.einsum("bnd,kd->bnk", coords, self.unit_directions)
        proj_max = projections.amax(dim=1, keepdim=True)
        proj_min = projections.amin(dim=1, keepdim=True)
        support_span = (proj_max - proj_min).clamp_min(1e-6)

        dist_max = (proj_max - projections).clamp_min(0.0) / support_span
        dist_min = (projections - proj_min).clamp_min(0.0) / support_span
        directional_distance = torch.minimum(dist_max, dist_min)

        pair_distance = torch.cat([dist_max, dist_min], dim=-1)
        nearest_index = pair_distance.argmin(dim=-1)
        num_dirs = self.unit_directions.shape[0]
        dir_index = nearest_index.remainder(num_dirs)
        gathered_dirs = self.unit_directions.index_select(0, dir_index.reshape(-1))
        boundary_normals = gathered_dirs.view(*dir_index.shape, self.spatial_dim)

        sign = coords.new_full((*dir_index.shape, 1), -1.0)
        sign = torch.where((nearest_index < num_dirs).unsqueeze(-1), -sign, sign)
        boundary_normals = sign * boundary_normals

        nearest_distance = pair_distance.amin(dim=-1, keepdim=True)
        boundary_proximity = torch.exp(-4.0 * nearest_distance)

        return torch.cat(
            [coords, centered_coords, radial_distance, directional_distance, boundary_proximity, boundary_normals],
            dim=-1,
        )

    def forward(self, inputs: Tensor, coords: Tensor, t_norm: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Build node tokens and geometry tokens.

        Args:
            inputs (Tensor): Input node features. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            t_norm (Optional[Tensor]): Normalized rollout time. (B,).

        Returns:
            Tuple[Tensor, Tensor]: Node tokens. (B, N, C). Geometry tokens. (B, N, C_LATENT).
        """
        B, N, _ = coords.shape
        hidden_dtype = self.node_proj.weight.dtype

        inputs = inputs.to(dtype=hidden_dtype)
        coords = coords.to(dtype=hidden_dtype)
        geometry_features = self._boundary_geometry(coords)

        if self.time_encoder is None:
            time_features = None
        else:
            if t_norm is None:
                t_norm = torch.zeros(B, device=coords.device, dtype=hidden_dtype)
            else:
                t_norm = t_norm.to(device=coords.device, dtype=hidden_dtype)
            time_features = self.time_encoder(t_norm, N).to(dtype=hidden_dtype)

        components = [inputs, geometry_features]
        if time_features is not None:
            components.append(time_features)

        node_tokens = self.node_norm(self.node_proj(torch.cat(components, dim=-1)))
        geometry_tokens = self.geometry_norm(self.geometry_proj(geometry_features))
        return node_tokens, geometry_tokens


# ============================================================
# Slice Blocks
# ============================================================


class SliceWriter(nn.Module):
    """
    Shared-basis soft slice writer for node tokens.
    """

    def __init__(self, width: int, latent_dim: int, num_slices: int) -> None:
        """
        Initialize the slice writer.

        Args:
            width (int): Node token width.
            latent_dim (int): Shared bottleneck width.
            num_slices (int): Number of soft slices.
        """
        super().__init__()
        self.width = width
        self.latent_dim = latent_dim
        self.num_slices = num_slices

        self.shared_proj = nn.Linear(width, latent_dim)
        self.feature_proj = nn.Linear(latent_dim, width)
        self.assignment_proj = nn.Linear(2 * latent_dim, num_slices)

        nn.init.orthogonal_(self.assignment_proj.weight)

    def forward(self, node_tokens: Tensor, geometry_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Aggregate node tokens into shared-assignment slice tokens.

        Args:
            node_tokens (Tensor): Node tokens. (B, N, C).
            geometry_tokens (Tensor): Geometry tokens. (B, N, C_LATENT).

        Returns:
            Tuple[Tensor, Tensor]: Slice tokens. (B, S, C). Slice weights. (B, N, S).
        """
        shared_tokens = self.shared_proj(node_tokens)
        feature_tokens = self.feature_proj(shared_tokens)
        slice_logits = self.assignment_proj(torch.cat([shared_tokens, geometry_tokens], dim=-1))
        slice_weights = torch.softmax(slice_logits, dim=-1)

        slice_tokens = torch.einsum("bns,bnc->bsc", slice_weights, feature_tokens)
        slice_norm = slice_weights.sum(dim=1, keepdim=False).unsqueeze(-1).clamp_min(1e-6)
        slice_tokens = slice_tokens / slice_norm
        return slice_tokens, slice_weights


class LatentTransition(nn.Module):
    """
    Anchor-coupled latent state transition for slice tokens.
    """

    def __init__(self, width: int, latent_dim: int, num_anchors: int) -> None:
        """
        Initialize the latent transition module.

        Args:
            width (int): Slice token width.
            latent_dim (int): Latent state width.
            num_anchors (int): Number of anchor states.
        """
        super().__init__()
        self.width = width
        self.latent_dim = latent_dim
        self.num_anchors = num_anchors

        self.down_proj = nn.Linear(width, latent_dim)
        self.down_norm = nn.LayerNorm(latent_dim)
        self.anchor_proj = nn.Linear(latent_dim, num_anchors)
        self.self_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.context_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.out_norm = nn.LayerNorm(latent_dim)
        self.up_proj = nn.Linear(latent_dim, width)

    def forward(self, slice_tokens: Tensor) -> Tensor:
        """
        Update slice tokens through low-rank latent dynamics.

        Args:
            slice_tokens (Tensor): Slice tokens. (B, S, C).

        Returns:
            Tensor: Updated slice tokens. (B, S, C).
        """
        latent_states = self.down_norm(self.down_proj(slice_tokens))
        anchor_logits = self.anchor_proj(latent_states)
        anchor_weights = torch.softmax(anchor_logits, dim=-1)

        anchor_states = torch.einsum("bsa,bsd->bad", anchor_weights, latent_states)
        anchor_norm = anchor_weights.sum(dim=1, keepdim=False).unsqueeze(-1).clamp_min(1e-6)
        anchor_states = anchor_states / anchor_norm

        latent_context = torch.einsum("bsa,bad->bsd", anchor_weights, anchor_states)
        updated_latent_states = latent_states + F.gelu(
            self.self_proj(latent_states) + self.context_proj(latent_context)
        )
        return self.up_proj(self.out_norm(updated_latent_states))


class SliceReader(nn.Module):
    """
    Shared-assignment slice reader for mapping slice states back to nodes.
    """

    def forward(self, slice_tokens: Tensor, slice_weights: Tensor) -> Tensor:
        """
        Read updated slice states back to node tokens.

        Args:
            slice_tokens (Tensor): Slice tokens. (B, S, C).
            slice_weights (Tensor): Slice weights. (B, N, S).

        Returns:
            Tensor: Node updates. (B, N, C).
        """
        return torch.einsum("bns,bsc->bnc", slice_weights, slice_tokens)


class ChannelMixer(nn.Module):
    """
    Low-rank GLU channel mixer for node tokens.
    """

    def __init__(self, width: int, latent_dim: int) -> None:
        """
        Initialize the channel mixer.

        Args:
            width (int): Node token width.
            latent_dim (int): Low-rank bottleneck width.
        """
        super().__init__()
        self.width = width
        self.latent_dim = latent_dim

        self.in_proj = nn.Linear(width, 2 * latent_dim)
        self.out_proj = nn.Linear(latent_dim, width)

    def forward(self, node_tokens: Tensor) -> Tensor:
        """
        Mix token channels through a low-rank GLU.

        Args:
            node_tokens (Tensor): Node tokens. (B, N, C).

        Returns:
            Tensor: Mixed node tokens. (B, N, C).
        """
        value, gate = self.in_proj(node_tokens).chunk(2, dim=-1)
        return self.out_proj(F.gelu(value) * torch.sigmoid(gate))


class HyperFlowBlock(nn.Module):
    """
    Shared recurrent slice block for irregular mesh dynamics.
    """

    def __init__(self, width: int, latent_dim: int, num_slices: int, num_anchors: int) -> None:
        """
        Initialize the shared recurrent block.

        Args:
            width (int): Node token width.
            latent_dim (int): Latent bottleneck width.
            num_slices (int): Number of soft slices.
            num_anchors (int): Number of anchor states.
        """
        super().__init__()
        self.slice_norm = nn.LayerNorm(width)
        self.slice_writer = SliceWriter(width, latent_dim, num_slices)
        self.latent_transition = LatentTransition(width, latent_dim, num_anchors)
        self.slice_reader = SliceReader()
        self.channel_norm = nn.LayerNorm(width)
        self.channel_mixer = ChannelMixer(width, latent_dim)

    def forward(self, node_tokens: Tensor, geometry_tokens: Tensor) -> Tensor:
        """
        Refine node tokens through one shared recurrent update.

        Args:
            node_tokens (Tensor): Node tokens. (B, N, C).
            geometry_tokens (Tensor): Geometry tokens. (B, N, C_LATENT).

        Returns:
            Tensor: Refined node tokens. (B, N, C).
        """
        slice_tokens, slice_weights = self.slice_writer(self.slice_norm(node_tokens), geometry_tokens)
        node_tokens = node_tokens + self.slice_reader(self.latent_transition(slice_tokens), slice_weights)
        node_tokens = node_tokens + self.channel_mixer(self.channel_norm(node_tokens))
        return node_tokens


# ============================================================
# HyperFlowNet
# ============================================================


class HyperFlowNet(nn.Module):
    """
    Recurrent slice-state operator for autoregressive flow prediction on irregular meshes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        width: int = 128,
        depth: int = 6,
        num_slices: int = 24,
        latent_dim: int = 32,
        num_anchors: int = 8,
        time_features: int = 4,
        freq_base: int = 1000,
    ) -> None:
        """
        Initialize the HyperFlowNet architecture.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Node token width.
            depth (int): Number of recurrent refinement steps.
            num_slices (int): Number of soft slices.
            latent_dim (int): Latent state width.
            num_anchors (int): Number of anchor states.
            time_features (int): Number of temporal sinusoidal frequency pairs.
            freq_base (int): Base for temporal frequencies.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.width = width
        self.depth = depth
        self.num_slices = num_slices
        self.latent_dim = latent_dim
        self.num_anchors = num_anchors

        self.node_stem = NodeStem(
            in_channels=in_channels,
            spatial_dim=spatial_dim,
            width=width,
            latent_dim=latent_dim,
            time_features=time_features,
            freq_base=freq_base,
        )
        self.block = HyperFlowBlock(width, latent_dim, num_slices, num_anchors)
        self.output_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, out_channels)

    def forward(self, inputs: Tensor, coords: Tensor, t_norm: Optional[Tensor] = None) -> Tensor:
        """
        Predict the next state from the current flow field.

        Args:
            inputs (Tensor): Current node features. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            t_norm (Optional[Tensor]): Normalized rollout time. (B,).

        Returns:
            Tensor: Predicted next state. (B, N, C_OUT).
        """
        node_tokens, geometry_tokens = self.node_stem(inputs, coords, t_norm=t_norm)

        for _ in range(self.depth):
            node_tokens = self.block(node_tokens, geometry_tokens)

        return self.output_head(self.output_norm(node_tokens))

    def predict(
        self,
        inputs: Tensor,
        coords: Tensor,
        steps: int,
        t0_norm: Optional[Tensor] = None,
        dt_norm: Optional[Tensor] = None,
        boundary_condition=None,
    ) -> Tensor:
        """
        Run autoregressive rollout for inference.

        Args:
            inputs (Tensor): Initial rollout state. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            steps (int): Number of rollout steps.
            t0_norm (Optional[Tensor]): Initial normalized time index. (B,).
            dt_norm (Optional[Tensor]): Normalized time increment per rollout step. (B,).
            boundary_condition: Optional boundary-condition object exposing an `enforce` method.

        Returns:
            Tensor: Rollout sequence including the initial state. (B, T + 1, N, C_OUT).
        """
        device = next(self.parameters()).device
        input_state = inputs.to(device)
        coords = coords.to(device)

        if t0_norm is None:
            t0_norm = torch.zeros(input_state.shape[0], device=device, dtype=input_state.dtype)
        else:
            t0_norm = t0_norm.to(device=device, dtype=input_state.dtype)

        if dt_norm is None:
            dt_norm = torch.full((input_state.shape[0],), 1.0 / max(steps, 1), device=device, dtype=input_state.dtype)
        else:
            dt_norm = dt_norm.to(device=device, dtype=input_state.dtype)

        preds: List[Tensor] = [inputs.cpu()]

        with torch.no_grad():
            for step_idx in tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True):
                step_t_norm = t0_norm + step_idx * dt_norm
                next_state = self(input_state, coords, t_norm=step_t_norm)

                if boundary_condition is not None:
                    next_state = boundary_condition.enforce(next_state)

                preds.append(next_state.cpu())
                input_state = next_state

        return torch.stack(preds, dim=1)

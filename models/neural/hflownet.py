# HyperFlowNet for spatio-temporal irregular-mesh flow prediction
# Author: Shengning Wang

from typing import List, Optional

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


# ============================================================
# Encoding Blocks
# ============================================================


class SpatialEncoder(nn.Module):
    """
    Hybrid spatial encoder for irregular mesh coordinates.
    """

    def __init__(
        self,
        spatial_dim: int,
        num_fixed_bands: int = 8,
        num_learned_features: int = 8,
        learned_scale: float = 1.0,
    ):
        """
        Initialize the spatial encoder.

        Args:
            spatial_dim (int): Spatial coordinate dimension.
            num_fixed_bands (int): Number of fixed Fourier bands per coordinate axis.
            num_learned_features (int): Number of learnable Fourier projections.
            learned_scale (float): Initialization scale of the learnable frequency matrix.
        """
        super().__init__()

        self.spatial_dim = spatial_dim
        self.num_fixed_bands = num_fixed_bands
        self.num_learned_features = num_learned_features

        bands = 2.0 ** torch.arange(num_fixed_bands, dtype=torch.float32)
        self.register_buffer("bands", bands, persistent=False)
        self.learned_freq_matrix = nn.Parameter(learned_scale * torch.randn(spatial_dim, num_learned_features))

        out_dim = spatial_dim
        out_dim += 4 if spatial_dim == 2 else 7
        out_dim += 2 * spatial_dim * num_fixed_bands
        out_dim += 2 * num_learned_features
        self.out_dim = out_dim

    def forward(self, coords: Tensor) -> Tensor:
        """
        Encode mesh coordinates into multi-scale spatial features.

        Args:
            coords (Tensor): Node coordinates. (B, N, D).

        Returns:
            Tensor: Encoded spatial features. (B, N, C).
        """
        coords = coords.to(dtype=self.learned_freq_matrix.dtype)
        features = [coords]

        if self.spatial_dim == 2:
            x = coords[..., 0:1]
            y = coords[..., 1:2]
            r = torch.sqrt((x * x + y * y).clamp_min(1e-12))
            features.extend([x * x, y * y, x * y, r])
        elif self.spatial_dim == 3:
            x = coords[..., 0:1]
            y = coords[..., 1:2]
            z = coords[..., 2:3]
            r = torch.sqrt((x * x + y * y + z * z).clamp_min(1e-12))
            features.extend([x * x, y * y, z * z, x * y, y * z, x * z, r])

        if self.num_fixed_bands > 0:
            fixed_proj = (2.0 * torch.pi) * coords.unsqueeze(-1) * self.bands
            features.extend([
                torch.sin(fixed_proj).flatten(start_dim=-2),
                torch.cos(fixed_proj).flatten(start_dim=-2),
            ])

        if self.num_learned_features > 0:
            learned_proj = (2.0 * torch.pi) * (coords @ self.learned_freq_matrix)
            features.extend([torch.sin(learned_proj), torch.cos(learned_proj)])

        return torch.cat(features, dim=-1)


class TemporalEncoder(nn.Module):
    """
    Sinusoidal temporal encoder for normalized rollout time.
    """

    def __init__(self, time_features: int = 4, freq_base: int = 1000):
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
            Tensor: Time features. (B, N, C).
        """
        t_scaled = t_norm.float() * self.freq_base
        angles = self.omega.unsqueeze(0) * t_scaled.unsqueeze(1)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embedding.unsqueeze(1).expand(-1, num_nodes, -1)


# ============================================================
# Token Blocks
# ============================================================


class SliceAttention(nn.Module):
    """
    Slice linear attention for irregular mesh node tokens.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_slices: int):
        """
        Initialize the mesh slice attention module.

        Args:
            hidden_dim (int): Hidden token width.
            num_heads (int): Number of attention heads.
            num_slices (int): Number of slice tokens used to compress the node set.
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_slices = num_slices
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)
        self.assignment_proj = nn.Linear(hidden_dim, hidden_dim)
        self.feature_proj = nn.Linear(hidden_dim, hidden_dim)
        self.slice_proj = nn.Linear(self.head_dim, num_slices)
        nn.init.orthogonal_(self.slice_proj.weight)

        self.query_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, num_nodes, _ = x.shape
        x = x.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply slice attention with linear complexity in the number of nodes.

        Args:
            x (Tensor): Input node tokens. (B, N, H).

        Returns:
            Tensor: Updated node tokens. (B, N, H).
        """
        batch_size, num_nodes, _ = x.shape

        feature_tokens = self._reshape_heads(self.feature_proj(x))
        assignment_tokens = self._reshape_heads(self.assignment_proj(x))

        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weights = torch.softmax(self.slice_proj(assignment_tokens) / temperature, dim=-1)

        slice_norm = slice_weights.sum(dim=2)
        slice_tokens = torch.einsum("bhnc,bhng->bhgc", feature_tokens, slice_weights)
        slice_tokens = slice_tokens / (slice_norm.unsqueeze(-1) + 1e-6)

        query = self.query_proj(slice_tokens)
        key = self.key_proj(slice_tokens)
        value = self.value_proj(slice_tokens)

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out_slice = torch.matmul(attn_weights, value)

        out = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.hidden_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Token-wise feed-forward network used inside HyperFlow blocks.
    """

    def __init__(self, hidden_dim: int, ffn_ratio: float = 4.0):
        """
        Initialize the feed-forward network.

        Args:
            hidden_dim (int): Hidden token width.
            ffn_ratio (float): Expansion ratio of the intermediate hidden layer.
        """
        super().__init__()
        inner_dim = max(1, int(round(hidden_dim * ffn_ratio)))
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the feed-forward network on token features.

        Args:
            x (Tensor): Input tokens. (B, L, H).

        Returns:
            Tensor: Output tokens. (B, L, H).
        """
        return self.net(x)


class HyperFlowBlock(nn.Module):
    """
    Pre-norm slice-attention block for irregular mesh dynamics.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_slices: int, ffn_ratio: float):
        """
        Initialize one HyperFlow block.

        Args:
            hidden_dim (int): Hidden token width.
            num_heads (int): Number of attention heads.
            num_slices (int): Number of slice tokens.
            ffn_ratio (float): Expansion ratio of the feed-forward network.
        """
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = SliceAttention(hidden_dim, num_heads, num_slices)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_ratio=ffn_ratio)

    def forward(self, node_tokens: Tensor) -> Tensor:
        """
        Update node tokens with slice attention and feed-forward residual blocks.

        Args:
            node_tokens (Tensor): Node tokens. (B, N, H).

        Returns:
            Tensor: Updated node tokens. (B, N, H).
        """
        node_tokens = node_tokens + self.attn(self.attn_norm(node_tokens))
        node_tokens = node_tokens + self.ffn(self.ffn_norm(node_tokens))
        return node_tokens


# ============================================================
# HyperFlowNet
# ============================================================


class HyperFlowNet(nn.Module):
    """
    Spatio-temporal linear-attention operator for autoregressive flow prediction on irregular meshes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        width: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        num_slices: int = 64,
        ffn_ratio: float = 4.0,
        use_spatial_encoding: bool = True,
        use_temporal_encoding: bool = True,
        num_fixed_bands: int = 8,
        num_learned_features: int = 8,
        time_features: int = 4,
        freq_base: int = 1000,
        predict_delta: bool = False,
        delta_scale: float = 1.0,
    ):
        """
        Initialize the HyperFlowNet architecture.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Hidden token width.
            depth (int): Number of stacked HyperFlow blocks.
            num_heads (int): Number of attention heads.
            num_slices (int): Number of slice tokens used by the linear attention operator.
            ffn_ratio (float): Expansion ratio of the feed-forward network.
            use_spatial_encoding (bool): Whether to encode coordinates before concatenation.
            use_temporal_encoding (bool): Whether to append a sinusoidal time embedding.
            num_fixed_bands (int): Number of fixed Fourier bands per coordinate axis.
            num_learned_features (int): Number of learnable Fourier projections.
            time_features (int): Number of temporal sinusoidal frequency pairs.
            freq_base (int): Reference time scale used by the temporal encoder.
            predict_delta (bool): Whether to predict a residual update on top of the input state.
            delta_scale (float): Scaling factor applied to the residual update when predict_delta is enabled.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.width = width
        self.num_slices = num_slices
        self.predict_delta = predict_delta
        self.delta_scale = delta_scale

        if use_spatial_encoding:
            self.spatial_encoder = SpatialEncoder(
                spatial_dim=spatial_dim,
                num_fixed_bands=num_fixed_bands,
                num_learned_features=num_learned_features,
            )
            spatial_width = self.spatial_encoder.out_dim
        else:
            self.spatial_encoder = None
            spatial_width = spatial_dim

        if use_temporal_encoding and time_features > 0:
            self.time_encoder = TemporalEncoder(time_features=time_features, freq_base=freq_base)
            time_width = self.time_encoder.out_dim
        else:
            self.time_encoder = None
            time_width = 0

        embed_in = in_channels + spatial_width + time_width
        self.input_embed = nn.Linear(embed_in, width)
        self.input_norm = nn.LayerNorm(width)

        self.blocks = nn.ModuleList([
            HyperFlowBlock(width, num_heads, self.num_slices, ffn_ratio=ffn_ratio)
            for _ in range(depth)
        ])

        self.output_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, out_channels)

    def _forward_step(self, inputs: Tensor, coords: Tensor, t_norm: Tensor) -> Tensor:
        _, num_nodes, _ = inputs.shape
        hidden_dtype = self.input_embed.weight.dtype

        inputs = inputs.to(dtype=hidden_dtype)
        coords = coords.to(dtype=hidden_dtype)

        components = [inputs]
        if self.spatial_encoder is not None:
            components.append(self.spatial_encoder(coords).to(dtype=hidden_dtype))
        else:
            components.append(coords)

        if self.time_encoder is not None:
            t_norm = t_norm.to(device=coords.device, dtype=hidden_dtype)
            components.append(self.time_encoder(t_norm, num_nodes).to(dtype=hidden_dtype))

        node_inputs = torch.cat(components, dim=-1)
        node_tokens = self.input_norm(self.input_embed(node_inputs))

        for block in self.blocks:
            node_tokens = block(node_tokens)

        output = self.output_head(self.output_norm(node_tokens))
        if self.predict_delta:
            return inputs + self.delta_scale * output
        return output

    def forward(
        self,
        inputs: Tensor,
        coords: Tensor,
        t_norm: Tensor,
        dt_norm: Tensor,
        targets: Tensor,
        teacher_forcing_ratio: float = 0.0,
        noise_std: float = 0.0,
        boundary_condition=None,
    ) -> Tensor:
        """
        Run autoregressive rollout for training.

        Args:
            inputs (Tensor): Input node features. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            t_norm (Tensor): Starting normalized time indices. (B,).
            dt_norm (Tensor): Normalized time increment per rollout step. (B,).
            targets (Tensor): Training rollout targets. (B, T, N, C_OUT).
            teacher_forcing_ratio (float): Probability of feeding ground-truth states during rollout mode.
            noise_std (float): Gaussian noise standard deviation injected into the rollout input during training.
            boundary_condition: Optional boundary-condition object exposing an `enforce` method.

        Returns:
            Tensor: Rollout preds. (B, T, N, C_OUT).
        """
        input_state = inputs
        batch_size = input_state.shape[0]
        num_steps = targets.shape[1]
        base_time = t_norm.to(device=input_state.device, dtype=input_state.dtype)
        step_dt_norm = dt_norm.to(device=input_state.device, dtype=input_state.dtype)
        preds = []

        for step_idx in range(num_steps):
            step_t_norm = base_time + step_idx * step_dt_norm

            if self.training and noise_std > 0.0:
                input_state = input_state + noise_std * torch.randn_like(input_state)

            next_state = self._forward_step(input_state, coords, step_t_norm)

            if boundary_condition is not None:
                next_state = boundary_condition.enforce(next_state)

            preds.append(next_state)

            if step_idx < num_steps - 1:
                if teacher_forcing_ratio >= 1.0:
                    input_state = targets[:, step_idx]
                elif teacher_forcing_ratio <= 0.0:
                    input_state = next_state
                else:
                    use_truth = (torch.rand(batch_size, device=next_state.device) < teacher_forcing_ratio)
                    use_truth = use_truth.to(dtype=next_state.dtype).view(-1, 1, 1)
                    input_state = use_truth * targets[:, step_idx] + (1.0 - use_truth) * next_state

        return torch.stack(preds, dim=1)

    def predict(self, inputs: Tensor, coords: Tensor, steps: int, boundary_condition=None) -> Tensor:
        """
        Run autoregressive rollout for inference.

        Args:
            inputs (Tensor): Initial rollout state. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            steps (int): Number of rollout steps.
            boundary_condition: Optional boundary-condition object exposing an `enforce` method.

        Returns:
            Tensor: Rollout sequence including the initial state. (B, T + 1, N, C_OUT).
        """
        device = next(self.parameters()).device
        input_state = inputs.to(device)
        coords = coords.to(device)
        preds: List[Tensor] = [inputs.cpu()]

        with torch.no_grad():
            for step_idx in tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True):
                step_t_norm = torch.full((input_state.shape[0],), step_idx / max(steps, 1),
                    device=device, dtype=input_state.dtype)

                next_state = self._forward_step(input_state, coords, step_t_norm)

                if boundary_condition is not None:
                    next_state = boundary_condition.enforce(next_state)

                preds.append(next_state.cpu())

                input_state = next_state

        return torch.stack(preds, dim=1)

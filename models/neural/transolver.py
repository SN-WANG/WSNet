# Transolver: Physics-Attention Neural Operator
# Author: Shengning Wang

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


def _build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in ACTIVATIONS:
        raise NotImplementedError(f"Unsupported activation: {name}")
    factory = ACTIVATIONS[key]
    return factory() if isinstance(factory, type) else factory()


def _trunc_normal_(tensor: Tensor, std: float = 0.02) -> Tensor:
    with torch.no_grad():
        tensor.normal_(0.0, std)
        while True:
            mask = tensor.abs() > 2 * std
            if not mask.any():
                break
            tensor[mask] = torch.empty_like(tensor[mask]).normal_(0.0, std)
    return tensor


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Build sinusoidal timestep embeddings.

    Args:
        timesteps (Tensor): Time indices. (B,).
        dim (int): Embedding dimension.
        max_period (int): Minimum-frequency control.

    Returns:
        Tensor: Time embeddings. (B, C).
    """
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / max(half_dim, 1)
    )
    angles = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ============================================================
# Token Blocks
# ============================================================


class TransolverMLP(nn.Module):
    """
    Token-wise MLP used by the original Transolver blocks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
        act: str = "gelu",
        res: bool = True,
    ):
        """
        Initialize the token-wise MLP.

        Args:
            in_channels (int): Input token width.
            hidden_channels (int): Hidden token width.
            out_channels (int): Output token width.
            num_layers (int): Number of hidden residual layers after the input lift.
            act (str): Activation name.
            res (bool): Whether to use residual hidden updates.
        """
        super().__init__()
        self.num_layers = num_layers
        self.res = res
        self.input_proj = nn.Sequential(nn.Linear(in_channels, hidden_channels), _build_activation(act))
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_channels, hidden_channels), _build_activation(act))
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the token-wise MLP.

        Args:
            x (Tensor): Input tokens. (B, N, C_IN).

        Returns:
            Tensor: Output tokens. (B, N, C_OUT).
        """
        x = self.input_proj(x)
        for layer in self.hidden_layers:
            x = layer(x) + x if self.res else layer(x)
        return self.output_proj(x)


class _PhysicsAttentionBase(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int,
        dropout: float,
        num_slices: int,
        clamp_temperature: bool,
    ):
        super().__init__()
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.clamp_temperature = clamp_temperature

        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)
        self.slice_proj = nn.Linear(dim_head, num_slices)
        nn.init.orthogonal_(self.slice_proj.weight)

        self.query_proj = nn.Linear(dim_head, dim_head, bias=False)
        self.key_proj = nn.Linear(dim_head, dim_head, bias=False)
        self.value_proj = nn.Linear(dim_head, dim_head, bias=False)
        self.out_proj = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, num_nodes, _ = x.shape
        x = x.reshape(batch_size, num_nodes, self.num_heads, self.dim_head)
        return x.permute(0, 2, 1, 3).contiguous()

    def _apply_slice_attention(self, feature_tokens: Tensor, assignment_tokens: Tensor) -> Tensor:
        temperature = self.temperature
        if self.clamp_temperature:
            temperature = torch.clamp(temperature, min=0.1, max=5.0)

        slice_weights = torch.softmax(self.slice_proj(assignment_tokens) / temperature, dim=-1)
        slice_norm = slice_weights.sum(dim=2)
        slice_tokens = torch.einsum("bhnc,bhng->bhgc", feature_tokens, slice_weights)
        slice_tokens = slice_tokens / (slice_norm[..., None] + 1e-5)

        query = self.query_proj(slice_tokens)
        key = self.key_proj(slice_tokens)
        value = self.value_proj(slice_tokens)

        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))
        out_slice_tokens = torch.matmul(attn_weights, value)

        batch_size, _, num_nodes, _ = feature_tokens.shape
        out = torch.einsum("bhgc,bhng->bhnc", out_slice_tokens, slice_weights)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        return self.out_proj(out)


# ============================================================
# Physics Attention
# ============================================================


class PhysicsAttentionIrregularMesh(_PhysicsAttentionBase):
    """
    Physics Attention for irregular meshes in 1D, 2D, or 3D.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        num_slices: int = 64,
    ):
        """
        Initialize irregular-mesh physics attention.

        Args:
            dim (int): Input token width.
            num_heads (int): Number of attention heads.
            dim_head (int): Hidden width per head.
            dropout (float): Dropout rate on slice attention outputs.
            num_slices (int): Number of learned physical states.
        """
        super().__init__(dim, num_heads, dim_head, dropout, num_slices, clamp_temperature=False)
        inner_dim = num_heads * dim_head
        self.assignment_proj = nn.Linear(dim, inner_dim)
        self.feature_proj = nn.Linear(dim, inner_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply physics attention on irregular mesh tokens.

        Args:
            x (Tensor): Input tokens. (B, N, C).

        Returns:
            Tensor: Updated tokens. (B, N, C).
        """
        feature_tokens = self._reshape_heads(self.feature_proj(x))
        assignment_tokens = self._reshape_heads(self.assignment_proj(x))
        return self._apply_slice_attention(feature_tokens, assignment_tokens)


class PhysicsAttentionStructuredMesh2D(_PhysicsAttentionBase):
    """
    Physics Attention for structured 2D meshes.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        num_slices: int = 64,
        num_rows: int = 85,
        num_cols: int = 85,
        kernel_size: int = 3,
    ):
        """
        Initialize structured 2D physics attention.

        Args:
            dim (int): Input token width.
            num_heads (int): Number of attention heads.
            dim_head (int): Hidden width per head.
            dropout (float): Dropout rate on slice attention outputs.
            num_slices (int): Number of learned physical states.
            num_rows (int): Grid height.
            num_cols (int): Grid width.
            kernel_size (int): Spatial projection kernel size.
        """
        super().__init__(dim, num_heads, dim_head, dropout, num_slices, clamp_temperature=True)
        inner_dim = num_heads * dim_head
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.assignment_proj = nn.Conv2d(dim, inner_dim, kernel_size, 1, kernel_size // 2)
        self.feature_proj = nn.Conv2d(dim, inner_dim, kernel_size, 1, kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply physics attention on structured 2D mesh tokens.

        Args:
            x (Tensor): Input tokens. (B, H * W, C).

        Returns:
            Tensor: Updated tokens. (B, H * W, C).
        """
        batch_size, _, num_channels = x.shape
        x = x.reshape(batch_size, self.num_rows, self.num_cols, num_channels).permute(0, 3, 1, 2).contiguous()

        feature_tokens = self.feature_proj(x).permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_heads * self.dim_head)
        assignment_tokens = self.assignment_proj(x).permute(0, 2, 3, 1).reshape(
            batch_size, -1, self.num_heads * self.dim_head
        )
        return self._apply_slice_attention(self._reshape_heads(feature_tokens), self._reshape_heads(assignment_tokens))


class PhysicsAttentionStructuredMesh3D(_PhysicsAttentionBase):
    """
    Physics Attention for structured 3D meshes.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        num_slices: int = 32,
        num_rows: int = 32,
        num_cols: int = 32,
        num_depth: int = 32,
        kernel_size: int = 3,
    ):
        """
        Initialize structured 3D physics attention.

        Args:
            dim (int): Input token width.
            num_heads (int): Number of attention heads.
            dim_head (int): Hidden width per head.
            dropout (float): Dropout rate on slice attention outputs.
            num_slices (int): Number of learned physical states.
            num_rows (int): Grid size along the first axis.
            num_cols (int): Grid size along the second axis.
            num_depth (int): Grid size along the third axis.
            kernel_size (int): Spatial projection kernel size.
        """
        super().__init__(dim, num_heads, dim_head, dropout, num_slices, clamp_temperature=True)
        inner_dim = num_heads * dim_head
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_depth = num_depth
        self.assignment_proj = nn.Conv3d(dim, inner_dim, kernel_size, 1, kernel_size // 2)
        self.feature_proj = nn.Conv3d(dim, inner_dim, kernel_size, 1, kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply physics attention on structured 3D mesh tokens.

        Args:
            x (Tensor): Input tokens. (B, H * W * D, C).

        Returns:
            Tensor: Updated tokens. (B, H * W * D, C).
        """
        batch_size, _, num_channels = x.shape
        x = x.reshape(batch_size, self.num_rows, self.num_cols, self.num_depth, num_channels)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        feature_tokens = self.feature_proj(x).permute(0, 2, 3, 4, 1).reshape(
            batch_size, -1, self.num_heads * self.dim_head
        )
        assignment_tokens = self.assignment_proj(x).permute(0, 2, 3, 4, 1).reshape(
            batch_size, -1, self.num_heads * self.dim_head
        )
        return self._apply_slice_attention(self._reshape_heads(feature_tokens), self._reshape_heads(assignment_tokens))


# ============================================================
# Transolver Blocks
# ============================================================


class TransolverBlock(nn.Module):
    """
    Pre-norm Transolver block with Physics Attention and token MLP.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_slices: int,
        dropout: float,
        act: str = "gelu",
        mlp_ratio: int = 1,
        mesh_type: str = "irregular",
        spatial_dim: int = 2,
        mesh_size: Optional[Sequence[int]] = None,
        kernel_size: int = 3,
        last_layer: bool = False,
        out_channels: int = 1,
    ):
        """
        Initialize one Transolver block.

        Args:
            hidden_dim (int): Hidden token width.
            num_heads (int): Number of attention heads.
            num_slices (int): Number of learned physical states.
            dropout (float): Dropout rate on attention outputs.
            act (str): Activation name for the MLP.
            mlp_ratio (int): Expansion ratio in the token MLP.
            mesh_type (str): Mesh type, either `irregular` or `structured`.
            spatial_dim (int): Spatial coordinate dimension.
            mesh_size (Optional[Sequence[int]]): Structured mesh resolution.
            kernel_size (int): Projection kernel size for structured attention.
            last_layer (bool): Whether this block includes the output head.
            out_channels (int): Output token width of the final block.
        """
        super().__init__()
        dim_head = hidden_dim // num_heads

        self.attn_norm = nn.LayerNorm(hidden_dim)
        if mesh_type == "structured":
            if spatial_dim == 2:
                self.attn = PhysicsAttentionStructuredMesh2D(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    num_slices=num_slices,
                    num_rows=int(mesh_size[0]),
                    num_cols=int(mesh_size[1]),
                    kernel_size=kernel_size,
                )
            else:
                self.attn = PhysicsAttentionStructuredMesh3D(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    num_slices=num_slices,
                    num_rows=int(mesh_size[0]),
                    num_cols=int(mesh_size[1]),
                    num_depth=int(mesh_size[2]),
                    kernel_size=kernel_size,
                )
        else:
            self.attn = PhysicsAttentionIrregularMesh(
                dim=hidden_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                num_slices=num_slices,
            )

        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = TransolverMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, num_layers=0, act=act, res=False)

        self.last_layer = last_layer
        if self.last_layer:
            self.out_norm = nn.LayerNorm(hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Update Transolver tokens.

        Args:
            x (Tensor): Input tokens. (B, N, H).

        Returns:
            Tensor: Updated tokens. (B, N, H) or final predictions. (B, N, C_OUT).
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        if self.last_layer:
            return self.out_proj(self.out_norm(x))
        return x


# ============================================================
# Transolver
# ============================================================


class Transolver(nn.Module):
    """
    Transolver from the first paper, refactored in WSNet style.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        width: int = 256,
        depth: int = 5,
        num_slices: int = 32,
        num_heads: int = 8,
        mlp_ratio: int = 1,
        dropout: float = 0.0,
        act: str = "gelu",
        use_time_input: bool = False,
        unified_pos: bool = False,
        ref: int = 8,
        mesh_type: str = "irregular",
        mesh_size: Optional[Sequence[int]] = None,
        kernel_size: int = 3,
        ref_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        """
        Initialize the Transolver architecture.

        Args:
            in_channels (int): Input feature width.
            out_channels (int): Output feature width.
            spatial_dim (int): Coordinate dimension.
            width (int): Hidden token width.
            depth (int): Number of Transolver blocks.
            num_slices (int): Number of learned physical states.
            num_heads (int): Number of attention heads.
            mlp_ratio (int): Expansion ratio in the block MLP.
            dropout (float): Dropout rate on attention outputs.
            act (str): Activation name in token MLPs.
            use_time_input (bool): Whether to add sinusoidal time embeddings.
            unified_pos (bool): Whether to use Transolver unified position encoding.
            ref (int): Reference resolution per axis for unified positions.
            mesh_type (str): Mesh type, either `irregular` or `structured`.
            mesh_size (Optional[Sequence[int]]): Structured mesh resolution.
            kernel_size (int): Projection kernel size for structured attention.
            ref_bounds (Optional[Sequence[Tuple[float, float]]]): Coordinate bounds for unified positions.
        """
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")

        self.__name__ = "Transolver"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.width = width
        self.depth = depth
        self.num_slices = num_slices
        self.num_heads = num_heads
        self.use_time_input = use_time_input
        self.unified_pos = unified_pos
        self.ref = ref
        self.mesh_type = mesh_type.lower()
        self.ref_bounds = self._normalize_ref_bounds(ref_bounds)
        if self.mesh_type not in {"irregular", "structured"}:
            raise ValueError(f"Unsupported mesh_type: {mesh_type}")

        if self.mesh_type == "structured":
            if mesh_size is None:
                raise ValueError("mesh_size is required when mesh_type='structured'")
            if spatial_dim not in (2, 3):
                raise ValueError("structured Transolver only supports spatial_dim=2 or 3")
            if len(mesh_size) != spatial_dim:
                raise ValueError(f"Expected mesh_size to have {spatial_dim} entries, got {len(mesh_size)}")
            self.mesh_size = tuple(int(size) for size in mesh_size)
        else:
            self.mesh_size = None

        coord_dim = spatial_dim if not unified_pos else ref ** spatial_dim
        self.preprocess = TransolverMLP(in_channels + coord_dim, width * 2, width, num_layers=0, act=act, res=False)

        if self.use_time_input:
            self.time_fc = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        else:
            self.time_fc = None

        self.blocks = nn.ModuleList([
            TransolverBlock(
                hidden_dim=width,
                num_heads=num_heads,
                num_slices=num_slices,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                mesh_type=self.mesh_type,
                spatial_dim=spatial_dim,
                mesh_size=self.mesh_size,
                kernel_size=kernel_size,
                last_layer=(layer_idx == depth - 1),
                out_channels=out_channels,
            )
            for layer_idx in range(depth)
        ])

        self.placeholder = nn.Parameter((1.0 / width) * torch.rand(width, dtype=torch.float32))

        structured_pos = None
        if self.mesh_type == "structured" and self.unified_pos:
            structured_pos = self._build_structured_unified_pos(torch.device("cpu"), torch.float32)
        self.register_buffer("structured_pos", structured_pos, persistent=False)

        self._initialize_weights()

    def _normalize_ref_bounds(
        self,
        ref_bounds: Optional[Sequence[Tuple[float, float]]],
    ) -> Tuple[Tuple[float, float], ...]:
        if ref_bounds is None:
            return tuple((0.0, 1.0) for _ in range(self.spatial_dim))
        if len(ref_bounds) != self.spatial_dim:
            raise ValueError(f"Expected {self.spatial_dim} coordinate bounds, got {len(ref_bounds)}")
        return tuple((float(lower), float(upper)) for lower, upper in ref_bounds)

    def _initialize_weights(self) -> None:
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def _build_reference_points(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        axes = [torch.linspace(lower, upper, self.ref, device=device, dtype=dtype) for lower, upper in self.ref_bounds]
        mesh = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(mesh, dim=-1).reshape(-1, self.spatial_dim)

    def _build_structured_unified_pos(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        axes = [
            torch.linspace(lower, upper, size, device=device, dtype=dtype)
            for size, (lower, upper) in zip(self.mesh_size, self.ref_bounds)
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        mesh_points = torch.stack(mesh, dim=-1).reshape(-1, self.spatial_dim)
        ref_points = self._build_reference_points(device, dtype)
        return torch.sqrt(torch.sum((mesh_points[:, None, :] - ref_points[None, :, :]) ** 2, dim=-1))

    def _get_coordinate_features(
        self,
        input_features: Optional[Tensor],
        physical_coords: Optional[Tensor],
    ) -> Tensor:
        reference = physical_coords if physical_coords is not None else input_features
        if reference is None:
            raise ValueError("Transolver needs physical_coords or input_features to infer batch size and device")

        batch_size = reference.shape[0]
        device = reference.device
        dtype = reference.dtype

        if not self.unified_pos:
            if physical_coords is None:
                raise ValueError("physical_coords is required when unified_pos=False")
            return physical_coords

        if self.mesh_type == "structured":
            return self.structured_pos.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

        if physical_coords is None:
            raise ValueError("physical_coords is required for irregular unified_pos encoding")

        ref_points = self._build_reference_points(device, dtype)
        return torch.sqrt(torch.sum((physical_coords[:, :, None, :] - ref_points[None, None, :, :]) ** 2, dim=-1))

    def _add_placeholder(self, hidden: Tensor, input_features: Optional[Tensor]) -> Tensor:
        if self.mesh_type == "irregular" or input_features is None:
            hidden = hidden + self.placeholder.view(1, 1, -1)
        return hidden

    def forward(
        self,
        input_features: Optional[Tensor],
        physical_coords: Optional[Tensor],
        t_norm: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply Transolver on mesh tokens.

        Args:
            input_features (Optional[Tensor]): Input node features. (B, N, C_IN).
            physical_coords (Optional[Tensor]): Node coordinates. (B, N, D).
            t_norm (Optional[Tensor]): Normalized time indices. (B,).

        Returns:
            Tensor: Output node features. (B, N, C_OUT).
        """
        coord_features = self._get_coordinate_features(input_features, physical_coords)

        if input_features is None:
            hidden = self.preprocess(coord_features)
        else:
            hidden = self.preprocess(torch.cat([coord_features, input_features], dim=-1))

        hidden = self._add_placeholder(hidden, input_features)

        if self.time_fc is not None and t_norm is not None:
            time_emb = timestep_embedding(t_norm, self.width).unsqueeze(1).expand(-1, hidden.shape[1], -1)
            hidden = hidden + self.time_fc(time_emb)

        for block in self.blocks:
            hidden = block(hidden)

        return hidden

    def predict(self, initial_state: Tensor, coords: Optional[Tensor], steps: int) -> Tensor:
        """
        Run autoregressive inference with Transolver.

        Args:
            initial_state (Tensor): Initial state. (B, N, C_IN).
            coords (Optional[Tensor]): Node coordinates. (B, N, D).
            steps (int): Number of rollout steps.

        Returns:
            Tensor: Predicted sequence including the initial state. (B, T + 1, N, C_OUT).

        Note:
            This rollout assumes the model output can be fed back as the next input state.
        """
        device = next(self.parameters()).device
        current_state = initial_state.to(device)
        coords = coords.to(device) if coords is not None else None

        sequence: List[Tensor] = [current_state.cpu()]

        with torch.no_grad():
            for step_idx in tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True):
                step_t_norm = None
                if self.time_fc is not None:
                    step_t_norm = torch.full(
                        (current_state.shape[0],),
                        step_idx / max(steps, 1),
                        device=device,
                        dtype=current_state.dtype,
                    )
                next_state = self.forward(current_state, coords, t_norm=step_t_norm)
                sequence.append(next_state.cpu())
                current_state = next_state

        return torch.stack(sequence, dim=1)

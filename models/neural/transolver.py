# Transolver: Physics-Attention Neural Operator
# Author: Shengning Wang

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


ACTIVATIONS = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'elu': nn.ELU,
    'silu': nn.SiLU,
}


def _build_activation(name: str) -> nn.Module:
    key = name.lower()
    if key not in ACTIVATIONS:
        raise NotImplementedError(f'Unsupported activation: {name}')
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
        timesteps (Tensor): Time indices with shape (batch_size,) and dtype float32-compatible.
        dim (int): Embedding dimension.
        max_period (int): Minimum frequency control used by the original implementation.

    Returns:
        Tensor: Time embeddings with shape (batch_size, dim) and dtype float32.
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
# MLP
# ============================================================


class TransolverMLP(nn.Module):
    """
    MLP used by the original Transolver blocks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 1,
        act: str = 'gelu',
        res: bool = True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(in_channels, hidden_channels), _build_activation(act))
        self.linear_post = nn.Linear(hidden_channels, out_channels)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_channels, hidden_channels), _build_activation(act))
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_pre(x)
        for layer in self.hidden_layers:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


# ============================================================
# Physics Attention
# ============================================================


class PhysicsAttentionIrregularMesh(nn.Module):
    """
    Physics Attention for irregular meshes.
    """

    def __init__(self, dim: int, num_heads: int = 8, dim_head: int = 64, dropout: float = 0.0, num_slices: int = 64):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, num_slices)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_nodes, _ = x.shape

        fx_mid = self.in_project_fx(x).reshape(batch_size, num_nodes, self.num_heads, self.dim_head)
        fx_mid = fx_mid.permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).reshape(batch_size, num_nodes, self.num_heads, self.dim_head)
        x_mid = x_mid.permute(0, 2, 1, 3).contiguous()

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(dim=2)
        slice_token = torch.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm[..., None] + 1e-5)

        q_slice = self.to_q(slice_token)
        k_slice = self.to_k(slice_token)
        v_slice = self.to_v(slice_token)
        attn = self.softmax(torch.matmul(q_slice, k_slice.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v_slice)

        out = torch.einsum('bhgc,bhng->bhnc', out_slice, slice_weights)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        return self.to_out(out)


class PhysicsAttentionStructuredMesh2D(nn.Module):
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
        super().__init__()
        inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel_size, 1, kernel_size // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel_size, 1, kernel_size // 2)
        self.in_project_slice = nn.Linear(dim_head, num_slices)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_nodes, num_channels = x.shape
        expected_nodes = self.num_rows * self.num_cols
        if num_nodes != expected_nodes:
            raise ValueError(f'Expected {expected_nodes} nodes for mesh_size=({self.num_rows}, {self.num_cols}), got {num_nodes}')

        x = x.reshape(batch_size, self.num_rows, self.num_cols, num_channels).permute(0, 3, 1, 2).contiguous()

        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).reshape(batch_size, num_nodes, self.num_heads, self.dim_head)
        fx_mid = fx_mid.permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).reshape(batch_size, num_nodes, self.num_heads, self.dim_head)
        x_mid = x_mid.permute(0, 2, 1, 3).contiguous()

        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weights = self.softmax(self.in_project_slice(x_mid) / temperature)
        slice_norm = slice_weights.sum(dim=2)
        slice_token = torch.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm[..., None] + 1e-5)

        q_slice = self.to_q(slice_token)
        k_slice = self.to_k(slice_token)
        v_slice = self.to_v(slice_token)
        attn = self.softmax(torch.matmul(q_slice, k_slice.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v_slice)

        out = torch.einsum('bhgc,bhng->bhnc', out_slice, slice_weights)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        return self.to_out(out)


class PhysicsAttentionStructuredMesh3D(nn.Module):
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
        super().__init__()
        inner_dim = dim_head * num_heads
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_depth = num_depth

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel_size, 1, kernel_size // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel_size, 1, kernel_size // 2)
        self.in_project_slice = nn.Linear(dim_head, num_slices)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_nodes, num_channels = x.shape
        expected_nodes = self.num_rows * self.num_cols * self.num_depth
        if num_nodes != expected_nodes:
            raise ValueError(
                f'Expected {expected_nodes} nodes for mesh_size=({self.num_rows}, {self.num_cols}, {self.num_depth}), '
                f'got {num_nodes}'
            )

        x = x.reshape(batch_size, self.num_rows, self.num_cols, self.num_depth, num_channels)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1)
        fx_mid = fx_mid.reshape(batch_size, num_nodes, self.num_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1)
        x_mid = x_mid.reshape(batch_size, num_nodes, self.num_heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weights = self.softmax(self.in_project_slice(x_mid) / temperature)
        slice_norm = slice_weights.sum(dim=2)
        slice_token = torch.einsum('bhnc,bhng->bhgc', fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm[..., None] + 1e-5)

        q_slice = self.to_q(slice_token)
        k_slice = self.to_k(slice_token)
        v_slice = self.to_v(slice_token)
        attn = self.softmax(torch.matmul(q_slice, k_slice.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)
        out_slice = torch.matmul(attn, v_slice)

        out = torch.einsum('bhgc,bhng->bhnc', out_slice, slice_weights)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, -1)
        return self.to_out(out)


# ============================================================
# Transolver Block
# ============================================================


class TransolverBlock(nn.Module):
    """
    One Transolver block with pre-norm attention and feedforward residuals.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        act: str = 'gelu',
        mlp_ratio: int = 4,
        num_slices: int = 32,
        mesh_type: str = 'irregular',
        spatial_dim: int = 2,
        mesh_size: Optional[Sequence[int]] = None,
        kernel_size: int = 3,
        last_layer: bool = False,
        out_channels: int = 1,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = TransolverMLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, act=act, res=False)

        if mesh_type == 'irregular':
            self.attn = PhysicsAttentionIrregularMesh(
                dim=hidden_dim,
                num_heads=num_heads,
                dim_head=hidden_dim // num_heads,
                dropout=dropout,
                num_slices=num_slices,
            )
        elif mesh_type == 'structured':
            if mesh_size is None:
                raise ValueError('mesh_size must be provided when mesh_type="structured"')
            if spatial_dim == 2:
                self.attn = PhysicsAttentionStructuredMesh2D(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    dim_head=hidden_dim // num_heads,
                    dropout=dropout,
                    num_slices=num_slices,
                    num_rows=int(mesh_size[0]),
                    num_cols=int(mesh_size[1]),
                    kernel_size=kernel_size,
                )
            elif spatial_dim == 3:
                self.attn = PhysicsAttentionStructuredMesh3D(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    dim_head=hidden_dim // num_heads,
                    dropout=dropout,
                    num_slices=num_slices,
                    num_rows=int(mesh_size[0]),
                    num_cols=int(mesh_size[1]),
                    num_depth=int(mesh_size[2]),
                    kernel_size=kernel_size,
                )
            else:
                raise ValueError(f'structured mesh only supports spatial_dim=2 or 3, got {spatial_dim}')
        else:
            raise ValueError(f'Unsupported mesh_type: {mesh_type}')

        if self.last_layer:
            self.norm3 = nn.LayerNorm(hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        if self.last_layer:
            return self.out_proj(self.norm3(x))
        return x


# ============================================================
# Transolver
# ============================================================


class Transolver(nn.Module):
    """
    WSNet-style Transolver implementation aligned with the original repository.

    Args:
        in_channels (int): Input feature dimension with shape (batch_size, num_nodes, in_channels).
        out_channels (int): Output feature dimension with shape (batch_size, num_nodes, out_channels).
        spatial_dim (int): Coordinate dimension.
        width (int): Hidden channel width.
        depth (int): Number of Transolver blocks.
        num_slices (int): Number of slice tokens in Physics Attention.
        num_heads (int): Number of attention heads.
        mlp_ratio (int): Hidden expansion ratio inside each block MLP.
        dropout (float): Dropout rate used by attention outputs.
        act (str): Activation name used by the original MLP.
        use_time_input (bool): Whether to add original timestep embeddings.
        unified_pos (bool): Whether to replace raw coordinates with unified position distances.
        ref (int): Number of reference points per spatial axis for unified_pos.
        mesh_type (str): Either 'irregular' or 'structured'.
        mesh_size (Optional[Sequence[int]]): Structured mesh resolution for 2D or 3D attention.
        kernel_size (int): Structured mesh convolution kernel size.
        ref_bounds (Optional[Sequence[Tuple[float, float]]]): Reference coordinate bounds for unified_pos.
        append_coords_when_unified (bool): If True, concatenate raw coordinates before unified_pos distances.
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
        act: str = 'gelu',
        use_time_input: bool = False,
        unified_pos: bool = False,
        ref: int = 8,
        mesh_type: str = 'irregular',
        mesh_size: Optional[Sequence[int]] = None,
        kernel_size: int = 3,
        ref_bounds: Optional[Sequence[Tuple[float, float]]] = None,
        append_coords_when_unified: bool = False,
    ):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f'width={width} must be divisible by num_heads={num_heads}')

        mesh_type = mesh_type.lower()
        if mesh_type not in {'irregular', 'structured'}:
            raise ValueError(f'Unsupported mesh_type: {mesh_type}')

        if mesh_type == 'structured':
            if mesh_size is None:
                raise ValueError('mesh_size must be provided when mesh_type="structured"')
            if spatial_dim not in (2, 3):
                raise ValueError(f'structured mesh only supports spatial_dim=2 or 3, got {spatial_dim}')
            if len(mesh_size) != spatial_dim:
                raise ValueError(f'mesh_size must have {spatial_dim} entries, got {len(mesh_size)}')
            mesh_size = tuple(int(size) for size in mesh_size)

        self.__name__ = 'Transolver'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.width = width
        self.depth = depth
        self.num_slices = num_slices
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.act = act
        self.use_time_input = use_time_input
        self.unified_pos = unified_pos
        self.ref = ref
        self.mesh_type = mesh_type
        self.mesh_size = mesh_size
        self.kernel_size = kernel_size
        self.append_coords_when_unified = append_coords_when_unified
        self.ref_bounds = self._normalize_ref_bounds(ref_bounds)
        self.num_ref_points = ref ** spatial_dim

        coord_dim = self.num_ref_points if unified_pos else spatial_dim
        if unified_pos and append_coords_when_unified:
            coord_dim += spatial_dim

        self.preprocess = TransolverMLP(
            in_channels=in_channels + coord_dim,
            hidden_channels=width * 2,
            out_channels=width,
            n_layers=0,
            act=act,
            res=False,
        )

        if use_time_input:
            self.time_fc = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        else:
            self.time_fc = None

        self.blocks = nn.ModuleList([
            TransolverBlock(
                hidden_dim=width,
                num_heads=num_heads,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                num_slices=num_slices,
                mesh_type=mesh_type,
                spatial_dim=spatial_dim,
                mesh_size=mesh_size,
                kernel_size=kernel_size,
                last_layer=(layer_index == depth - 1),
                out_channels=out_channels,
            )
            for layer_index in range(depth)
        ])

        self.placeholder = nn.Parameter((1.0 / width) * torch.rand(width, dtype=torch.float32))

        structured_pos = None
        if self.mesh_type == 'structured' and self.unified_pos:
            structured_pos = self._build_structured_unified_pos(device=torch.device('cpu'), dtype=torch.float32)
        self.register_buffer('structured_pos', structured_pos, persistent=False)

        self._initialize_weights()

    def _normalize_ref_bounds(
        self,
        ref_bounds: Optional[Sequence[Tuple[float, float]]],
    ) -> Tuple[Tuple[float, float], ...]:
        if ref_bounds is None:
            return tuple((0.0, 1.0) for _ in range(self.spatial_dim))
        if len(ref_bounds) != self.spatial_dim:
            raise ValueError(f'ref_bounds must have {self.spatial_dim} entries, got {len(ref_bounds)}')
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
        axes = [
            torch.linspace(lower, upper, self.ref, device=device, dtype=dtype)
            for lower, upper in self.ref_bounds
        ]
        mesh = torch.meshgrid(*axes, indexing='ij')
        return torch.stack(mesh, dim=-1).reshape(-1, self.spatial_dim)

    def _build_structured_unified_pos(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        axes = [
            torch.linspace(lower, upper, size, device=device, dtype=dtype)
            for size, (lower, upper) in zip(self.mesh_size, self.ref_bounds)
        ]
        mesh = torch.meshgrid(*axes, indexing='ij')
        mesh_points = torch.stack(mesh, dim=-1).reshape(-1, self.spatial_dim)
        ref_points = self._build_reference_points(device=device, dtype=dtype)
        return torch.sqrt(torch.sum((mesh_points[:, None, :] - ref_points[None, :, :]) ** 2, dim=-1))

    def _validate_inputs(self, input_features: Optional[Tensor], physical_coords: Optional[Tensor]) -> None:
        if input_features is None and self.in_channels != 0:
            raise ValueError(f'Expected input_features with last dimension {self.in_channels}, but got None')

        if input_features is not None and input_features.shape[-1] != self.in_channels:
            raise ValueError(
                f'Expected input_features.shape[-1] == {self.in_channels}, got {input_features.shape[-1]}'
            )

        if physical_coords is not None and physical_coords.shape[-1] != self.spatial_dim:
            raise ValueError(
                f'Expected physical_coords.shape[-1] == {self.spatial_dim}, got {physical_coords.shape[-1]}'
            )

        if self.mesh_type == 'structured':
            expected_nodes = math.prod(self.mesh_size)
            if input_features is not None and input_features.shape[1] != expected_nodes:
                raise ValueError(
                    f'Expected input_features.shape[1] == {expected_nodes} for mesh_size={self.mesh_size}, '
                    f'got {input_features.shape[1]}'
                )
            if physical_coords is not None and physical_coords.shape[1] != expected_nodes:
                raise ValueError(
                    f'Expected physical_coords.shape[1] == {expected_nodes} for mesh_size={self.mesh_size}, '
                    f'got {physical_coords.shape[1]}'
                )

    def _get_coordinate_features(self, input_features: Optional[Tensor], physical_coords: Optional[Tensor]) -> Tensor:
        reference = physical_coords if physical_coords is not None else input_features
        if reference is None:
            raise ValueError('At least one of input_features or physical_coords must be provided')

        batch_size = reference.shape[0]
        device = reference.device
        dtype = reference.dtype

        if not self.unified_pos:
            if physical_coords is None:
                raise ValueError('physical_coords is required when unified_pos=False')
            return physical_coords

        if self.mesh_type == 'structured':
            if self.structured_pos is None:
                raise RuntimeError('structured_pos cache was not initialized')
            coord_features = self.structured_pos.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if physical_coords is None:
                raise ValueError('physical_coords is required when mesh_type="irregular" and unified_pos=True')
            ref_points = self._build_reference_points(device=device, dtype=dtype)
            coord_features = torch.sqrt(
                torch.sum((physical_coords[:, :, None, :] - ref_points[None, None, :, :]) ** 2, dim=-1)
            )

        if self.append_coords_when_unified:
            if physical_coords is None:
                raise ValueError('physical_coords is required when append_coords_when_unified=True')
            coord_features = torch.cat([physical_coords, coord_features], dim=-1)

        return coord_features

    def forward(
        self,
        input_features: Optional[Tensor],
        physical_coords: Optional[Tensor],
        t_norm: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Forward pass on node features.

        Args:
            input_features (Optional[Tensor]): Input features with shape (batch_size, num_nodes, in_channels).
            physical_coords (Optional[Tensor]): Coordinates with shape (batch_size, num_nodes, spatial_dim).
            t_norm (Optional[Tensor]): Time indices with shape (batch_size,). Used only when use_time_input=True.

        Returns:
            Tensor: Predictions with shape (batch_size, num_nodes, out_channels).
        """
        self._validate_inputs(input_features, physical_coords)

        coord_features = self._get_coordinate_features(input_features, physical_coords)

        if input_features is None:
            hidden = self.preprocess(coord_features)
        else:
            hidden = self.preprocess(torch.cat([coord_features, input_features], dim=-1))

        hidden = hidden + self.placeholder.view(1, 1, -1)

        if self.time_fc is not None and t_norm is not None:
            time_emb = timestep_embedding(t_norm, self.width)
            time_emb = self.time_fc(time_emb).unsqueeze(1).expand(-1, hidden.shape[1], -1)
            hidden = hidden + time_emb

        for block in self.blocks:
            hidden = block(hidden)

        return hidden

    def predict(self, initial_state: Tensor, coords: Tensor, steps: int) -> Tensor:
        """
        Autoregressive rollout.

        Args:
            initial_state (Tensor): Initial state with shape (batch_size, num_nodes, in_channels).
            coords (Tensor): Coordinates with shape (batch_size, num_nodes, spatial_dim).
            steps (int): Number of rollout steps.

        Returns:
            Tensor: Sequence with shape (batch_size, steps + 1, num_nodes, out_channels).
        """
        device = next(self.parameters()).device
        batch_size = initial_state.shape[0]
        current_state = initial_state.to(device)
        coords = coords.to(device)

        sequence: List[Tensor] = [current_state.cpu()]

        with torch.no_grad():
            for step in tqdm(range(steps), desc='Predicting', leave=False, dynamic_ncols=True):
                t_norm = torch.full((batch_size,), step / max(steps, 1), device=device)
                next_state = self.forward(current_state, coords, t_norm=t_norm)
                sequence.append(next_state.cpu())
                current_state = next_state

        return torch.stack(sequence, dim=1)

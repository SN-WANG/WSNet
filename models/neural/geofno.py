# Geometry-aware Fourier Neural Operator (Geo-FNO)
# Author: Shengning Wang

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import List, Dict, Optional

# ============================================================
# Deformation Network (Coordinate Mapping)
# ============================================================

class DeformationNet(nn.Module):
    """
    Learnable coordinate deformation network f_theta.
    Maps physical coordinates x to latent coordinates x_hat: x_hat = x + f_theta(x).
    This network is based on a simple Multi-Layer Perceptron (MLP).
    """

    def __init__(self, spatial_dim: int, num_layers: int = 3, hidden_dim: int = 32):
        """
        Args:
        - spatial_dim (int): Dimension of the coordinate space (1, 2 or 3)
        - num_layers (int): Total number of layers in the MLP
        - hidden_dim (int): Hidden layer size of the MLP
        """
        super().__init__()

        layers: List[nn.Module] = []
        in_dim: int = spatial_dim

        # Build MLP structure
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim

        # Output dimension matches spatial dimension (coordinate offsets)
        layers += [nn.Linear(in_dim, spatial_dim)]
        self.net = nn.Sequential(*layers)

        # Initialize the last layer to ensure deformation starts near Identity (minimal offset)
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs coordinate deformation.

        Args:
        - x (Tensor): Input coordinates. Shape (batch_size, num_nodes, spatial_dim)

        Returns:
        - Tensor: Latent coordinates after deformation. Shape (batch_size, num_nodes, spatial_dim)
        """
        offsets = self.net(x)
        return x + offsets


# ============================================================
# Spectral Convolution Layer
# ============================================================

class SpectralConv(nn.Module):
    """
    Fourier Spectral Convolution Layer.
    Performs global convolution by multiplying in the Fourier domain.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: List[int]):
        """
        Args:
        - in_channels (int): Number of input feature channels.
        - out_channels (int): Number of output feature channels.
        - modes (List[int]): Number of Fourier modes to keep along dimensions.
        """
        super().__init__()
        self.spatial_dim = len(modes)

        if self.spatial_dim == 1:
            self.impl = _SpectralConv1d(in_channels, out_channels, modes[0])
        elif self.spatial_dim == 2:
            self.impl = _SpectralConv2d(in_channels, out_channels, modes[0], modes[1])
        elif self.spatial_dim == 3:
            self.impl = _SpectralConv3d(in_channels, out_channels, modes[0], modes[1], modes[2])
        else:
            raise ValueError(f'Only 1D, 2D and 3D are supported. Got {self.spatial_dim} modes')

    def forward(self, x: Tensor) -> Tensor:
        return self.impl(x)

class _SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.out_channels = out_channels
        self.modes = modes

        scale = 1 / (in_channels * out_channels)

        self.weights = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, num_channels, length)
        B, _, L = x.shape

        # Fourier transform (real-to-complex FFT)
        x_ft = torch.fft.rfft(x, n=L, dim=-1)  # (batch_size, num_channels, length // 2 + 1)

        # Filter and linear transform (truncation and multiplication)
        out_ft = torch.zeros(B, self.out_channels, L // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = \
            torch.einsum('bix, iox -> box', x_ft[:, :, :self.modes], self.weights)

        # Inverse Fourier transform (complex-to-real IFFT)
        return torch.fft.irfft(out_ft, n=L, dim=-1)

class _SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, num_channels, height, width)
        B, _, H, W = x.shape

        # Fourier transform (real-to-complex FFT)
        x_ft = torch.fft.rfft2(x, s=(H, W))  # (batch_size, num_channels, height, width // 2 + 1)

        # Filter and linear transform (truncation and multiplication)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum('bixy, ioxy -> boxy', x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum('bixy, ioxy -> boxy', x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Inverse Fourier transform (complex-to-real IFFT)
        return torch.fft.irfft2(out_ft, s=(H, W))

class _SpectralConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(
            in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, num_channels, len_x, len_y, len_z)
        B, _, X, Y, Z = x.shape

        # Fourier transform (real-to-complex FFT)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Filter and linear transform (truncation and multiplication)
        out_ft = torch.zeros(B, self.out_channels, X, Y, Z // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            torch.einsum('bixyz, ioxyz -> boxyz', x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            torch.einsum('bixyz, ioxyz -> boxyz', x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            torch.einsum('bixyz, ioxyz -> boxyz', x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            torch.einsum('bixyz, ioxyz -> boxyz', x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Inverse Fourier transform (complex-to-real IFFT)
        return torch.fft.irfftn(out_ft, s=(X, Y, Z))


# ============================================================
# FNO Block
# ============================================================

class FNOBlock(nn.Module):
    """
    Standard FNO residual block:
    y = Act(Norm(SpectralConv(x) + PointwiseConv(x)))
    """
    def __init__(self, width: int, modes: List[int]):
        super().__init__()
        self.spatial_dim = len(modes)

        # Spectral branch
        self.spectral_conv = SpectralConv(width, width, modes)

        # Pointwise branch
        if self.spatial_dim == 1:
            self.pointwise_conv = nn.Conv1d(width, width, 1)  # Linear map across channels
            self.norm = nn.InstanceNorm1d(width)  # Instance norm for stability
        elif self.spatial_dim == 2:
            self.pointwise_conv = nn.Conv2d(width, width, 1)  # Linear map across channels
            self.norm = nn.InstanceNorm2d(width)  # Instance norm for stability
        elif self.spatial_dim == 3:
            self.pointwise_conv = nn.Conv3d(width, width, 1)  # Linear map across channels
            self.norm = nn.InstanceNorm3d(width)  # Instance norm for stability

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
        - x (Tensor): Input feature grid. Shape (batch_size, width, len_x, (len_y), (len_z))

        Returns:
        - Tensor: Output feature grid. Shape (batch_size, width, len_x, (len_y), (len_z))
        """
        # 1. Spectral branch (Global)
        spectral_out = self.spectral_conv(x)

        # 2. Pointwise branch (Local/Residual connection)
        pointwise_out = self.pointwise_conv(x)

        # 3. Residual connection
        output = spectral_out + pointwise_out

        # 4. Normalization and activation
        output = self.norm(output)
        output = F.gelu(output)

        return output


# ============================================================
# Geo-FNO Network
# ============================================================

class GeoFNO(nn.Module):
    """
    Geometry-Aware Fourier Neural Operator (Geo-FNO).

    Pipeline:
    1. Lifting (Features + Coords -> Hidden)
    2. Deformation (Coords -> Latent Coords)
    3. P2G (Scatter-Mean to Latent Grid)
    4. FNO Processing (Spectral Convolutions)
    5. G2P (Grid Sample from Latent Grid)
    6. Projection (Hidden -> Output)
    """
    def __init__(self, in_channels: int, out_channels: int, modes: List[int], latent_grid_size: List[int],
                 depth: int, width: int, deformation_kwargs: Optional[Dict] = {'num_layers': 3, 'hidden_dim': 32}):
        """
        Args:
        - in_channels (int): Number of input channels
        - out_channels (int): Number of output channels
        - modes (List[int]): Number of Fourier modes to keep per dimension
        - latent_grid_size (List[int]): Size of the latent FFT grid
        - depth (int): Number of FNO blocks
        - width (int): Hidden channel dimension
        - deformation_kwargs (Optional[Dict]): kwargs for deformation net
        """
        super().__init__()

        assert len(modes) == len(latent_grid_size), 'Modes and Grid Size dimension must match'
        self.spatial_dim = len(modes)
        self.modes = modes
        self.latent_grid_size = latent_grid_size
        self.width = width

        # Deformation network (learns to coordinate map)
        if self.spatial_dim > 1:
            self.deformation_net = DeformationNet(spatial_dim=self.spatial_dim, **deformation_kwargs)
        else:
            self.deformation_net = None

        # # P: Lifting layer (Input -> Hidden). Concatenates feature + spatial coordinates
        # self.fc_lift = nn.Linear(in_channels + self.spatial_dim, width)

        # P: Lifting layer (Input -> Hidden).
        self.fc_lift = nn.Linear(in_channels, width)

        # FNO layers (Processing in latent space)
        self.fno_blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(depth)])

        # Q: Projection layer (Hidden -> Output)
        self.fc_proj1 = nn.Linear(width, 128)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_proj2 = nn.Linear(128, out_channels)

    def _compute_latent_coords(self, physical_coords: Tensor) -> Tensor:
        """Apply learned coordinate deformation to get latent grid coordinates.

        Args:
        - physical_coords (Tensor): Normalized node coordinates in [-1, 1].
                                    Shape (batch_size, num_nodes, spatial_dim).

        Returns:
        - Tensor: Latent coordinates in [-1, 1]. Shape (batch_size, num_nodes, spatial_dim).
        """
        if self.spatial_dim > 1 and self.deformation_net is not None:
            return torch.clamp(self.deformation_net(physical_coords), -1.0, 1.0)
        return physical_coords

    def _p2g_sample(self, features: Tensor, latent_coords: Tensor) -> Tensor:
        """
        Point-to-Grid (P2G) Encoding: Maps unstructured features to a regular grid using
        Nearest Neighbor accumulation (Scatter-Mean)

        Args:
        - features (Tensor): Lifted features. Shape (batch_size, num_nodes, width)
        - latent_coords (Tensor): Latent coordinates in [-1, 1]. Shape (batch_size, num_nodes, spatial_dim)

        Returns:
        - Tensor: Feature grid (batch_size, width, len_x, (len_y), (len_z))
        """
        batch_size, num_nodes, _ = features.shape
        device = features.device

        # 1. Map coords from [-1, 1] to Nearest Neighbor indices [0, len - 1]
        dims_tensor = torch.tensor(self.latent_grid_size, device=device).view(1, 1, -1)  # (1, 1, spatial_dim)

        # (coords + 1) / 2 * (len - 1)
        unnormalized = (latent_coords + 1) / 2 * (dims_tensor - 1)
        indices = unnormalized.round().long()
        for i, dim_size in enumerate(self.latent_grid_size):
            indices[..., i] = indices[..., i].clamp(0, dim_size - 1)  # (batch_size, num_nodes)

        # 2. Calculate flat indices for scatter (Row-Major Encoding)
        strides = []
        cum_stride = 1
        for dim_size in reversed(self.latent_grid_size):
            strides.append(cum_stride)
            cum_stride *= dim_size
        strides = torch.tensor(strides[::-1], device=device).view(1, 1, -1)  # (1, 1, spatial_dim)
        total_grid_size = cum_stride

        # Shape: (batch_size, num_nodes)
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num_nodes)

        # Global 1D Index (Flattened index) = b * stride_b + i_x * stride_x + (i_y * stride_y) + (i_z * 1)
        # Shape: (batch_size * num_nodes)
        flat_indices = ((batch_indices * total_grid_size) + (indices * strides).sum(dim=-1)).view(-1)

        # 3. Scatter Mean. Shape: (batch_size, num_nodes, width) -> (batch_size * num_nodes, width)
        flat_features = features.view(-1, self.width)

        # Initialize flat grid accumulator (total_voxels, width)
        grid_flat = torch.zeros(batch_size * total_grid_size, self.width, device=device)

        # Initialize counts accumulator (total_voxels, 1)
        counts_flat = torch.zeros(batch_size * total_grid_size, 1, device=device)

        # Accumulate features
        grid_flat.index_add_(0, flat_indices, flat_features)

        # Accumulate counts for averaging
        ones = torch.ones_like(flat_indices, dtype=features.dtype).unsqueeze(-1)
        counts_flat.index_add_(0, flat_indices, ones)

        # Perform mean
        grid_flat = grid_flat / counts_flat.clamp(min=1)

        # 4. Reshape: (total_voxels, width) -> (batch_size, len_x, (len_y), (len_z), width)
        #          -> (batch_size, width, len_x, (len_y), (len_z))
        target_shape = [batch_size] + self.latent_grid_size + [self.width]
        grid_features = grid_flat.view(*target_shape)
        permute_order = [0, self.spatial_dim + 1] + list(range(1, self.spatial_dim + 1))
        grid_features = grid_features.permute(*permute_order)

        return grid_features

    def _g2p_sample(self, grid_features: Tensor, latent_coords: Tensor) -> Tensor:
        """
        Grid-to-Point (G2P) Decoding: Maps grid features back to original geometries using Interpolation

        Args:
        - grid_features (Tensor): Latent grid features. Shape: (batch_size, num_channels, len_x, (len_y), (len_z))
        - latent_coords (Tensor): Latent coordinates in [-1, 1]. Shape: (batch_size, num_nodes, spatial_dim)

        Returns:
        - Tensor: Resampled features (batch_size, num_nodes, num_channels)
        """
        batch_size, num_nodes, _ = latent_coords.shape

        if self.spatial_dim == 1:
            grid_input = grid_features.unsqueeze(2)
            zeros = torch.zeros_like(latent_coords)
            coords_input = torch.cat([latent_coords, zeros], dim=-1).view(batch_size, num_nodes, 1, 2)
            sampled = F.grid_sample(grid_input, coords_input, align_corners=True, padding_mode='border')
            return sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        elif self.spatial_dim == 2:
            # F.grid_sample expects coords in (x, y) = (W-axis, H-axis) order,
            # but latent_coords are stored as (dim0=H, dim1=W). Flip last dim.
            coords_input = latent_coords[..., [1, 0]].view(batch_size, num_nodes, 1, 2)
            sampled = F.grid_sample(grid_features, coords_input, align_corners=True, padding_mode='border')
            return sampled.squeeze(-1).permute(0, 2, 1)

        elif self.spatial_dim == 3:
            # F.grid_sample expects coords in (x, y, z) = (W, H, D) order,
            # but latent_coords are stored as (dim0=D, dim1=H, dim2=W). Reverse last dim.
            coords_input = latent_coords[..., [2, 1, 0]].view(batch_size, num_nodes, 1, 1, 3)
            sampled = F.grid_sample(grid_features, coords_input, align_corners=True, padding_mode='border')
            return sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)

    def forward(self, input_features: Tensor, physical_coords: Tensor,
                latent_coords: Optional[Tensor] = None) -> Tensor:
        """
        Geo-FNO forward pass.

        Args:
        - input_features (Tensor): Input fields at nodes. Shape (batch_size, num_nodes, in_channels)
        - physical_coords (Tensor): Normalized coordinates in [-1, 1]. Shape (batch_size, num_nodes, spatial_dim)
        - latent_coords (Optional[Tensor]): Pre-computed deformed coordinates in [-1, 1].
                                            Shape (batch_size, num_nodes, spatial_dim). If None, computed
                                            internally via deformation_net. Pass a cached
                                            value from _compute_latent_coords() to avoid
                                            redundant deformation network calls during rollout.

        Returns:
        - Tensor: Prediction at nodes. Shape (batch_size, num_nodes, out_channels)
        """
        # 1. Lifting (P)
        lifted_features = self.fc_lift(input_features)

        # 2. Coordinate deformation (use cached value if provided)
        if latent_coords is None:
            latent_coords = self._compute_latent_coords(physical_coords)

        # 3. Point-to-Grid (P2G) Encoding
        grid_features = self._p2g_sample(lifted_features, latent_coords)

        # 4. FNO Processing (Latent Space Convolution)
        for block in self.fno_blocks:
            grid_features = block(grid_features)

        # 5. Grid-to-Point (G2P) Decoding
        recovered_features = self._g2p_sample(grid_features, latent_coords)

        # 6. Projection (Q)
        output = self.fc_proj1(recovered_features)
        output = F.gelu(output)
        output = self.dropout(output)
        output = self.fc_proj2(output)

        return output

    def predict(self, initial_state: Tensor, coords: Tensor, steps: int) -> Tensor:
        """
        Autoregressive inference for time-dependent PDE evolution.

        Note: caller must set model.eval() before calling this method.

        Args:
        - initial_state (Tensor): State at t=0. Shape (batch_size, num_nodes, in_channels)
        - coords (Tensor): Node coordinates in [-1, 1]. Shape (batch_size, num_nodes, spatial_dim)
        - steps (int): Number of future steps to generate.

        Returns:
        - Tensor: Predicted sequence. Shape (batch_size, steps+1, num_nodes, out_channels).
        """
        device = next(self.parameters()).device
        current_state = initial_state.to(device)
        coords = coords.to(device)

        # Pre-compute latent coords once: physical_coords are constant across steps,
        # and deformation_net weights are fixed during inference.
        latent_coords = self._compute_latent_coords(coords)

        seq: List[Tensor] = [current_state.cpu()]

        with torch.no_grad():
            for _ in tqdm(range(steps), desc='Predicting', leave=False, dynamic_ncols=True):
                next_state = self.forward(current_state, coords, latent_coords=latent_coords)
                seq.append(next_state.cpu())
                current_state = next_state

        return torch.stack(seq, dim=1)

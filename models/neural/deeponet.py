# Deep Operator Network (DeepONet) Surrogate Model
# Author: Shengning Wang

import torch
from torch import nn, Tensor
from typing import List, Optional, Callable, Tuple


class DeepONet(nn.Module):
    """
    Deep Operator Network (DeepONet) for learning operators between function spaces.

    The Branch Net encodes the input function 'u' at fixed sensors.
    The Trunk Net encodes the random coordinate/location 'y' for the output function.
    The output is the dot product of the Branch and Trunk feature embeddings + bias.
    """

    def __init__(self, in_channels: int, out_channels: int, num_trunk_features: int,
                 branch_hidden_sizes: List[int], trunk_hidden_sizes: List[int], num_basis_functions: int,
                 branch_net_type: str = 'mlp', num_sensors: Optional[int] = None,
                 branch_grid_size: Optional[Tuple[int, int]] = None,
                 activation: Callable = nn.ReLU, dropout_rate: float = 0.0):
        """
        Initializes the DeepONet structure.

        Args:
        - in_channels (int): Number of input channels for Branch Net
        - out_channels (int): Number of output channels for G(u)
        - num_trunk_features (int): Input dimension for Trunk Net (random coordinate/location dimension)
        - branch_hidden_sizes (List[int]): Hidden layer sizes for Branch Net
        - trunk_hidden_sizes (List[int]): Hidden layer sizes for Trunk Net
        - num_basis_functions (int): The output dimension of both sub-networks (p)
        - branch_net_type (str): Type of Branch Net, either 'mlp' or 'cnn'
        - num_sensors (int): Total sensors for MLP branch
        - branch_grid_size (Tuple[int, int]): Grid size (H, W) for CNN branch
        - activation (Callable): Activation function class
        - dropout_rate (float): Dropout probability
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_basis_functions = num_basis_functions
        self.branch_net_type = branch_net_type.lower()
        self.branch_grid_size = branch_grid_size

        # Branch input dimension for MLP branch: Coefficients for each (sensor， in_channel) pair
        if num_sensors:
            branch_mlp_in_channels = num_sensors * in_channels

        # Branch output dimension: Coefficients for each (basis_function， out_channel) pair
        branch_out_channels = num_basis_functions * out_channels

        # 1. Build Branch Net
        if branch_net_type == 'cnn':
            if branch_grid_size is None:
                raise ValueError('branch_grid_size must be provided for CNN branch')
            self.branch_net = self._build_conv_net(
                in_channels=in_channels, hidden_channels=branch_hidden_sizes, out_channels=branch_out_channels,
                activation=activation, dropout_rate=dropout_rate)
        else:
            if num_sensors is None:
                raise ValueError('num_sensors must be provided for MLP branch')
            self.branch_net= self._build_dense_net(
                in_channels=branch_mlp_in_channels, hidden_channels=branch_hidden_sizes,
                out_channels=branch_out_channels, activation=activation, dropout_rate=dropout_rate,
                activate_last=False)

        # 2. Build Trunk Net
        self.trunk_net = self._build_dense_net(
            in_channels=num_trunk_features, hidden_channels=trunk_hidden_sizes, out_channels=num_basis_functions,
            activation=activation, dropout_rate=dropout_rate, activate_last=True)

        # Trainable bias for each output channel
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self._init_weights()

    def _build_dense_net(self, in_channels: int, hidden_channels: List[int], out_channels: int,
                   activation: Callable, dropout_rate: float, activate_last: bool) -> nn.Sequential:
        """
        Helper to construct a sub-network (MLP style)
        """
        layers = []
        current_dim = in_channels

        for next_dim in hidden_channels:
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            current_dim = next_dim

        layers.append(nn.Linear(current_dim, out_channels))

        if activate_last:
            layers.append(activation())

        return nn.Sequential(*layers)

    def _build_conv_net(self, in_channels: int, hidden_channels: List[int], out_channels: int,
                   activation: Callable, dropout_rate: float) -> nn.Sequential:
        """
        Helper to construct a sub-network (CNN style)
        """
        layers = []
        current_dim = in_channels

        for next_dim in hidden_channels:
            layers.append(nn.Conv2d(current_dim, next_dim, kernel_size=3, padding=1))
            layers.append(activation())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.MaxPool2d(kernel_size=2))
            current_dim = next_dim

        layers.append(nn.Flatten())

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *self.branch_grid_size)
            flat_dim = nn.Sequential(*layers)(dummy).shape[1]

        layers.append(nn.Linear(flat_dim, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        Helper to initialize weights using He/Kaiming Uniform
        """
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_branch: Tensor, x_trunk: Tensor) -> Tensor:
        """
        Forward pass of DeepONet.

        Args:
        - x_branch (Tensor): Input for Branch Net
            - Shape for MLP branch: (batch_size, num_sensors, in_channels)
            - Shape for CNN branch: (batch_size, in_channels, H, W)
        - x_trunk (Tensor): Input for Trunk Net. Shape: (batch_size, num_trunk_features)

        Returns:
        - Tensor: Prediction G(u)(y). Shape: (batch_size, out_channels)
        """

        # Reshape B for MLP branch: (batch_size, num_sensors * out_channels)
        if self.branch_net_type == 'mlp':
            x_branch = x_branch.view(x_branch.shape[0], -1)

        # Branch output B: (batch_size, num_basis_functions * out_channels)
        b_out = self.branch_net(x_branch)

        # Reshape B to split into basis weights for each output channel
        # Shape: (batch_size, num_basis_functions, out_channels)
        b_out = b_out.view(-1, self.num_basis_functions, self.out_channels)

        # Trunk output T: (batch_size, num_basis_functions)
        t_out = self.trunk_net(x_trunk)

        # Einstein summation for dot product over basis functions p
        # Equation: G(u)_bo = sum_p(B_bpo * T_bp), where b: batch_size, p: num_basis_functions, o: out_channels
        # Shape: (batch_size, out_channels)
        out = torch.einsum('bpo, bp -> bo', b_out, t_out) + self.bias

        return out

    def predict(self, x_branch: Tensor, x_trunk: Tensor) -> Tensor:
        """
        Performs inference on new inputs.
        """

        self.eval()
        device = next(self.parameters()).device
        x_branch = x_branch.to(device)
        x_trunk = x_trunk.to(device)

        with torch.no_grad():
            preds = self.forward(x_branch, x_trunk)

        return preds

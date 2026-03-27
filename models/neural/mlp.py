# Multi-Layer Perceptron (MLP) Surrogate Model
# Author: Shengning Wang

from typing import List, Dict, Any, Optional, Callable, Union, Tuple

import torch
from torch import nn, Tensor


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Regressor with configurable hidden layers and Dropout
    """

    def __init__(self, num_features: int, num_outputs: int, hidden_sizes: List[int] = [128, 128],
                 activation: Callable = nn.ReLU, dropout_rate: float = 0.1):
        """
        Initializes the MLP structure

        Args:
        - num_features (int): Number of input features
        - num_outputs (int): Number of target outputs
        - hidden_sizes (List[int]): Sizes of the hidden layers
        - activation (Callable): Activation function
        - dropout_rate (float): Dropout probability applied after each hidden layer
        """
        super().__init__()

        layers = []
        current_dim = num_features

        for size in hidden_sizes:
            layers.append(nn.Linear(current_dim, size))
            layers.append(activation())

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))

            current_dim = size

        self.net = nn.Sequential(*layers)

        self.output_head = nn.Linear(current_dim, num_outputs)

        self._init_weights()

    def _init_weights(self):
        """
        Helper to initialize weights using He/Kaiming Uniform
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:

        return self.output_head(self.net(x))

    def predict(self, inputs: Tensor) -> Tensor:
        """
        Performs inference on new inputs
        """

        self.eval()
        device = next(self.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            preds = self.forward(inputs)

        return preds

# Scalers for Standardization and Normalization
# Author: Shengning Wang

import numpy as np
from typing import Optional, Dict, Literal, TYPE_CHECKING

# Soft dependency for PyTorch
if TYPE_CHECKING:
    import torch
    from torch import Tensor
    _HAS_TORCH = True
else:
    try:
        import torch
        from torch import Tensor
        _HAS_TORCH = True
    except ImportError:
        torch = None
        Tensor = None
        _HAS_TORCH = False


class BaseScaler:
    """
    Base class for data scalers.

    This class defines the common interface for all scalers,
    including fit, transform, and inverse_transform.

    Subclasses must implement the core logic.
    """

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "BaseScaler":
        """
        Fits the scaler statistics from data.

        Args:
            x (np.ndarray): Input data array. Shape: (num_samples, ..., num_channels)
            channel_dim (int): Dimension index representing channel dimension.

        Returns:
            BaseScaler: Self instance for method chaining.
        """
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms input data using fitted statistics.

        Args:
            x (np.ndarray): Input data array. Shape must match data passed to fit().

        Returns:
            np.ndarray: Scaled data with the same shape and dtype as input.
        """
        raise NotImplementedError

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Inversely transforms scaled data back to original space.

        Args:
            x (np.ndarray): Scaled data array. Shape must match data passed to transform().

        Returns:
            np.ndarray: Data restored to original scale.
        """
        raise NotImplementedError

    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns scaler internal state for serialization.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing scaler parameters.
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """
        Loads scaler internal state from dictionary.

        Args:
            state_dict (Dict[str, np.ndarray]): State dictionary.
        """
        raise NotImplementedError


# ======================================================================
# Standardization Scaler (NumPy)
# ======================================================================

class StandardScalerNP(BaseScaler):
    """
    NumPy implementation of standardization scaler.

    Performs channel-wise standardization:
        x_scaled = (x - mean) / std

    Statistics are computed over all dimensions except channel_dim.
    """

    def __init__(self, eps: float = 1e-7):
        """
        Initializes the StandardScalerNP.

        Args:
            eps (float): Small constant to avoid division by zero.
        """
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps
        self.channel_dim: Optional[int] = None

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "StandardScalerNP":
        """
        Computes mean and standard deviation for standardization.

        Args:
            x (np.ndarray): Input data. Shape: (num_samples, ..., num_channels)
            channel_dim (int): Dimension index representing channel dimension.

        Returns:
            StandardScalerNP: Self instance.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        self.channel_dim = channel_dim % x.ndim

        reduce_dims = tuple(d for d in range(x.ndim) if d != self.channel_dim)

        self.mean = np.mean(x, axis=reduce_dims, keepdims=True)
        self.std = np.std(x, axis=reduce_dims, keepdims=True)

        # Avoid numerical instability
        self.std[self.std < self.eps] = 1.0

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Applies standardization to input data.

        Args:
            x (np.ndarray): Input data. Shape matches fit() input.

        Returns:
            np.ndarray: Standardized data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")

        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Restores standardized data to original scale.

        Args:
            x (np.ndarray): Standardized data. Shape matches transform() input.

        Returns:
            np.ndarray: Original scale data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")

        return x * self.std + self.mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "std": self.std,
            "channel_dim": np.array(self.channel_dim, dtype=np.int64),
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.channel_dim = int(state_dict["channel_dim"])


# ======================================================================
# Min-Max Normalization Scaler (NumPy)
# ======================================================================

class MinMaxScalerNP(BaseScaler):
    """
    NumPy implementation of min-max normalization scaler.

    Supports normalization to:
        - [0, 1]
        - [-1, 1]

    Scaling rule:
        x_scaled = a + (x - min) * (b - a) / (max - min)
    """

    def __init__(self, norm_range: Literal["unit", "bipolar"] = "unit", eps: float = 1e-7):
        """
        Initializes the MinMaxScalerNP.

        Args:
            norm_range (Literal): Target normalization range.
                - "unit" -> [0, 1]
                - "bipolar" -> [-1, 1]
            eps (float): Small constant to avoid division by zero.
        """
        self.norm_range = norm_range
        self.eps = eps
        self.data_min: Optional[np.ndarray] = None
        self.data_max: Optional[np.ndarray] = None
        self.channel_dim: Optional[int] = None

        if norm_range == "unit":
            self.a, self.b = 0.0, 1.0
        elif norm_range == "bipolar":
            self.a, self.b = -1.0, 1.0
        else:
            raise ValueError("Invalid norm_range.")

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "MinMaxScalerNP":
        """
        Computes min and max statistics for normalization.

        Args:
            x (np.ndarray): Input data. Shape: (num_samples, ..., num_channels)
            channel_dim (int): Dimension index representing channel dimension.

        Returns:
            MinMaxScalerNP: Self instance.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        self.channel_dim = channel_dim % x.ndim

        reduce_dims = tuple(d for d in range(x.ndim) if d != self.channel_dim)

        self.data_min = np.min(x, axis=reduce_dims, keepdims=True)
        self.data_max = np.max(x, axis=reduce_dims, keepdims=True)

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Applies min-max normalization.

        Args:
            x (np.ndarray): Input data. Shape matches fit() input.

        Returns:
            np.ndarray: Normalized data.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler has not been fitted.")

        scale = self.data_max - self.data_min
        scale[scale < self.eps] = 1.0

        return self.a + (x - self.data_min) * (self.b - self.a) / scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Restores normalized data to original scale.

        Args:
            x (np.ndarray): Normalized data. Shape matches transform() input.

        Returns:
            np.ndarray: Original scale data.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler has not been fitted.")

        scale = self.data_max - self.data_min
        scale[scale < self.eps] = 1.0

        return (x - self.a) * scale / (self.b - self.a) + self.data_min

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "data_min": self.data_min,
            "data_max": self.data_max,
            "channel_dim": np.array(self.channel_dim, dtype=np.int64),
            "norm_range": np.array(
                0 if self.norm_range == "unit" else 1, dtype=np.int64
            ),
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.data_min = state_dict["data_min"]
        self.data_max = state_dict["data_max"]
        self.channel_dim = int(state_dict["channel_dim"])
        self.norm_range = (
            "unit" if int(state_dict["norm_range"]) == 0 else "bipolar"
        )
        self.a, self.b = (0.0, 1.0) if self.norm_range == "unit" else (-1.0, 1.0)


# ======================================================================
# Below is defined only if torch is available
# ======================================================================

if _HAS_TORCH:

    # ======================================================================
    # Standardization Scaler (PyTorch Tensor)
    # ======================================================================

    class StandardScalerTensor:
        """
        PyTorch Tensor implementation of standardization scaler.

        Performs channel-wise standardization:
            x_scaled = (x - mean) / std

        Statistics are computed over all dimensions except channel_dim.
        Automatically handles device migration for cross-device inference.
        """

        def __init__(self, eps: float = 1e-7):
            """
            Initializes the StandardScalerTensor.

            Args:
                eps (float): Small constant to avoid division by zero.
            """
            self.mean: Optional[Tensor] = None
            self.std: Optional[Tensor] = None
            self.eps = eps
            self.channel_dim: Optional[int] = None

        def fit(self, x: Tensor, channel_dim: int = -1) -> "StandardScalerTensor":
            """
            Computes mean and standard deviation for standardization.

            Args:
                x (Tensor): Input data.
                    Shape: (batch_size, ..., num_channels)
                    Dtype: float32 or float64
                channel_dim (int): Dimension index representing channel dimension.

            Returns:
                StandardScalerTensor: Self instance for method chaining.
            """
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a PyTorch Tensor.")

            self.channel_dim = channel_dim % x.ndim

            # Compute statistics over all dimensions except channel_dim
            reduce_dims = [d for d in range(x.ndim) if d != self.channel_dim]

            self.mean = torch.mean(x, dim=reduce_dims, keepdim=True)
            self.std = torch.std(x, dim=reduce_dims, keepdim=True)

            # Avoid numerical instability
            self.std = torch.where(self.std < self.eps, torch.ones_like(self.std), self.std)

            return self

        def transform(self, x: Tensor) -> Tensor:
            """
            Applies standardization to input data.

            Args:
                x (Tensor): Input data.
                    Shape: (batch_size, ..., num_channels)
                    Dtype: Same as input to fit()

            Returns:
                Tensor: Standardized data.
                    Shape: Same as input
                    Dtype: Same as input
            """
            if self.mean is None or self.std is None:
                raise RuntimeError("Scaler has not been fitted.")

            # Ensure statistics are on the same device as input
            if self.mean.device != x.device:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)

            return (x - self.mean) / self.std

        def inverse_transform(self, x: Tensor) -> Tensor:
            """
            Restores standardized data to original scale.

            Args:
                x (Tensor): Standardized data.
                    Shape: (batch_size, ..., num_channels)
                    Dtype: Same as output from transform()

            Returns:
                Tensor: Original scale data.
                    Shape: Same as input
                    Dtype: Same as input
            """
            if self.mean is None or self.std is None:
                raise RuntimeError("Scaler has not been fitted.")

            # Ensure statistics are on the same device as input
            if self.mean.device != x.device:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)

            return x * self.std + self.mean

        def state_dict(self) -> Dict[str, Tensor]:
            """
            Returns scaler internal state for serialization.

            Returns:
                Dict[str, Tensor]: Dictionary containing scaler parameters.
                    - "mean": Tensor, shape matches computed statistics
                    - "std": Tensor, shape matches computed statistics
                    - "channel_dim": Tensor of shape (), dtype int64
            """
            return {
                "mean": self.mean,
                "std": self.std,
                "channel_dim": torch.tensor(self.channel_dim, dtype=torch.int64),
            }

        def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
            """
            Loads scaler internal state from dictionary.

            Args:
                state_dict (Dict[str, Tensor]): State dictionary containing:
                    - "mean": Tensor
                    - "std": Tensor
                    - "channel_dim": Tensor
            """
            self.mean = state_dict["mean"]
            self.std = state_dict["std"]
            self.channel_dim = int(state_dict["channel_dim"].item())


    # ======================================================================
    # Min-Max Normalization Scaler (PyTorch Tensor)
    # ======================================================================

    class MinMaxScalerTensor:
        """
        PyTorch Tensor implementation of min-max normalization scaler.

        Supports normalization to:
            - [0, 1] ("unit")
            - [-1, 1] ("bipolar")

        Scaling rule:
            x_scaled = a + (x - min) * (b - a) / (max - min)

        Automatically handles device migration for cross-device inference.
        """

        def __init__(self, norm_range: Literal["unit", "bipolar"] = "unit", eps: float = 1e-7):
            """
            Initializes the MinMaxScalerTensor.

            Args:
                norm_range (Literal): Target normalization range.
                    - "unit" -> [0, 1]
                    - "bipolar" -> [-1, 1]
                eps (float): Small constant to avoid division by zero.
            """
            self.norm_range = norm_range
            self.eps = eps
            self.data_min: Optional[Tensor] = None
            self.data_max: Optional[Tensor] = None
            self.channel_dim: Optional[int] = None

            if norm_range == "unit":
                self.a, self.b = 0.0, 1.0
            elif norm_range == "bipolar":
                self.a, self.b = -1.0, 1.0
            else:
                raise ValueError("Invalid norm_range. Must be \"unit\" or \"bipolar\".")

        def fit(self, x: Tensor, channel_dim: int = -1) -> "MinMaxScalerTensor":
            """
            Computes min and max statistics for normalization.

            Args:
                x (Tensor): Input data.
                    Shape: (batch_size, ..., num_channels)
                    Dtype: float32 or float64
                channel_dim (int): Dimension index representing channel dimension.

            Returns:
                MinMaxScalerTensor: Self instance for method chaining.
            """
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a PyTorch Tensor.")

            self.channel_dim = channel_dim % x.ndim

            # Compute statistics over all dimensions except channel_dim
            reduce_dims = [d for d in range(x.ndim) if d != self.channel_dim]

            self.data_min = torch.min(x, dim=reduce_dims[0], keepdim=True)[0]
            self.data_max = torch.max(x, dim=reduce_dims[0], keepdim=True)[0]

            # Iteratively reduce over remaining dimensions
            for dim in reduce_dims[1:]:
                self.data_min = torch.min(self.data_min, dim=dim, keepdim=True)[0]
                self.data_max = torch.max(self.data_max, dim=dim, keepdim=True)[0]

            return self

        def transform(self, x: Tensor) -> Tensor:
            """
            Applies min-max normalization.

            Args:
                x (Tensor): Input data.
                    Shape: (batch_size, ..., num_channels)
                    Dtype: Same as input to fit()

            Returns:
                Tensor: Normalized data.
                    Shape: Same as input
                    Dtype: Same as input
            """
            if self.data_min is None or self.data_max is None:
                raise RuntimeError("Scaler has not been fitted.")

            # Ensure statistics are on the same device as input
            if self.data_min.device != x.device:
                self.data_min = self.data_min.to(x.device)
                self.data_max = self.data_max.to(x.device)

            scale = self.data_max - self.data_min
            scale = torch.where(scale < self.eps, torch.ones_like(scale), scale)

            return self.a + (x - self.data_min) * (self.b - self.a) / scale

        def inverse_transform(self, x: Tensor) -> Tensor:
            """
            Restores normalized data to original scale.

            Args:
                x (Tensor): Normalized data.
                    Shape: (batch_size, ..., num_channels)
                    Dtype: Same as output from transform()

            Returns:
                Tensor: Original scale data.
                    Shape: Same as input
                    Dtype: Same as input
            """
            if self.data_min is None or self.data_max is None:
                raise RuntimeError("Scaler has not been fitted.")

            # Ensure statistics are on the same device as input
            if self.data_min.device != x.device:
                self.data_min = self.data_min.to(x.device)
                self.data_max = self.data_max.to(x.device)

            scale = self.data_max - self.data_min
            scale = torch.where(scale < self.eps, torch.ones_like(scale), scale)

            return (x - self.a) * scale / (self.b - self.a) + self.data_min

        def state_dict(self) -> Dict[str, Tensor]:
            """
            Returns scaler internal state for serialization.

            Returns:
                Dict[str, Tensor]: Dictionary containing scaler parameters.
                    - "data_min": Tensor, shape matches computed statistics
                    - "data_max": Tensor, shape matches computed statistics
                    - "channel_dim": Tensor of shape (), dtype int64
                    - "norm_range": Tensor of shape (), dtype int64
            """
            return {
                "data_min": self.data_min,
                "data_max": self.data_max,
                "channel_dim": torch.tensor(self.channel_dim, dtype=torch.int64),
                "norm_range": torch.tensor(
                    0 if self.norm_range == "unit" else 1, dtype=torch.int64
                ),
            }

        def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
            """
            Loads scaler internal state from dictionary.

            Args:
                state_dict (Dict[str, Tensor]): State dictionary containing:
                    - "data_min": Tensor
                    - "data_max": Tensor
                    - "channel_dim": Tensor
                    - "norm_range": Tensor
            """
            self.data_min = state_dict["data_min"]
            self.data_max = state_dict["data_max"]
            self.channel_dim = int(state_dict["channel_dim"].item())
            self.norm_range = (
                "unit" if int(state_dict["norm_range"].item()) == 0 else "bipolar"
            )
            self.a, self.b = (0.0, 1.0) if self.norm_range == "unit" else (-1.0, 1.0)

else:
    # Define as None when torch is not available for safe import
    StandardScalerTensor = None
    MinMaxScalerTensor = None

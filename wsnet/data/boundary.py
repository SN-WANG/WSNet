# Boundary Condition Detection and Enforcement for CFD Neural Operators
# Author: Shengning Wang

import torch
from torch import Tensor
from typing import List, Dict, Any

from wsnet.utils.hue_logger import hue, logger


class BoundaryCondition:
    """Data-driven wall detection and hard boundary condition enforcement.

    Identifies no-slip wall nodes by detecting mesh nodes whose velocity
    components remain below a threshold across all timesteps and all training
    cases. At each autoregressive rollout step, enforce() replaces the model's
    wall-node velocity predictions with the known Dirichlet BC values in
    standardized feature space, breaking error accumulation at boundaries.

    This class is a data utility (not an nn.Module) and is serialized
    alongside scalers in the training checkpoint.

    Typical usage::

        bc = BoundaryCondition()
        bc.fit(train_raw, feature_scaler, velocity_channels=[0, 1])

        # During training / inference rollout:
        pred = model(input_state, coords, t_norm=t_norm)
        pred = bc.enforce(pred)   # hard-set wall velocities
    """

    def __init__(self) -> None:
        self.wall_mask: Tensor = torch.empty(0, dtype=torch.bool)
        self.wall_values_std: Tensor = torch.empty(0)
        self.enforce_channels: List[int] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        raw_dataset: Any,
        feature_scaler: Any,
        velocity_channels: List[int] = [0, 1],
        velocity_threshold: float = 1e-4,
    ) -> "BoundaryCondition":
        """Detect wall nodes and compute standardized BC values.

        A node is classified as a wall node if its velocity magnitude stays
        below *velocity_threshold* across every timestep of every training case.

        Args:
            raw_dataset: FlowData instance with ``seqs: List[Tensor]``.
                         Each tensor has shape ``(seq_len, num_nodes, num_channels)``.
            feature_scaler: Fitted StandardScalerTensor with ``.mean_`` and
                            ``.std_`` attributes (Tensor, shape ``(num_channels,)``).
            velocity_channels: Channel indices of velocity components to
                               enforce (default ``[0, 1]`` for Vx, Vy).
            velocity_threshold: Absolute velocity below which a node is
                                considered stationary. Default: ``1e-4``.

        Returns:
            Self, for method chaining.
        """
        seqs: List[Tensor] = raw_dataset.seqs
        num_nodes = seqs[0].shape[1]

        # Start with all nodes as candidates; exclude any that exceed threshold
        is_wall = torch.ones(num_nodes, dtype=torch.bool)

        for seq in seqs:  # seq: (T, N, C)
            for ch in velocity_channels:
                max_abs = seq[:, :, ch].abs().max(dim=0).values  # (N,)
                is_wall &= max_abs < velocity_threshold

        self.wall_mask = is_wall
        self.enforce_channels = list(velocity_channels)

        # Compute BC values in standardized space: (0 - mean) / std
        mean = feature_scaler.mean.squeeze()    # (C,)
        std = feature_scaler.std.squeeze()      # (C,)
        num_channels = mean.shape[0]
        wall_vals = torch.zeros(num_channels)
        for ch in self.enforce_channels:
            wall_vals[ch] = (0.0 - float(mean[ch])) / float(std[ch])

        self.wall_values_std = wall_vals
        self._fitted = True

        n_wall = int(is_wall.sum().item())
        n_total = num_nodes
        logger.info(
            f"boundary condition: detected {hue.m}{n_wall}{hue.q} wall nodes "
            f"out of {hue.m}{n_total}{hue.q} ({100 * n_wall / n_total:.1f}%), "
            f"enforce channels: {hue.b}{self.enforce_channels}{hue.q}"
        )
        return self

    # ------------------------------------------------------------------
    # Enforcement
    # ------------------------------------------------------------------

    def enforce(self, pred: Tensor) -> Tensor:
        """Replace wall-node predictions with known BC values.

        Uses ``clone()`` so that BPTT gradients flow normally through
        non-wall nodes during training.

        Args:
            pred: Model prediction in standardized space.
                  Shape: ``(B, N, C)``.

        Returns:
            Tensor with wall-node velocity channels overwritten.
            Shape: ``(B, N, C)``.
        """
        if not self._fitted or self.wall_mask.sum() == 0:
            return pred

        mask = self.wall_mask.to(pred.device)
        vals = self.wall_values_std.to(device=pred.device, dtype=pred.dtype)

        out = pred.clone()
        for ch in self.enforce_channels:
            out[:, mask, ch] = vals[ch]
        return out

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return internal state for checkpoint serialization.

        Returns:
            Dictionary containing wall mask, BC values, and channel list.
        """
        return {
            "wall_mask": self.wall_mask,
            "wall_values_std": self.wall_values_std,
            "enforce_channels": self.enforce_channels,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore internal state from a checkpoint.

        Args:
            state_dict: Dictionary previously returned by ``state_dict()``.
        """
        self.wall_mask = state_dict["wall_mask"]
        self.wall_values_std = state_dict["wall_values_std"]
        self.enforce_channels = state_dict["enforce_channels"]
        self._fitted = state_dict["fitted"]

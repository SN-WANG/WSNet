# Base Criterion Framework with Simple Loss and Metrics
# Author: Shengning Wang

import torch
from torch import nn, Tensor
from typing import Dict, List, Optional, Any


class BaseCriterion(nn.Module):
    """Abstract base class for loss functions.

    Provides a consistent interface for all loss implementations.
    Subclasses must implement the forward method.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor:
        """Compute loss between predictions and targets.

        Args:
            pred: Predicted values. Shape (batch_size, seq_len, num_nodes, num_channels)
            target: Ground truth values. Shape same as pred.
            **kwargs: Additional arguments (e.g., coords, time for physics losses).

        Returns:
            Tensor: Scalar loss tensor. Shape (1,)

        Raises:
            ValueError: If tensor shapes are incompatible.
        """
        raise NotImplementedError("Subclasses must implement forward method.")


class NMSECriterion(BaseCriterion):
    """Per-channel Normalized Mean Squared Error loss with optional channel weighting.

    Computes per-channel NMSE and returns the (weighted) average across channels:
        L = sum_c (w_c * NMSE_c) / sum_c(w_c)
    where NMSE_c = ||target_c - pred_c||^2 / (||target_c||^2 + eps).

    Each channel is independently normalized by its own energy, preventing
    high-energy channels from dominating the gradient signal. Optional
    channel_weights control the relative importance of each channel without
    affecting the overall loss magnitude (e.g., up-weighting a sparse
    Y-velocity field).
    """

    def __init__(self, eps: float = 1e-8, channel_weights: Optional[List[float]] = None):
        super().__init__()
        self.eps = eps
        if channel_weights is not None:
            self.register_buffer(
                "channel_weights",
                torch.tensor(channel_weights, dtype=torch.float32),
            )
        else:
            self.channel_weights = None

    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        C = pred.shape[-1]
        # Flatten all non-channel dims: (*dims, C) -> (N_total, C)
        sq_err = (target - pred) ** 2
        mse_c = sq_err.reshape(-1, C).sum(0)              # (C,)
        norm_c = (target ** 2).reshape(-1, C).sum(0) + self.eps  # (C,)
        nmse_c = mse_c / norm_c  # (C,)
        if self.channel_weights is not None:
            nmse_c = nmse_c * self.channel_weights.to(nmse_c.device)
            return nmse_c.sum() / self.channel_weights.sum()
        return nmse_c.mean()


class Metrics:
    """Evaluation metrics for autoregressive sequence forecasting.

    Computes comprehensive metrics including global and step-wise breakdowns
    for each channel in multivariate predictions.
    """

    SUPPORTED_METRICS = ("nmse", "mse", "rmse", "mae", "r2", "max_error")

    def __init__(self, channel_names: List[str], metrics: Optional[List[str]] = None):
        if metrics is None:
            metrics = list(self.SUPPORTED_METRICS)

        for metric in metrics:
            if metric not in self.SUPPORTED_METRICS:
                available = ", ".join(self.SUPPORTED_METRICS)
                raise ValueError(f"Unknown metric '{metric}'. Available: {available}")

        self.channel_names = channel_names
        self.metrics = metrics

    def compute(self, pred: Tensor, target: Tensor) -> Dict[str, Dict[str, Any]]:
        """Compute comprehensive AR evaluation metrics.

        Args:
            pred: Predicted sequence. Shape (seq_len, num_nodes, num_channels)
            target: Ground truth sequence. Shape (seq_len, num_nodes, num_channels)

        Returns:
            Nested dictionary of metrics per channel and aggregate.
        """
        if pred.dim() != 3:
            raise ValueError(f"pred must be 3D (S, N, C), got {pred.dim()}D")

        S, N, C = pred.shape

        if target.shape != (S, N, C):
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        if C != len(self.channel_names):
            raise ValueError(
                f"Channel count mismatch: tensor has {C} channels "
                f"but {len(self.channel_names)} names provided"
            )

        results: Dict[str, Dict[str, Any]] = {}

        for c, ch_name in enumerate(self.channel_names):
            pred_c = pred[..., c]      # (S, N)
            target_c = target[..., c]    # (S, N)

            channel_result: Dict[str, Any] = {
                "global": {},
                "step_wise": {}
            }

            # Global metrics (aggregate over all timesteps and nodes)
            abs_diff = torch.abs(target_c - pred_c)
            sq_diff = (target_c - pred_c) ** 2

            for metric_name in self.metrics:
                if metric_name == "nmse":
                    nmse_val = torch.mean(sq_diff) / (torch.mean(target_c ** 2) + 1e-8)
                    channel_result["global"][metric_name] = nmse_val.item()
                elif metric_name == "mse":
                    mse_val = torch.mean(sq_diff)
                    channel_result["global"][metric_name] = mse_val.item()
                elif metric_name == "rmse":
                    rmse_val = torch.sqrt(torch.mean(sq_diff))
                    channel_result["global"][metric_name] = rmse_val.item()
                elif metric_name == "mae":
                    mae_val = torch.mean(abs_diff)
                    channel_result["global"][metric_name] = mae_val.item()
                elif metric_name == "r2":
                    ss_res = torch.sum(sq_diff).item()
                    ss_tot = torch.sum((target_c - torch.mean(target_c)) ** 2).item()
                    r2_val = 1.0 - (ss_res / (ss_tot + 1e-8))
                    channel_result["global"][metric_name] = r2_val
                elif metric_name == "max_error":
                    max_val = torch.max(abs_diff).item()
                    channel_result["global"][metric_name] = max_val

            # Step-wise metrics (per timestep, aggregate over nodes)
            for metric_name in self.metrics:
                if metric_name == "nmse":
                    step_nmse = (
                        torch.mean(sq_diff, dim=1) /
                        (torch.mean(target_c ** 2, dim=1) + 1e-8)
                    )
                    channel_result["step_wise"][metric_name] = step_nmse.tolist()
                elif metric_name == "mse":
                    channel_result["step_wise"][metric_name] = torch.mean(sq_diff, dim=1).tolist()
                elif metric_name == "rmse":
                    channel_result["step_wise"][metric_name] = torch.sqrt(
                        torch.mean(sq_diff, dim=1)
                    ).tolist()
                elif metric_name == "mae":
                    channel_result["step_wise"][metric_name] = torch.mean(abs_diff, dim=1).tolist()
                elif metric_name == "r2":
                    step_ss_res = torch.sum(sq_diff, dim=1)
                    step_mean = torch.mean(target_c, dim=1, keepdim=True)
                    step_ss_tot = torch.sum((target_c - step_mean) ** 2, dim=1)
                    channel_result["step_wise"][metric_name] = (
                        (1.0 - step_ss_res / (step_ss_tot + 1e-8)).tolist()
                    )
                elif metric_name == "max_error":
                    channel_result["step_wise"][metric_name] = torch.max(
                        abs_diff, dim=1
                    ).values.tolist()

            results[ch_name] = channel_result

        return results

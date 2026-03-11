# Teacher Forcing Trainer for Baseline Comparison Experiments
# Author: Shengning Wang

import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any

from wsnet.training.base_trainer import BaseTrainer
from wsnet.training.base_criterion import NMSECriterion


class TeacherForcingTrainer(BaseTrainer):
    """Trainer using teacher forcing for baseline comparison.

    At every step, the ground-truth state is fed as input (no autoregressive
    rollout). Loss is the unweighted average over all prediction steps.

    Loss formulation (T = number of prediction steps):
        L = (1/T) * sum_{t=0}^{T-1} criterion(model(x_t), x_{t+1})
    """

    def __init__(self, model: nn.Module,
                 lr: float = 1e-3, max_epochs: int = 500,
                 weight_decay: float = 1e-5, eta_min: float = 1e-6,
                 **kwargs):
        """Initialize TeacherForcingTrainer.

        Args:
            model: The neural network.
            lr: Initial learning rate for AdamW.
            max_epochs: Total training epochs; sets CosineAnnealingLR period.
            weight_decay: L2 regularization coefficient for AdamW.
            eta_min: Minimum learning rate for cosine annealing.
            **kwargs: Passed to BaseTrainer (e.g., scalers, output_dir, device).
        """
        optimizer = kwargs.pop("optimizer", None)
        scheduler = kwargs.pop("scheduler", None)
        criterion = kwargs.pop("criterion", None)

        if optimizer is None:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if scheduler is None:
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)
        if criterion is None:
            criterion = NMSECriterion()

        super().__init__(model, lr=lr, max_epochs=max_epochs,
                         optimizer=optimizer, scheduler=scheduler,
                         criterion=criterion, **kwargs)

    def _compute_loss(self, batch: Any) -> Tensor:
        """Compute teacher-forcing loss averaged over all steps.

        At each step t, the ground-truth x_t is used as input to predict x_{t+1}.

        Args:
            batch: Tuple (seq, coords, start_t_norm, dt_norm) where
                seq: (B, T, N, C) — sequence of states.
                coords: (B, N, spatial_dim) — spatial coordinates.
                start_t_norm: (B,) — normalized start time.
                dt_norm: (B,) — normalized time step.

        Returns:
            Scalar loss averaged over all prediction steps.
        """
        seq, coords, start_t_norm, dt_norm = batch
        num_steps = seq.shape[1] - 1
        loss = torch.tensor(0.0, device=self.device)

        for t in range(num_steps):
            input_state = seq[:, t]

            if coords is not None:
                if hasattr(self.model, "time_encoder"):
                    t_norm = start_t_norm + t * dt_norm
                    pred = self.model(input_state, coords, t_norm=t_norm)
                else:
                    pred = self.model(input_state, coords)
            else:
                pred = self.model(input_state)

            loss = loss + self.criterion(pred, seq[:, t + 1])

        return loss / num_steps

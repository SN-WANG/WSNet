# Rollout Trainer with Weighted Full-Rollout and Curriculum Learning
# Author: Shengning Wang

import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Optional

from wsnet.training.base_trainer import BaseTrainer
from wsnet.training.base_criterion import NMSECriterion
from wsnet.utils.hue_logger import hue, logger


class RolloutTrainer(BaseTrainer):
    """
    Trainer for autoregressive sequence prediction with weighted full-rollout and curriculum learning.

    Assumes the model follows an autoregressive pattern: x_{t+1} = model(x_t, conditions).

    Training strategy:
    - Full BPTT rollout: gradients flow through all rollout steps without detach. The
      cross-step Jacobian product teaches the model that early prediction errors are
      amplified at later steps — critical for high-pressure, nonlinear flow regimes.
    - Noise injection: adds Gaussian noise to each step's input to expose the model to
      the realistic input distribution it faces during inference (partially addresses the
      distribution shift between training and inference inputs).
    - Linear step weighting: loss at step t is weighted by w_t = 2t / (k*(k+1)), so
      later steps contribute proportionally more to the total gradient signal.
      This emphasizes long-range accuracy over short-step accuracy, consistent with the
      deployment objective of stable 1000-step autoregressive prediction.
    - Curriculum schedule: rollout steps increase by 1 every `rollout_patience` epochs,
      starting from 1. Noise std is decayed multiplicatively at each curriculum advance.

    Loss formulation (k = current_rollout_steps):
        L = sum_{t=1}^{k} w_t * NMSE(x_hat_t, x_t),   w_t = 2t / (k*(k+1))

    Configs:
    1. Default optimizer: AdamW
    2. Default scheduler: CosineAnnealingLR
    3. Default criterion: NMSECriterion (per-channel normalized MSE)
    4. Curriculum: epoch-counter-based (not loss-based), advances every rollout_patience epochs
    """

    def __init__(self, model: nn.Module,
                 # optimization params
                 lr: float = 1e-3, max_epochs: int = 500,
                 weight_decay: float = 1e-5, eta_min: float = 1e-6,
                 # curriculum params
                 max_rollout_steps: int = 5, rollout_patience: int = 40,
                 noise_std_init: float = 0.05, noise_decay: float = 0.9,
                 # boundary condition
                 boundary_condition: Optional[Any] = None,
                 # base params
                 **kwargs):
        """
        Args:
            model (nn.Module): The neural network.
            lr (float): Initial learning rate for AdamW.
            max_epochs (int): Total training epochs; sets CosineAnnealingLR period.
            weight_decay (float): L2 regularization coefficient for AdamW.
            eta_min (float): Minimum learning rate for cosine annealing.
            max_rollout_steps (int): Maximum autoregressive rollout steps (curriculum ceiling).
            rollout_patience (int): Number of epochs between curriculum advances.
            noise_std_init (float): Initial std dev of Gaussian noise injected into inputs.
            noise_decay (float): Multiplicative decay applied to noise_std at each curriculum advance.
            boundary_condition: Optional BoundaryCondition instance for hard BC enforcement
                during rollout. When provided, wall-node predictions are replaced with known
                Dirichlet values after each forward step, preventing error accumulation.
            **kwargs: Passed to BaseTrainer (e.g., scalers, output_dir, device, criterion).
        """

        # 1. Initialize BaseTrainer defaults
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
                         optimizer=optimizer, scheduler=scheduler, criterion=criterion, **kwargs)

        # 2. Store curriculum hyperparameters
        self.max_rollout_steps = max_rollout_steps
        self.rollout_patience = rollout_patience
        self.noise_std_init = noise_std_init
        self.noise_decay = noise_decay

        # 3. Initialize curriculum state
        self.rollout_counter = 0
        self.log_update_info = False
        self.current_rollout_steps = 1
        self.current_noise_std = noise_std_init

        # 4. Boundary condition enforcement
        self.boundary_condition = boundary_condition

    def _update_curriculum(self) -> None:
        """
        Advance curriculum state every rollout_patience epochs.

        Strategy: increase rollout difficulty (more steps) and decrease training
        assistance (less noise). The epoch counter resets after each advance so
        the model has a fixed number of epochs to stabilize at each difficulty level.
        """
        self.rollout_counter += 1

        if self.rollout_counter >= self.rollout_patience:
            if self.current_rollout_steps < self.max_rollout_steps:
                self.current_rollout_steps += 1
                self.current_noise_std *= self.noise_decay
                self.rollout_counter = 0
                self.log_update_info = True

        if self.log_update_info and self.rollout_counter == 1:
            logger.info(f"{hue.y}curriculum update:{hue.q} "
                        f"steps = {hue.m}{self.current_rollout_steps}{hue.q}, "
                        f"noise = {hue.m}{self.current_noise_std:.4f}{hue.q}")
            self.log_update_info = False

    def _on_epoch_end(self, train_loss=None, val_loss=None, **kwargs) -> None:
        """Advance curriculum state at the end of each epoch."""
        self._update_curriculum()

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Compute weighted full-rollout loss with noise injection.

        Algorithm:
        1. Parse batch: (seq, coords).
        2. Initialize rollout state from x_0.
        3. For each step t in {1, ..., k}:
            a. Noise injection: noisy_input = x_t + eps  (training only, eps ~ N(0, sigma^2))
            b. Predict: x_hat_{t+1} = f(noisy_input, coords, [step=t])
            c. Accumulate: L += w_t * criterion(x_hat_{t+1}, x_{t+1})
            d. State update: x_{t+1} = x_hat_{t+1}  (full BPTT — no detach)

        Linear step weights w_t = 2t / (k*(k+1)) for t in {1, ..., k}:
            - sum(w_t) = 1 (normalized)
            - w_k / w_1 = k  (last step weighted k times the first step)
            - Rationale: deployment objective is long-horizon accuracy; the model
              should not sacrifice step-k stability for marginally better step-1 loss.

        Args:
            batch: Tuple (seq, coords) where
                   seq shape: (batch_size, win_len, num_nodes, num_channels)
                   coords shape: (batch_size, num_nodes, spatial_dim) or None.

        Returns:
            Scalar weighted loss. Shape: ()
        """

        seq, coords, start_t_norm, dt_norm = batch

        k = self.current_rollout_steps

        # Linear step weights: w_t = 2t / (k*(k+1)), t = 1..k (sum = 1)
        total_weight = k * (k + 1)
        step_weights = [2.0 * (t + 1) / total_weight for t in range(k)]

        input_state = seq[:, 0]
        loss = torch.tensor(0.0, device=self.device)

        for t in range(k):
            # a. Noise injection (training only, skip if std is negligible)
            if self.model.training and self.current_noise_std > 1e-6:
                input_state = input_state + torch.randn_like(input_state) * self.current_noise_std

            # b. Forward prediction
            if coords is not None:
                if hasattr(self.model, "time_encoder"):
                    t_norm = start_t_norm + t * dt_norm  # (B,) float tensor
                    pred_state = self.model(input_state, coords, t_norm=t_norm)
                else:
                    pred_state = self.model(input_state, coords)
            else:
                pred_state = self.model(input_state)

            # b2. Hard BC enforcement: replace wall-node predictions with known values
            if self.boundary_condition is not None:
                pred_state = self.boundary_condition.enforce(pred_state)

            # c. Weighted loss accumulation
            target_state = seq[:, t + 1]
            loss = loss + step_weights[t] * self.criterion(pred_state, target_state)

            # d. State update — full BPTT, no detach
            input_state = pred_state

        return loss

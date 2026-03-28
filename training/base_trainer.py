# Base Trainer with Training Loop, Checkpointing and Evaluation
# Author: Shengning Wang

import json
import time
import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from utils.scaler import StandardScalerTensor
from utils.hue_logger import hue, logger

class BaseTrainer:
    """
    Base class encapsulating the training loop, checkpointing, and evaluation logic.
    Subclasses must implement 'compute_loss' to define specific task logic.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, max_epochs: int = 100, patience: int = None,
                 scalers: Optional[Dict[str, StandardScalerTensor]] = None,
                 output_dir: Optional[Union[str, Path]] = "./runs",
                 optimizer: Optional[Optimizer] = None, scheduler: Optional[_LRScheduler] = None,
                 criterion: Optional[nn.Module] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The neural network.
            lr (float): Initial learning rate for the optimizer, defaults to 1e-3.
            max_epochs (int): Maximum training epochs, defaults to 100.
            patience (int): Epochs to wait before early stopping if no improvement, defaults to max_epochs.
            scalers (Optional[Dict[str, StandardScalerTensor]]): Dictionary of scalers to save.
            output_dir (Union[str, Path]): Directory to save artifacts, defaults to "./runs".
            optimizer (Optional[Optimizer]): Optimizer instance, defaults to Adam.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler, defaults to None.
            criterion (Optional[nn.Module]): Loss function, defaults to MSELoss.
            device (str): Computation device, defaults to "cuda" or "cpu".
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.scalers = scalers

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optimizer if optimizer else Adam(self.model.parameters(), lr=lr)
        self.scheduler = scheduler
        self.criterion = criterion if criterion else nn.MSELoss()

        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.patience = patience if patience else max_epochs

        self.best_loss = float("inf")
        self.history: List[Dict[str, Any]] = []

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Abstract method: Calculate the loss for a single batch.
        Must be implemented by subclasses.

        Args:
            batch (Any): Data batch from DataLoader.

        Returns:
            Tensor: Scalar loss tensor (Attached to graph). Shape: (1,)
        """
        raise NotImplementedError('Subclasses must implement _compute_loss.')

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> float:
        """
        Runs a single epoch of training or validation.

        Args:
            loader (DataLoader): The data loader.
            is_training (bool): Whether gradients should be computed.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train(is_training)
        losses = []

        context = torch.enable_grad() if is_training else torch.no_grad()

        with context:
            pbar = tqdm(loader, desc="Training" if is_training else "Validating", leave=False, dynamic_ncols=True)
            for batch in pbar:
                # move batch to device (handling lists/tuples generic logic)
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if isinstance(b, Tensor) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

                loss = self._compute_loss(batch)

                if is_training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                loss_val = loss.item()
                losses.append(loss_val)
                pbar.set_postfix({"loss": f"{loss_val:.4e}"})

        return float(np.mean(losses))

    def _on_epoch_start(self, train_loss=None, val_loss=None, **kwargs) -> None:
        """
        Optional hook called at the start of each epoch.
        Default implementation is a no-op.
        """
        pass

    def _on_epoch_end(self, train_loss=None, val_loss=None, **kwargs) -> None:
        """
        Optional hook called at the end of each epoch.
        Default implementation is a no-op.
        """
        pass

    def _save_checkpoint(self, val_loss: float, is_best: bool = False, extra_state: Dict = {}) -> None:
        """
        Save the training state.
        """
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            **extra_state
        }
        if self.scalers:
            state["scaler_state_dict"] = {k: v.state_dict() for k, v in self.scalers.items()}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state, self.output_dir / "ckpt.pt")
        if is_best:
            torch.save(state, self.output_dir / "best.pt")

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """
        Main training loop.
        """
        logger.info(f"start training on {hue.m}{self.device}{hue.q} with {hue.m}{self.max_epochs}{hue.q} epochs")
        start_time = time.time()
        patience_counter = 0

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch + 1
            ep_start = time.time()

            # call hook function
            self._on_epoch_start()

            # train & validate
            train_loss = self._run_epoch(train_loader, is_training=True)
            val_loss = self._run_epoch(val_loader, is_training=False) if val_loader else None

            # call hook function
            self._on_epoch_end(train_loss, val_loss)

            # scheduler step
            if self.scheduler:
                self.scheduler.step()

            # check best model
            is_best = val_loss and val_loss < self.best_loss
            if is_best:
                val_str = f" | val loss: {hue.m}{val_loss:.4e} {hue.y}(best){hue.q}"
                self.best_loss = val_loss
                patience_counter = 0
            else:
                val_str = f" | val loss: {hue.m}{val_loss:.4e}{hue.q}" if val_loss else ""
                patience_counter += 1

            # save checkpoint
            self._save_checkpoint(val_loss, is_best)

            # log info
            duration = time.time() - ep_start
            logger.info(f'epoch {hue.b}{self.current_epoch:03d}{hue.q} | time: {hue.c}{duration:.1f}s{hue.q} '
                        f'| train loss: {hue.m}{train_loss:.4e}{hue.q}{val_str}')
            self.history.append({'epoch': self.current_epoch, 'train_loss': train_loss,
                                 'val_loss': val_loss, 'lr': self.optimizer.param_groups[0]['lr']})

            # early stop
            if patience_counter >= self.patience:
                logger.info(f"early stopping triggered at epoch {hue.m}{self.current_epoch}{hue.q}")
                break

        # save history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"{hue.g}training finished in {time.time() - start_time:.1f}s{hue.q}")

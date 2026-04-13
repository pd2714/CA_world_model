"""Training loop and checkpoint management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import compute_training_loss
from src.utils.io import ensure_dir
from src.utils.logging_utils import CSVLogger, save_run_config


@dataclass
class Trainer:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader
    val_loader: DataLoader
    config: dict[str, Any]
    run_dir: Path
    device: torch.device

    def __post_init__(self) -> None:
        self.run_dir = ensure_dir(self.run_dir)
        save_run_config(self.run_dir, self.config)
        self.logger = CSVLogger(self.run_dir)
        self.best_val = float("inf")

    def _run_epoch(self, loader: DataLoader, training: bool, epoch: int) -> dict[str, float]:
        self.model.train(training)
        stats: dict[str, list[float]] = {}
        desc = "train" if training else "val"
        for batch in tqdm(loader, desc=desc, leave=False):
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_training_loss(self.model, batch, self.config, self.device, epoch=epoch)
            if training:
                loss.backward()
                grad_clip = float(self.config["train"].get("grad_clip", 0.0))
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()
            for key, value in metrics.items():
                stats.setdefault(key, []).append(value)
        return {key: float(sum(values) / max(1, len(values))) for key, values in stats.items()}

    def save_checkpoint(self, name: str, epoch: int, val_loss: float) -> Path:
        path = self.run_dir / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": self.config,
            },
            path,
        )
        return path

    def fit(self) -> Path:
        epochs = int(self.config["train"].get("epochs", 10))
        patience = int(self.config["train"].get("early_stopping_patience", 0))
        stale_epochs = 0
        best_path = self.run_dir / "best.ckpt"
        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, training=True, epoch=epoch)
            val_metrics = self._run_epoch(self.val_loader, training=False, epoch=epoch)
            val_loss = val_metrics.get("loss_total", float("inf"))
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self.logger.log(row)
            self.save_checkpoint("last.ckpt", epoch, val_loss)
            if val_loss < self.best_val:
                self.best_val = val_loss
                stale_epochs = 0
                best_path = self.save_checkpoint("best.ckpt", epoch, val_loss)
            else:
                stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                break
        return best_path

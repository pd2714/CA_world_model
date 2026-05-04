"""Baseline models and utilities for baseline comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.ca.datasets import NextStepDataset
from src.models.pixel_predictor import PixelPredictor
from src.utils.io import ensure_dir
from src.utils.seed import set_seed

from analysis.utils import DEFAULT_SEED, RunSpec, dataset_path, log


class CopyPreviousBaseline:
    name = "copy_previous"

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, steps: int) -> tuple[torch.Tensor, None]:
        states = [x0]
        current = x0
        for _ in range(steps):
            states.append(current.clone())
        return torch.stack(states, dim=1), None


class MajorityLocalBaseline:
    name = "majority_local"

    @staticmethod
    def step(x: torch.Tensor) -> torch.Tensor:
        left = torch.roll(x, shifts=1, dims=-1)
        right = torch.roll(x, shifts=-1, dims=-1)
        votes = left + x + right
        return (votes >= 2.0).float()

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, steps: int) -> tuple[torch.Tensor, None]:
        states = [x0]
        current = x0
        for _ in range(steps):
            current = self.step(current)
            states.append(current)
        return torch.stack(states, dim=1), None


class FittedRuleTableBaseline:
    name = "fitted_rule_table"

    def __init__(self, lookup: torch.Tensor) -> None:
        self.lookup = lookup.float()

    @classmethod
    def fit(cls, train_trajectories: torch.Tensor) -> "FittedRuleTableBaseline":
        x = train_trajectories[:, :-1, 0]
        y = train_trajectories[:, 1:, 0]
        left = torch.roll(x, shifts=1, dims=-1)
        right = torch.roll(x, shifts=-1, dims=-1)
        neighborhoods = ((left.long() << 2) | (x.long() << 1) | right.long()).reshape(-1)
        targets = y.reshape(-1).long()
        counts = torch.zeros(8, 2, dtype=torch.float32)
        for neighborhood, target in zip(neighborhoods, targets):
            counts[int(neighborhood), int(target)] += 1.0
        total = counts.sum(dim=1)
        default_value = int(counts.sum(dim=0).argmax().item()) if counts.sum().item() > 0 else 0
        lookup = torch.full((8,), float(default_value))
        for index in range(8):
            if total[index] > 0:
                lookup[index] = float(counts[index].argmax().item())
        return cls(lookup=lookup)

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, steps: int) -> tuple[torch.Tensor, None]:
        states = [x0]
        current = x0
        lookup = self.lookup.to(x0.device)
        for _ in range(steps):
            left = torch.roll(current, shifts=1, dims=-1)
            right = torch.roll(current, shifts=-1, dims=-1)
            neighborhoods = ((left.long() << 2) | (current.long() << 1) | right.long()).long()
            current = lookup[neighborhoods].float()
            states.append(current)
        return torch.stack(states, dim=1), None


@dataclass
class LearnedCNNBaseline:
    model: PixelPredictor
    metrics: pd.DataFrame
    name: str = "learned_cnn"

    @classmethod
    def fit_or_load(
        cls,
        spec: RunSpec,
        data_dir: str | Path,
        device: torch.device,
        cache_dir: str | Path,
        epochs: int = 8,
        seed: int = DEFAULT_SEED,
    ) -> "LearnedCNNBaseline" | None:
        train_path = dataset_path(spec.config, data_dir, "train")
        val_path = dataset_path(spec.config, data_dir, "val")
        if not train_path.exists() or not val_path.exists():
            log(f"Skipping learned CNN baseline for {spec.run_name}: missing train/val dataset.")
            return None

        cache_dir = ensure_dir(cache_dir)
        cache_path = cache_dir / f"{spec.rule_label}_{spec.dataset_prefix}_learned_cnn.pt"
        metrics_path = cache_dir / f"{spec.rule_label}_{spec.dataset_prefix}_learned_cnn_metrics.csv"
        model = PixelPredictor(dimension=spec.dimension, hidden_channels=32, depth=3, kernel_size=3).to(device)

        if cache_path.exists():
            payload = torch.load(cache_path, map_location=device)
            model.load_state_dict(payload["model_state"])
            metric_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
            model.eval()
            return cls(model=model, metrics=metric_df)

        set_seed(seed)
        train_loader = DataLoader(NextStepDataset.from_npz(train_path), batch_size=128, shuffle=True, num_workers=0)
        val_loader = DataLoader(NextStepDataset.from_npz(val_path), batch_size=128, shuffle=False, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        rows: list[dict[str, float | int]] = []

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            train_batches = 0
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                train_batches += 1

            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    y = batch["y"].to(device)
                    logits = model(x)
                    val_loss += float(criterion(logits, y).item())
                    val_batches += 1

            train_mean = train_loss / max(1, train_batches)
            val_mean = val_loss / max(1, val_batches)
            rows.append({"epoch": epoch, "train_bce": train_mean, "val_bce": val_mean})
            if val_mean < best_loss:
                best_loss = val_mean
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        metric_df = pd.DataFrame(rows)
        metric_df.to_csv(metrics_path, index=False)
        if best_state is None:
            return None
        model.load_state_dict(best_state)
        model.eval()
        torch.save({"model_state": best_state, "metrics": rows}, cache_path)
        return cls(model=model, metrics=metric_df)

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        states = [x0]
        logits_seq = [torch.full_like(x0, float("nan"))]
        current = x0
        for _ in range(steps):
            logits = self.model(current)
            current = (torch.sigmoid(logits) >= 0.5).float()
            logits_seq.append(logits)
            states.append(current)
        return torch.stack(states, dim=1), torch.stack(logits_seq, dim=1)


def build_baselines(
    spec: RunSpec,
    train_trajectories: torch.Tensor,
    data_dir: str | Path,
    device: torch.device,
    cache_dir: str | Path,
) -> list[tuple[str, Any]]:
    baselines: list[tuple[str, Any]] = [
        (CopyPreviousBaseline.name, CopyPreviousBaseline()),
        (MajorityLocalBaseline.name, MajorityLocalBaseline()),
        (FittedRuleTableBaseline.name, FittedRuleTableBaseline.fit(train_trajectories)),
    ]
    learned = LearnedCNNBaseline.fit_or_load(
        spec=spec,
        data_dir=data_dir,
        device=device,
        cache_dir=cache_dir,
    )
    if learned is not None:
        baselines.append((learned.name, learned))
    return baselines


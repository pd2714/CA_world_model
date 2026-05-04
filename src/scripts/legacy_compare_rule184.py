"""Compare the legacy Rule 184 rollout-only checkpoint against exact rule rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from src.ca.datasets import CADataset
from src.ca.visualization import plot_trajectory_comparison
from src.training.rollout import model_rollout
from src.utils.factory import build_automaton
from src.models.decoder import ConvDecoder1D
from src.models.encoder import ConvEncoder1D


class LegacySpatialLatentDynamics1D(nn.Module):
    """Exact 1D latent dynamics used by the initial checkpointed Rule 184 runs."""

    def __init__(
        self,
        channels: int = 32,
        depth: int = 3,
        step_size: float = 1.0,
        use_post_norm: bool = False,
        clamp_delta: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Conv1d(channels, channels, 3, padding=1, padding_mode="circular"), nn.GELU()])
        self.net = nn.Sequential(*layers)
        self.step_size = float(step_size)
        self.post_norm = nn.GroupNorm(1, channels) if use_post_norm else nn.Identity()
        self.clamp_delta = float(clamp_delta)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.net(z)
        if self.clamp_delta > 0.0:
            delta = torch.clamp(delta, min=-self.clamp_delta, max=self.clamp_delta)
        return self.post_norm(z + self.step_size * delta)


class LegacyDenseWorldModel(nn.Module):
    """Minimal legacy dense world model matching the original Rule 184 checkpoint layout."""

    def __init__(
        self,
        channels: int = 32,
        depth: int = 3,
        step_size: float = 1.0,
        use_post_norm: bool = False,
        clamp_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = ConvEncoder1D(latent_channels=channels, depth=depth)
        self.dynamics = LegacySpatialLatentDynamics1D(
            channels=channels,
            depth=depth,
            step_size=step_size,
            use_post_norm=use_post_norm,
            clamp_delta=clamp_delta,
        )
        self.decoder = ConvDecoder1D(latent_channels=channels, depth=depth)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.dynamics(z)


def build_legacy_rule184_model(config: dict) -> LegacyDenseWorldModel:
    model_cfg = config["model"]
    return LegacyDenseWorldModel(
        channels=int(model_cfg["latent_channels"]),
        depth=int(model_cfg["depth"]),
        step_size=float(model_cfg.get("dynamics_step_size", 1.0)),
        use_post_norm=bool(model_cfg.get("dynamics_use_post_norm", False)),
        clamp_delta=float(model_cfg.get("dynamics_clamp_delta", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Legacy Rule 184 run directory with best.ckpt and config.json.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--sample_idx", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with (run_dir / "config.json").open() as fh:
        config = json.load(fh)

    model = build_legacy_rule184_model(config).to(args.device)
    ckpt = torch.load(run_dir / "best.ckpt", map_location=args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset_prefix = config.get("dataset_name", config["experiment_name"])
    dataset = CADataset.from_npz(Path(args.data_dir) / f"{dataset_prefix}_test.npz")
    sample = dataset.trajectories[args.sample_idx : args.sample_idx + 1].to(args.device)

    automaton = build_automaton(config)
    exact = automaton.rollout(
        dataset.trajectories[args.sample_idx, 0, 0].numpy().astype("uint8"),
        steps=int(args.horizon),
    ).astype("float32")
    pred = model_rollout(model, sample[:, 0], steps=int(args.horizon))["states"][0, :, 0].cpu().numpy()

    output_dir = Path(args.output_dir or run_dir / f"legacy_compare_h{args.horizon}")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"compare_sample_{args.sample_idx}_h{args.horizon}.png"
    plot_trajectory_comparison(
        exact,
        pred,
        title=f"rule184 legacy sample {args.sample_idx} horizon {args.horizon}",
        path=out_path,
    )
    print(out_path)


if __name__ == "__main__":
    main()

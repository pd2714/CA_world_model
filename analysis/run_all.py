"""Run the full evaluation and latent-analysis suite."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.latent_probes import run_linear_probes, run_locality_probe, run_rule184_conservation_test
from analysis.latent_viz import run_latent_channel_visualization, run_latent_embedding_analysis
from analysis.plotting import set_analysis_style
from analysis.sanity_checks import (
    run_baseline_comparison,
    run_lattice_size_generalization,
    run_ood_density_test,
    run_rollout_sanity_checks,
)
from analysis.utils import DEFAULT_SEED, discover_run_specs, load_model_from_checkpoint, load_trajectories, log, prepare_output_dir, save_manifest
from src.utils.runtime import configure_runtime
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="*", default=None, help="Experiment config(s) or saved run config.json path(s).")
    parser.add_argument("--checkpoint", nargs="*", default=None, help="Checkpoint path(s) to analyze.")
    parser.add_argument("--search_root", nargs="*", default=["outputs"], help="Search root(s) used when config/checkpoint are omitted.")
    parser.add_argument("--rules", nargs="*", type=int, default=None, help="Only analyze these rules.")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_root", default="analysis/outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu_threads", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--probe_points", type=int, default=12000)
    parser.add_argument("--checkpoint_name", default="best.ckpt")
    args = parser.parse_args()

    configure_runtime(cpu_threads=args.cpu_threads, interop_threads=1)
    set_seed(DEFAULT_SEED)
    set_analysis_style()
    device = torch.device(args.device)
    rules = set(args.rules) if args.rules else None

    specs = discover_run_specs(
        config_paths=args.config,
        checkpoint_paths=args.checkpoint,
        search_roots=args.search_root,
        output_root=args.output_root,
        rules=rules,
        checkpoint_name=args.checkpoint_name,
    )
    if not specs:
        log("No runs found for analysis.")
        return

    manifest_rows: list[dict[str, object]] = []
    for spec in specs:
        log(f"Starting analysis for {spec.run_name} ({spec.rule_label}, {spec.model_name})")
        run_dir = prepare_output_dir(spec.output_dir)
        errors_path = run_dir / "errors.csv"
        if errors_path.exists():
            errors_path.unlink()
        run_errors: list[dict[str, str]] = []
        try:
            model = load_model_from_checkpoint(spec, device)
        except Exception as exc:  # pragma: no cover - defensive logging
            log(f"Failed to load {spec.run_name}: {exc}")
            run_errors.append({"stage": "load_model", "error": str(exc)})
            save_manifest(run_errors, run_dir / "errors.csv")
            continue

        try:
            train_trajectories = load_trajectories(spec.config, args.data_dir, "train")[: args.max_samples]
            test_trajectories = load_trajectories(spec.config, args.data_dir, "test")[: args.max_samples]
        except Exception as exc:
            log(f"Missing dataset for {spec.run_name}: {exc}")
            run_errors.append({"stage": "load_data", "error": str(exc)})
            save_manifest(run_errors, run_dir / "errors.csv")
            continue

        stages = [
            ("sanity_checks", lambda: run_rollout_sanity_checks(spec, model, test_trajectories, device, run_dir / "sanity_checks")),
            (
                "baseline_comparison",
                lambda: run_baseline_comparison(
                    spec,
                    model,
                    train_trajectories=train_trajectories,
                    test_trajectories=test_trajectories,
                    data_dir=args.data_dir,
                    device=device,
                    output_dir=run_dir / "baseline_comparison",
                ),
            ),
            ("ood_density", lambda: run_ood_density_test(spec, model, device, run_dir / "ood_density", num_samples=args.max_samples)),
            ("lattice_generalization", lambda: run_lattice_size_generalization(spec, model, device, run_dir / "lattice_generalization", num_samples=args.max_samples)),
            ("latent_channels", lambda: run_latent_channel_visualization(spec, model, test_trajectories, device, run_dir / "latent_viz")),
            ("latent_embeddings", lambda: run_latent_embedding_analysis(spec, model, test_trajectories, run_dir / "latent_embeddings", max_points=args.probe_points)),
            ("linear_probes", lambda: run_linear_probes(spec, model, test_trajectories, run_dir / "linear_probes", max_points=args.probe_points)),
            ("locality_probe", lambda: run_locality_probe(spec, model, test_trajectories, run_dir / "locality_probe", max_points=max(args.probe_points, 20000))),
            ("rule184_conservation", lambda: run_rule184_conservation_test(spec, model, test_trajectories, device, run_dir / "rule184_conservation")),
        ]

        for stage_name, stage_fn in stages:
            log(f"{spec.run_name}: {stage_name}")
            try:
                result = stage_fn()
                manifest_rows.append(
                    {
                        "run_name": spec.run_name,
                        "rule": spec.rule,
                        "model_name": spec.model_name,
                        "stage": stage_name,
                        "status": result.get("status", "ok"),
                        "output_dir": str(run_dir / stage_name if stage_name != "latent_channels" else run_dir / "latent_viz"),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                log(f"{spec.run_name}: {stage_name} failed: {exc}")
                run_errors.append({"stage": stage_name, "error": str(exc)})
                manifest_rows.append(
                    {
                        "run_name": spec.run_name,
                        "rule": spec.rule,
                        "model_name": spec.model_name,
                        "stage": stage_name,
                        "status": "failed",
                        "output_dir": str(run_dir),
                    }
                )

        if run_errors:
            save_manifest(run_errors, errors_path)

    save_manifest(manifest_rows, Path(args.output_root) / "analysis_manifest.csv")
    log(f"Finished analysis for {len(specs)} run(s).")


if __name__ == "__main__":
    main()

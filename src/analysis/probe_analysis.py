"""Probe analysis for latent sufficiency."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.probes import fit_linear_probe


def run_probe_suite(features: dict[str, object], out_path: str | Path) -> pd.DataFrame:
    rows = []
    z = features["latent"]
    for name, target in features.items():
        if name == "latent":
            continue
        _, result = fit_linear_probe(z, target, name=name)
        rows.append({"probe": result.name, "mse": result.mse, "r2": result.r2})
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df

"""Runtime controls for conservative CPU usage."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def configure_runtime(
    cpu_threads: int = 2,
    interop_threads: int = 1,
    mpl_config_dir: str | None = "outputs/.mplconfig",
) -> None:
    """Set conservative defaults so experiments do not saturate the machine."""
    cpu_threads = max(1, int(cpu_threads))
    interop_threads = max(1, int(interop_threads))

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(cpu_threads)

    if mpl_config_dir:
        path = Path(mpl_config_dir)
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(path))

    torch.set_num_threads(cpu_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            pass


def dataloader_num_workers(config: dict | None = None) -> int:
    if config is None:
        return 0
    return int(config.get("train", {}).get("num_workers", 0))

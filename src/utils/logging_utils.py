"""Lightweight run logging."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, save_json


@dataclass
class CSVLogger:
    """Append-only CSV logger for metrics."""

    run_dir: Path
    filename: str = "metrics.csv"
    fieldnames: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        ensure_dir(self.run_dir)
        self.path = self.run_dir / self.filename

    def log(self, row: dict[str, Any]) -> None:
        row_keys = list(row.keys())
        if not self.fieldnames:
            self.fieldnames = row_keys
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        new_fields = [key for key in row_keys if key not in self.fieldnames]
        if new_fields:
            self.fieldnames.extend(new_fields)
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                existing_rows = list(csv.DictReader(handle))
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()
                for existing in existing_rows:
                    writer.writerow(existing)
                writer.writerow(row)
            return

        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_run_config(run_dir: str | Path, config: dict[str, Any]) -> None:
    ensure_dir(run_dir)
    save_json(config, Path(run_dir) / "config.json")

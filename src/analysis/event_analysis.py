"""Event-focused analysis helpers for coherent structures in 1D CA."""

from __future__ import annotations

import numpy as np

from src.ca.observables import simple_event_score_1d


RULE54_PATTERNS = {
    "double_seed": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    "collision_seed": [1, 1, 0, 1, 0, 0, 1, 1, 0],
}

RULE110_PATTERNS = {
    "moving_packet": [1, 1, 0, 1, 1, 0, 0, 1],
    "collision_seed": [1, 0, 1, 1, 0, 1, 1, 0, 0],
}


def curated_seed(rule: int, length: int, name: str) -> np.ndarray:
    patterns = RULE54_PATTERNS if rule == 54 else RULE110_PATTERNS
    pattern = patterns[name]
    state = np.zeros(length, dtype=np.uint8)
    start = max(0, (length - len(pattern)) // 2)
    state[start : start + len(pattern)] = np.array(pattern, dtype=np.uint8)
    return state


def event_summary(trajectory: np.ndarray) -> dict[str, float]:
    score = simple_event_score_1d(trajectory)
    return {
        "mean_event_score": float(score.mean()),
        "max_event_score": float(score.max()) if len(score) > 0 else 0.0,
    }

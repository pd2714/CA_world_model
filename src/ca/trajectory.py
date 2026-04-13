"""Trajectory generation for cellular automata."""

from __future__ import annotations

from typing import Callable

import numpy as np

from src.ca.base import CellularAutomaton


def bernoulli_initial_state_1d(length: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(1, p, size=(length,)).astype(np.uint8)


def bernoulli_initial_state_2d(height: int, width: int, p: float, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(1, p, size=(height, width)).astype(np.uint8)


def sparse_seed_1d(length: int, num_active: int, rng: np.random.Generator) -> np.ndarray:
    state = np.zeros(length, dtype=np.uint8)
    indices = rng.choice(length, size=min(num_active, length), replace=False)
    state[indices] = 1
    return state


def centered_pattern_1d(length: int, pattern: list[int]) -> np.ndarray:
    state = np.zeros(length, dtype=np.uint8)
    start = max(0, (length - len(pattern)) // 2)
    stop = min(length, start + len(pattern))
    state[start:stop] = np.array(pattern[: stop - start], dtype=np.uint8)
    return state


def resolve_bernoulli_p(config: dict, rng: np.random.Generator) -> float:
    if "p_range" in config:
        low, high = config["p_range"]
        return float(rng.uniform(float(low), float(high)))
    if "p_list" in config:
        values = [float(value) for value in config["p_list"]]
        if not values:
            raise ValueError("initial_condition.p_list must not be empty")
        return float(rng.choice(values))
    return float(config.get("p", 0.5))


def sample_initial_state(config: dict, rng: np.random.Generator) -> np.ndarray:
    kind = config.get("kind", "bernoulli")
    dimension = config.get("dimension", "1d")
    if dimension == "1d":
        length = int(config["size"])
        if kind == "bernoulli":
            return bernoulli_initial_state_1d(length, resolve_bernoulli_p(config, rng), rng)
        if kind == "sparse":
            return sparse_seed_1d(length, int(config.get("num_active", 4)), rng)
        if kind == "pattern":
            return centered_pattern_1d(length, list(config.get("pattern", [1, 1, 1])))
    height, width = config["size"]
    if kind == "bernoulli":
        return bernoulli_initial_state_2d(int(height), int(width), resolve_bernoulli_p(config, rng), rng)
    raise ValueError(f"Unsupported initial condition config: {config}")


def rollout_batch(
    automaton: CellularAutomaton,
    batch_size: int,
    steps: int,
    sampler: Callable[[], np.ndarray],
) -> np.ndarray:
    trajectories = [automaton.rollout(sampler(), steps=steps) for _ in range(batch_size)]
    return np.stack(trajectories, axis=0).astype(np.float32)

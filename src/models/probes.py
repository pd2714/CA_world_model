"""Linear probe helpers for latent analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:  # pragma: no cover - fallback path
    Ridge = None

    def mean_squared_error(targets: np.ndarray, preds: np.ndarray) -> float:
        return float(np.mean((targets - preds) ** 2))

    def r2_score(targets: np.ndarray, preds: np.ndarray) -> float:
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-8))


@dataclass
class ProbeResult:
    name: str
    mse: float
    r2: float


def fit_linear_probe(features: np.ndarray, targets: np.ndarray, name: str) -> tuple[Ridge, ProbeResult]:
    if Ridge is not None:
        model = Ridge(alpha=1.0)
        model.fit(features, targets)
        preds = model.predict(features)
    else:
        features_aug = np.concatenate([features, np.ones((features.shape[0], 1), dtype=features.dtype)], axis=1).astype(np.float64)
        target_array = np.asarray(targets, dtype=np.float64)
        scale = np.std(features_aug, axis=0, keepdims=True)
        scale[scale < 1e-6] = 1.0
        normalized = features_aug / scale
        coef, _, _, _ = np.linalg.lstsq(normalized, target_array, rcond=1e-4)
        coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)

        class NumpyRidge:
            def __init__(self, coef_: np.ndarray, scale_: np.ndarray) -> None:
                self.coef_ = coef_
                self.scale_ = scale_

            def predict(self, x: np.ndarray) -> np.ndarray:
                x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1).astype(np.float64)
                normalized_x = np.nan_to_num(x_aug / self.scale_, nan=0.0, posinf=0.0, neginf=0.0)
                coef = np.clip(np.nan_to_num(self.coef_, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    preds = normalized_x @ coef
                return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

        model = NumpyRidge(coef, scale)
        preds = model.predict(features)
    result = ProbeResult(name=name, mse=float(mean_squared_error(targets, preds)), r2=float(r2_score(targets, preds)))
    return model, result

from __future__ import annotations

import numpy as np


def joint_gaussian_loglik(observed: np.ndarray, predicted: np.ndarray, sigma: float | np.ndarray) -> float:
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    sig = np.maximum(np.asarray(sigma, dtype=float), 1.0e-8)
    residual = obs - pred
    return float(np.sum(-0.5 * (residual / sig) ** 2 - np.log(sig) - 0.5 * np.log(2.0 * np.pi)))


def joint_gaussian_loglik_batch(observed: np.ndarray, predicted: np.ndarray, sigma: float | np.ndarray) -> np.ndarray:
    obs = np.asarray(observed, dtype=float)[None, :, :]
    pred = np.asarray(predicted, dtype=float)
    sig = np.maximum(np.asarray(sigma, dtype=float), 1.0e-8)
    residual = obs - pred
    return np.sum(-0.5 * (residual / sig) ** 2 - np.log(sig) - 0.5 * np.log(2.0 * np.pi), axis=(1, 2))


def normalized_weights(loglik: np.ndarray) -> np.ndarray:
    arr = np.asarray(loglik, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full(arr.shape, 1.0 / max(arr.size, 1), dtype=float)
    shifted = np.where(finite, arr - np.max(arr[finite]), -np.inf)
    w = np.exp(shifted)
    total = float(np.sum(w))
    return w / total if total > 0 and np.isfinite(total) else np.full(arr.shape, 1.0 / max(arr.size, 1), dtype=float)


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(w**2))


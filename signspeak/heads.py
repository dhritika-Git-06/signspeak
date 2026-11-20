"""Softmax head used to add a sign without retraining the whole network."""

from __future__ import annotations

import numpy as np


def fit_softmax(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    epochs: int = 120,
    lr: float = 0.35,
    l2: float = 1e-3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression. features are (N, D), labels are ints."""
    rng = np.random.default_rng(seed)
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    n, dim = x.shape
    w = rng.normal(0, 0.01, size=(dim, num_classes)).astype(np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)
    eye = np.eye(num_classes, dtype=np.float32)[y]
    for _ in range(epochs):
        logits = x @ w + b
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        grad = (probs - eye) / n
        w -= lr * (x.T @ grad + l2 * w)
        b -= lr * grad.sum(axis=0)
        lr *= 0.985
    return w.astype(np.float32), b.astype(np.float32)


def predict_proba(features: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    logits = np.asarray(features, dtype=np.float32) @ w + b
    logits -= logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)

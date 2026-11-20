"""NumPy forward passes that match the Keras models (inference mode)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from signspeak.models import DYNAMIC_ORDER, STATIC_ORDER

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def conv1d_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Channels-last conv1d. x is (N, T, C), kernel is (K, C, F)."""
    n, t, _ = x.shape
    k = kernel.shape[0]
    pad = k // 2
    xp = np.pad(x, ((0, 0), (pad, pad), (0, 0)))
    out = np.zeros((n, t, kernel.shape[2]), dtype=np.float32)
    for i in range(k):
        out += xp[:, i : i + t, :] @ kernel[i]
    out += bias
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def lstm_layer(x: np.ndarray, kernel, recurrent, bias, return_sequences: bool = False) -> np.ndarray:
    """Keras LSTM, implementation 2, gate order i, f, c, o. x is (N, T, D)."""
    n, timesteps, _ = x.shape
    units = recurrent.shape[0]
    h = np.zeros((n, units), dtype=np.float32)
    c = np.zeros((n, units), dtype=np.float32)
    outputs = []
    for t in range(timesteps):
        z = x[:, t, :] @ kernel + h @ recurrent + bias
        i, f, g, o = np.split(z, 4, axis=1)
        i = _sigmoid(i)
        f = _sigmoid(f)
        o = _sigmoid(o)
        g = np.tanh(g)
        c = f * c + i * g
        h = o * np.tanh(c)
        if return_sequences:
            outputs.append(h.astype(np.float32))
    if return_sequences:
        return np.stack(outputs, axis=1)
    return h.astype(np.float32)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class StaticNet:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights

    @classmethod
    def from_file(cls, path: Path | None = None) -> "StaticNet":
        return cls(load_npz(path or MODELS / "static.npz"))

    def embed(self, x: np.ndarray) -> np.ndarray:
        h = _relu(conv1d_same(x, self.w["conv1_k"], self.w["conv1_b"]))
        h = _relu(conv1d_same(h, self.w["conv2_k"], self.w["conv2_b"]))
        h = _relu(conv1d_same(h, self.w["conv3_k"], self.w["conv3_b"]))
        h = h.mean(axis=1)
        return _relu(h @ self.w["embed_k"] + self.w["embed_b"])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        emb = self.embed(x)
        return _softmax(emb @ self.w["probs_k"] + self.w["probs_b"])


class DynamicNet:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights

    @classmethod
    def from_file(cls, path: Path | None = None) -> "DynamicNet":
        return cls(load_npz(path or MODELS / "dynamic.npz"))

    def embed(self, x: np.ndarray) -> np.ndarray:
        h = lstm_layer(x, self.w["lstm1_k"], self.w["lstm1_r"], self.w["lstm1_b"], True)
        h = lstm_layer(h, self.w["lstm2_k"], self.w["lstm2_r"], self.w["lstm2_b"], False)
        return _relu(h @ self.w["embed_k"] + self.w["embed_b"])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        emb = self.embed(x)
        return _softmax(emb @ self.w["probs_k"] + self.w["probs_b"])


def topk(probs: np.ndarray, labels: list[str], k: int = 3):
    row = probs[0]
    order = np.argsort(row)[::-1][:k]
    return [{"label": labels[int(i)], "confidence": float(row[int(i)])} for i in order]


def assert_static_parity(keras_model, weights: dict[str, np.ndarray], seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(4, 21, 3)).astype(np.float32)
    keras_probs = keras_model.predict(x, verbose=0)
    numpy_probs = StaticNet(weights).predict_proba(x)
    if not np.allclose(keras_probs, numpy_probs, atol=1e-4):
        diff = np.max(np.abs(keras_probs - numpy_probs))
        raise AssertionError(f"Static NumPy forward pass drifted from Keras by {diff}")


def assert_dynamic_parity(keras_model, weights: dict[str, np.ndarray], seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3, 30, 63)).astype(np.float32)
    keras_probs = keras_model.predict(x, verbose=0)
    numpy_probs = DynamicNet(weights).predict_proba(x)
    if not np.allclose(keras_probs, numpy_probs, atol=1e-4):
        diff = np.max(np.abs(keras_probs - numpy_probs))
        raise AssertionError(f"Dynamic NumPy forward pass drifted from Keras by {diff}")


def pack_order(weights: dict[str, np.ndarray], order: list[str]) -> tuple[np.ndarray, list[dict]]:
    blobs = []
    manifest = []
    cursor = 0
    for name in order:
        arr = np.asarray(weights[name], dtype=np.float32)
        flat = arr.reshape(-1)
        manifest.append(
            {"name": name, "shape": list(arr.shape), "offset": cursor, "length": int(flat.size)}
        )
        blobs.append(flat)
        cursor += int(flat.size)
    return np.concatenate(blobs), manifest

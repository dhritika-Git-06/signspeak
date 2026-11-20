"""Kinematic ASL word sequences seeded by real static handshapes.

The public landmark file is photos, not videos. Each dynamic example takes a
real normalised handshape (an actual row from that file) and moves it along a
class-specific path: a J stroke, a Z stroke, a wave, a nod, a wag, and so on.

Train and test use different source photos, so the reported accuracy is on
handshapes the optimiser did not see. It is not a webcam accuracy and it is
not a claim about a recorded word dataset. The evaluation page says so.
"""

from __future__ import annotations

import numpy as np

from signspeak.labels import DYNAMIC_HANDSHAPE, DYNAMIC_LABELS, SEQUENCE_LENGTH

T = SEQUENCE_LENGTH


def _rot_z(points: np.ndarray, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    out = points.copy()
    x, y = out[:, 0], out[:, 1]
    out[:, 0] = c * x - s * y
    out[:, 1] = s * x + c * y
    return out


def _warp_path(path: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    amp = float(rng.uniform(0.75, 1.3))
    path = path * amp
    ang = float(rng.uniform(-0.45, 0.45))
    c, s = np.cos(ang), np.sin(ang)
    xy = path[:, :2].copy()
    path = path.copy()
    path[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    path[:, 1] = s * xy[:, 0] + c * xy[:, 1]
    path += rng.normal(0, 0.02, path.shape).astype(np.float32)
    path -= path[0]
    return path.astype(np.float32)


def _polyline(points: list[tuple[float, float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(segs.sum()) + 1e-6
    edges = np.concatenate([[0.0], np.cumsum(segs) / total])
    t = np.linspace(0, 1, T, dtype=np.float32)
    out = np.zeros((T, 3), dtype=np.float32)
    for i, ti in enumerate(t):
        k = int(np.searchsorted(edges, ti, side="right") - 1)
        k = min(max(k, 0), len(segs) - 1)
        span = edges[k + 1] - edges[k]
        u = 0.0 if span < 1e-6 else (ti - edges[k]) / span
        xy = (1 - u) * pts[k] + u * pts[k + 1]
        out[i, 0] = xy[0]
        out[i, 1] = xy[1]
    return out


def make_path(kind: str, rng: np.random.Generator) -> np.ndarray | None:
    """Return a (T, 3) translation path, or None when the class is a rotation."""
    t = np.linspace(0, 1, T, dtype=np.float32)
    if kind == "J":
        path = _polyline([(0.05, 0.0), (0.02, 1.25), (-0.85, 1.45)])
    elif kind == "Z":
        path = _polyline([(0.0, 0.0), (1.15, 0.0), (-0.05, 0.95), (1.1, 0.95)])
    elif kind == "HELLO":
        cycles = float(rng.uniform(1.6, 2.8))
        phase = float(rng.uniform(0, np.pi))
        path = np.zeros((T, 3), dtype=np.float32)
        path[:, 0] = np.sin(2 * np.pi * cycles * t + phase)
        path[:, 1] = 0.12 * np.sin(2 * np.pi * t)
    elif kind == "YES":
        cycles = float(rng.uniform(1.8, 3.0))
        path = np.zeros((T, 3), dtype=np.float32)
        path[:, 1] = np.sin(2 * np.pi * cycles * t)
    elif kind == "THANKS":
        path = np.zeros((T, 3), dtype=np.float32)
        path[:, 1] = t * 1.15
        path[:, 0] = 0.08 * np.sin(2 * np.pi * t)
    elif kind == "PLEASE":
        path = np.zeros((T, 3), dtype=np.float32)
        loops = float(rng.uniform(1.0, 1.8))
        path[:, 0] = np.sin(2 * np.pi * loops * t)
        path[:, 1] = np.cos(2 * np.pi * loops * t) - 1.0
    elif kind == "SORRY":
        path = np.zeros((T, 3), dtype=np.float32)
        loops = float(rng.uniform(0.8, 1.4))
        path[:, 0] = 0.55 * np.sin(2 * np.pi * loops * t)
        path[:, 1] = 0.55 * (np.cos(2 * np.pi * loops * t) - 1.0)
    elif kind == "NO":
        return None
    else:
        raise KeyError(kind)
    return _warp_path(path.astype(np.float32), rng)


def _sequence_from_base(base: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    pose = float(rng.uniform(-0.3, 0.3))
    shaped = _rot_z(base, pose)
    noise = rng.normal(0, 0.035, (T, 21, 3)).astype(np.float32)
    frames = np.empty((T, 21, 3), dtype=np.float32)
    if kind == "NO":
        amp = float(rng.uniform(0.35, 0.7))
        cycles = float(rng.uniform(1.6, 2.6))
        phase = float(rng.uniform(0, np.pi))
        t = np.linspace(0, 1, T, dtype=np.float32)
        for i, ti in enumerate(t):
            ang = amp * np.sin(2 * np.pi * cycles * ti + phase)
            frames[i] = _rot_z(shaped, ang) + noise[i]
    else:
        path = make_path(kind, rng)
        assert path is not None
        for i in range(T):
            frames[i] = shaped + path[i] + noise[i]
    return frames


def make_dataset(by_letter: dict[str, np.ndarray], seed: int = 42, train_n: int = 220, test_n: int = 50, train_aug: int = 3, test_aug: int = 2):
    """Build train and test tensors.

    by_letter maps a static letter to normalised (N, 21, 3) handshapes.
    """
    rng = np.random.default_rng(seed)
    x_train, y_train, x_test, y_test = [], [], [], []
    for class_index, name in enumerate(DYNAMIC_LABELS):
        pool = by_letter[DYNAMIC_HANDSHAPE[name]]
        order = rng.permutation(len(pool))
        if len(order) < train_n + test_n:
            raise RuntimeError(f"Not enough handshapes for {name}")
        test_ids = order[:test_n]
        train_ids = order[test_n : test_n + train_n]
        for src in train_ids:
            for _ in range(train_aug):
                seq = _sequence_from_base(pool[src], name, rng)
                x_train.append(seq.reshape(T, -1))
                y_train.append(class_index)
        for src in test_ids:
            for _ in range(test_aug):
                seq = _sequence_from_base(pool[src], name, rng)
                x_test.append(seq.reshape(T, -1))
                y_test.append(class_index)
    x_train = np.stack(x_train).astype(np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    x_test = np.stack(x_test).astype(np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)
    perm = rng.permutation(len(y_train))
    return x_train[perm], y_train[perm], x_test, y_test


def sample_sequences(by_letter: dict[str, np.ndarray], per_class: int = 2, seed: int = 7):
    """A few full (T, 21, 3) sequences for the in-browser replay control."""
    rng = np.random.default_rng(seed)
    samples = {}
    for name in DYNAMIC_LABELS:
        pool = by_letter[DYNAMIC_HANDSHAPE[name]]
        picked = []
        # Use the high end of the pool so these are not the first test ids,
        # but replay does not need to be disjoint from training to be useful.
        ids = rng.choice(len(pool), size=per_class, replace=False)
        for src in ids:
            picked.append(_sequence_from_base(pool[int(src)], name, rng))
        samples[name] = picked
    return samples

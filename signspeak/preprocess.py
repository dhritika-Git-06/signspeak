"""Landmark normalisation shared by training, the API, and the browser.

Static signs are wrist-centred and scaled by the palm (wrist to middle MCP)
so the CNN sees a 21×3 shape, not where the hand sits in the frame.

Dynamic signs keep the wrist path. Every frame is shifted by the first
frame's wrist and scaled by the median palm size, then resampled to a fixed
length. Pure translation would vanish if each frame were re-centred.
"""

from __future__ import annotations

import numpy as np

from signspeak.labels import N_LANDMARKS, SEQUENCE_LENGTH

WRIST = 0
MIDDLE_MCP = 9


def landmarks_to_array(landmarks) -> np.ndarray:
    """Return float32 array of shape (21, 3).

    Accepts (21, 2), (21, 3), a flat vector of 42 or 63 numbers, or a list of
    {x, y, z} dicts. Missing z is filled with 0.
    """
    if isinstance(landmarks, (list, tuple)) and landmarks and isinstance(landmarks[0], dict):
        pts = [[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in landmarks]
        arr = np.asarray(pts, dtype=np.float32)
    else:
        arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size == N_LANDMARKS * 2:
            xy = arr.reshape(N_LANDMARKS, 2)
            z = np.zeros((N_LANDMARKS, 1), dtype=np.float32)
            arr = np.concatenate([xy, z], axis=1)
        elif arr.size == N_LANDMARKS * 3:
            arr = arr.reshape(N_LANDMARKS, 3)
        else:
            raise ValueError(
                f"Expected {N_LANDMARKS * 2} or {N_LANDMARKS * 3} values, got {arr.size}"
            )
    elif arr.shape == (N_LANDMARKS, 2):
        z = np.zeros((N_LANDMARKS, 1), dtype=np.float32)
        arr = np.concatenate([arr.astype(np.float32), z], axis=1)
    elif arr.shape != (N_LANDMARKS, 3):
        raise ValueError(f"Expected shape (21, 2) or (21, 3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Landmarks contain NaN or infinity")
    return arr.astype(np.float32)


def palm_scale(points: np.ndarray) -> float:
    wrist = points[WRIST, :2]
    middle = points[MIDDLE_MCP, :2]
    return float(np.linalg.norm(middle - wrist) + 1e-6)


def normalize_landmarks(landmarks, mirror: bool = False, zero_z: bool = False) -> np.ndarray:
    """Wrist-centred, palm-scaled (21, 3) landmarks."""
    pts = landmarks_to_array(landmarks)
    if mirror:
        pts = pts.copy()
        pts[:, 0] *= -1.0
    if zero_z:
        pts = pts.copy()
        pts[:, 2] = 0.0
    scale = palm_scale(pts)
    centred = pts - pts[WRIST]
    return (centred / scale).astype(np.float32)


def resample_sequence(seq: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Linear resample of (T, 21, 3) to a fixed length."""
    seq = np.asarray(seq, dtype=np.float32)
    t_count = seq.shape[0]
    if t_count == length:
        return seq
    if t_count == 1:
        return np.repeat(seq, length, axis=0)
    t = np.linspace(0, t_count - 1, length, dtype=np.float32)
    lo = np.floor(t).astype(np.int32)
    hi = np.minimum(lo + 1, t_count - 1)
    alpha = (t - lo).astype(np.float32)[:, None, None]
    return ((1.0 - alpha) * seq[lo] + alpha * seq[hi]).astype(np.float32)


def normalize_sequence(
    frames,
    length: int = SEQUENCE_LENGTH,
    mirror: bool = False,
    zero_z: bool = False,
) -> np.ndarray:
    """Return (length, 21, 3) with motion kept relative to the first wrist."""
    if not frames:
        raise ValueError("Sequence is empty")
    pts = [landmarks_to_array(frame) for frame in frames]
    if mirror:
        for p in pts:
            p[:, 0] *= -1.0
    if zero_z:
        for p in pts:
            p[:, 2] = 0.0
    scale = float(np.median([palm_scale(p) for p in pts]))
    origin = pts[0][WRIST].copy()
    seq = np.stack([(p - origin) / scale for p in pts], axis=0).astype(np.float32)
    return resample_sequence(seq, length)


def flatten_sequence(seq: np.ndarray) -> np.ndarray:
    """(T, 21, 3) -> (T, 63)."""
    seq = np.asarray(seq, dtype=np.float32)
    return seq.reshape(seq.shape[0], -1)


def motion_energy(seq: np.ndarray) -> float:
    """Mean per-landmark displacement between consecutive frames."""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.shape[0] < 2:
        return 0.0
    delta = np.linalg.norm(seq[1:] - seq[:-1], axis=-1)
    return float(delta.mean())

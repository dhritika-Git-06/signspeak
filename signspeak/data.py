"""Load the public ASL alphabet landmark CSV."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

from signspeak.labels import STATIC_LABELS
from signspeak.preprocess import normalize_landmarks

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "hand_dataset_1000_24.csv"
DATASET_URL = (
    "https://raw.githubusercontent.com/MinorvaFalk/KNN_Alphabet/main/Dataset/"
    "hand_dataset_1000_24.csv"
)
DATASET_SHA256 = "8150c9277aec32c3b89d9df40a074b6601806f5c1f219d9e58b097fe4e6e7ffa"

# Landmark CSV published by MinorvaFalk/KNN_Alphabet, converted with MediaPipe
# Hands from the ASL Alphabet image set (Akash Nagaraj, Kaggle, GPL-2.0).
DATASET_CITATION = {
    "name": "ASL Alphabet landmarks (24 static letters)",
    "images": "ASL Alphabet, Akash Nagaraj (grassknoted), Kaggle, 2018",
    "images_url": "https://www.kaggle.com/datasets/grassknoted/asl-alphabet",
    "images_license": "GPL-2.0",
    "landmarks": "MinorvaFalk/KNN_Alphabet hand_dataset_1000_24.csv",
    "landmarks_url": DATASET_URL,
    "notes": (
        "1,000 MediaPipe Hands landmark rows for each of 24 static ASL letters "
        "(A–Y except J and Z). Columns are class plus x1,y1 … x21,y21 in image "
        "coordinates. J and Z are excluded in this file because they are motions."
    ),
}


def download_static_csv(path: Path = RAW_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and _sha256(path) == DATASET_SHA256:
        return path
    print(f"Downloading {DATASET_URL}")
    req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "signspeak-ai"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if digest != DATASET_SHA256:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for landmark CSV: {digest}")
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_static_raw(path: Path | None = None):
    """Return raw (N, 21, 2) coordinates, integer labels, and the label list."""
    import pandas as pd

    csv_path = path or download_static_csv()
    frame = pd.read_csv(csv_path)
    if "class" not in frame.columns:
        raise RuntimeError("Landmark CSV is missing the class column")
    unknown = sorted(set(frame["class"].unique()) - set(STATIC_LABELS))
    if unknown:
        raise RuntimeError(f"Unexpected labels in CSV: {unknown}")
    coords = []
    for i in range(1, 22):
        coords.append(frame[f"x{i}"].to_numpy(np.float32))
        coords.append(frame[f"y{i}"].to_numpy(np.float32))
    raw = np.stack(coords, axis=1).reshape(-1, 21, 2).astype(np.float32)
    label_to_index = {name: i for i, name in enumerate(STATIC_LABELS)}
    y = np.array([label_to_index[c] for c in frame["class"]], dtype=np.int64)
    return raw, y, STATIC_LABELS


def raw_to_model_input(raw_xy: np.ndarray, zero_z: bool = False) -> np.ndarray:
    """(N, 21, 2) image coords -> (N, 21, 3) normalised landmarks."""
    out = np.empty((raw_xy.shape[0], 21, 3), dtype=np.float32)
    for i, row in enumerate(raw_xy):
        out[i] = normalize_landmarks(row, zero_z=zero_z)
    return out


def stratified_split(y: np.ndarray, seed: int = 42, fracs=(0.70, 0.15, 0.15)):
    """Return train, val, test index arrays. Membership is deterministic."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for cls in range(int(y.max()) + 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(fracs[0] * n))
        n_val = int(round(fracs[1] * n))
        train.append(idx[:n_train])
        val.append(idx[n_train : n_train + n_val])
        test.append(idx[n_train + n_val :])
    train_idx = np.concatenate(train)
    val_idx = np.concatenate(val)
    test_idx = np.concatenate(test)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

"""Write npz, a float32 blob, and a JSON manifest the browser can load."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from signspeak.inference import pack_order
from signspeak.models import DYNAMIC_ORDER, STATIC_ORDER


def save_pack(weights: dict[str, np.ndarray], order: list[str], dest_dir: Path, stem: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    np.savez(dest_dir / f"{stem}.npz", **{k: weights[k] for k in order})
    blob, manifest = pack_order(weights, order)
    (dest_dir / f"{stem}.bin").write_bytes(blob.astype("<f4").tobytes())
    (dest_dir / f"{stem}.json").write_text(
        json.dumps({"file": f"{stem}.bin", "tensors": manifest}, indent=2)
    )


def save_static(weights, dest_dir: Path) -> None:
    save_pack(weights, STATIC_ORDER, dest_dir, "static")


def save_dynamic(weights, dest_dir: Path) -> None:
    save_pack(weights, DYNAMIC_ORDER, dest_dir, "dynamic")

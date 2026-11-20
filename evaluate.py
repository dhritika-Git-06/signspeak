#!/usr/bin/env python3
"""Evaluate the saved Keras models and write the report the UI shows.

Usage:
    python evaluate.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from signspeak.data import DATASET_CITATION, load_static_raw, raw_to_model_input, stratified_split
from signspeak.dynamic_data import make_dataset
from signspeak.labels import DYNAMIC_LABELS, STATIC_LABELS
from signspeak.metrics import classification_payload, save_confusion_png

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluation"
PUBLIC_EVAL = ROOT / "frontend" / "public" / "evaluation"
PUBLIC_MODELS = ROOT / "frontend" / "public" / "models"


def _copy_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    import keras

    raw, y, labels = load_static_raw()
    assert labels == STATIC_LABELS
    x = raw_to_model_input(raw, zero_z=False)
    _, _, test_idx = stratified_split(y, seed=42)
    model = keras.models.load_model(MODEL_DIR / "static.keras")
    probs = model.predict(x[test_idx], verbose=0)
    pred = probs.argmax(axis=1)
    static = classification_payload(y[test_idx], pred, STATIC_LABELS)

    noisy = x[test_idx].copy()
    rng = np.random.default_rng(99)
    noisy[:, :, 2] = rng.normal(0, 0.1, size=noisy[:, :, 2].shape).astype(np.float32)
    z_pred = model.predict(noisy, verbose=0).argmax(axis=1)
    z_acc = float((z_pred == y[test_idx]).mean())
    static["z_noise_accuracy"] = z_acc
    static["z_noise_std"] = 0.1
    # Real MediaPipe depth is informative, but this training file has no z.
    # If extra z noise changes predictions, the browser sends z = 0 so live
    # input matches the photographs the network actually saw.
    static["z_mode"] = "passthrough" if z_acc >= static["accuracy"] - 0.03 else "zero"
    static["train_split"] = {"train": 0.70, "val": 0.15, "test": 0.15, "seed": 42}
    print(f"static accuracy {static['accuracy']:.4f}  macro F1 {static['macro_f1']:.4f}  z-noise {z_acc:.4f}  mode {static['z_mode']}")

    grouped = {}
    for index, name in enumerate(STATIC_LABELS):
        grouped[name] = x[y == index]
    _, _, x_test, y_test = make_dataset(grouped, seed=42)
    dynamic_model = keras.models.load_model(MODEL_DIR / "dynamic.keras")
    d_probs = dynamic_model.predict(x_test, verbose=0)
    d_pred = d_probs.argmax(axis=1)
    dynamic = classification_payload(y_test, d_pred, DYNAMIC_LABELS)
    hit = int(np.where(d_pred == y_test)[0][0])
    (MODEL_DIR / "parity_dynamic.json").write_text(
        json.dumps(
            {
                "label": DYNAMIC_LABELS[int(y_test[hit])],
                "input": x_test[hit].astype(float).tolist(),
                "probs": d_probs[hit].astype(float).tolist(),
            }
        )
    )
    dynamic["data"] = (
        "Kinematic sequences seeded by real ASL Alphabet handshapes. "
        "Train and test use different source photographs. "
        "This number is not a recording of people signing whole words."
    )
    print(f"dynamic accuracy {dynamic['accuracy']:.4f}  macro F1 {dynamic['macro_f1']:.4f}")

    motion = json.loads((MODEL_DIR / "motion.json").read_text())

    save_confusion_png(
        np.asarray(static["confusion_matrix"]),
        STATIC_LABELS,
        EVAL_DIR / "static_confusion.png",
        "Static ASL letters — test set",
    )
    save_confusion_png(
        np.asarray(dynamic["confusion_matrix"]),
        DYNAMIC_LABELS,
        EVAL_DIR / "dynamic_confusion.png",
        "Dynamic signs — held-out synthetic test",
    )
    _copy_text(EVAL_DIR / "static_report.txt", static["report"])
    _copy_text(EVAL_DIR / "dynamic_report.txt", dynamic["report"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "static": {k: v for k, v in static.items() if k != "confusion_matrix"},
        "dynamic": {k: v for k, v in dynamic.items() if k != "confusion_matrix"},
        "motion_threshold": motion["threshold"],
        "motion": motion,
        "datasets": {
            "static": DATASET_CITATION,
            "dynamic": {
                "name": "Kinematic ASL sequences seeded by the static landmark set",
                "license": "GPL-2.0 (derived from the ASL Alphabet images)",
                "classes": DYNAMIC_LABELS,
                "handshapes": {
                    "J": "I",
                    "Z": "D",
                    "HELLO": "B",
                    "YES": "S",
                    "NO": "V",
                    "THANKS": "B",
                    "PLEASE": "B",
                    "SORRY": "A",
                },
                "notes": dynamic["data"],
            },
        },
    }
    # Keep the matrix out of the web JSON (the PNG is the figure) but store
    # it beside the report for anyone re-plotting.
    (EVAL_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    (EVAL_DIR / "static_confusion.json").write_text(json.dumps(static["confusion_matrix"]))
    (EVAL_DIR / "dynamic_confusion.json").write_text(json.dumps(dynamic["confusion_matrix"]))

    PUBLIC_EVAL.mkdir(parents=True, exist_ok=True)
    for name in (
        "metrics.json",
        "static_confusion.png",
        "dynamic_confusion.png",
        "static_report.txt",
        "dynamic_report.txt",
    ):
        target = PUBLIC_EVAL / name
        target.write_bytes((EVAL_DIR / name).read_bytes())

    meta = {
        "version": "1.0.0",
        "static_labels": STATIC_LABELS,
        "dynamic_labels": DYNAMIC_LABELS,
        "z_mode": static["z_mode"],
        "sequence_length": 30,
        "static_embed_dim": 128,
        "dynamic_embed_dim": 32,
        "static_filters": [64, 128, 64],
        "dynamic_lstm": [64, 32],
        "motion_threshold": motion["threshold"],
        "static_accuracy": static["accuracy"],
        "static_macro_f1": static["macro_f1"],
        "dynamic_accuracy": dynamic["accuracy"],
        "dynamic_macro_f1": dynamic["macro_f1"],
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    PUBLIC_MODELS.mkdir(parents=True, exist_ok=True)
    (PUBLIC_MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Wrote", EVAL_DIR / "metrics.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the static ASL letter CNN on MediaPipe landmarks.

Usage:
    python train_static.py
    python train_static.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from signspeak.data import RAW_PATH, load_static_raw, raw_to_model_input, stratified_split
from signspeak.export_weights import save_static
from signspeak.inference import assert_static_parity
from signspeak.labels import STATIC_LABELS
from signspeak.models import build_static, static_weight_dict
from signspeak.preprocess import normalize_landmarks

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
PUBLIC_MODELS = ROOT / "frontend" / "public" / "models"
PUBLIC_SAMPLES = ROOT / "frontend" / "public" / "samples"


def augment_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    ang = rng.uniform(-0.55, 0.55, size=n).astype(np.float32)
    c = np.cos(ang)[:, None]
    s = np.sin(ang)[:, None]
    out = x.copy()
    px, py = out[:, :, 0].copy(), out[:, :, 1].copy()
    out[:, :, 0] = c * px - s * py
    out[:, :, 1] = s * px + c * py
    out += rng.normal(0, 0.028, out.shape).astype(np.float32)
    out[:, :, 2] += rng.normal(0, 0.08, size=(n, 21)).astype(np.float32)
    out *= rng.uniform(0.85, 1.15, size=(n, 1, 1)).astype(np.float32)
    return out.astype(np.float32)


def write_normalize_cases(raw: np.ndarray) -> None:
    cases = []
    for index, mirror in ((0, False), (12, True), (40, False)):
        row = raw[index]
        pts = np.concatenate([row, np.zeros((21, 1), np.float32)], axis=1)
        norm = normalize_landmarks(pts, mirror=mirror, zero_z=False)
        cases.append(
            {
                "raw": pts.tolist(),
                "mirror": mirror,
                "zero_z": False,
                "normalized": norm.tolist(),
            }
        )
    dest = ROOT / "frontend" / "src" / "lib" / "normalize_cases.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(cases))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the static landmark CNN")
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import keras

    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading", RAW_PATH)
    raw, y, labels = load_static_raw()
    print(f"{len(y)} rows, {len(labels)} classes")
    x = raw_to_model_input(raw, zero_z=False)
    write_normalize_cases(raw)

    train_idx, val_idx, test_idx = stratified_split(y, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    parts_x = [x_train]
    parts_y = [y_train]
    for _ in range(4):
        parts_x.append(augment_batch(x_train, rng))
        parts_y.append(y_train)
    x_fit = np.concatenate(parts_x)
    y_fit = np.concatenate(parts_y)
    order = rng.permutation(len(y_fit))
    x_fit, y_fit = x_fit[order], y_fit[order]
    print(f"train {len(train_idx)} (+aug {len(y_fit)})  val {len(val_idx)}  test {len(test_idx)}")

    model, embed_model = build_static(len(labels))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]
    model.fit(
        x_fit,
        y_fit,
        validation_data=(x[val_idx], y[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    weights = static_weight_dict(model)
    assert_static_parity(model, weights)
    print("NumPy parity check passed")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "static.keras")
    embed_model.save(MODEL_DIR / "static_embed.keras")
    save_static(weights, MODEL_DIR)
    save_static(weights, PUBLIC_MODELS)

    probs = model.predict(x[test_idx], verbose=0)
    pred = probs.argmax(axis=1)
    correct = pred == y[test_idx]
    print(f"clean test accuracy {correct.mean():.4f}")

    support_x = []
    support_y = []
    support_rng = np.random.default_rng(43)
    embeddings = embed_model.predict(x[train_idx], verbose=0)
    train_labels = y[train_idx]
    for cls in range(len(labels)):
        choices = np.where(train_labels == cls)[0]
        take = support_rng.choice(choices, size=12, replace=False)
        support_x.append(embeddings[take])
        support_y.append(np.full(len(take), labels[cls]))
    support = {
        "dim": int(embeddings.shape[1]),
        "labels": np.concatenate(support_y).tolist(),
        "vectors": np.concatenate(support_x).astype(float).round(6).tolist(),
    }
    for dest in (MODEL_DIR, PUBLIC_MODELS):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "support_static.json").write_text(json.dumps(support))

    samples = {}
    raw_test = raw[test_idx]
    y_test = y[test_idx]
    for cls, name in enumerate(labels):
        hits = np.where((y_test == cls) & correct)[0]
        if len(hits) == 0:
            hits = np.where(y_test == cls)[0]
        chosen = hits[:8]
        samples[name] = raw_test[chosen].reshape(len(chosen), -1).tolist()
    PUBLIC_SAMPLES.mkdir(parents=True, exist_ok=True)
    (PUBLIC_SAMPLES / "static_samples.json").write_text(json.dumps({"labels": labels, "samples": samples}))

    fixture_i = int(np.where(correct)[0][0])
    fixture = {
        "label": labels[int(y_test[fixture_i])],
        "raw_xy": raw_test[fixture_i].reshape(-1).tolist(),
        "normalized": x[test_idx][fixture_i].reshape(-1).tolist(),
        "probs": probs[fixture_i].astype(float).tolist(),
    }
    (MODEL_DIR / "parity_static.json").write_text(json.dumps(fixture))
    print("Saved static model, support set, and sample landmarks")


if __name__ == "__main__":
    main()

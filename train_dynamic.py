#!/usr/bin/env python3
"""Train the LSTM on kinematic sequences seeded by real ASL handshapes.

Usage:
    python train_dynamic.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from signspeak.data import load_static_raw, raw_to_model_input
from signspeak.dynamic_data import make_dataset, sample_sequences
from signspeak.export_weights import save_dynamic
from signspeak.inference import assert_dynamic_parity
from signspeak.labels import DYNAMIC_LABELS, STATIC_LABELS
from signspeak.models import build_dynamic, dynamic_weight_dict
from signspeak.preprocess import motion_energy

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
PUBLIC_MODELS = ROOT / "frontend" / "public" / "models"
PUBLIC_SAMPLES = ROOT / "frontend" / "public" / "samples"


def handshapes_by_letter(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    grouped = {}
    for index, name in enumerate(STATIC_LABELS):
        grouped[name] = x[y == index]
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dynamic LSTM")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import keras

    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    raw, y, _ = load_static_raw()
    x = raw_to_model_input(raw, zero_z=False)
    grouped = handshapes_by_letter(x, y)
    print("Synthesising sequences from real handshapes…")
    x_train, y_train, x_test, y_test = make_dataset(grouped, seed=args.seed)
    print(f"dynamic train {len(y_train)}  test {len(y_test)}  classes {len(DYNAMIC_LABELS)}")

    # Motion of a still hand versus the generated signs, for the auto switch.
    still = grouped["B"][:40].copy()
    rng = np.random.default_rng(1)
    still_energy = []
    for shape in still:
        frames = np.repeat(shape[None, :, :], 30, axis=0)
        frames = frames + rng.normal(0, 0.012, frames.shape).astype(np.float32)
        still_energy.append(motion_energy(frames))
    class_energy = {}
    for index, name in enumerate(DYNAMIC_LABELS):
        seqs = x_test[y_test == index][:30].reshape(-1, 30, 21, 3)
        class_energy[name] = float(np.mean([motion_energy(seq) for seq in seqs]))
    still_mean = float(np.mean(still_energy))
    dyn_min = min(class_energy.values())
    threshold = float((still_mean + dyn_min) / 2)
    print("still motion", round(still_mean, 4), "dynamic", {k: round(v, 4) for k, v in class_energy.items()})
    print("suggested motion threshold", round(threshold, 4))

    model, embed_model = build_dynamic(len(DYNAMIC_LABELS))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        )
    ]
    # Hold out 15% of the training sequences for early stopping. The official
    # test set above is a different set of source photographs.
    rng = np.random.default_rng(args.seed + 1)
    perm = rng.permutation(len(y_train))
    cut = int(len(perm) * 0.85)
    fit_idx, val_idx = perm[:cut], perm[cut:]
    model.fit(
        x_train[fit_idx],
        y_train[fit_idx],
        validation_data=(x_train[val_idx], y_train[val_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    weights = dynamic_weight_dict(model)
    assert_dynamic_parity(model, weights)
    print("NumPy parity check passed")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "dynamic.keras")
    embed_model.save(MODEL_DIR / "dynamic_embed.keras")
    save_dynamic(weights, MODEL_DIR)
    save_dynamic(weights, PUBLIC_MODELS)

    probs = model.predict(x_test, verbose=0)
    pred = probs.argmax(axis=1)
    print(f"held-out synthetic test accuracy {(pred == y_test).mean():.4f}")

    embeddings = embed_model.predict(x_train[fit_idx], verbose=0)
    labels_fit = y_train[fit_idx]
    support_rng = np.random.default_rng(44)
    vectors, names = [], []
    for cls, name in enumerate(DYNAMIC_LABELS):
        choices = np.where(labels_fit == cls)[0]
        take = support_rng.choice(choices, size=min(12, len(choices)), replace=False)
        vectors.append(embeddings[take])
        names.extend([name] * len(take))
    support = {
        "dim": int(embeddings.shape[1]),
        "labels": names,
        "vectors": np.concatenate(vectors).astype(float).round(6).tolist(),
    }
    for dest in (MODEL_DIR, PUBLIC_MODELS):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "support_dynamic.json").write_text(json.dumps(support))

    replay = sample_sequences(grouped, per_class=2, seed=7)
    payload = {
        "labels": DYNAMIC_LABELS,
        "length": 30,
        "samples": {name: [seq.tolist() for seq in seqs] for name, seqs in replay.items()},
    }
    PUBLIC_SAMPLES.mkdir(parents=True, exist_ok=True)
    (PUBLIC_SAMPLES / "dynamic_samples.json").write_text(json.dumps(payload))

    (MODEL_DIR / "motion.json").write_text(
        json.dumps(
            {
                "still_mean": still_mean,
                "class_energy": class_energy,
                "threshold": threshold,
            },
            indent=2,
        )
    )
    print("Saved dynamic model")


if __name__ == "__main__":
    main()

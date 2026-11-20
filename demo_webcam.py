#!/usr/bin/env python3
"""OpenCV demo.

With a camera:
    python demo_webcam.py --camera 0

Without a camera, replay landmark samples and write a contact sheet:
    python demo_webcam.py --samples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from signspeak.inference import StaticNet, topk  # noqa: E402
from signspeak.labels import STATIC_LABELS  # noqa: E402
from signspeak.preprocess import normalize_landmarks  # noqa: E402

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_hand(image, points, color=(30, 90, 200)):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[-1] == 2:
        xy = pts[:, :2]
    else:
        xy = pts[:, :2]
    h, w = image.shape[:2]
    # Fit the hand into the frame.
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    margin = 28
    scale = min((w - 2 * margin) / span[0], (h - 2 * margin) / span[1])
    origin = np.array([margin, margin], dtype=np.float32) - lo * scale
    pix = xy * scale + origin
    for a, b in HAND_EDGES:
        pa = tuple(np.round(pix[a]).astype(int))
        pb = tuple(np.round(pix[b]).astype(int))
        cv2.line(image, pa, pb, color, 2, cv2.LINE_AA)
    for p in pix:
        cv2.circle(image, tuple(np.round(p).astype(int)), 3, (20, 20, 20), -1, cv2.LINE_AA)
    return pix


def samples_path() -> Path:
    return ROOT / "frontend" / "public" / "samples" / "static_samples.json"


def run_samples(out: Path) -> None:
    payload = json.loads(samples_path().read_text())
    net = StaticNet.from_file()
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    zero = meta.get("z_mode") == "zero"
    cols, rows = 6, 4
    cell = 180
    sheet = np.full((rows * cell, cols * cell, 3), 245, np.uint8)
    correct = 0
    total = 0
    print(f"{'true':<8}{'pred':<8}{'conf':>8}")
    for index, name in enumerate(STATIC_LABELS):
        raw = np.asarray(payload["samples"][name][0], dtype=np.float32).reshape(21, 2)
        norm = normalize_landmarks(raw, zero_z=zero)
        probs = net.predict_proba(norm[None, :, :])
        ranked = topk(probs, STATIC_LABELS, k=1)[0]
        total += 1
        correct += int(ranked["label"] == name)
        print(f"{name:<8}{ranked['label']:<8}{ranked['confidence']:8.3f}")
        r, c = divmod(index, cols)
        tile = sheet[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell]
        draw_hand(tile, norm, color=(40, 110, 190))
        cv2.putText(
            tile,
            f"{name}->{ranked['label']}",
            (8, cell - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(f"sample accuracy {correct}/{total}")
    print(f"wrote {out}")


def run_camera(index: int) -> None:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print("No camera opened. Re-run with --samples.")
        return
    net = StaticNet.from_file()
    print("Press q to quit. This window shows the raw camera; landmark overlay needs MediaPipe,")
    print("which the browser app runs. Here we only confirm OpenCV capture.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(frame, "SignSpeak camera", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 180, 255), 2)
        cv2.imshow("SignSpeak", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="SignSpeak OpenCV demo")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--samples", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "evaluation" / "opencv_demo.png",
    )
    args = parser.parse_args()
    if args.camera is not None:
        run_camera(args.camera)
        return
    # Default, and --samples: no webcam required.
    started = time.time()
    run_samples(args.out)
    print(f"done in {time.time() - started:.2f}s")


if __name__ == "__main__":
    main()

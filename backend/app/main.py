"""SignSpeak REST API.

The browser runs the same models itself. These endpoints exist for a server
deployment, for gTTS audio, and for retraining a custom head from uploaded
landmarks.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from signspeak import __version__
from signspeak.heads import fit_softmax, predict_proba
from signspeak.inference import DynamicNet, StaticNet, topk
from signspeak.labels import DYNAMIC_LABELS, STATIC_LABELS
from signspeak.preprocess import (
    flatten_sequence,
    landmarks_to_array,
    normalize_landmarks,
    normalize_sequence,
)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RUNTIME = ROOT / "backend" / "runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SignSpeak AI", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static: StaticNet | None = None
_dynamic: DynamicNet | None = None
_meta: dict = {}


def meta() -> dict:
    global _meta
    if not _meta:
        path = MODELS / "meta.json"
        _meta = json.loads(path.read_text()) if path.exists() else {"z_mode": "zero"}
    return _meta


def static_net() -> StaticNet:
    global _static
    if _static is None:
        _static = StaticNet.from_file(MODELS / "static.npz")
    return _static


def dynamic_net() -> DynamicNet:
    global _dynamic
    if _dynamic is None:
        _dynamic = DynamicNet.from_file(MODELS / "dynamic.npz")
    return _dynamic


def zero_z() -> bool:
    return meta().get("z_mode", "zero") == "zero"


class PredictIn(BaseModel):
    landmarks: list | None = None
    sequence: list | None = None
    mirror: bool = False


class TTSIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    lang: str = "en"


class TrainIn(BaseModel):
    name: str = Field(min_length=1, max_length=24)
    kind: str
    samples: list


NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,23}$")


def _load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    return json.loads(path.read_text())


def _signs() -> dict:
    return _load_json(RUNTIME / "signs.json", {"static": {}, "dynamic": {}})


def _save_signs(data: dict) -> None:
    (RUNTIME / "signs.json").write_text(json.dumps(data))


def _support(kind: str) -> tuple[np.ndarray, list[str]]:
    path = MODELS / f"support_{kind}.json"
    payload = json.loads(path.read_text())
    vectors = np.asarray(payload["vectors"], dtype=np.float32)
    return vectors, list(payload["labels"])


def _custom_names(kind: str) -> list[str]:
    return list(_signs().get(kind, {}).keys())


def _fit_and_store(kind: str) -> dict:
    signs = _signs().get(kind, {})
    base_x, base_y = _support(kind)
    names = []
    chunks_x = [base_x]
    chunks_y = [base_y]
    label_names = list(dict.fromkeys(base_y))
    index = {name: i for i, name in enumerate(label_names)}
    net = static_net() if kind == "static" else dynamic_net()
    for name, samples in signs.items():
        if name not in index:
            index[name] = len(label_names)
            label_names.append(name)
        embedded = []
        for sample in samples:
            if kind == "static":
                norm = normalize_landmarks(sample, zero_z=zero_z())[None, :, :]
                embedded.append(net.embed(norm)[0])
            else:
                seq = normalize_sequence(sample, zero_z=zero_z())
                flat = flatten_sequence(seq)[None, :, :]
                embedded.append(net.embed(flat)[0])
        arr = np.stack(embedded).astype(np.float32)
        # Repeat custom samples so a handful of recordings can compete with
        # the stored support set of the original classes.
        reps = max(1, int(np.ceil(len(base_x) / max(len(label_names), 1) / len(arr))))
        chunks_x.append(np.concatenate([arr] * reps, axis=0))
        chunks_y.append([name] * (len(arr) * reps))
        names.append(name)
    features = np.concatenate(chunks_x, axis=0)
    text_labels = []
    for chunk in chunks_y:
        text_labels.extend(list(chunk))
    y = np.array([index[item] for item in text_labels], dtype=np.int64)
    w, b = fit_softmax(features, y, num_classes=len(label_names))
    probs = predict_proba(features, w, b)
    acc = float((probs.argmax(axis=1) == y).mean())
    np.savez(RUNTIME / f"head_{kind}.npz", w=w, b=b, labels=np.array(label_names))
    return {"labels": label_names, "train_accuracy": acc, "custom": names}


def _head(kind: str):
    path = RUNTIME / f"head_{kind}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        labels = [str(item) for item in data["labels"].tolist()]
        return data["w"], data["b"], labels


def _maybe_custom(kind: str, embedding: np.ndarray, base_label: str, base_conf: float):
    loaded = _head(kind)
    custom = set(_custom_names(kind))
    if loaded is None or not custom:
        return base_label, base_conf, None
    w, b, labels = loaded
    probs = predict_proba(embedding[None, :], w, b)[0]
    best = int(np.argmax(probs))
    label = labels[best]
    conf = float(probs[best])
    if label in custom and conf >= 0.55:
        ranked = topk(probs[None, :], labels, k=min(3, len(labels)))
        return label, conf, ranked
    return base_label, base_conf, None


@app.get("/health")
def health():
    signs = _signs()
    ready = (MODELS / "static.npz").exists() and (MODELS / "dynamic.npz").exists()
    return {
        "status": "ok" if ready else "models-missing",
        "version": __version__,
        "static_labels": STATIC_LABELS,
        "dynamic_labels": DYNAMIC_LABELS,
        "custom_static": list(signs.get("static", {})),
        "custom_dynamic": list(signs.get("dynamic", {})),
        "z_mode": meta().get("z_mode", "zero"),
        "models_loaded": ready,
    }


@app.post("/predict")
def predict(body: PredictIn):
    if body.landmarks is None and body.sequence is None:
        raise HTTPException(400, "Send landmarks or a sequence")
    try:
        if body.sequence is not None:
            seq = normalize_sequence(body.sequence, mirror=body.mirror, zero_z=zero_z())
            flat = flatten_sequence(seq)[None, :, :]
            probs = dynamic_net().predict_proba(flat)[0]
            best = int(np.argmax(probs))
            label = DYNAMIC_LABELS[best]
            conf = float(probs[best])
            ranked = topk(probs[None, :], DYNAMIC_LABELS)
            custom_label, custom_conf, custom_ranked = _maybe_custom(
                "dynamic", dynamic_net().embed(flat)[0], label, conf
            )
            return {
                "kind": "dynamic",
                "label": custom_label,
                "confidence": custom_conf,
                "topk": custom_ranked or ranked,
            }
        points = landmarks_to_array(body.landmarks)
        norm = normalize_landmarks(points, mirror=body.mirror, zero_z=zero_z())[None, :, :]
        probs = static_net().predict_proba(norm)[0]
        best = int(np.argmax(probs))
        label = STATIC_LABELS[best]
        conf = float(probs[best])
        ranked = topk(probs[None, :], STATIC_LABELS)
        custom_label, custom_conf, custom_ranked = _maybe_custom(
            "static", static_net().embed(norm)[0], label, conf
        )
        return {
            "kind": "static",
            "label": custom_label,
            "confidence": custom_conf,
            "topk": custom_ranked or ranked,
        }
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/tts")
def tts(body: TTSIn):
    text = " ".join(body.text.split())
    if not text:
        raise HTTPException(400, "Text is empty")
    lang = body.lang if re.fullmatch(r"[a-z]{2}(-[A-Z]{2})?", body.lang) else "en"
    try:
        from gtts import gTTS

        buffer = io.BytesIO()
        gTTS(text=text, lang=lang.split("-")[0]).write_to_fp(buffer)
    except Exception as exc:  # network or gTTS failure
        raise HTTPException(503, f"Text to speech is unavailable: {exc}") from exc
    return Response(content=buffer.getvalue(), media_type="audio/mpeg")


@app.post("/train")
def train(body: TrainIn):
    if body.kind not in {"static", "dynamic"}:
        raise HTTPException(422, "kind must be static or dynamic")
    name = body.name.strip().upper().replace(" ", "_")
    if not NAME_RE.match(body.name.strip()):
        raise HTTPException(422, "Name must start with a letter and stay short")
    if len(body.samples) < (6 if body.kind == "dynamic" else 12):
        raise HTTPException(422, "Not enough samples")
    if len(body.samples) > 80:
        raise HTTPException(422, "Too many samples")
    try:
        if body.kind == "static":
            cleaned = [landmarks_to_array(sample).tolist() for sample in body.samples]
        else:
            cleaned = []
            for seq in body.samples:
                if not isinstance(seq, list) or len(seq) < 4:
                    raise ValueError("Each dynamic sample needs at least 4 frames")
                cleaned.append([landmarks_to_array(frame).tolist() for frame in seq])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    store = _signs()
    store.setdefault(body.kind, {})
    store[body.kind][name] = cleaned
    _save_signs(store)
    summary = _fit_and_store(body.kind)
    return {"ok": True, "name": name, "kind": body.kind, **summary}


@app.delete("/train/{kind}/{name}")
def delete_sign(kind: str, name: str):
    if kind not in {"static", "dynamic"}:
        raise HTTPException(422, "kind must be static or dynamic")
    store = _signs()
    key = name.strip().upper()
    if key not in store.get(kind, {}):
        raise HTTPException(404, "Sign not found")
    del store[kind][key]
    _save_signs(store)
    head_path = RUNTIME / f"head_{kind}.npz"
    if store[kind]:
        summary = _fit_and_store(kind)
    else:
        head_path.unlink(missing_ok=True)
        summary = {"labels": [], "train_accuracy": None, "custom": []}
    return {"ok": True, **summary}

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import RUNTIME, app

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate custom-sign files from the repo and from other tests.
    monkeypatch.setattr("backend.app.main.RUNTIME", tmp_path)
    for stale in tmp_path.glob("*"):
        stale.unlink()
    return TestClient(app)


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert len(body["static_labels"]) == 24
    assert "HELLO" in body["dynamic_labels"]
    assert "J" not in body["static_labels"]


def test_predict_fixture_letter(client):
    fixture = json.loads((ROOT / "models" / "parity_static.json").read_text())
    raw = np.asarray(fixture["raw_xy"], dtype=np.float32).reshape(21, 2)
    landmarks = np.concatenate([raw, np.zeros((21, 1), np.float32)], axis=1).tolist()
    res = client.post("/predict", json={"landmarks": landmarks})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] == fixture["label"]
    assert body["confidence"] > 0.4
    assert body["topk"][0]["label"] == fixture["label"]


def test_predict_rejects_garbage(client):
    res = client.post("/predict", json={"landmarks": [1, 2, 3]})
    assert res.status_code == 422
    res = client.post("/predict", json={})
    assert res.status_code == 400


def test_dynamic_sample_shape(client):
    samples = json.loads(
        (ROOT / "frontend" / "public" / "samples" / "dynamic_samples.json").read_text()
    )
    seq = samples["samples"]["HELLO"][0]
    res = client.post("/predict", json={"sequence": seq})
    assert res.status_code == 200
    assert res.json()["kind"] == "dynamic"
    assert res.json()["label"] in samples["labels"]


def test_tts_mocked(client, monkeypatch):
    class Fake:
        def __init__(self, text, lang):
            self.text = text

        def write_to_fp(self, fp):
            fp.write(b"ID3fake-mp3")

    monkeypatch.setattr("gtts.gTTS", Fake)
    res = client.post("/tts", json={"text": "hello there"})
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("audio/mpeg")
    assert res.content.startswith(b"ID3")


def test_tts_empty(client):
    res = client.post("/tts", json={"text": "   "})
    assert res.status_code in {400, 422}


def test_train_custom_static_sign(client):
    # A pose that is not a natural letter: landmarks stretched far on x.
    base = np.zeros((21, 3), dtype=np.float32)
    base[9] = (0, 1, 0)
    base[4] = (3, 0.2, 0)
    base[8] = (2.4, -0.4, 0)
    base[12] = (1.2, -1.5, 0)
    samples = []
    rng = np.random.default_rng(0)
    for _ in range(16):
        jitter = base + rng.normal(0, 0.01, base.shape).astype(np.float32)
        samples.append(jitter.tolist())
    res = client.post(
        "/train",
        json={"name": "Snap", "kind": "static", "samples": samples},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["name"] == "SNAP"
    assert "SNAP" in body["custom"]
    assert body["train_accuracy"] > 0.8
    health = client.get("/health").json()
    assert "SNAP" in health["custom_static"]
    pred = client.post("/predict", json={"landmarks": samples[0]})
    assert pred.status_code == 200
    assert pred.json()["label"] == "SNAP"
    gone = client.delete("/train/static/SNAP")
    assert gone.status_code == 200
    assert "SNAP" not in client.get("/health").json()["custom_static"]

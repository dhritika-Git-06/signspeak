"""
Train a small word classifier on REAL signers from PopSign ASL v1.0 (CC BY 4.0, via the
sign/popsign-images frames on Hugging Face). MediaPipe HandLandmarker landmarks were extracted
from the frames by training/extract_popsign.py. Only the dominant hand is used, because the live
app sees hands only (no face or body), so location signs like WATER-at-chin lose that cue.

Evaluation is signer independent: no participant in the test set appears in training.
Writes frontend/public/models/asl_words.json (weights) and frontend/public/evaluation/asl_words.json (metrics).
"""
import glob, json, re, sys, collections, random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

T = 8
SRC = sys.argv[1] if len(sys.argv) > 1 else "/workspace/popsign/out"
OUT_MODEL = "/workspace/ss/frontend/public/models/asl_words.json"
OUT_METRICS = "/workspace/ss/frontend/public/evaluation/asl_words.json"


def participant(path):
    m = re.search(r"(\d+[a-z]?\.\d+)-", path.split("/")[-1])
    return m.group(1) if m else path


def features(frames, handedness_of=lambda h: h["h"]):
    """frames: list of list of hands ({h, p}). Returns a fixed-length vector or None."""
    counts = collections.Counter(handedness_of(h) for f in frames for h in f)
    if not counts:
        return None
    dom = counts.most_common(1)[0][0]
    seq, other = [], []
    for f in frames:
        hand = next((h for h in f if handedness_of(h) == dom), None)
        if hand is None:
            continue
        p = np.array(hand["p"], dtype=np.float32)[:, :2]
        if dom == "Left":
            p[:, 0] = -p[:, 0]
        seq.append(p)
        other.append(1.0 if len(f) > 1 else 0.0)
    if len(seq) < 2:
        return None
    return encode(seq, other)


def encode(seq, other):
    seq = np.stack(seq)  # n, 21, 2 (image coords, mirrored so the dominant hand looks like a right hand)
    palm = np.linalg.norm(seq[:, 9] - seq[:, 0], axis=1)
    scale = float(np.median(palm)) or 1e-3
    idx = np.linspace(0, len(seq) - 1, T)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    a = (idx - lo)[:, None, None]
    res = seq[lo] * (1 - a) + seq[hi] * a
    shape = (res - res[:, :1]) / scale  # handshape per step, wrist centred
    path = (res[:, 0] - res[0, 0]) / scale  # wrist path from the first step
    oth = np.interp(idx, np.arange(len(other)), other)
    return np.concatenate([shape.reshape(-1), path.reshape(-1), oth]).astype(np.float32)


def load():
    clips = []
    for f in glob.glob(f"{SRC}/*.json"):
        clips += json.load(open(f))
    seen = set()
    out = []
    for c in clips:
        if c["file"] in seen:
            continue
        seen.add(c["file"])
        out.append(c)
    return out


def main():
    clips = load()
    labels = sorted({c["word"] for c in clips})
    test_people = {participant(c["file"]) for c in clips if c["split"] == "test"}
    X, y, split, people = [], [], [], []
    dropped = 0
    for c in clips:
        v = features(c["frames"])
        if v is None:
            dropped += 1
            continue
        p = participant(c["file"])
        X.append(v); y.append(labels.index(c["word"])); people.append(p)
        # Signer independent: anyone who appears in the official test split is test only.
        split.append("test" if p in test_people else "train")
    X = np.stack(X); y = np.array(y); split = np.array(split)
    tr, te = split == "train", split == "test"
    print(f"clips {len(clips)}, usable {len(X)}, no hand {dropped}, train {tr.sum()} from {len(set(np.array(people)[tr]))} signers, test {te.sum()} from {len(set(np.array(people)[te]))} signers")
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xs = (X - mu) / sd
    # Pick the model on held-out TRAIN signers only; the test signers are used once at the end.
    ppl = np.array(people)
    train_people = sorted(set(ppl[tr]))
    random.Random(0).shuffle(train_people)
    val_people = set(train_people[: max(4, len(train_people) // 5)])
    va = tr & np.isin(ppl, list(val_people))
    fit = tr & ~va
    candidates = {"logistic": lambda: LogisticRegression(max_iter=3000, C=0.5)}
    for hidden, alpha in [((128,), 1e-3), ((256,), 1e-2), ((128, 64), 1e-2), ((256, 128), 3e-2)]:
        candidates[f"mlp{hidden}_a{alpha}"] = (lambda h=hidden, a_=alpha: MLPClassifier(hidden_layer_sizes=h, alpha=a_, max_iter=400, early_stopping=True, random_state=0))
    results = {}
    for name, make in candidates.items():
        results[name] = accuracy_score(y[va], make().fit(Xs[fit], y[fit]).predict(Xs[va]))
    choice = max((k for k in results if k.startswith("mlp")), key=results.get)
    print("validation", results, "->", choice)
    best = candidates[choice]().fit(Xs[tr], y[tr])
    best_acc = accuracy_score(y[te], best.predict(Xs[te]))
    pred = best.predict(Xs[te])
    prob = best.predict_proba(Xs[te])
    report = classification_report(y[te], pred, target_names=labels, digits=4, output_dict=True)
    cm = confusion_matrix(y[te], pred).tolist()
    top3 = float(np.mean([y[te][i] in np.argsort(-prob[i])[:3] for i in range(len(pred))]))
    metrics = {
        "source": "PopSign ASL v1.0 (Starner et al., NeurIPS 2023), CC BY 4.0, frames from huggingface.co/datasets/sign/popsign-images (game subset). Landmarks extracted with MediaPipe HandLandmarker.",
        "real_data": True,
        "signer_independent": True,
        "labels": labels,
        "train_clips": int(tr.sum()), "test_clips": int(te.sum()),
        "train_signers": len(set(np.array(people)[tr])), "test_signers": len(set(np.array(people)[te])),
        "clips_without_hands": dropped,
        "accuracy": float(best_acc), "macro_f1": float(f1_score(y[te], pred, average="macro")), "top3_accuracy": top3,
        "chance": 1 / len(labels),
        "validation_accuracy_by_candidate": {k: float(v) for k, v in results.items()},
        "chosen": choice,
        "per_class": {l: {"precision": report[l]["precision"], "recall": report[l]["recall"], "f1": report[l]["f1-score"], "support": report[l]["support"]} for l in labels},
        "confusion": cm,
        "features": f"Dominant hand only, {T} resampled steps: wrist-centred 21-point shape and wrist path in palm units, plus a second-hand flag.",
    }
    r = lambda a: [float("%.4g" % x) for x in np.asarray(a).reshape(-1)]
    weights = [{"W": [r(row) for row in w], "b": r(b)} for w, b in zip(best.coefs_, best.intercepts_)]
    json.dump({"labels": labels, "steps": T, "mean": r(mu), "std": r(sd), "layers": weights, "activation": best.activation}, open(OUT_MODEL, "w"), separators=(",", ":"))
    json.dump(metrics, open(OUT_METRICS, "w"), indent=1)
    print(json.dumps({k: metrics[k] for k in ["accuracy", "macro_f1", "top3_accuracy", "train_clips", "test_clips", "train_signers", "test_signers"]}))


def export_test_clips(path="/workspace/popsign/test_clips.json"):
    """Dominant-hand frames of the test signers, for checking the hand-written rules on real signing."""
    clips = load()
    test_people = {participant(c["file"]) for c in clips if c["split"] == "test"}
    out = []
    for c in clips:
        if participant(c["file"]) not in test_people:
            continue
        counts = collections.Counter(h["h"] for f in c["frames"] for h in f)
        if not counts:
            continue
        dom = counts.most_common(1)[0][0]
        kept = [(next((h["p"] for h in f if h["h"] == dom), None), 1 if len(f) > 1 else 0) for f in c["frames"]]
        kept = [(p, o) for p, o in kept if p]
        if len(kept) >= 2:
            out.append({"word": c["word"], "hand": dom, "frames": [p for p, _ in kept], "other": [o for _, o in kept]})
    json.dump(out, open(path, "w"))
    print("exported", len(out))


if __name__ == "__main__":
    main()
    export_test_clips()

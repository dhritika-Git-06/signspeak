"""Extract MediaPipe hand landmarks from PopSign ASL (CC BY 4.0) frames for words that overlap SignSpeak."""
import json, io, sys, os
import numpy as np, pyarrow.parquet as pq, fsspec
from PIL import Image
from multiprocessing import Pool
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

WORDS = {"hello": "HELLO", "yes": "YES", "no": "NO", "thankyou": "THANKS", "please": "PLEASE", "water": "WATER",
         "drink": "DRINK", "go": "GO", "home": "HOME", "like": "LIKE", "fine": "FINE", "bad": "BAD", "where": "WHERE", "weus": "WE"}
BASE = "https://huggingface.co/datasets/sign/popsign-images/resolve/main/data/"
INDEX = json.load(open("/workspace/popsign/index.json"))
MODEL = "/workspace/ss/frontend/public/models/hand_landmarker.task"

def jobs():
    out = []
    for name, groups in INDEX.items():
        if isinstance(groups, str): continue
        split = "train" if "/train-" in name else "test"
        for gi, g in enumerate(groups):
            if not g: continue
            lo, hi, _ = g
            hits = [w for w in WORDS if lo <= w <= hi]
            if hits: out.append((name, gi, split, hits))
    return out

def work(job):
    name, gi, split, hits = job
    tag = f"{name.replace('/', '_')}_{gi}"
    dest = f"/workspace/popsign/out/{tag}.json"
    if os.path.exists(dest): return tag, "cached"
    detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL), num_hands=2,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4))
    fs = fsspec.filesystem("https")
    with fs.open(BASE + name, block_size=8 << 20) as fh:
        t = pq.ParquetFile(fh).read_row_group(gi, columns=["file", "text", "images"])
    texts = t.column("text").to_pylist()
    files = t.column("file").to_pylist()
    rows = [i for i, x in enumerate(texts) if x in hits]
    images = t.column("images")
    clips = []
    for i in rows:
        frames = []
        for img in images[i].as_py():
            arr = np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
            hands = []
            for lm, hd in zip(res.hand_landmarks, res.handedness):
                hands.append({"h": hd[0].category_name, "s": round(hd[0].score, 3), "p": [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in lm]})
            frames.append(hands)
        clips.append({"word": WORDS[texts[i]], "file": files[i], "split": split, "frames": frames})
    json.dump(clips, open(dest, "w"))
    return tag, len(clips)

if __name__ == "__main__":
    js = jobs()
    print(len(js), "row groups", flush=True)
    with Pool(7) as p:
        for tag, n in p.imap_unordered(work, js):
            print(tag, n, flush=True)

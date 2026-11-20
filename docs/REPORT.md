# SignSpeak AI — objective map

Project: **SignSpeak AI – Sign Language to Speech Converter**, by Dhritika Sahu.

The synopsis file supplied with the brief lists these objectives. Each one below is implemented and can be exercised without a GPU server. Where a behaviour depends on a webcam, the same code path accepts recorded landmarks (`Play sample` on Recognize, `python demo_webcam.py --samples`, and the API tests).

| Synopsis objective | Where it lives | How to see it work |
| --- | --- | --- |
| Real-time webcam capture | `frontend/src/hooks/useHandTracker.ts`, Recognize and Train pages | Start camera. Permission errors stay on the page and do not block sample replay. |
| MediaPipe hand landmarks | `@mediapipe/tasks-vision` `HandLandmarker`, model `frontend/public/models/hand_landmarker.task`, overlay in `frontend/src/lib/hands.ts` | Skeleton is drawn on the camera. WASM is served from `/wasm` so the page does not need a CDN. |
| CNN gesture classifier | `train_static.py`, `signspeak/models.py` `build_static`, browser `frontend/src/lib/model.ts` | 1D convolutions over 21×3 normalized landmarks. Test accuracy **94.19%**, macro F1 **94.23%** (`evaluation/metrics.json`). |
| LSTM gesture classifier | `train_dynamic.py`, `signspeak/dynamic_data.py`, `build_dynamic` | 30-frame sequences. Held-out synthetic accuracy **99.12%**, macro F1 **99.13%**. The data is real handshapes plus documented motions, not filmed words. See README. |
| Recognized sign to text | `frontend/src/lib/stabilizer.ts`, `frontend/src/lib/sentence.ts`, Recognize page | Majority vote, confidence slider, debounce so a held letter is committed once, then again after the hand drops. Space, Delete, Clear. |
| Text to speech | `frontend/src/lib/speech.ts` (`speechSynthesis` by default), `POST /tts` in `backend/app/main.py` (gTTS) | Speak and Auto-speak. If `/health` fails, the page says it is on the browser voice and still speaks. |
| Simple responsive web UI | `frontend/src/components/Layout.tsx` and the four routes | Stacks on a narrow viewport. Dark / light toggle. Skip link, live region on the prediction, labelled controls. |
| Custom sign training | Train page, `trainLinearHead` in `frontend/src/lib/model.ts`, IndexedDB `frontend/src/lib/storage.ts`, `POST /train` and `DELETE /train/{kind}/{name}` | Name a sign, record bursts, fit a softmax head on frozen embeddings, export / import JSON. The new label is preferred when that head is confident. |
| Evaluation: accuracy and confusion matrix | `evaluate.py`, `signspeak/metrics.py`, Evaluation page | `evaluation/static_confusion.png`, `evaluation/dynamic_confusion.png`, classification reports, and the same files under `frontend/public/evaluation/`. |

## Stack, and where it was adjusted

The brief allows Flask or FastAPI, TensorFlow/Keras, and gTTS or pyttsx3, with a React front end.

- React + Vite + TypeScript, rather than Next.js, so the site is a static bundle and TensorFlow.js never has to survive server rendering.
- FastAPI for `/health`, `/predict`, `/tts`, `/train`.
- Keras trains the CNN and LSTM. The serving image runs the same math in NumPy so a free host does not have to load TensorFlow. `backend/tests` and `frontend/src/lib/model.test.ts` check that NumPy and TensorFlow.js match Keras.
- gTTS on the API. The browser default is `speechSynthesis`, because a visitor with only a browser cannot depend on the API.
- OpenCV is used by `demo_webcam.py` to draw the hand skeleton and, with `--camera`, to open a capture device.

## Reproduce

```bash
PYTHONPATH=. python train_static.py
PYTHONPATH=. python train_dynamic.py
PYTHONPATH=. python evaluate.py
PYTHONPATH=. pytest
cd frontend && npm test && npm run build
```

`train_static.py` checks the landmark CSV against SHA-256 `8150c9277aec32c3b89d9df40a074b6601806f5c1f219d9e58b097fe4e6e7ffa`.

## Honest limits (also on the How it works page)

- Not full ASL. No facial grammar, no two-handed signs in the shipped vocabulary, no sentence-level translation.
- Letter photos are one signer (GPL-2.0 ASL Alphabet via the KNN_Alphabet landmark CSV).
- Word and J/Z scores are from kinematic sequences seeded by those photos. Different photographs are held out; different people were not filmed.
- A visitor’s own signs are first-class: that is what the training module is for.

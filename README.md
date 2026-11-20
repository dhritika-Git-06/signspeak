# SignSpeak AI

Built by Dhritika Sahu.

SignSpeak AI turns a hand in a webcam into letters, a short English sentence, and speech. The page runs in the browser: MediaPipe finds the hand, a TensorFlow.js CNN reads still ASL letters, geometric rules read a set of common words, and `speechSynthesis` speaks the sentence. A small Python API is optional. If it is down, the page still recognizes signs and still speaks.

[Run it locally](#run-the-app) on port **43123** after `npm run dev` in `frontend/`.

![Landing page with the presentation guide](docs/screenshots/landing.png)

![Scripted demo producing “I want water.”](docs/screenshots/presentation-demo.png)

![A recorded B, classified at 100% and added to the sentence](docs/screenshots/02-letter-b.png)

The Evaluation page shows the held-out numbers and both confusion matrices.

![Held-out accuracy and confusion matrices](docs/screenshots/06-evaluation.png)

## What it recognizes

**Static ASL letters** (1D CNN on one frame): A B C D E F G H I K L M N O P Q R S T U V W X Y.

J and Z are motions, so they are not in the still-letter model.

**Words** (geometric rules, not a trained accuracy): HELLO, YES, NO, THANKS, PLEASE, SORRY, ME, YOU, WE, LOVE, STOP, WATER, MORE, DRINK, EAT, GO, COME, GOOD, BAD, LIKE, WANT, NEED, WHERE, WHAT, NAME, FINE, HOME. HELP and SCHOOL need two hands and are not detected. A presentation button plays ME → WANT → WATER and speaks “I want water.” if the camera fails.

An earlier LSTM on kinematic sequences scored 99.12% on its synthetic holdout. That model is not used in the live demo. Do not quote 99.12% as webcam word accuracy.

You can add your own pose or motion from the Train page. Those signs stay in this browser (IndexedDB) and can be exported.

## Architecture

```mermaid
flowchart LR
  subgraph browser [Browser, no GPU server]
    cam[Webcam]
    mp[MediaPipe HandLandmarker]
    norm[Normalize 21 x 3 landmarks]
    cnn[1D CNN]
    lstm[LSTM]
    head[Your softmax head]
    stable[Confidence and debounce]
    text[Sentence]
    voice[speechSynthesis]
    cam --> mp --> norm
    norm --> cnn
    norm --> lstm
    cnn --> head
    lstm --> head
    head --> stable --> text --> voice
  end
  subgraph api [Optional FastAPI]
    predict["/predict"]
    tts["/tts gTTS"]
    train["/train"]
  end
  text -. if the API is up .-> tts
  norm -. same weights in NumPy .-> predict
  head -. landmark upload .-> train
```

| Piece | Where |
| --- | --- |
| Camera, overlay, sentence, theme | `frontend/src/pages/RecognizePage.tsx` |
| Landmark math, shared with Python | `frontend/src/lib/normalize.ts`, `signspeak/preprocess.py` |
| Debounce | `frontend/src/lib/stabilizer.ts` |
| CNN and LSTM in the browser | `frontend/src/lib/model.ts` |
| Custom signs | `frontend/src/pages/TrainPage.tsx`, IndexedDB in `frontend/src/lib/storage.ts` |
| Keras training | `train_static.py`, `train_dynamic.py` |
| Official metrics | `evaluate.py` → `evaluation/` and the Evaluation page |
| API | `backend/app/main.py` (`/health`, `/predict`, `/tts`, `/train`) |
| OpenCV demo | `python demo_webcam.py` |

## Real test-set numbers

From `python evaluate.py` on the held-out splits (seed 42). These are the figures on the Evaluation page. They are not rounded.

| Model | Test accuracy | Macro F1 | What the test is |
| --- | --- | --- | --- |
| Static 1D CNN | **94.19%** | **94.23%** | 3,600 landmark rows, 15% stratified holdout |
| Dynamic LSTM | **99.12%** | **99.13%** | 800 sequences whose source photographs were not in training |

Static depth check: adding Gaussian noise of σ = 0.1 on z changes accuracy to 94.22%, so the browser may pass real MediaPipe depth through (`z_mode: passthrough`).

The usual static confusions are M↔N, then P→Q and R→U. Those pairs look alike as 21 points.

The dynamic score is high because the motions are generated. See [Dynamic signs](#dynamic-signs) before treating 99% as a webcam result.

## Dataset and license

**Static letters.** 1,000 MediaPipe Hands rows for each of 24 letters, file `hand_dataset_1000_24.csv` from [MinorvaFalk/KNN_Alphabet](https://github.com/MinorvaFalk/KNN_Alphabet). Those rows were extracted from the [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) photographs by Akash Nagaraj (Kaggle, 2018), which is **GPL-2.0**. The photographs are one signer in even light, often augmented. A random split therefore shares that signer’s appearance between train and test. Live accuracy on a new hand will be lower until you record a few bursts on the Train page.

Columns are `class, x1, y1, … x21, y21`. There is no z in the file. The network still takes a 21×3 tensor; z is filled and lightly noised during training.

**Dynamic signs.** There is no small, freely downloadable ASL *word* set of MediaPipe landmarks that can be fetched without a Kaggle login. The LSTM is trained on kinematic sequences **seeded by real handshapes** from the letter file:

| Sign | Handshape donor | Motion |
| --- | --- | --- |
| J | I | J stroke |
| Z | D | Z stroke |
| HELLO | B | side-to-side wave |
| YES | S | nod |
| NO | V | wag (rotation) |
| THANKS | B | drift downward |
| PLEASE | B | circle |
| SORRY | A | smaller circle |

Train and test use different source photographs (220 vs 50 per class, with several random speeds and sizes). The number above is that held-out synthetic test. It is not a recording of people signing those words. The Train page is the supported way to add a motion the way *you* sign it.

The code and the weights derived from the GPL-2.0 photographs are released under GPL-2.0. See `LICENSE`.

## Run the app

### In VS Code

1. Install [Node.js 20 or newer](https://nodejs.org/) if `node -v` fails in a terminal.
2. Unzip `SignSpeak-AI.zip` and open that folder in VS Code (**File → Open Folder**). Do not open the zip itself.
3. Open a terminal in VS Code (**Terminal → New Terminal**) and run:

```bash
cd frontend
npm ci
npm run dev
```

4. VS Code will offer to open the local server. If it does not, go to http://127.0.0.1:43123 .
5. Allow the camera, or press **Play sample B** / **Play sample HELLO** if this computer has no webcam.

**Terminal → Run Task → SignSpeak: start dev server** runs the same two commands.

The browser app does not need Python. The optional API is below.

```bash
cd frontend
npm ci
npm run dev
```

Open http://127.0.0.1:43123 . Allow the camera, or press **Play sample B** / **Play sample HELLO** if you have no webcam. Those buttons feed recorded landmarks through the same normalizer, model, and sentence builder.

```bash
cd frontend
npm test          # Vitest: normalize, sentence, debounce, TF.js parity
npm run build     # writes ../dist
```

### API (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port 8765
PYTHONPATH=. pytest
python demo_webcam.py --samples
```

On localhost the page probes `http://127.0.0.1:8765/health` and then `/api/health`. When the probe succeeds, **Speak** asks `POST /tts` for an mp3 and falls back to the browser voice if gTTS cannot reach Google. Set `VITE_API_URL` at build time to point a static deploy at a public API.

`python demo_webcam.py --camera 0` opens an OpenCV window. Without a camera, `--samples` classifies one landmark row per letter and writes `evaluation/opencv_demo.png`.

### Train the models again

```bash
pip install -r backend/requirements.txt
PYTHONPATH=. python train_static.py
PYTHONPATH=. python train_dynamic.py
PYTHONPATH=. python evaluate.py
```

`train_static.py` downloads the landmark CSV (SHA-256 pinned in `signspeak/data.py`) into `data/raw/`, which is gitignored. The checked-in weights are the result of that run.

## Deploy

No deploy token was available in this environment (no Vercel, Netlify, Cloudflare, or Hugging Face credential), so there is no public URL yet. The repo is set up so a static host and a free API host can be attached without code changes.

**Frontend, static.** `frontend` builds to `/dist`.

- GitHub Pages for `dhritika-Git-06/signspeak`: `.github/workflows/deploy-pages.yml` builds with `VITE_BASE=/signspeak/` and deploys `dist`. The site is then at `https://dhritika-git-06.github.io/signspeak/`. Local `npm run dev` leaves the base at `/`.
- Vercel: import the repo. `vercel.json` sets the install, build, output directory `dist`, and the SPA rewrite. Do not set `VITE_BASE` there.
- Netlify: `netlify.toml` publishes `dist`.
- Cloudflare Pages: build command `cd frontend && npm ci && npm run build`, output directory `dist`.

Do not put a secret in the frontend. If you deploy the API, rebuild with `VITE_API_URL=https://your-api.example`.

**API.**

- Render: `render.yaml` builds `backend/Dockerfile` (Python 3.11, NumPy inference, no TensorFlow in the image) and checks `/health`.
- Hugging Face Spaces: the repo-root `Dockerfile` listens on port **7860**, which Spaces expects. Create a Docker Space and point it at this repo.
- Both together: `docker compose up --build` serves the API on port 8765 and nginx on port 43123, with `/api` proxied to the API.

The serving image does not include TensorFlow. `evaluate.py` and the training scripts do.

## Pages

- **Recognize** — camera, landmark overlay, live letter, confidence, top alternatives, alphabet strip, sentence, space / delete / clear, speak, auto-speak, letter / word / auto mode.
- **Train** — name a sign, record bursts, fit a TF.js head, export and import JSON. If the API is up, the same samples are posted to `/train`.
- **How it works** — the pipeline in plain language, including what the project does not claim.
- **Evaluation** — accuracy, macro F1, confusion matrices, classification reports, dataset citation.

Light and dark follow the system theme until you toggle them. The choice is stored locally.

## Limitations

- Fingerspelling and a handful of isolated motions, not conversational ASL. No face, eyebrows, or body grammar.
- The public photograph set is a single signer. M and N, and a few other pairs, stay easy to confuse.
- J and Z are only on the motion model, and those motions are templates, not a filmed corpus.
- Auto mode switches on hand motion. A very still word or a shaky letter can pick the wrong network. The Letters / Words toggle overrides it.
- Custom signs are only as good as the bursts you record, and they live in that browser until you export them.
- gTTS needs the API and outbound network. The browser voice does not.
- The first visit downloads the hand landmarker (~8 MB) and the MediaPipe WASM (~14 MB).

## Future scope

These are the natural next steps for this project, in the same spirit as a final-year synopsis:

- Continuous signing, not one letter or one isolated word at a time.
- Indian Sign Language vocabulary beside the ASL alphabet.
- Non-manual signals: face, head, and a second hand.
- Speech-to-sign the other way, with an avatar.
- A multi-signer landmark set so the reported accuracy matches new hands.
- On-device speech that does not call Google, and a phone layout that is the primary UI rather than a responsive page.
- A grammar pass over the fingerspelled string before it is spoken.

`docs/REPORT.md` maps each required objective to the file that implements it.

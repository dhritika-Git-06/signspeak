# SignSpeak AI

**Built by Dhritika Sahu**

## Abstract

SignSpeak AI is a browser application that turns American Sign Language fingerspelling and a defined set of word gestures (one- and two-handed) into an English or Hindi sentence and spoken audio. Hand landmarks come from MediaPipe. Still letters are classified by a 1D convolutional network trained on a public landmark table. Word gestures are geometric rules on those same landmarks, and an optional word model trained on real signers (PopSign ASL) is offered as a beta mode. A small offline grammar turns glosses such as ME WANT WATER into “I want water.”

## Objective

The practical objective for the demonstration is a laptop page that works without a GPU server: show a hand, confirm a sign, build a sentence, and hear it. If the camera or the room lighting fails, a scripted landmark sequence still plays the same sentence.

## Method

1. The webcam frame is passed to MediaPipe HandLandmarker, which returns 21 landmarks.
2. Still letters are wrist-centred, scaled by the palm, and passed to a CNN trained on `hand_dataset_1000_24.csv` (MinorvaFalk/KNN_Alphabet, derived from the ASL Alphabet photographs of Akash Nagaraj, GPL-2.0). The classes are A–Y excluding J and Z. J and Z are motions and are not in that file.
3. Word signs are rules. For example, WATER is the W handshape (index, middle and ring extended, pinky closed), ME is an index finger pointing down, and WANT is an open hand pulled back toward the signer, read from the palm getting smaller on camera (MediaPipe gives the wrist no usable depth). MediaPipe tracks up to two hands: HELP is a fist resting on a flat palm and SCHOOL is two flat hands clapping twice. Still signs are ignored while the hand is moving so a motion word does not first add a letter.
3a. A separate word model (MLP, 256 hidden units) was trained on MediaPipe landmarks extracted from PopSign ASL v1.0 (CC BY 4.0) for 14 words, split by signer. It is available as the “ASL beta” mode.
4. A sign is committed after it stays stable for about 650 ms, or for 8 frames. Letters can be corrected against a small dictionary (WATR becomes WATER). Glosses are mapped into an English sentence without calling an external language API.
5. A phrase table also renders the glosses in Hindi. `speechSynthesis` speaks the sentence, in Hindi when the device has a hi-IN voice. If the optional FastAPI service is running, Speak can request a gTTS recording and falls back to the browser voice.

## Results

Held-out test of the letter CNN, stratified 15% split, seed 42, 3,600 rows:

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Static 1D CNN | 94.19% | 94.23% |

The largest confusions are M predicted as N (23), N predicted as M (18), P as Q (9) and R as U (6). Adding depth noise of standard deviation 0.1 left accuracy at 94.22%.

Signer-independent test of the PopSign word model (real data), 1,431 clips from 8 signers not used in training:

| Model | Accuracy | Macro F1 | Top-3 |
| --- | --- | --- | --- |
| PopSign word MLP, 14 words | 89.38% | 88.87% | 94.83% |
| Hand-written rules on the same clips | 9.8% | – | – |

The rules fail on real ASL forms because they were written for simplified one-hand versions; they produce no word on 58.1% of the clips.

Synthetic per-sign test (a 3D hand model, 20 random variations per sign, pass only if the right sign is added exactly once): words 93.3% (weakest LIKE 45%, DRINK 70%), letters in Letters mode 92.3% (weakest M 65%, N 75%), letters in Auto mode 71% because W, O, F and X are read as words. HELP and SCHOOL pass at least 80% of 24 variations. This is a regression test, not a user study.

A previous LSTM, trained on kinematic paths applied to real handshapes, scored 99.12% on a synthetic holdout. That experiment is retained on the Evaluation page and is **not** the live word recognizer. It must not be reported as the accuracy of the presentation demo.

## Demonstration

The presentation order is ME, WANT, WATER. The expected sentence is “I want water.” (Hindi: “मुझे पानी चाहिए।”). The same three signs are stored in the page as a fallback when the webcam cannot be used. A Learn page gives a reference figure, graded practice, a quiz and text-to-sign. Copy, Share and Download .txt export the sentence. Every control is covered by 27 automated browser tests that must pass before the site deploys.

## Limits

The system handles isolated signs and a small grammar. It does not translate continuous ASL or facial grammar. The rule words are simplified forms, and the real-signer model covers only 14 words and needs a pause between signs. Hindi speech depends on the device's voices. The letter photographs are a single signer, so a new hand can be less accurate until a few samples are recorded on the Train page.

## References

- Akash Nagaraj, ASL Alphabet, Kaggle, GPL-2.0.
- MinorvaFalk/KNN_Alphabet, MediaPipe landmark conversion of that set.
- Starner et al., PopSign ASL v1.0, NeurIPS 2023 Datasets and Benchmarks, CC BY 4.0.
- Google MediaPipe Hand Landmarker.
- Dhritika Sahu.

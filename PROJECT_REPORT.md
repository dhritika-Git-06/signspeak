# SignSpeak AI

**Built by Dhritika Sahu**

## Abstract

SignSpeak AI is a browser application that turns one-hand American Sign Language fingerspelling, and a defined set of word gestures, into an English sentence and spoken audio. Hand landmarks come from MediaPipe. Still letters are classified by a 1D convolutional network trained on a public landmark table. Word gestures are geometric rules on those same landmarks. A small offline grammar turns glosses such as ME WANT WATER into “I want water.”

## Objective

The practical objective for the demonstration is a laptop page that works without a GPU server: show a hand, confirm a sign, build a sentence, and hear it. If the camera or the room lighting fails, a scripted landmark sequence still plays the same sentence.

## Method

1. The webcam frame is passed to MediaPipe HandLandmarker, which returns 21 landmarks.
2. Still letters are wrist-centred, scaled by the palm, and passed to a CNN trained on `hand_dataset_1000_24.csv` (MinorvaFalk/KNN_Alphabet, derived from the ASL Alphabet photographs of Akash Nagaraj, GPL-2.0). The classes are A–Y excluding J and Z. J and Z are motions and are not in that file.
3. Word signs are rules. For example, WATER is the W handshape (index, middle and ring extended, pinky closed), ME is an index finger pointing down, and WANT is an open hand moving toward the camera. HELP and SCHOOL are listed as out of scope because they need two hands. These rules are not assigned a dataset accuracy.
4. A sign is committed after it stays stable for about 650 ms, or for 8 frames. Letters can be corrected against a small dictionary (WATR becomes WATER). Glosses are mapped into an English sentence without calling an external language API.
5. `speechSynthesis` speaks the sentence. If the optional FastAPI service is running, Speak can request a gTTS recording and falls back to the browser voice.

## Results

Held-out test of the letter CNN, stratified 15% split, seed 42, 3,600 rows:

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| Static 1D CNN | 94.19% | 94.23% |

The largest confusions are M predicted as N (23), N predicted as M (18), P as Q (9) and R as U (6). Adding depth noise of standard deviation 0.1 left accuracy at 94.22%.

A previous LSTM, trained on kinematic paths applied to real handshapes, scored 99.12% on a synthetic holdout. That experiment is retained on the Evaluation page and is **not** the live word recognizer. It must not be reported as the accuracy of the presentation demo.

## Demonstration

The presentation order is ME, WANT, WATER. The expected sentence is “I want water.” The same three poses are stored in the page as a fallback when the webcam cannot be used.

## Limits

The system is one hand, isolated signs, and a small grammar. It does not translate continuous ASL, facial grammar, or two-handed signs. The letter photographs are a single signer, so a new hand can be less accurate until a few samples are recorded on the Train page.

## References

- Akash Nagaraj, ASL Alphabet, Kaggle, GPL-2.0.
- MinorvaFalk/KNN_Alphabet, MediaPipe landmark conversion of that set.
- Google MediaPipe Hand Landmarker.
- Dhritika Sahu.

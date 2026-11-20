# Presentation notes

**Dhritika Sahu**
**SignSpeak AI · about 5 minutes**

Say only what the page actually does. The letter model has a measured accuracy. The words in the demo are rules.

## Talk outline

1. **Problem (40 seconds).** A hearing person who does not know sign language cannot read a fingerspelled word or a short signed request. This project is a laptop page that turns one hand into text and speech, without sending the video to a server.
2. **What the camera sees (40 seconds).** MediaPipe finds 21 hand landmarks. The drawing on the video is those points. The page mirrors the preview so the presenter sees their own hand naturally.
3. **Letters (60 seconds).** Still letters go through a 1D CNN trained on a public table of MediaPipe landmarks from the ASL Alphabet photographs (Akash Nagaraj, GPL-2.0). The test accuracy is **94.19%**, macro F1 **94.23%**, on 3,600 held-out rows. J and Z are motions, so they are not in that model. M and N are the usual confusion.
4. **Words (50 seconds).** HELLO, YES, ME, WANT, WATER and the other words on the sign board are geometric rules: which fingers are extended, and how the wrist moves. They are not a second neural network and they do not have a dataset accuracy. HELP and SCHOOL need two hands, so the page does not claim to see them.
5. **Sentence (50 seconds).** Confirmed signs become glosses. ME WANT WATER is rewritten, offline, to “I want water.” A misspelled fingerspelled word such as WATR can snap to WATER if it is in the small dictionary. The browser speaks the line.
6. **Demo (60 seconds).** Follow the script below. If the camera fails, press **Play presentation demo**.

## Demo script

Open the page, scroll to the demo, and keep Presentation mode visible.

1. Say: “If the light is bad I will use the recorded poses. I will try the camera first.”
2. Start the camera. Sign **ME** (index finger pointing down), hold it until the chip appears.
3. Sign **WANT** (open hand pulled toward you). Hold.
4. Sign **WATER** (index, middle and ring up, pinky down). Hold.
5. The sentence should read **I want water.** Press **Speak**.
6. If any step fails, press **Play presentation demo**. It plays those three poses and speaks the same sentence. Say that this fallback is built into the page so the demo does not depend on the room.

Optional letter check, only if time remains: press **Play sample B**. That sample is a real row from the letter dataset, not a drawing.

## Viva questions

**What is the measured accuracy?**
94.19% accuracy and 94.23% macro F1 on the held-out letter rows. Do not quote 99% for the live words. That 99.12% figure is an older test of synthetic motions and is not used on the demo page.

**Why are words not given an accuracy?**
I could not download a public hand-landmark set, with a known layout, for those exact words in the time available. Inventing a test score would be wrong. The words are rules I can explain: finger extension and wrist motion.

**Which dataset is the letter model trained on?**
`hand_dataset_1000_24.csv`, 1,000 MediaPipe rows for each of 24 letters, from MinorvaFalk/KNN_Alphabet, taken from the ASL Alphabet images by Akash Nagaraj, GPL-2.0. The split is 70/15/15, seed 42.

**Why do M and N get confused?**
As 21 points they are both closed fists with the thumb tucked. The test set confuses them more than any other pair.

**Does the video leave the laptop?**
No. MediaPipe and the letter network run in the browser. Speech uses the browser voice unless a separate API is running.

**What does the grammar do?**
It is a small rule list. ME becomes I, WANT stays want, WATER stays water, and the line is capitalised and given a full stop. WHAT plus NAME becomes “What is your name?”

**What can it not do?**
Continuous ASL, facial expression, two-handed signs such as HELP and SCHOOL, and a new signer whose hand does not look like the training photographs. The Train page can add a personal sign in the browser.

**Why GitHub Pages under /signspeak/?**
The static build can set its base path to `/signspeak/` so assets still load when the site is hosted as a project page. Local `npm run dev` keeps the base at `/`.

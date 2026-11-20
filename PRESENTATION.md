# Presentation notes

**Dhritika Sahu**
**SignSpeak AI · about 7 minutes**

Say only what the page actually does. Every accuracy number on the Evaluation page is labelled *Real data* or *Synthetic test*. Quote them that way.

## Talk outline

1. **Problem (40 s).** A hearing person who does not know sign language cannot read a fingerspelled word or a short signed request. SignSpeak is a web page that turns hand signs into text and speech, in English or Hindi, without sending the video to a server.
2. **What the camera sees (40 s).** MediaPipe finds 21 landmarks on each of up to two hands. The drawing on the video is those points. Live hints say if the hand is too far, too close, out of frame or the room is too dark.
3. **Letters (50 s).** Still letters go through a 1D CNN trained on a public landmark table from the ASL Alphabet photographs (Akash Nagaraj, GPL-2.0). Held-out accuracy **94.19%**, macro F1 **94.23%**, on 3,600 rows. M and N are the usual confusion.
4. **Words (60 s).** The 29 words on the sign board are geometric rules on the landmarks, including two-hand HELP and SCHOOL. A synthetic test (a 3D hand model doing each sign 20 times with random size, angle and speed) gives **93.3%** for words and **92.3%** for letters. That is a regression test, not real people.
5. **Real signers (40 s).** To get an honest real-world number I trained a separate model on the PopSign ASL dataset (CC BY 4.0). On 8 signers it never saw, it reads 14 words with **89.38%** accuracy. My simple rules score only 9.8% on those same clips, because real ASL forms differ from my simplified forms. The model is in the app as **ASL beta** mode.
6. **Sentence and speech (40 s).** Signs become glosses; ME WANT WATER becomes “I want water.” and, in Hindi, “मुझे पानी चाहिए।”. The sentence can be spoken, copied, shared or downloaded.
7. **Learn page (40 s).** Reference hand, practice with Correct / Try again and a score, a 5-sign quiz, and text to sign.
8. **Demo (90 s).** Follow the script below.

## Demo script

Open https://dhritika-git-06.github.io/signspeak/ in Chrome. Allow the camera.

1. Press **Tips for best accuracy**, read one tip, press **Got it**. Start the camera. Point at the hint at the bottom of the video (“No hand found”, then “Good framing” when the hand is in view).
2. Sign **ME** (index finger pointing down), hold until the chip appears.
3. Sign **WANT**: open hand facing the camera, then pull it back toward your chest. Hold until the chip appears.
4. Sign **WATER** (index, middle and ring up, pinky down). Hold. Each held sign adds one word; drop the hand out of view to repeat a word.
5. The sentence reads **I want water.** Press **Speak**.
6. Change **Voice** to **हिन्दी (Hindi)**. The Hindi line appears. Press **Speak** again. (If the laptop has no Hindi voice the page says so and speaks English; the Hindi text is still on screen.)
7. Press **Copy**, then **Download .txt** to show the transcript.
8. If any camera step fails, press **Play presentation demo**. It plays the three signs and speaks the sentence. Say that the fallback is built in so the demo does not depend on the room.
9. Two hands (optional): make a fist resting on your other open palm and hold it: **HELP** (the badge says Two hands). For **SCHOOL** clap two flat hands twice.
10. Go to **Learn**. Choose **Hello**, press **Show me**, then **Play example**: “Correct! That was HELLO.” and the score goes up. Start the camera and sign it yourself.
11. Press **Start quiz (5 signs)**. Show **Show hint**, sign or press **Play example**, then **Skip** to the end to show the result.
12. In **Text to sign** type “thank you water” and press **Show signs**. Press a **Replay** button.
13. Open **Evaluation** and point at the Real data and Synthetic test labels.

## Viva questions

**What is the measured accuracy?**
Real data: letters 94.19% (held-out photos, one signer) and the PopSign word model 89.38% on 1,431 clips from 8 unseen signers. Synthetic: rules 93.3% for words and 92.3% for letters. Do not quote the old 99.12% LSTM figure; it was a synthetic motion test and is not used live.

**Why do you still use rules if the trained model is better on real signers?**
The rules cover 29 words including the ones in the demo sentence (ME, WANT) that PopSign does not have, they react while the hand is still, and they are explainable. The PopSign model only knows 14 words and has no “no sign” class, so it reads one sign between pauses. Both are in the app: Auto for the rules, ASL beta for the model.

**Why do the rules score 9.8% on PopSign?**
They were written for simplified one-hand forms. PopSign signers use real ASL: HELLO is a salute from the forehead, FINE is a 5-hand tapping the chest. Reporting that low number is the honest result.

**How is the synthetic test made?**
A 3D hand model with real finger proportions makes each sign with random left or right hand, size, position, rotation, speed, camera shape and tracker jitter, and the frames go through the real recognizer. A sample passes only if the right sign is added exactly once.

**How do two-hand signs work?**
MediaPipe returns up to two hands. HELP is a fist resting just above a flat palm for 9 of the last 12 frames. SCHOOL is two open hands whose gap closes and opens at least twice. While two hands are close together the one-hand rules are paused so a clap is not read as THANKS.

**How is the Hindi made?**
A phrase table maps gloss sequences to Hindi (ME WANT WATER → मुझे पानी चाहिए।), with a word-by-word fallback. Speech uses the device's hi-IN voice when one exists.

**Which datasets and licences?**
Letters: `hand_dataset_1000_24.csv` (MinorvaFalk/KNN_Alphabet), from ASL Alphabet by Akash Nagaraj, GPL-2.0, split 70/15/15, seed 42. Words: PopSign ASL v1.0 (Starner et al., NeurIPS 2023), CC BY 4.0, split by signer.

**Why do M and N get confused?**
As 21 points they are both closed fists with the thumb tucked.

**Does the video leave the laptop?**
No. MediaPipe and both networks run in the browser.

**How do you know every button works?**
27 Playwright tests drive every button, select and input with a fake webcam, and they run on GitHub before each deploy. If one fails, the site is not updated.

**What can it not do?**
Continuous ASL, facial grammar, and new signers whose hands differ a lot from the training data. Letters in Auto mode clash with WATER/MORE/FINE/NEED, so spelling uses Letters mode.

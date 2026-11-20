// Writes the accuracy reports shown on the Evaluation page. Run with: REPORT=1 npx vitest run src/test/reports.test.ts
import { existsSync, readFileSync, writeFileSync } from "node:fs"
import path from "node:path"
import { describe, it } from "vitest"
import type { Point } from "@/lib/hands"
import { WORD_MOTIONS } from "@/lib/handModel"
import { encodeSegment, predictAslWords, type AslWordsModel } from "@/lib/aslWords"
import { recognizeWord } from "@/lib/words"
import { resampleSequence } from "@/lib/normalize"
import { loadShippedModels } from "./models"
import { letterStream, parseLetterCsv, ruleConfusion, runAll, tally, wordStream } from "./harness"

const pub = path.resolve(__dirname, "../../public")
const CLIPS = process.env.REAL_CLIPS ?? "/workspace/popsign/test_clips.json"
const LETTERS = process.env.LETTER_CSV ?? "/workspace/popsign/letters.csv"

function pct(n: number, d: number) {
  return d ? Math.round((1000 * n) / d) / 10 : 0
}

describe.skipIf(!process.env.REPORT)("accuracy reports", () => {
  it("real PopSign test signers: trained model vs the hand-written rules", () => {
    if (!existsSync(CLIPS)) throw new Error(`missing ${CLIPS}`)
    const clips = JSON.parse(readFileSync(CLIPS, "utf8")) as Array<{ word: string; hand: string; frames: number[][][]; other?: number[] }>
    const model = JSON.parse(readFileSync(path.join(pub, "models/asl_words.json"), "utf8")) as AslWordsModel
    const per: Record<string, { n: number; model: number; rules: number; rulesNone: number }> = {}
    let modelOk = 0
    let rulesOk = 0
    let none = 0
    for (const clip of clips) {
      const frames = clip.frames.map((f) => f.map((p) => [p[0], p[1], p[2] ?? 0] as Point))
      const mirror = clip.hand === "Left"
      const reading = predictAslWords(model, encodeSegment(frames, mirror, clip.other ?? null, model.steps))
      // The rules expect about 30 fps; the clips are 5 fps, so fill in the frames between.
      const dense = resampleSequence(frames, Math.max(8, (frames.length - 1) * 6 + 1))
      const votes: Record<string, number> = {}
      for (let i = 6; i <= dense.length; i++) {
        const hit = recognizeWord(dense.slice(Math.max(0, i - 48), i))
        if (hit) votes[hit.id] = (votes[hit.id] ?? 0) + 1
      }
      const ruled = Object.entries(votes).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null
      const row = (per[clip.word] ??= { n: 0, model: 0, rules: 0, rulesNone: 0 })
      row.n += 1
      if (reading.label === clip.word) { row.model += 1; modelOk += 1 }
      if (ruled === clip.word) { row.rules += 1; rulesOk += 1 }
      if (!ruled) { row.rulesNone += 1; none += 1 }
    }
    const file = path.join(pub, "evaluation/asl_words.json")
    const metrics = JSON.parse(readFileSync(file, "utf8"))
    metrics.app_check = {
      what: "The same held-out test clips run through the app's own code (TypeScript model with the shipped rounded weights, and the hand-written rules).",
      clips: clips.length,
      model_accuracy: pct(modelOk, clips.length),
      rules_accuracy: pct(rulesOk, clips.length),
      rules_no_word: pct(none, clips.length),
      per_class: Object.fromEntries(Object.entries(per).sort().map(([w, r]) => [w, { clips: r.n, model: pct(r.model, r.n), rules: pct(r.rules, r.n), rules_no_word: pct(r.rulesNone, r.n) }])),
      note: "The rules were written for this app's simplified one-hand versions of the signs, while PopSign signers use real ASL forms (for example HELLO is a salute and FINE is a 5-hand at the chest), so a low rules score on these clips is expected.",
    }
    writeFileSync(file, JSON.stringify(metrics, null, 1))
    console.log(JSON.stringify({ model: metrics.app_check.model_accuracy, rules: metrics.app_check.rules_accuracy, none: metrics.app_check.rules_no_word }))
  }, 600000)

  it("synthetic per-sign harness through the full live pipeline", async () => {
    const { deps } = await loadShippedModels()
    const N = Number(process.env.N ?? 20)
    const words = Object.keys(WORD_MOTIONS)
    const wordStreams = words.flatMap((w) => Array.from({ length: N }, (_, k) => wordStream(w, k, 3)))
    const rows = parseLetterCsv(readFileSync(LETTERS, "utf8"), Math.floor(1000 / N), 17)
    const letterStreams = rows.map((r, k) => letterStream(r.label, r.points, k, 3))
    const sum = (t: ReturnType<typeof tally>) => {
      const v = Object.values(t)
      return pct(v.reduce((a, b) => a + b.exact, 0), v.reduce((a, b) => a + b.n, 0))
    }
    const shape = (t: ReturnType<typeof tally>) =>
      Object.fromEntries(Object.entries(t).map(([k, v]) => [k, { n: v.n, exactly_once: pct(v.exact, v.n), extra: v.extra, missed: v.missed, wrong: v.wrong }]))
    const w = tally(runAll(deps, wordStreams, "auto"))
    const la = tally(runAll(deps, letterStreams, "auto"))
    const ll = tally(runAll(deps, letterStreams, "letters"))
    const rules = ruleConfusion(N, 3)
    const report = {
      real_data: false,
      what: "Synthetic tests, not real signers. Words: a 3D hand model performs each one-hand word with random hand (left or right), size, position, rotation, speed, camera shape (4:3 or 16:9) and tracker jitter. Letters: real MediaPipe letter landmarks (MinorvaFalk ASL alphabet CSV, may overlap the letter model's training data) with random rotation, size, position, mirroring and jitter. Every sample goes through the same Recognizer, letter model and stabilizer as the live camera.",
      score: "A sample passes only when exactly one sign is added and it is the right one.",
      samples_per_sign: N,
      words_auto: { overall: sum(w), per_sign: shape(w) },
      letters_letters_mode: { overall: sum(ll), per_sign: shape(ll) },
      letters_auto_mode: { overall: sum(la), per_sign: shape(la), note: "In Auto, W, O, F and X read as the words WATER, MORE, FINE and NEED because they are the same handshape. Use Letters mode to spell with them." },
      rule_confusion: rules,
    }
    writeFileSync(path.join(pub, "evaluation/synthetic_signs.json"), JSON.stringify(report, null, 1))
    console.log(JSON.stringify({ words: report.words_auto.overall, letters: report.letters_letters_mode.overall, lettersAuto: report.letters_auto_mode.overall }))
  }, 1200000)
})

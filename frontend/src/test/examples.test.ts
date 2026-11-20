// The Learn page's "Play example" must be graded correct for every sign.
import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"
import { squareFrame, type Point } from "@/lib/hands"
import { performTwoHands } from "@/lib/handModel"
import { Recognizer } from "@/lib/recognizer"
import { STILL_LETTERS, exampleRow, letterFigure } from "@/lib/signLibrary"
import { DEMO_ASPECT, TWO_HAND_WORDS, WORD_SIGNS, demoSquare } from "@/lib/words"
import { loadShippedModels } from "./models"

const samples = JSON.parse(readFileSync(path.resolve(__dirname, "../../public/samples/static_samples.json"), "utf8")).samples as Record<string, number[][]>

describe("built-in examples", () => {
  it("every word and still letter example is recognised as itself, once", async () => {
    const { deps } = await loadShippedModels()
    const bad: string[] = []
    for (const id of [...WORD_SIGNS.map((s) => s.id), ...STILL_LETTERS]) {
      const e = new Recognizer(deps)
      const commits: string[] = []
      const feed = (f: Point[] | null, mirror: boolean, second?: Point[] | null, letter = false) => {
        const r = e.push(f, { mode: letter ? "letters" : "auto", mirror, second })
        if (r.commit) commits.push(r.commit.label)
      }
      if (/^[A-Z]$/.test(id)) {
        const fr = letterFigure(exampleRow(samples[id])!).frames[0]
        for (let i = 0; i < 30; i++) feed(fr, false, null, true)
      } else if (TWO_HAND_WORDS.has(id)) {
        const { main, second } = performTwoHands(id as "HELP" | "SCHOOL", { rand: () => 0.5, jitter: 0, aspect: DEMO_ASPECT })
        for (let i = 0; i < main.length + 10; i++) {
          const k = Math.min(i, main.length - 1)
          feed(squareFrame(main[k], DEMO_ASPECT), true, squareFrame(second[k], DEMO_ASPECT))
        }
      } else for (const f of demoSquare(id)) feed(f, true)
      for (let i = 0; i < 12; i++) feed(null, false)
      if (commits.join() !== id) bad.push(`${id}:${commits.join("+")}`)
    }
    expect(bad).toEqual([])
  }, 120000)
})

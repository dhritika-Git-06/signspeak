import { describe, expect, it } from "vitest"
import { squareFrame, type Point } from "@/lib/hands"
import { WORD_MOTIONS, performMotion, performTwoHands, seededRandom } from "@/lib/handModel"
import { Recognizer } from "@/lib/recognizer"
import { recognizeTwoHands } from "@/lib/words"

const deps = { letter: () => ({ label: "A", confidence: 0.3, top: [] }), dynamic: () => null, motion: () => 0, motionThreshold: 1 }

function run(main: Point[][], second: Array<Point[] | null>) {
  const engine = new Recognizer(deps)
  const commits: string[] = []
  for (let i = 0; i < main.length; i++) {
    const r = engine.push(main[i], { mode: "auto", mirror: false, second: second[i] })
    if (r.commit) commits.push(r.commit.label)
  }
  for (let i = 0; i < 12; i++) engine.push(null, { mode: "auto", mirror: false })
  return commits
}

describe("two-hand words", () => {
  for (const id of ["HELP", "SCHOOL"] as const) {
    it(`${id} is added once, left- or right-handed, at any size`, () => {
      let ok = 0
      const N = 24
      for (let k = 0; k < N; k++) {
        const aspect = k % 3 ? 16 / 9 : 4 / 3
        const { main, second } = performTwoHands(id, { rand: seededRandom(k * 31 + id.length), hand: k % 2 ? "Left" : "Right", aspect })
        const hold = 10
        const m = [...main, ...Array(hold).fill(main.at(-1))].map((f) => squareFrame(f, aspect))
        const s = [...second, ...Array(hold).fill(second.at(-1))].map((f) => squareFrame(f, aspect))
        const commits = run(m, s)
        if (commits.length === 1 && commits[0] === id) ok += 1
        else if (process.env.DBG) console.log(id, k, commits)
      }
      expect(ok / N).toBeGreaterThan(0.8)
    })
  }

  it("one-hand words are not mistaken for two-hand words", () => {
    for (const word of ["HELLO", "WANT", "WATER", "YES"]) {
      const frames = performMotion(WORD_MOTIONS[word], { rand: seededRandom(3), aspect: 16 / 9 }).map((f) => squareFrame(f, 16 / 9))
      expect(recognizeTwoHands(frames, frames.map(() => null))).toBeNull()
    }
  })
})

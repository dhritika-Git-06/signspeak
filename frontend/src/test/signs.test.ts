// Quick per-sign accuracy checks on synthetic performances, run in CI. The full report is reports.test.ts.
import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"
import type { Point } from "@/lib/hands"
import { WORD_MOTIONS } from "@/lib/handModel"
import { PRESENTATION_SCRIPT } from "@/lib/words"
import { loadShippedModels } from "./models"
import { letterStream, runAll, tally, wordStream } from "./harness"

const samples = JSON.parse(readFileSync(path.resolve(__dirname, "../../public/samples/static_samples.json"), "utf8")) as {
  samples: Record<string, number[][]>
}

function overall(t: ReturnType<typeof tally>) {
  const v = Object.values(t)
  return v.reduce((a, b) => a + b.exact, 0) / v.reduce((a, b) => a + b.n, 0)
}

describe("per-sign accuracy on synthetic performances", () => {
  it("each one-hand word is added exactly once, left or right hand, at any size and angle", async () => {
    const { deps } = await loadShippedModels()
    const streams = Object.keys(WORD_MOTIONS).flatMap((w) => Array.from({ length: 4 }, (_, k) => wordStream(w, k, 11)))
    const t = tally(runAll(deps, streams, "auto"))
    expect(overall(t)).toBeGreaterThan(0.85)
    for (const id of PRESENTATION_SCRIPT) expect(t[id].exact, id).toBe(t[id].n)
  }, 120000)

  it("still letters from real landmark rows are spelled once each in Letters mode", async () => {
    const { deps } = await loadShippedModels()
    const streams = Object.entries(samples.samples).flatMap(([label, rows]) =>
      rows.slice(0, 3).map((row, k) => letterStream(label, Array.from({ length: 21 }, (_, j) => [row[2 * j], row[2 * j + 1], 0] as Point), k, 5)),
    )
    const t = tally(runAll(deps, streams, "letters"))
    expect(overall(t)).toBeGreaterThan(0.85)
  }, 120000)
})

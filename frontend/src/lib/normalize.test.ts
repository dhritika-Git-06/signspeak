import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"
import { normalizeLandmarks } from "./normalize"

interface Case {
  raw: number[][]
  mirror: boolean
  zero_z: boolean
  normalized: number[][]
}

const cases = JSON.parse(
  readFileSync(path.resolve(__dirname, "normalize_cases.json"), "utf8"),
) as Case[]

describe("normalizeLandmarks", () => {
  it("matches the Python preprocessing cases", () => {
    for (const item of cases) {
      const got = normalizeLandmarks(item.raw as [number, number, number][], {
        mirror: item.mirror,
        zeroZ: item.zero_z,
      })
      item.normalized.forEach((row, i) => {
        expect(got[i][0]).toBeCloseTo(row[0], 4)
        expect(got[i][1]).toBeCloseTo(row[1], 4)
        expect(got[i][2]).toBeCloseTo(row[2], 4)
      })
    }
  })

  it("places the wrist at the origin", () => {
    const raw = Array.from({ length: 21 }, () => [0.2, 0.4, 0.1] as [number, number, number])
    raw[9] = [0.2, 0.8, 0.1]
    const out = normalizeLandmarks(raw)
    expect(out[0][0]).toBeCloseTo(0, 5)
    expect(out[0][1]).toBeCloseTo(0, 5)
  })
})

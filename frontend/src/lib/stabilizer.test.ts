import { describe, expect, it } from "vitest"
import { Stabilizer } from "./stabilizer"

describe("Stabilizer", () => {
  it("commits a letter only after a confident majority", () => {
    const stabilizer = new Stabilizer({ window: 4, majority: 0.75, minConfidence: 0.7, releaseFrames: 2 })
    expect(stabilizer.push("A", 0.9)).toBeNull()
    expect(stabilizer.push("A", 0.91)).toBeNull()
    expect(stabilizer.push("B", 0.8)).toBeNull()
    expect(stabilizer.push("A", 0.95)).toBe("A")
    expect(stabilizer.push("A", 0.99)).toBeNull()
  })

  it("lets the same letter be signed again after the hand drops", () => {
    const stabilizer = new Stabilizer({ window: 2, majority: 1, minConfidence: 0.6, releaseFrames: 2 })
    expect(stabilizer.push("L", 0.9)).toBeNull()
    expect(stabilizer.push("L", 0.9)).toBe("L")
    expect(stabilizer.push(null, 0)).toBeNull()
    expect(stabilizer.push(null, 0)).toBeNull()
    expect(stabilizer.push("L", 0.9)).toBeNull()
    expect(stabilizer.push("L", 0.9)).toBe("L")
  })

  it("ignores low confidence frames", () => {
    const stabilizer = new Stabilizer({ window: 2, majority: 1, minConfidence: 0.8 })
    expect(stabilizer.push("A", 0.4)).toBeNull()
    expect(stabilizer.push("A", 0.5)).toBeNull()
  })
})
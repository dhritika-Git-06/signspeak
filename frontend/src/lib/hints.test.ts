import { describe, expect, it } from "vitest"
import type { Point } from "./hands"
import { framingHint } from "./hints"

function box(cx: number, cy: number, size: number): Point[] {
  return Array.from({ length: 21 }, (_, i) => [cx + ((i % 5) / 4 - 0.5) * size, cy + (Math.floor(i / 5) / 4 - 0.5) * size, 0] as Point)
}

describe("framing hints", () => {
  it("says too far, too close, out of frame, too dark, and good", () => {
    expect(framingHint(box(0.5, 0.5, 0.1), 120, 0)?.text).toMatch(/Too far/)
    expect(framingHint(box(0.5, 0.5, 0.9), 120, 0)?.text).toMatch(/Too close|inside the frame/)
    expect(framingHint(box(0.95, 0.5, 0.3), 120, 0)?.text).toMatch(/inside the frame/)
    expect(framingHint(box(0.5, 0.5, 0.35), 30, 0)?.text).toMatch(/Too dark/)
    expect(framingHint(box(0.5, 0.5, 0.35), 120, 0)?.tone).toBe("ok")
    expect(framingHint(null, 120, 100)?.text).toMatch(/No hand/)
    expect(framingHint(null, 120, 5)).toBeNull()
  })
})

import type { Point } from "./hands"

const WRIST = 0
const MIDDLE_MCP = 9

export function normalizeLandmarks(
  raw: Point[],
  opts?: { mirror?: boolean; zeroZ?: boolean },
): Point[] {
  const pts: Point[] = raw.map((p) => [p[0], p[1], p[2] ?? 0])
  if (opts?.mirror) {
    for (const p of pts) p[0] = -p[0]
  }
  if (opts?.zeroZ) {
    for (const p of pts) p[2] = 0
  }
  const wrist = pts[WRIST]
  const mid = pts[MIDDLE_MCP]
  const scale = Math.hypot(mid[0] - wrist[0], mid[1] - wrist[1]) || 1e-6
  return pts.map((p) => [
    (p[0] - wrist[0]) / scale,
    (p[1] - wrist[1]) / scale,
    (p[2] - wrist[2]) / scale,
  ])
}

export function resampleSequence(seq: Point[][], length: number): Point[][] {
  const count = seq.length
  if (count === length) return seq
  if (count === 1) return Array.from({ length }, () => seq[0])
  const out: Point[][] = []
  for (let i = 0; i < length; i++) {
    const t = (i * (count - 1)) / (length - 1)
    const lo = Math.floor(t)
    const hi = Math.min(lo + 1, count - 1)
    const a = t - lo
    const frame: Point[] = []
    for (let j = 0; j < 21; j++) {
      const p = seq[lo][j]
      const q = seq[hi][j]
      frame.push([
        (1 - a) * p[0] + a * q[0],
        (1 - a) * p[1] + a * q[1],
        (1 - a) * p[2] + a * q[2],
      ])
    }
    out.push(frame)
  }
  return out
}

export function normalizeSequence(
  frames: Point[][],
  length = 30,
  opts?: { mirror?: boolean; zeroZ?: boolean },
): Point[][] {
  if (!frames.length) throw new Error("Sequence is empty")
  const pts = frames.map((frame) => frame.map((p) => [p[0], p[1], p[2] ?? 0] as Point))
  if (opts?.mirror) {
    for (const frame of pts) for (const p of frame) p[0] = -p[0]
  }
  if (opts?.zeroZ) {
    for (const frame of pts) for (const p of frame) p[2] = 0
  }
  const scales = pts.map((frame) => {
    const w = frame[WRIST]
    const m = frame[MIDDLE_MCP]
    return Math.hypot(m[0] - w[0], m[1] - w[1]) || 1e-6
  })
  const sorted = [...scales].sort((a, b) => a - b)
  const scale = sorted[Math.floor(sorted.length / 2)]
  const origin = pts[0][WRIST]
  const seq = pts.map((frame) =>
    frame.map(
      (p) =>
        [(p[0] - origin[0]) / scale, (p[1] - origin[1]) / scale, (p[2] - origin[2]) / scale] as Point,
    ),
  )
  return resampleSequence(seq, length)
}

export function flattenSequence(seq: Point[][]): number[][] {
  return seq.map((frame) => frame.flat())
}

export function motionEnergy(seq: Point[][]): number {
  if (seq.length < 2) return 0
  let total = 0
  let n = 0
  for (let t = 1; t < seq.length; t++) {
    for (let j = 0; j < seq[t].length; j++) {
      const a = seq[t][j]
      const b = seq[t - 1][j]
      total += Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
      n += 1
    }
  }
  return n ? total / n : 0
}

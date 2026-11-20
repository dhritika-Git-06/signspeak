import type { Point } from "./hands"
import type { Ranked, Reading } from "./recognizer"

/**
 * A small word classifier trained on real signers from PopSign ASL v1.0 (CC BY 4.0), tested on
 * signers it never saw (training/popsign_words.py). Plain JSON weights, run here without TF.js.
 */
export interface AslWordsModel {
  labels: string[]
  steps: number
  mean: number[]
  std: number[]
  layers: Array<{ W: number[][]; b: number[] }>
  activation: "relu" | "tanh" | "logistic"
}

/** Same features as the Python training code: dominant hand, `steps` resampled frames, palm units. */
export function encodeSegment(frames: Point[][], mirror: boolean, other: number[] | null, steps: number): number[] {
  const seq = frames.map((frame) => frame.map((p) => [mirror ? -p[0] : p[0], p[1]] as [number, number]))
  const palms = seq.map((f) => Math.hypot(f[9][0] - f[0][0], f[9][1] - f[0][1])).sort((a, b) => a - b)
  const scale = palms[Math.floor(palms.length / 2)] || 1e-3
  const n = seq.length
  const res: Array<Array<[number, number]>> = []
  const oth: number[] = []
  for (let i = 0; i < steps; i++) {
    const t = steps === 1 ? 0 : (i * (n - 1)) / (steps - 1)
    const lo = Math.floor(t)
    const hi = Math.min(lo + 1, n - 1)
    const a = t - lo
    res.push(seq[lo].map((p, j) => [p[0] * (1 - a) + seq[hi][j][0] * a, p[1] * (1 - a) + seq[hi][j][1] * a]))
    const o = other ?? []
    oth.push(o.length ? (o[lo] ?? 0) * (1 - a) + (o[hi] ?? 0) * a : 0)
  }
  const shape: number[] = []
  for (const f of res) for (const p of f) shape.push((p[0] - f[0][0]) / scale, (p[1] - f[0][1]) / scale)
  const path: number[] = []
  for (const f of res) path.push((f[0][0] - res[0][0][0]) / scale, (f[0][1] - res[0][0][1]) / scale)
  return [...shape, ...path, ...oth]
}

export function predictAslWords(model: AslWordsModel, features: number[]): Reading {
  let x = features.map((v, i) => (v - model.mean[i]) / model.std[i])
  model.layers.forEach((layer, k) => {
    const out = layer.b.slice()
    for (let i = 0; i < x.length; i++) {
      const xi = x[i]
      if (xi === 0) continue
      const row = layer.W[i]
      for (let j = 0; j < out.length; j++) out[j] += xi * row[j]
    }
    const last = k === model.layers.length - 1
    x = last ? out : out.map((v) => (model.activation === "relu" ? Math.max(0, v) : model.activation === "tanh" ? Math.tanh(v) : 1 / (1 + Math.exp(-v))))
  })
  const max = Math.max(...x)
  const exp = x.map((v) => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  const top: Ranked[] = exp.map((v, i) => ({ label: model.labels[i], confidence: v / sum })).sort((a, b) => b.confidence - a.confidence)
  return { label: top[0].label, confidence: top[0].confidence, top: top.slice(0, 3) }
}

/**
 * Live use: the model has no "nothing" class, so it only reads a finished movement. A segment
 * starts when the hand starts moving and ends when the hand holds still or leaves the view.
 * The segment is resampled to the model's fixed number of steps, so its frame rate does not matter.
 */
export class SegmentReader {
  private frames: Point[][] = []
  private moved = 0
  private still = 0
  private missing = 0

  reset() {
    this.frames = []
    this.moved = 0
    this.still = 0
    this.missing = 0
  }

  /** Push one frame and whether the hand is moving; returns the frames of a finished movement, if any. */
  push(frame: Point[] | null, moving: boolean): Point[][] | null {
    if (!frame) {
      this.missing += 1
      if (this.missing >= 6) return this.finish()
      return null
    }
    this.missing = 0
    this.frames.push(frame)
    if (this.frames.length > 90) this.frames.shift()
    if (moving) {
      this.moved += 1
      this.still = 0
    } else if (this.moved) {
      this.still += 1
      if (this.still >= 10) return this.finish()
    } else {
      // Keep a short lead-in from before the movement started.
      this.frames = this.frames.slice(-6)
    }
    return null
  }

  private finish(): Point[][] | null {
    const done = this.moved >= 6 && this.frames.length >= 4 ? this.frames : null
    this.reset()
    return done
  }
}

let cached: Promise<AslWordsModel> | null = null

export function loadAslWords(url: string): Promise<AslWordsModel> {
  cached ??= fetch(url).then((r) => {
    if (!r.ok) throw new Error(`Could not load ${url}`)
    return r.json() as Promise<AslWordsModel>
  })
  return cached
}

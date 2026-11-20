export interface StabilizerOptions {
  window?: number
  majority?: number
  minConfidence?: number
  releaseFrames?: number
  /** Frames to ignore after a commit unless the hand is released first. */
  cooldownFrames?: number
}

export class Stabilizer {
  private buf: Array<{ label: string; confidence: number }> = []
  private held: string | null = null
  private gap = 0
  private readonly window: number
  private readonly majority: number
  private minConfidence: number
  private readonly releaseFrames: number
  private readonly cooldownFrames: number
  private cooldown = 0

  constructor(opts: StabilizerOptions = {}) {
    this.window = opts.window ?? 10
    this.majority = opts.majority ?? 0.7
    this.minConfidence = opts.minConfidence ?? 0.65
    this.releaseFrames = opts.releaseFrames ?? 8
    this.cooldownFrames = opts.cooldownFrames ?? 0
  }

  reset() {
    this.buf = []
    this.held = null
    this.gap = 0
    this.cooldown = 0
  }

  setMinConfidence(value: number) {
    this.minConfidence = value
  }

  push(label: string | null, confidence: number): string | null {
    if (!label || confidence < this.minConfidence) {
      this.gap += 1
      this.buf = []
      if (this.gap >= this.releaseFrames) {
        this.held = null
        this.cooldown = 0
      }
      return null
    }
    this.gap = 0
    if (this.cooldown > 0) {
      this.cooldown -= 1
      this.buf = []
      return null
    }
    this.buf.push({ label, confidence })
    if (this.buf.length > this.window) this.buf.shift()
    if (this.buf.length < this.window) return null
    const counts = new Map<string, { n: number; conf: number }>()
    for (const item of this.buf) {
      const row = counts.get(item.label) ?? { n: 0, conf: 0 }
      row.n += 1
      row.conf += item.confidence
      counts.set(item.label, row)
    }
    let best = ""
    let bestN = 0
    let bestConf = 0
    for (const [name, row] of counts) {
      if (row.n > bestN) {
        best = name
        bestN = row.n
        bestConf = row.conf / row.n
      }
    }
    if (bestN / this.buf.length < this.majority || bestConf < this.minConfidence) return null
    if (best === this.held) return null
    this.held = best
    this.buf = []
    this.cooldown = this.cooldownFrames
    return best
  }
}

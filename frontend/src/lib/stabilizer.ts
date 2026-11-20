export interface StabilizerOptions {
  /** Frames that must agree before a word commits. */
  window?: number
  /** Frames that must agree before a single letter commits. Letters wait longer so a hand passing through a shape does not spell it. */
  letterWindow?: number
  majority?: number
  minConfidence?: number
  /** Frames with no hand before the held sign is released and can be signed again. */
  releaseFrames?: number
  /** Frames to ignore after a commit unless the hand is released first. */
  cooldownFrames?: number
}

/** A vote that says "a hand is here but the reading is not trustworthy". It never commits and never releases the held sign. */
export const UNSURE = "\u0000unsure"

export function isLetterLabel(label: string): boolean {
  return /^[A-Z]$/.test(label)
}

export class Stabilizer {
  private buf: Array<{ label: string; confidence: number }> = []
  private held: string | null = null
  private gap = 0
  private readonly window: number
  private readonly letterWindow: number
  private readonly majority: number
  private minConfidence: number
  private readonly releaseFrames: number
  private readonly cooldownFrames: number
  private cooldown = 0

  constructor(opts: StabilizerOptions = {}) {
    this.window = opts.window ?? 10
    this.letterWindow = Math.max(this.window, opts.letterWindow ?? this.window)
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

  /** The sign that was committed last and is still held (not yet released by a hand drop). */
  get current(): string | null {
    return this.held
  }

  /**
   * Push one frame. `label = null` means no hand in view. A low-confidence label counts as an
   * unsure vote: it blocks a commit but does not release the held sign, so a shaky hold cannot
   * add the same word twice.
   */
  push(label: string | null, confidence: number): string | null {
    if (!label) {
      this.gap += 1
      // One or two lost tracker frames are normal; a longer gap starts the count again.
      if (this.gap >= 3) this.buf = []
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
    const vote = confidence >= this.minConfidence ? label : UNSURE
    this.buf.push({ label: vote, confidence })
    if (this.buf.length > this.letterWindow) this.buf.shift()
    if (this.buf.length < this.window) return null
    const best = winner(this.buf.slice(-this.window))
    if (!best || best.label === UNSURE) return null
    if (best.share < this.majority || best.conf < this.minConfidence) return null
    if (isLetterLabel(best.label) && this.letterWindow > this.window) {
      if (this.buf.length < this.letterWindow) return null
      const longer = winner(this.buf)
      if (!longer || longer.label !== best.label || longer.share < this.majority) return null
    }
    if (best.label === this.held) return null
    this.held = best.label
    this.buf = []
    this.cooldown = this.cooldownFrames
    return best.label
  }
}

function winner(items: Array<{ label: string; confidence: number }>) {
  const counts = new Map<string, { n: number; conf: number }>()
  for (const item of items) {
    const row = counts.get(item.label) ?? { n: 0, conf: 0 }
    row.n += 1
    row.conf += item.confidence
    counts.set(item.label, row)
  }
  let best: { label: string; share: number; conf: number } | null = null
  let bestN = 0
  for (const [label, row] of counts) {
    if (row.n > bestN) {
      bestN = row.n
      best = { label, share: row.n / items.length, conf: row.conf / row.n }
    }
  }
  return best
}

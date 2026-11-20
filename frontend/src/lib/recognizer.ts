import type { Point } from "./hands"
import { Stabilizer, UNSURE, isLetterLabel, type StabilizerOptions } from "./stabilizer"
import { recognizeWord, shapeSignature } from "./words"

export type SignKind = "letter" | "word"
export type RecognitionMode = "auto" | "letters" | "words"

export interface Ranked {
  label: string
  confidence: number
}

export interface Reading {
  label: string
  confidence: number
  top: Ranked[]
}

export interface Prediction extends Reading {
  kind: SignKind
}

export interface RecognizerDeps {
  /** Trained still-letter model on one frame. */
  letter: (frame: Point[], mirror: boolean) => Reading
  /** Trained motion model on the frame buffer (J, Z and a few motion words). */
  dynamic: (frames: Point[][], mirror: boolean, sampleKind?: SignKind) => Reading | null
  /** Motion energy of the frame buffer. */
  motion: (frames: Point[][], mirror: boolean) => number
  motionThreshold: number
}

export interface FrameOptions {
  mode: RecognitionMode
  mirror: boolean
  sampleKind?: SignKind
}

export interface FrameResult {
  live: Prediction | null
  commit: { label: string; kind: SignKind } | null
}

/**
 * Tuned for about 30 fps. A word needs roughly half a second of agreement, a letter roughly
 * three quarters of a second, and after any commit there is a short pause so the move to the
 * next sign is not read as a sign. Dropping the hand out of view ends the pause at once.
 */
export const LIVE_STABILIZER: StabilizerOptions = {
  window: 14,
  letterWindow: 20,
  majority: 0.7,
  minConfidence: 0.62,
  releaseFrames: 10,
  cooldownFrames: 20,
}

/** Frames with a different handshape before a committed word stops covering letter readings. */
const STICKY_RELEASE = 8
/** Confidence the motion model needs before J or Z is accepted in auto mode. */
const MOTION_LETTER_CONFIDENCE = 0.9

export function kindOf(label: string): SignKind {
  return isLetterLabel(label) ? "letter" : "word"
}

/**
 * The live pipeline, kept free of React and TF.js so it can be tested with recorded or
 * simulated frame streams: frame buffer, letter / motion / word-rule choice, and the stabilizer.
 */
export class Recognizer {
  readonly stabilizer: Stabilizer
  private buffer: Point[][] = []
  private wordVotes = 0
  private missing = 0
  private readonly releaseFrames: number
  private sticky: { label: string; sig: string; misses: number } | null = null
  readonly deps: RecognizerDeps

  constructor(deps: RecognizerDeps, opts: StabilizerOptions = LIVE_STABILIZER) {
    this.deps = deps
    this.stabilizer = new Stabilizer(opts)
    this.releaseFrames = opts.releaseFrames ?? 8
  }

  reset() {
    this.buffer = []
    this.wordVotes = 0
    this.missing = 0
    this.sticky = null
    this.stabilizer.reset()
  }

  setThreshold(value: number) {
    this.stabilizer.setMinConfidence(value)
  }

  push(frame: Point[] | null, options: FrameOptions): FrameResult {
    const { mode, mirror, sampleKind } = options
    if (!frame) {
      // A single lost frame is common while the hand moves; only a real drop clears the context.
      this.missing += 1
      if (this.missing >= this.releaseFrames) {
        this.buffer = []
        this.wordVotes = 0
        this.sticky = null
      }
      this.stabilizer.push(null, 0)
      return { live: null, commit: null }
    }
    this.missing = 0
    this.buffer.push(frame)
    if (this.buffer.length > 48) this.buffer.shift()

    let motion = 0
    if (this.buffer.length >= 8 && sampleKind !== "word") {
      try {
        motion = this.deps.motion(this.buffer, mirror)
      } catch {
        motion = 0
      }
    }
    if (sampleKind === "word") this.wordVotes = 12
    else if (motion > this.deps.motionThreshold) this.wordVotes = Math.min(12, this.wordVotes + 1)
    else this.wordVotes = Math.max(0, this.wordVotes - 1)

    const useMotionModel =
      sampleKind === "word" ||
      mode === "words" ||
      (mode === "auto" && sampleKind !== "letter" && this.wordVotes >= 6 && this.buffer.length >= 12)

    let prediction: Prediction | null = null
    let unsure = false
    if (useMotionModel && this.buffer.length >= 8) {
      const reading = this.deps.dynamic(this.buffer, mirror, sampleKind)
      if (reading) {
        prediction = { ...reading, kind: kindOf(reading.label) }
        // In auto mode the hand is usually just moving between signs. Only a confident J or Z
        // from the motion model is trusted there; motion words come from the rules below.
        if (mode === "auto" && !sampleKind && (prediction.kind === "word" || reading.confidence < MOTION_LETTER_CONFIDENCE)) {
          unsure = true
        }
      }
    }
    if (!prediction) {
      prediction = { ...this.deps.letter(frame, mirror), kind: "letter" }
    }
    if (mode !== "letters") {
      const hit = recognizeWord(this.buffer)
      if (hit) {
        prediction = { label: hit.id, confidence: hit.score, top: [{ label: hit.id, confidence: hit.score }], kind: "word" }
        unsure = false
      }
    }

    // While the hand keeps the shape of the word just added (for example the open hand left over
    // after WANT, or W while holding WATER), letter readings belong to that word, not to spelling.
    const sig = shapeSignature(frame)
    if (this.sticky) {
      if (sig === this.sticky.sig) this.sticky.misses = 0
      else if (++this.sticky.misses >= STICKY_RELEASE) this.sticky = null
    }
    let vote = unsure ? UNSURE : prediction.label
    if (this.sticky && prediction.kind === "letter" && sig === this.sticky.sig) {
      vote = this.sticky.label
      prediction = { ...prediction, label: this.sticky.label, kind: "word" }
      unsure = false
    }

    const stable = this.stabilizer.push(vote, prediction.confidence)
    if (!stable) return { live: prediction, commit: null }
    const kind = kindOf(stable)
    this.sticky = kind === "word" ? { label: stable, sig, misses: 0 } : null
    return { live: prediction, commit: { label: stable, kind } }
  }

  /** Feed a ready-made reading (recorded motion samples) through the same stabilizer. */
  pushReading(reading: Reading): FrameResult {
    const prediction: Prediction = { ...reading, kind: kindOf(reading.label) }
    const stable = this.stabilizer.push(reading.label, reading.confidence)
    return { live: prediction, commit: stable ? { label: stable, kind: kindOf(stable) } : null }
  }
}

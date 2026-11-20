import type { Point } from "./hands"
import { Stabilizer, UNSURE, isLetterLabel, type StabilizerOptions } from "./stabilizer"
import { STATIC_WORDS, handsTogether, palmSize, recognizeTwoHands, recognizeWord, shapeSignature } from "./words"

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
  /** The second hand, when two are in view (square units, like the main hand). */
  second?: Point[] | null
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
const MOTION_LETTERS = new Set(["J", "Z"])

/**
 * How much the hand is moving right now: wrist travel over the last few frames in palm lengths
 * (smoothed so tracker jitter does not count), plus any push or pull. Letters and still word
 * shapes are only read while the hand is held still, so the start of a wave or a push is not
 * spelled as a letter first.
 */
export function recentMotion(frames: Point[][], span = 8): number {
  if (frames.length < span + 2) return 0
  const recent = frames.slice(-(span + 2))
  const unit = recent.map(palmSize).sort((a, b) => a - b)[Math.floor(recent.length / 2)] || 1e-3
  const smooth: Array<[number, number]> = []
  for (let i = 1; i < recent.length - 1; i++) {
    smooth.push([
      (recent[i - 1][0][0] + recent[i][0][0] + recent[i + 1][0][0]) / 3,
      (recent[i - 1][0][1] + recent[i][0][1] + recent[i + 1][0][1]) / 3,
    ])
  }
  let path = 0
  for (let i = 1; i < smooth.length; i++) path += Math.hypot(smooth[i][0] - smooth[i - 1][0], smooth[i][1] - smooth[i - 1][1])
  const grow = Math.abs(Math.log(palmSize(recent[recent.length - 1]) / (palmSize(recent[0]) || 1e-6)))
  return path / unit + grow * 4
}

/** Straight-line wrist travel over the last `span` frames, in palm lengths: slow steady moves show up here. */
export function recentDrift(frames: Point[][], span = 15): number {
  if (frames.length < span) return 0
  const a = frames[frames.length - span][0]
  const b = frames[frames.length - 1][0]
  const unit = palmSize(frames[frames.length - 1]) || 1e-3
  return Math.hypot(b[0] - a[0], b[1] - a[1]) / unit
}

/** Held signs may drift this far (palm lengths) in half a second. */
export const DRIFT = 0.25

/** Above this the hand counts as moving (about half a palm of travel in a quarter second). */
export const MOVING = 0.5

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
  private secondBuffer: Array<Point[] | null> = []
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
    this.secondBuffer = []
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
        this.secondBuffer = []
        this.wordVotes = 0
        this.sticky = null
      }
      this.stabilizer.push(null, 0)
      return { live: null, commit: null }
    }
    this.missing = 0
    this.buffer.push(frame)
    this.secondBuffer.push(options.second ?? null)
    if (this.buffer.length > 48) this.buffer.shift()
    if (this.secondBuffer.length > 48) this.secondBuffer.shift()

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
    const moving = !sampleKind && (recentMotion(this.buffer) > MOVING || recentDrift(this.buffer) > DRIFT)
    // A still letter read from a moving hand is a hand on its way somewhere, not a letter.
    if (moving && prediction.kind === "letter" && !MOTION_LETTERS.has(prediction.label)) unsure = true
    if (mode !== "letters") {
      const pair = recognizeTwoHands(this.buffer, this.secondBuffer)
      const hit = pair ?? (handsTogether(this.buffer, this.secondBuffer) ? null : recognizeWord(this.buffer))
      if (hit && !(moving && STATIC_WORDS.has(hit.id))) {
        prediction = { label: hit.id, confidence: hit.score, top: [{ label: hit.id, confidence: hit.score }], kind: "word" }
        unsure = false
      } else if (hit) {
        unsure = true
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

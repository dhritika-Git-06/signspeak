import { squareFrame, type Point } from "./hands"
import { WORD_MOTIONS, performMotion } from "./handModel"
import { normalizeLandmarks } from "./normalize"

/**
 * Word signs in the live demo are geometric rules on the 21 MediaPipe hand
 * landmarks. They are not a trained classifier and they do not have a dataset
 * accuracy. Each rule is the handshape and motion described below.
 */
export interface WordSign {
  id: string
  title: string
  method: "rule"
  show: string
  detail: string
}

export const WORD_SIGNS: WordSign[] = [
  { id: "HELLO", title: "Hello", method: "rule", show: "Open hand, wave side to side", detail: "Four fingers extended and the wrist crosses left and right at least twice." },
  { id: "YES", title: "Yes", method: "rule", show: "Fist nod", detail: "Closed fist whose wrist moves up and down at least twice." },
  { id: "NO", title: "No", method: "rule", show: "Two fingers wag", detail: "Index and middle extended, ring and pinky closed, fingertips wag sideways." },
  { id: "THANKS", title: "Thanks", method: "rule", show: "Flat hand moves down", detail: "Open hand travels downward without a wave." },
  { id: "PLEASE", title: "Please", method: "rule", show: "Open hand circle", detail: "Open hand whose wrist traces a circle." },
  { id: "SORRY", title: "Sorry", method: "rule", show: "Fist circle", detail: "Closed fist whose wrist traces a circle." },
  { id: "ME", title: "I / me", method: "rule", show: "Index points down", detail: "Only the index finger is out, and it points downward." },
  { id: "YOU", title: "You", method: "rule", show: "Index points up", detail: "Only the index finger is out, and it points upward." },
  { id: "WE", title: "We", method: "rule", show: "Index sweeps across", detail: "Index finger stays out and the hand travels sideways once." },
  { id: "LOVE", title: "Love", method: "rule", show: "I-love-you handshape", detail: "Thumb, index and pinky extended; middle and ring closed." },
  { id: "STOP", title: "Stop", method: "rule", show: "Open palm pushes out", detail: "Open hand pushes toward the camera, so it grows on screen." },
  { id: "WATER", title: "Water", method: "rule", show: "W handshape", detail: "Index, middle and ring extended and held together; pinky closed." },
  { id: "MORE", title: "More", method: "rule", show: "Fingertips together", detail: "Thumb and fingertips meet in a pinch, with little motion." },
  { id: "DRINK", title: "Drink", method: "rule", show: "C hand lifts", detail: "Curved C hand moves upward." },
  { id: "EAT", title: "Eat", method: "rule", show: "Pinch lifts", detail: "A pinch moves upward, as toward the mouth." },
  { id: "GO", title: "Go", method: "rule", show: "Index pushes forward", detail: "Only the index finger is out and the hand pushes toward the camera." },
  { id: "COME", title: "Come", method: "rule", show: "Hand pulls back", detail: "A bent or partly open hand pulls back toward you, so it shrinks on screen." },
  { id: "GOOD", title: "Good", method: "rule", show: "Thumb up", detail: "Thumb extended upward, other fingers closed. This is the thumbs-up gesture, not the chin-touch ASL GOOD." },
  { id: "BAD", title: "Bad", method: "rule", show: "Fingers flip down", detail: "Extended fingers rotate so the tips travel downward." },
  { id: "LIKE", title: "Like", method: "rule", show: "Open hand closes", detail: "The hand starts open and ends in a pinch." },
  { id: "WANT", title: "Want", method: "rule", show: "Open hand pulls in", detail: "Open hand pulls back toward you, so it shrinks on screen." },
  { id: "NEED", title: "Need", method: "rule", show: "Bent index", detail: "Index finger is hooked; the other fingers are closed." },
  { id: "WHERE", title: "Where", method: "rule", show: "Index wiggles", detail: "Only the index finger is out, and it wags sideways." },
  { id: "WHAT", title: "What", method: "rule", show: "Open hand shakes", detail: "Open hand shakes sideways with a smaller motion than HELLO." },
  { id: "NAME", title: "Name", method: "rule", show: "Two fingers tap", detail: "Index and middle extended together, hand taps up and down. One-hand stand-in for the two-hand ASL NAME." },
  { id: "FINE", title: "Fine", method: "rule", show: "OK handshape", detail: "Thumb and index touch; the other three fingers are extended. Used here for “I’m fine,” not the ASL 5-hand FINE." },
  { id: "HELP", title: "Help", method: "rule", show: "Fist rests on an open palm", detail: "Not detected from one hand. Use the Train page or fingerspell HELP." },
  { id: "SCHOOL", title: "School", method: "rule", show: "Two hands clap", detail: "Not detected from one hand. Fingerspell SCHOOL." },
  { id: "HOME", title: "Home", method: "rule", show: "Pinch, then a flat hand", detail: "First half of the gesture is a pinch and the second half is an open hand." },
]

/** Words that are a held handshape with no motion. They need the hand still and the shape held. */
export const STATIC_WORDS = new Set(["LOVE", "FINE", "WATER", "NEED", "MORE", "GOOD", "ME", "YOU"])

/** Frames a still word shape must be held (about 0.4 s) before it counts. */
const STATIC_HOLD = 12

const DETECTED = new Set(WORD_SIGNS.filter((sign) => !["HELP", "SCHOOL"].includes(sign.id)).map((sign) => sign.id))

function dist2(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1])
}

const FINGERS = [
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
] as const

/** Per-finger measures in the image plane, in palm lengths. Calibrated on real MediaPipe letter data. */
function fingerStats(pts: Point[], palm: number) {
  return FINGERS.map(([mcp, pip, dip, tip]) => {
    const bones = dist2(pts[mcp], pts[pip]) + dist2(pts[pip], pts[dip]) + dist2(pts[dip], pts[tip])
    return {
      /** Tip distance from the wrist over knuckle distance: about 1.9 straight, under 1 curled. */
      reach: dist2(pts[tip], pts[0]) / (dist2(pts[mcp], pts[0]) || 1e-6),
      /** 1 when the finger is a straight line. */
      straightness: dist2(pts[mcp], pts[tip]) / (bones || 1e-6),
      /** Visible finger length; short when it points at the camera. */
      length: bones / palm,
    }
  })
}

export interface HandShape {
  index: boolean
  middle: boolean
  ring: boolean
  pinky: boolean
  thumb: boolean
  /** Thumb straight and pointing up in the image, other fingers closed: a thumbs-up. */
  thumbUp: boolean
  pinch: boolean
  ok: boolean
  bentIndex: boolean
  /** Fingers half bent in a C, neither straight nor closed. */
  curved: boolean
  /** Index and middle held together (U) rather than spread (V). */
  together: boolean
  /** Thumb stretched well away from the fingers, as in L, Y or I-love-you. */
  thumbWide: boolean
}

/**
 * Landmarks are expected in square image units (x already multiplied by the frame's width / height),
 * the same space the letter model was trained in.
 */
export function shapeOf(raw: Point[]): HandShape {
  const pts = normalizeLandmarks(raw)
  const palm = dist2(pts[0], pts[9]) || 1e-6
  const f = fingerStats(pts, palm)
  const straight = (k: number) => f[k].reach > 1.55 && f[k].straightness > 0.86 && f[k].length > (k === 3 ? 0.55 : 0.72)
  const gap = dist2(pts[4], pts[8]) / palm
  const gapMiddle = dist2(pts[4], pts[12]) / palm
  const thumbOut = dist2(pts[4], pts[5]) / palm
  const thumb = dist2(pts[4], pts[0]) > dist2(pts[3], pts[0]) * 1.05 && thumbOut > 0.35
  // Fingertips meeting the thumb: F / OK when the other three stand up, otherwise a flat O pinch.
  const ok = gap < 0.4 && straight(1) && straight(2) && f[1].reach > 1.6 && f[2].reach > 1.6
  const pinch = !ok && gap < 0.42 && gapMiddle < 0.65 && f[0].reach > 1.1 && f[1].reach > 1.1
  // C: four fingers curve round with the thumb well out and its tip facing the fingertips.
  const curved =
    !pinch && !ok && f.every((s) => s.reach > 1.2) && thumbOut > 0.6 && gap > 0.42 && gap < 1.02 && (f[0].straightness + f[1].straightness) / 2 < 0.975
  const index = straight(0) && !pinch && !curved
  const middle = straight(1) && !pinch && !curved
  const ring = straight(2) && !pinch && !curved
  const pinky = straight(3) && !pinch && !curved
  const closedRest = f[1].reach < 1.1 && f[2].reach < 1.0
  // X: index half out and hooked (tip folded back near its middle joint), the rest closed.
  const bentIndex =
    !index && closedRest && f[0].reach > 1.15 && f[0].reach < 1.7 && dist2(pts[8], pts[6]) / palm < 0.42 && gapMiddle < 0.7
  const closed = !index && !middle && !ring && !pinky && !pinch && !curved
  // Thumbs-up: the thumb tip rises clearly above every finger knuckle and points up.
  const topKnuckle = Math.min(pts[5][1], pts[6][1], pts[9][1], pts[10][1], pts[13][1], pts[17][1])
  const thumbDx = pts[4][0] - pts[2][0]
  const thumbDy = pts[4][1] - pts[2][1]
  const thumbUp = closed && !bentIndex && (topKnuckle - pts[4][1]) / palm > 0.35 && thumbDy / (Math.hypot(thumbDx, thumbDy) || 1e-6) < -0.7
  const together = dist2(pts[8], pts[12]) / palm < 0.32
  const thumbWide = thumb && thumbOut > 0.6 && gapMiddle > 0.85
  return { index, middle, ring, pinky, thumb, thumbUp, pinch, ok, bentIndex, curved, together, thumbWide }
}

function open(shape: HandShape): boolean {
  return shape.index && shape.middle && shape.ring && shape.pinky
}

function fist(shape: HandShape): boolean {
  return !shape.index && !shape.middle && !shape.ring && !shape.pinky && !shape.ok && !shape.thumbUp && !shape.bentIndex && !shape.pinch && !shape.curved
}

function indexOnly(shape: HandShape): boolean {
  return shape.index && !shape.middle && !shape.ring && !shape.pinky
}

function wShape(shape: HandShape): boolean {
  return shape.index && shape.middle && shape.ring && !shape.pinky
}

function twoFingers(shape: HandShape): boolean {
  return shape.index && shape.middle && !shape.ring && !shape.pinky
}

function ily(shape: HandShape): boolean {
  return shape.thumbWide && shape.index && !shape.middle && !shape.ring && shape.pinky
}

/**
 * Direction changes along a path, counted only when the hand travels back at least `swing`
 * (in palm lengths) from the last turning point. Smooth waves and webcam jitter both count right.
 */
export function reversals(values: number[], swing: number): number {
  if (values.length < 3) return 0
  let count = 0
  let dir = 0
  let hi = values[0]
  let lo = values[0]
  for (const v of values) {
    hi = Math.max(hi, v)
    lo = Math.min(lo, v)
    if (dir === 0) {
      if (hi - v > swing) {
        dir = -1
        lo = v
      } else if (v - lo > swing) {
        dir = 1
        hi = v
      }
    } else if (dir > 0 && hi - v > swing) {
      count += 1
      dir = -1
      lo = v
    } else if (dir < 0 && v - lo > swing) {
      count += 1
      dir = 1
      hi = v
    }
  }
  return count
}

function span(values: number[]): number {
  return Math.max(...values) - Math.min(...values)
}

/** Finger states as a short string, used to tell when the hand keeps the same shape. */
export function shapeSignature(raw: Point[]): string {
  const s = shapeOf(raw)
  return [s.index, s.middle, s.ring, s.pinky, s.pinch, s.ok, s.thumbUp, s.bentIndex].map((on) => (on ? "1" : "0")).join("")
}

/** Apparent palm size: mean of the wrist / knuckle triangle in the image. */
export function palmSize(frame: Point[]): number {
  return (dist2(frame[0], frame[5]) + dist2(frame[0], frame[17]) + dist2(frame[5], frame[17])) / 3
}

/**
 * Depth from the camera, larger when the hand is farther away. MediaPipe gives every wrist a z of
 * zero, so the apparent palm size is used instead: log of the inverse of the wrist / knuckle triangle.
 */
export function palmDepth(frame: Point[]): number {
  return -Math.log(Math.max(palmSize(frame), 1e-4))
}

/**
 * The newest frames that keep the current handshape, so motion from the previous sign or the
 * move between signs is not read as part of this one. Up to two stray frames in a row are allowed.
 */
function trailingRun(frames: Point[][], maxLength = 36): Point[][] {
  const target = shapeSignature(frames[frames.length - 1])
  let start = frames.length - 1
  let misses = 0
  for (let i = frames.length - 2; i >= 0 && frames.length - i <= maxLength; i--) {
    if (shapeSignature(frames[i]) === target) {
      misses = 0
      start = i
    } else if (++misses > 3) break
  }
  return frames.slice(start)
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)]
}

export interface WordHit {
  id: string
  score: number
}

/** Change in log palm size that counts as a push or pull (about 12 percent bigger or smaller). */
const DEPTH_STEP = 0.12

/** Unit direction of the index finger in the image (knuckle to tip). */
function indexDirection(frame: Point[]): [number, number] {
  const dx = frame[8][0] - frame[5][0]
  const dy = frame[8][1] - frame[5][1]
  const n = Math.hypot(dx, dy) || 1e-6
  return [dx / n, dy / n]
}

/** Unit direction from wrist to middle knuckle in the image. */
function handDirection(frame: Point[]): [number, number] {
  const dx = frame[9][0] - frame[0][0]
  const dy = frame[9][1] - frame[0][1]
  const n = Math.hypot(dx, dy) || 1e-6
  return [dx / n, dy / n]
}

/** Pick at most one word. Returns null when the pose is better left to the letter model. */
export function recognizeWord(frames: Point[][]): WordHit | null {
  if (frames.length < 6) return null
  return wordScores(frames).best
}

/** The rule inputs and every rule that fired, for tests and the harness report. */
export function wordScores(frames: Point[][]) {
  const lastFrame = frames[frames.length - 1]
  const last = shapeOf(lastFrame)
  // Motion is measured only over the frames that keep this handshape, in palm lengths.
  const run = trailingRun(frames)
  const unit = median(run.map(palmSize)) || 1e-3
  const xs = run.map((frame) => frame[0][0] / unit)
  const ys = run.map((frame) => frame[0][1] / unit)
  const depth = run.map(palmDepth)
  const xTurns = reversals(xs, 0.25)
  const yTurns = reversals(ys, 0.2)
  const xSpan = span(xs)
  const ySpan = span(ys)
  // Depth is measured after the hand stops travelling across the frame, so the move into a
  // sign (where a turning hand also changes its apparent size) is not read as a push or pull.
  let settledFrom = 0
  for (let i = 1; i < run.length; i++) {
    if (Math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) > 0.12) settledFrom = i
  }
  const settled = depth.slice(settledFrom)
  const zEdge = Math.max(1, Math.min(3, Math.floor(settled.length / 3)))
  // Positive when the hand moves away from the camera (toward the signer).
  const zDelta = settled.length >= 6 ? mean(settled.slice(-zEdge)) - mean(settled.slice(0, zEdge)) : 0
  const yDelta = ys[ys.length - 1] - ys[0]
  const xDelta = xs[xs.length - 1] - xs[0]
  const still = xSpan < 0.6 && ySpan < 0.6
  const [, indexY] = indexDirection(lastFrame)
  const indexPointsDown = indexY > 0.55
  const indexPointsUp = indexY < -0.6

  // Shape changes look back over the last second, across handshapes.
  const recent = frames.slice(-45)
  const earlier = recent.slice(0, Math.max(1, Math.floor(recent.length / 2))).map(shapeOf)
  // A shape change only counts when the first shape was really held, not passed through.
  const startedOpen = earlier.filter(open).length >= 5
  const startedPinch = earlier.filter((s) => s.pinch).length >= 5
  const flatRecent = recent.filter((frame) => {
    const s = shapeOf(frame)
    return s.index && s.middle && s.ring
  })
  let flip = false
  if (flatRecent.length >= 4) {
    const startY = handDirection(flatRecent[0])[1]
    const endY = handDirection(flatRecent[flatRecent.length - 1])[1]
    flip = startY < -0.2 && endY > 0.2
  }

  const scores: Record<string, number> = {}
  if (ily(last)) scores.LOVE = 0.95
  if (last.ok) scores.FINE = 0.93
  if (wShape(last) && xTurns < 2) scores.WATER = 0.9
  if (last.bentIndex && yTurns < 2) scores.NEED = 0.9
  if (last.pinch && still && !startedOpen) scores.MORE = 0.86
  if (last.thumbUp && still) scores.GOOD = 0.9
  if (open(last) && xTurns >= 2 && xSpan >= 0.7) scores.HELLO = 0.92
  if (open(last) && xTurns >= 2 && xSpan < 0.7) scores.WHAT = 0.86
  if (fist(last) && yTurns >= 2 && xTurns < 2) scores.YES = 0.92
  if (twoFingers(last) && !last.together && xTurns >= 2) scores.NO = 0.9
  if (indexOnly(last) && xTurns >= 2) scores.WHERE = 0.9
  if (twoFingers(last) && last.together && yTurns >= 2) scores.NAME = 0.86
  if (open(last) && yDelta > 0.5 && xTurns < 2 && xSpan < ySpan) scores.THANKS = 0.86
  // A circle: real travel on both axes, at least one turn, and neither axis dominating.
  const circle = xSpan > 0.4 && ySpan > 0.4 && xSpan < ySpan * 2.2 && ySpan < xSpan * 2.2 && xTurns + yTurns >= 1 && xTurns < 4
  if (open(last) && circle) scores.PLEASE = 0.88
  if (fist(last) && circle) scores.SORRY = 0.93
  if (indexOnly(last) && indexPointsDown && xTurns < 2) scores.ME = 0.9
  if (indexOnly(last) && !last.thumbWide && indexPointsUp && xTurns < 2 && zDelta > -DEPTH_STEP && xSpan < 0.8) scores.YOU = 0.88
  if (indexOnly(last) && Math.abs(xDelta) > 0.8 && xTurns < 2 && !indexPointsDown) scores.WE = 0.91
  // Pushes and pulls travel toward or away from the camera, not across the frame.
  const depthOnly = run.length >= 8 && xTurns < 2 && span(xs.slice(-15)) < 0.9 && span(ys.slice(-15)) < 0.9
  if (open(last) && zDelta < -DEPTH_STEP && depthOnly) scores.STOP = 0.89
  if (open(last) && zDelta > DEPTH_STEP && depthOnly) scores.WANT = 0.89
  if (indexOnly(last) && zDelta < -DEPTH_STEP && depthOnly) scores.GO = 0.89
  if (!indexOnly(last) && !open(last) && !last.pinch && !last.thumbUp && zDelta > DEPTH_STEP && depthOnly) scores.COME = 0.87
  if (last.pinch && yDelta < -0.4 && !startedOpen) scores.EAT = 0.89
  if (last.curved && yDelta < -0.4) scores.DRINK = 0.89
  if (flip) scores.BAD = 0.9
  if (startedOpen && last.pinch) scores.LIKE = 0.9
  if (startedPinch && open(last)) scores.HOME = 0.9

  let best: WordHit | null = null
  for (const [id, score] of Object.entries(scores)) {
    if (!DETECTED.has(id)) continue
    if (STATIC_WORDS.has(id) && run.length < STATIC_HOLD) continue
    if (!best || score > best.score) best = { id, score }
  }
  return {
    best: best && best.score >= 0.8 ? best : null,
    scores,
    features: { run: run.length, xTurns, yTurns, xSpan, ySpan, zDelta, yDelta, xDelta, indexY, flip, startedOpen, startedPinch, shape: last },
  }
}

export function still(pose: Point[], count = 10): Point[][] {
  return Array.from({ length: count }, () => pose.map((point) => [...point] as Point))
}

export function shift(frames: Point[][], at: (index: number, count: number) => [number, number, number]): Point[][] {
  return frames.map((frame, index) => {
    const [dx, dy, dz] = at(index, frames.length)
    return frame.map((point) => [point[0] + dx, point[1] + dy, point[2] + dz] as Point)
  })
}

/** Grow or shrink each frame about its wrist, as a hand does when it moves toward or away from the camera. */
export function scaleFrames(frames: Point[][], factor: (index: number, count: number) => number): Point[][] {
  return frames.map((frame, index) => {
    const k = factor(index, frames.length)
    const [wx, wy] = frame[0]
    return frame.map((point) => [wx + (point[0] - wx) * k, wy + (point[1] - wy) * k, point[2] * k] as Point)
  })
}

/** Aspect of the built-in demo frames (a 16:9 camera view). */
export const DEMO_ASPECT = 16 / 9

/**
 * A clean built-in performance of a word, in image coordinates of a 16:9 frame, followed by a short
 * hold. Uses the same hand model as the accuracy tests. Pass through `squareFrame` before the rules.
 */
export function demoSequence(id: string): Point[][] {
  const motion = WORD_MOTIONS[id] ?? WORD_MOTIONS.HELLO
  const frames = performMotion(motion, { rand: () => 0.5, jitter: 0, aspect: DEMO_ASPECT, hand: "Right" })
  return [...frames, ...Array.from({ length: 10 }, () => frames[frames.length - 1])]
}

/** The demo frames as the recognizer sees them. */
export function demoSquare(id: string): Point[][] {
  return demoSequence(id).map((frame) => squareFrame(frame, DEMO_ASPECT))
}

export const PRESENTATION_SCRIPT = ["ME", "WANT", "WATER"] as const

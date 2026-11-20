import type { Point } from "./hands"
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

const DETECTED = new Set(WORD_SIGNS.filter((sign) => !["HELP", "SCHOOL"].includes(sign.id)).map((sign) => sign.id))

function dist(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])
}

function extended(pts: Point[], mcp: number, _pip: number, tip: number): boolean {
  const base = dist(pts[mcp], pts[0])
  return dist(pts[tip], pts[0]) > base * 1.45 && dist(pts[tip], pts[mcp]) > base * 0.55
}

export interface HandShape {
  index: boolean
  middle: boolean
  ring: boolean
  pinky: boolean
  thumb: boolean
  pinch: boolean
  ok: boolean
  bentIndex: boolean
}

export function shapeOf(raw: Point[]): HandShape {
  const pts = normalizeLandmarks(raw)
  const index = extended(pts, 5, 6, 8)
  const middle = extended(pts, 9, 10, 12)
  const ring = extended(pts, 13, 14, 16)
  const pinky = extended(pts, 17, 18, 20)
  const thumb = dist(pts[4], pts[0]) > dist(pts[3], pts[0]) * 1.05 && dist(pts[4], pts[5]) > 0.35
  const pinch = dist(pts[4], pts[8]) < 0.45 && dist(pts[4], pts[12]) < 0.55 && !index && !middle
  const ok = dist(pts[4], pts[8]) < 0.4 && middle && ring && pinky
  const bentIndex = !index && dist(pts[8], pts[5]) < dist(pts[6], pts[5]) * 1.3 && !middle && !ring && !pinky
  return { index, middle, ring, pinky, thumb, pinch, ok, bentIndex }
}

function open(shape: HandShape): boolean {
  return shape.index && shape.middle && shape.ring && shape.pinky
}

function fist(shape: HandShape): boolean {
  return !shape.index && !shape.middle && !shape.ring && !shape.pinky && !shape.ok
}

function indexOnly(shape: HandShape): boolean {
  return shape.index && !shape.middle && !shape.ring && !shape.pinky
}

function wShape(shape: HandShape): boolean {
  return shape.index && shape.middle && shape.ring && !shape.pinky
}

function vShape(shape: HandShape): boolean {
  return shape.index && shape.middle && !shape.ring && !shape.pinky
}

function ily(shape: HandShape): boolean {
  return shape.thumb && shape.index && !shape.middle && !shape.ring && shape.pinky
}

function uShape(shape: HandShape): boolean {
  return shape.index && shape.middle && !shape.ring && !shape.pinky
}

function turns(values: number[], minStep = 0.012): number {
  let count = 0
  for (let i = 2; i < values.length; i++) {
    const previous = values[i - 1] - values[i - 2]
    const next = values[i] - values[i - 1]
    if (previous * next < 0 && Math.abs(next) > minStep) count += 1
  }
  return count
}

function span(values: number[]): number {
  return Math.max(...values) - Math.min(...values)
}

/** Finger states as a short string, used to tell when the hand keeps the same shape. */
export function shapeSignature(raw: Point[]): string {
  const s = shapeOf(raw)
  return [s.index, s.middle, s.ring, s.pinky, s.pinch, s.ok].map((on) => (on ? "1" : "0")).join("")
}

function dist2(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1])
}

/**
 * Depth from the camera, larger when the hand is farther away. MediaPipe gives every wrist a z of
 * zero, so the apparent palm size is used instead: log of the inverse of the wrist / knuckle triangle.
 */
export function palmDepth(frame: Point[]): number {
  const size = (dist2(frame[0], frame[5]) + dist2(frame[0], frame[17]) + dist2(frame[5], frame[17])) / 3
  return -Math.log(Math.max(size, 1e-4))
}

/**
 * The newest frames that keep the current handshape, so motion from the previous sign or the
 * move between signs is not read as part of this one. Up to two stray frames in a row are allowed.
 */
function trailingRun(frames: Point[][], maxLength = 30): Point[][] {
  const target = shapeSignature(frames[frames.length - 1])
  let start = frames.length - 1
  let misses = 0
  for (let i = frames.length - 2; i >= 0 && frames.length - i <= maxLength; i--) {
    if (shapeSignature(frames[i]) === target) {
      misses = 0
      start = i
    } else if (++misses > 2) break
  }
  return frames.slice(start)
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length
}

export interface WordHit {
  id: string
  score: number
}

/** Change in log palm size that counts as a push or pull (about 12 percent bigger or smaller). */
const DEPTH_STEP = 0.12

/** Pick at most one word. Returns null when the pose is better left to the letter model. */
export function recognizeWord(frames: Point[][]): WordHit | null {
  if (frames.length < 6) return null
  const first = shapeOf(frames[0])
  const last = shapeOf(frames[frames.length - 1])
  // Motion is measured only over the frames that keep this handshape.
  const run = trailingRun(frames)
  const xs = run.map((frame) => frame[0][0])
  const ys = run.map((frame) => frame[0][1])
  const depth = run.map(palmDepth)
  const edge = Math.max(1, Math.min(3, Math.floor(run.length / 3)))
  const xTurns = turns(xs)
  const yTurns = turns(ys)
  const xSpan = span(xs)
  const ySpan = span(ys)
  // Positive when the hand moves away from the camera (toward the signer).
  const zDelta = run.length >= 6 ? mean(depth.slice(-edge)) - mean(depth.slice(0, edge)) : 0
  const yDelta = ys[ys.length - 1] - ys[0]
  const tipsY = frames.map((frame) => frame[8][1])
  const tipDrop = tipsY[tipsY.length - 1] - tipsY[0]
  const norm = normalizeLandmarks(frames[frames.length - 1])
  const indexPointsDown = norm[8][1] > 0.35
  const indexPointsUp = norm[8][1] < -0.55

  const scores: Record<string, number> = {}
  if (ily(last)) scores.LOVE = 0.95
  if (last.ok) scores.FINE = 0.93
  if (wShape(last) && xTurns < 2) scores.WATER = 0.9
  if (last.bentIndex && yTurns < 2) scores.NEED = 0.9
  if (last.pinch && yTurns < 2 && yDelta > -0.04) scores.MORE = 0.86
  if (last.thumb && fist(last) && indexPointsUp) scores.GOOD = 0.9
  if (open(last) && xTurns >= 2 && xSpan >= 0.08) scores.HELLO = 0.92
  if (open(last) && xTurns >= 2 && xSpan < 0.08) scores.WHAT = 0.84
  if (fist(last) && yTurns >= 2) scores.YES = 0.92
  if (vShape(last) && xTurns >= 2) scores.NO = 0.9
  if (indexOnly(last) && xTurns >= 2) scores.WHERE = 0.9
  if (uShape(last) && yTurns >= 2 && !vShape(last)) scores.NAME = 0.84
  if (open(last) && yDelta > 0.06 && xTurns < 2) scores.THANKS = 0.86
  if (open(last) && xSpan > 0.05 && ySpan > 0.05 && xTurns >= 1 && yTurns >= 1) scores.PLEASE = 0.84
  if (fist(last) && xSpan > 0.04 && ySpan > 0.04 && xTurns >= 1 && yTurns >= 1 && yTurns < 3) scores.SORRY = 0.8
  if (indexOnly(last) && indexPointsDown && xTurns < 2) scores.ME = 0.9
  if (indexOnly(last) && indexPointsUp && xTurns < 2 && zDelta > -DEPTH_STEP) scores.YOU = 0.88
  if (indexOnly(last) && xSpan > 0.08 && xTurns < 2 && !indexPointsDown) scores.WE = 0.82
  if (open(last) && zDelta < -DEPTH_STEP && xTurns < 2) scores.STOP = 0.86
  if (open(last) && zDelta > DEPTH_STEP && xTurns < 2) scores.WANT = 0.88
  if (indexOnly(last) && zDelta < -DEPTH_STEP) scores.GO = 0.84
  if (!indexOnly(last) && zDelta > DEPTH_STEP && !open(last) && !fist(last)) scores.COME = 0.8
  if (last.pinch && yDelta < -0.05) scores.EAT = 0.88
  if (!open(last) && !fist(last) && !last.pinch && yDelta < -0.05) scores.DRINK = 0.8
  if (tipDrop > 0.08 && open(first)) scores.BAD = 0.86
  if (open(first) && last.pinch) scores.LIKE = 0.9
  if (first.pinch && open(last)) scores.HOME = 0.86

  let best: WordHit | null = null
  for (const [id, score] of Object.entries(scores)) {
    if (!DETECTED.has(id)) continue
    if (!best || score > best.score) best = { id, score }
  }
  return best && best.score >= 0.8 ? best : null
}

function blank(): Point[] {
  return Array.from({ length: 21 }, () => [0.5, 0.62, 0] as Point)
}

function placeFinger(pts: Point[], mcp: number, pip: number, dip: number, tip: number, reach: number) {
  const wrist = pts[0]
  const base = pts[mcp]
  const dx = base[0] - wrist[0]
  const dy = base[1] - wrist[1]
  const length = Math.hypot(dx, dy) || 0.01
  const ux = dx / length
  const uy = dy / length
  pts[pip] = [base[0] + ux * reach * 0.35, base[1] + uy * reach * 0.35, 0]
  pts[dip] = [base[0] + ux * reach * 0.7, base[1] + uy * reach * 0.7, 0]
  pts[tip] = [base[0] + ux * reach, base[1] + uy * reach, 0]
}

/** A right-hand pose in image coordinates. `reach` near 1 is straight; near 0.15 is curled. */
export function posedHand(reaches: { index: number; middle: number; ring: number; pinky: number; thumb: number }, pinch = false): Point[] {
  const pts = blank()
  pts[5] = [0.58, 0.5, 0]
  pts[9] = [0.5, 0.46, 0]
  pts[13] = [0.43, 0.49, 0]
  pts[17] = [0.37, 0.54, 0]
  pts[1] = [0.62, 0.6, 0]
  pts[2] = [0.66, 0.56, 0]
  pts[3] = [0.69, 0.52, 0]
  placeFinger(pts, 5, 6, 7, 8, 0.16 * reaches.index)
  placeFinger(pts, 9, 10, 11, 12, 0.18 * reaches.middle)
  placeFinger(pts, 13, 14, 15, 16, 0.16 * reaches.ring)
  placeFinger(pts, 17, 18, 19, 20, 0.14 * reaches.pinky)
  pts[4] = [0.66 + 0.08 * reaches.thumb, 0.48 - 0.06 * reaches.thumb, 0]
  if (pinch) pts[4] = [...pts[8]]
  return pts
}

const OPEN = { index: 1, middle: 1, ring: 1, pinky: 1, thumb: 1 }
const FIST = { index: 0.12, middle: 0.12, ring: 0.12, pinky: 0.12, thumb: 0.2 }
const INDEX = { index: 1, middle: 0.12, ring: 0.12, pinky: 0.12, thumb: 0.15 }

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

export function demoSequence(id: string): Point[][] {
  if (id === "ME") {
    const pose = posedHand(INDEX)
    pose[6] = [0.58, 0.7, 0]
    pose[7] = [0.58, 0.8, 0]
    pose[8] = [0.58, 0.92, 0]
    return still(pose)
  }
  if (id === "YOU") return still(posedHand(INDEX))
  if (id === "LOVE") return still(posedHand({ index: 1, middle: 0.12, ring: 0.12, pinky: 1, thumb: 1 }))
  if (id === "WATER") return still(posedHand({ index: 1, middle: 1, ring: 1, pinky: 0.12, thumb: 0.2 }))
  if (id === "FINE") return still(posedHand({ index: 0.2, middle: 1, ring: 1, pinky: 1, thumb: 0.2 }, true))
  if (id === "NEED") return still(posedHand({ index: 0.35, middle: 0.12, ring: 0.12, pinky: 0.12, thumb: 0.2 }))
  if (id === "MORE") return still(posedHand(FIST, true))
  if (id === "GOOD") return still(posedHand({ index: 0.12, middle: 0.12, ring: 0.12, pinky: 0.12, thumb: 1 }))
  if (id === "HELLO") return shift(still(posedHand(OPEN), 16), (i) => [Math.sin((i / 15) * Math.PI * 4) * 0.07, 0, 0])
  if (id === "YES") return shift(still(posedHand(FIST), 16), (i) => [0, Math.sin((i / 15) * Math.PI * 4) * 0.06, 0])
  if (id === "NO") return shift(still(posedHand({ index: 1, middle: 1, ring: 0.12, pinky: 0.12, thumb: 0.2 }), 16), (i) => [Math.sin((i / 15) * Math.PI * 4) * 0.06, 0, 0])
  if (id === "WANT") return scaleFrames(still(posedHand(OPEN), 10), (i, n) => 1 - 0.25 * (i / (n - 1)))
  if (id === "STOP") return scaleFrames(still(posedHand(OPEN), 10), (i, n) => 1 + 0.3 * (i / (n - 1)))
  if (id === "THANKS") return shift(still(posedHand(OPEN), 10), (i, n) => [0, 0.1 * (i / (n - 1)), 0])
  if (id === "WHERE") return shift(still(posedHand(INDEX), 16), (i) => [Math.sin((i / 15) * Math.PI * 4) * 0.05, 0, 0])
  if (id === "LIKE") return [...still(posedHand(OPEN), 5), ...still(posedHand(FIST, true), 5)]
  if (id === "EAT") return shift(still(posedHand(FIST, true), 10), (i, n) => [0, -0.08 * (i / (n - 1)), 0])
  if (id === "HOME") return [...still(posedHand(FIST, true), 6), ...still(posedHand(OPEN), 6)]
  return still(posedHand(OPEN))
}

export const PRESENTATION_SCRIPT = ["ME", "WANT", "WATER"] as const

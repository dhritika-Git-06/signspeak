// Test-only: synthetic sign performances and a runner that feeds them through the live pipeline.
import type { Point } from "@/lib/hands"
import { squareFrame } from "@/lib/hands"
import { WORD_MOTIONS, performMotion, seededRandom } from "@/lib/handModel"
import { Recognizer, type RecognitionMode, type RecognizerDeps } from "@/lib/recognizer"
import { recognizeWord } from "@/lib/words"

/** What the tracker hands on: x in the same units as y. */
export function square(frames: Point[][], aspect: number) {
  return frames.map((f) => squareFrame(f, aspect))
}

export interface Stream {
  label: string
  frames: Point[][]
  /** True when MediaPipe would call the hand "Left" (the app then mirrors it for the models). */
  mirror: boolean
}

/** A performed word: a right or left hand, random size, place, angle, speed and jitter, then a short hold. */
export function wordStream(word: string, k: number, seed = 1): Stream {
  const rand = seededRandom(seed * 7919 + k * 104729 + word.charCodeAt(0) * 31 + word.length)
  const aspect = k % 3 === 0 ? 4 / 3 : 16 / 9
  const hand = k % 2 ? "Left" : "Right"
  const jitter = 0.002 + rand() * 0.004
  const frames = square(performMotion(WORD_MOTIONS[word], { rand, hand, aspect, jitter }), aspect)
  const hold = Array.from({ length: 12 }, () => frames[frames.length - 1])
  // An unmirrored right hand looks like a mirrored left hand, so MediaPipe labels it "Left".
  return { label: word, frames: [...frames, ...hold], mirror: hand === "Right" }
}

/** A held letter from a real landmark row, rotated, scaled, moved, jittered and sometimes mirrored. */
export function letterStream(label: string, row: Point[], k: number, seed = 1): Stream {
  const rand = seededRandom(seed * 15485863 + k * 7 + label.charCodeAt(0) * 131)
  const angle = ((rand() - 0.5) * 30 * Math.PI) / 180
  const scale = 0.6 + rand() * 0.8
  const flip = k % 2 === 1
  const cx = 0.3 + rand() * 0.6
  const cy = 0.4 + rand() * 0.3
  const w = row[0]
  const base = row.map((p) => {
    let x = (p[0] - w[0]) * scale
    const y = (p[1] - w[1]) * scale
    if (flip) x = -x
    return [cx + x * Math.cos(angle) - y * Math.sin(angle), cy + x * Math.sin(angle) + y * Math.cos(angle), 0] as Point
  })
  const frames: Point[][] = []
  const n = 36
  for (let i = 0; i < n; i++) {
    frames.push(base.map((p) => [p[0] + (rand() - 0.5) * 0.006, p[1] + (rand() - 0.5) * 0.006, 0] as Point))
  }
  // Rows are stored as the model expects (a "Right" hand); a flipped row is a "Left" hand to mirror back.
  return { label, frames, mirror: flip }
}

/** Feed one sign, with the hand coming into view and leaving, and return every committed label. */
export function runStream(engine: Recognizer, stream: Stream, mode: RecognitionMode): string[] {
  engine.reset()
  const commits: string[] = []
  const feed = (frame: Point[] | null) => {
    const result = engine.push(frame, { mode, mirror: stream.mirror })
    if (result.commit) commits.push(result.commit.label)
  }
  for (let i = 0; i < 4; i++) feed(null)
  for (const frame of stream.frames) feed(frame)
  for (let i = 0; i < 12; i++) feed(null)
  return commits
}

export interface Tally {
  n: number
  exact: number
  extra: number
  missed: number
  wrong: Record<string, number>
}

export function tally(results: Array<{ label: string; commits: string[] }>) {
  const out: Record<string, Tally> = {}
  for (const { label, commits } of results) {
    const t = (out[label] ??= { n: 0, exact: 0, extra: 0, missed: 0, wrong: {} })
    t.n += 1
    if (commits.length === 1 && commits[0] === label) t.exact += 1
    else if (!commits.length) t.missed += 1
    else if (commits.includes(label)) t.extra += 1
    else {
      const key = commits.join("+")
      t.wrong[key] = (t.wrong[key] ?? 0) + 1
    }
  }
  return out
}

export function runAll(deps: RecognizerDeps, streams: Stream[], mode: RecognitionMode) {
  const engine = new Recognizer(deps)
  return streams.map((stream) => ({ label: stream.label, commits: runStream(engine, stream, mode) }))
}

export function parseLetterCsv(text: string, every = 50, offset = 7) {
  const rows: Array<{ label: string; points: Point[] }> = []
  text.trim().split("\n").slice(1).forEach((line, i) => {
    if (i % every !== offset) return
    const cells = line.split(",")
    const v = cells.slice(1).map(Number)
    rows.push({ label: cells[0], points: Array.from({ length: 21 }, (_, j) => [v[2 * j], v[2 * j + 1], 0] as Point) })
  })
  return rows
}

/** Rule-only confusion over synthetic performances of every one-hand word. */
export function ruleConfusion(perSign = 40, seed = 1) {
  const labels = Object.keys(WORD_MOTIONS)
  const matrix: Record<string, Record<string, number>> = {}
  for (const word of labels) {
    matrix[word] = {}
    for (let k = 0; k < perSign; k++) {
      const frames = wordStream(word, k, seed).frames
      const got = endOfSign(frames)
      matrix[word][got] = (matrix[word][got] ?? 0) + 1
    }
  }
  return { labels, matrix }
}

/** The label the rules give most over the final second of a performance. */
export function endOfSign(frames: Point[][]): string {
  const all = frames.slice(-48)
  const votes: Record<string, number> = {}
  for (let i = Math.max(6, all.length - 30); i <= all.length; i++) {
    const hit = recognizeWord(all.slice(0, i))?.id ?? "-"
    votes[hit] = (votes[hit] ?? 0) + 1
  }
  return Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0]
}

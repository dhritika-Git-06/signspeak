import { describe, expect, it } from "vitest"
import type { Point } from "./hands"
import { motionEnergy, normalizeSequence } from "./normalize"
import { Recognizer, type RecognitionMode, type SignKind } from "./recognizer"
import { SentenceSession } from "./sentenceEngine"
import { demoSequence, posedHand, recognizeWord } from "./words"

/**
 * Simulated webcam streams at 30 fps: hand-landmark jitter, slow drift, single lost frames,
 * a letter model that flickers between look-alike letters, and a motion model that guesses
 * at random while the hand moves. The point is to check the whole live pipeline, not one rule.
 */

const META_MOTION_THRESHOLD = 0.06283550059888512
const ALL_LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("")
const MOTION_LABELS = ["J", "Z", "HELLO", "YES", "NO", "THANKS", "PLEASE", "SORRY"]

function rng(seed: number) {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

type Frame = Point[] | null
interface Tagged {
  frames: Frame[]
  /** What the letter model "sees" for each frame: [main letter, look-alikes] or null for a transition. */
  letters: Array<[string, string[]] | null>
}

const LOOK_ALIKES: Record<string, [string, string[]]> = {
  ME: ["D", ["G", "X", "P"]],
  WANT: ["B", ["C", "E"]],
  WATER: ["W", ["F", "V", "K"]],
  C: ["C", ["O", "E"]],
  S: ["S", ["A", "E", "T"]],
  B: ["B", ["C"]],
}

function basePose(id: string): Point[] {
  if (id === "ME" || id === "WATER") return demoSequence(id)[0]
  if (id === "WANT" || id === "B") return posedHand({ index: 1, middle: 1, ring: 1, pinky: 1, thumb: 1 })
  if (id === "S") return posedHand({ index: 0.12, middle: 0.12, ring: 0.12, pinky: 0.12, thumb: 0 })
  if (id === "C") return posedHand({ index: 0.45, middle: 0.45, ring: 0.45, pinky: 0.45, thumb: 0 })
  throw new Error(id)
}

function scaled(pose: Point[], k: number): Point[] {
  const [wx, wy] = pose[0]
  return pose.map((p) => [wx + (p[0] - wx) * k, wy + (p[1] - wy) * k, p[2]] as Point)
}

function moved(pose: Point[], dx: number, dy: number): Point[] {
  return pose.map((p) => [p[0] + dx, p[1] + dy, p[2]] as Point)
}

function jitter(pose: Point[], rand: () => number, amount = 0.004): Point[] {
  return pose.map((p) => [p[0] + (rand() - 0.5) * 2 * amount, p[1] + (rand() - 0.5) * 2 * amount, p[2]] as Point)
}

/** Hold a sign for `frames` frames. WANT starts with a pull back toward the signer (hand shrinks). */
function hold(id: string, frames: number, rand: () => number, offset: [number, number]): Tagged {
  const out: Tagged = { frames: [], letters: [] }
  const pose = moved(basePose(id), offset[0], offset[1])
  for (let i = 0; i < frames; i++) {
    let p = pose
    if (id === "WANT") {
      const t = Math.min(1, i / 12)
      p = scaled(pose, 1 - 0.28 * t)
    }
    // Slow drift plus per-joint jitter.
    p = moved(p, Math.sin(i / 9) * 0.004, Math.cos(i / 11) * 0.004)
    p = jitter(p, rand)
    // About 3% of frames are lost by the tracker.
    out.frames.push(rand() < 0.03 ? null : p)
    out.letters.push(LOOK_ALIKES[id])
  }
  return out
}

/** Move from one sign to the next, hand in view, with the letter model guessing at random. */
function transition(from: Point[], to: Point[], frames: number, rand: () => number): Tagged {
  const out: Tagged = { frames: [], letters: [] }
  for (let i = 1; i <= frames; i++) {
    const t = i / (frames + 1)
    const p = from.map((a, j) => [a[0] + (to[j][0] - a[0]) * t, a[1] + (to[j][1] - a[1]) * t + Math.sin(t * Math.PI) * 0.05, 0] as Point)
    out.frames.push(jitter(p, rand))
    out.letters.push(null)
  }
  return out
}

function drop(frames: number): Tagged {
  return { frames: Array(frames).fill(null), letters: Array(frames).fill(null) }
}

function concat(...parts: Tagged[]): Tagged {
  return { frames: parts.flatMap((p) => p.frames), letters: parts.flatMap((p) => p.letters) }
}

function lastPose(t: Tagged): Point[] {
  for (let i = t.frames.length - 1; i >= 0; i--) if (t.frames[i]) return t.frames[i]!
  throw new Error("no pose")
}

function firstPose(t: Tagged): Point[] {
  for (const f of t.frames) if (f) return f
  throw new Error("no pose")
}

function run(stream: Tagged, rand: () => number, mode: RecognitionMode = "auto") {
  let index = 0
  const engine = new Recognizer({
    letter: () => {
      const tag = stream.letters[index]
      if (!tag) {
        // Moving hand: the letter model returns any letter, often with high confidence.
        return { label: ALL_LETTERS[Math.floor(rand() * ALL_LETTERS.length)], confidence: 0.45 + rand() * 0.5, top: [] }
      }
      const [main, alike] = tag
      const label = rand() < 0.18 ? alike[Math.floor(rand() * alike.length)] : main
      return { label, confidence: 0.68 + rand() * 0.3, top: [] }
    },
    dynamic: () => ({ label: MOTION_LABELS[Math.floor(rand() * MOTION_LABELS.length)], confidence: 0.55 + rand() * 0.44, top: [] }),
    motion: (frames, mirror) => motionEnergy(normalizeSequence(frames, 30, { mirror })),
    motionThreshold: META_MOTION_THRESHOLD,
  })
  const session = new SentenceSession()
  const commits: Array<{ label: string; kind: SignKind; frame: number }> = []
  const lives: string[] = []
  for (index = 0; index < stream.frames.length; index++) {
    const { commit, live } = engine.push(stream.frames[index], { mode, mirror: false })
    lives.push(live ? live.label : "-")
    if (commit) {
      commits.push({ ...commit, frame: index })
      if (commit.kind === "letter") session.confirmLetter(commit.label)
      else session.confirmWord(commit.label)
    }
  }
  return { commits, session, lives }
}

function sentenceStream(seed: number, handDrops: boolean, holdFrames = 60): Tagged {
  const rand = rng(seed * 7919)
  const offsets: Array<[number, number]> = [
    [(rand() - 0.5) * 0.1, (rand() - 0.5) * 0.06],
    [(rand() - 0.5) * 0.1, (rand() - 0.5) * 0.06],
    [(rand() - 0.5) * 0.1, (rand() - 0.5) * 0.06],
  ]
  const me = hold("ME", holdFrames, rand, offsets[0])
  const want = hold("WANT", holdFrames, rand, offsets[1])
  const water = hold("WATER", holdFrames, rand, offsets[2])
  const gap = () => 10 + Math.floor(rand() * 12)
  if (handDrops) return concat(drop(5), me, drop(gap() + 5), want, drop(gap() + 5), water, drop(15))
  return concat(
    drop(5),
    me,
    transition(lastPose(me), firstPose(want), gap(), rand),
    want,
    transition(lastPose(want), firstPose(water), gap(), rand),
    water,
    drop(15),
  )
}

describe("live recognizer: one held sign adds one word", () => {
  for (const handDrops of [true, false]) {
    it(`ME WANT WATER held ~2 s each, ${handDrops ? "dropping the hand" : "moving straight"} between signs, 40 random runs`, () => {
      for (let seed = 1; seed <= 40; seed++) {
        const stream = sentenceStream(seed, handDrops)
        const { session, commits } = run(stream, rng(seed))
        expect(session.glosses, `seed ${seed}: ${JSON.stringify(commits)}`).toEqual(["ME", "WANT", "WATER"])
        expect(session.previewEnglish()).toBe("I want water.")
      }
    })
  }

  it("a long 4 s hold of each sign still adds each word once", () => {
    for (let seed = 1; seed <= 15; seed++) {
      for (const handDrops of [true, false]) {
        const { session, commits } = run(sentenceStream(seed, handDrops, 120), rng(seed + 100))
        expect(session.glosses, `seed ${seed} drops=${handDrops}: ${JSON.stringify(commits)}`).toEqual(["ME", "WANT", "WATER"])
      }
    }
  })

  it("feels responsive: each word lands within about a second of the sign starting", () => {
    for (const handDrops of [true, false]) {
      const stream = sentenceStream(3, handDrops)
      const { commits } = run(stream, rng(3))
      // First frame of each sign block (letters tag changes from one sign to the next).
      const signStarts: number[] = []
      stream.letters.forEach((tag, i) => {
        if (tag && tag !== stream.letters[i - 1]) signStarts.push(i)
      })
      expect(commits.map((c) => c.label)).toEqual(["ME", "WANT", "WATER"])
      commits.forEach((c, i) => expect(c.frame - signStarts[i], `${c.label} drops=${handDrops}`).toBeLessThanOrEqual(30))
    }
  })

  it("the same sign twice needs the hand to drop in between", () => {
    const rand = rng(9)
    const once = run(concat(hold("WATER", 90, rand, [0, 0]), drop(15)), rng(9))
    expect(once.session.glosses).toEqual(["WATER"])
    const rand2 = rng(10)
    const twice = run(concat(hold("WATER", 50, rand2, [0, 0]), drop(15), hold("WATER", 50, rand2, [0, 0])), rng(10))
    expect(twice.session.glosses).toEqual(["WATER", "WATER"])
  })
})

describe("live recognizer: fingerspelling", () => {
  it("spells C-S-B letter by letter with the hand dropping between letters", () => {
    for (let seed = 1; seed <= 20; seed++) {
      const rand = rng(seed * 31)
      const stream = concat(hold("C", 45, rand, [0, 0]), drop(12), hold("S", 45, rand, [0, 0]), drop(12), hold("B", 45, rand, [0, 0]), drop(12))
      const { commits, session } = run(stream, rng(seed), "letters")
      expect(commits.map((c) => c.label), `seed ${seed} ${JSON.stringify(commits)}`).toEqual(["C", "S", "B"])
      expect(session.spelling).toBe("CSB")
    }
  })

  it("spells letters in auto mode too, moving straight from one letter to the next", () => {
    for (let seed = 1; seed <= 20; seed++) {
      const rand = rng(seed * 37)
      const c = hold("C", 40, rand, [0, 0])
      const a = hold("S", 40, rand, [0, 0])
      const b = hold("B", 40, rand, [0, 0])
      const stream = concat(c, transition(lastPose(c), firstPose(a), 8, rand), a, transition(lastPose(a), firstPose(b), 8, rand), b, drop(12))
      const { commits, lives } = run(stream, rng(seed), "auto")
      expect(commits.map((x) => x.label), `seed ${seed} ${JSON.stringify(commits)} ${lives.slice(70).join(",")}`).toEqual(["C", "S", "B"])
      expect(commits.every((x) => x.kind === "letter")).toBe(true)
    }
  })

  it("a held letter is added once, even for 3 s", () => {
    const rand = rng(77)
    const { commits } = run(concat(hold("S", 90, rand, [0, 0]), drop(12)), rng(77), "letters")
    expect(commits.map((c) => c.label)).toEqual(["S"])
  })
})

describe("word rules on live-like landmarks", () => {
  it("WANT is read from the hand shrinking as it pulls back, not from wrist z (always 0 in MediaPipe)", () => {
    const open = posedHand({ index: 1, middle: 1, ring: 1, pinky: 1, thumb: 1 })
    const frames = Array.from({ length: 14 }, (_, i) => scaled(open, 1 - 0.25 * Math.min(1, i / 10)))
    expect(frames.every((f) => f[0][2] === 0)).toBe(true)
    expect(recognizeWord(frames)?.id).toBe("WANT")
    const push = Array.from({ length: 14 }, (_, i) => scaled(open, 1 + 0.3 * Math.min(1, i / 10)))
    expect(recognizeWord(push)?.id).toBe("STOP")
    expect(recognizeWord(Array.from({ length: 14 }, () => open))).toBeNull()
  })

  it("motion from the previous sign does not leak into the next one", () => {
    // ME held low, then the open hand appears higher up: that move must not read as THANKS.
    const me = Array.from({ length: 30 }, () => demoSequence("ME")[0])
    const open = posedHand({ index: 1, middle: 1, ring: 1, pinky: 1, thumb: 1 })
    const later = Array.from({ length: 10 }, () => moved(open, 0, 0.12))
    expect(recognizeWord([...me, ...later])).toBeNull()
  })
})

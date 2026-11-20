import { squareFrame, type Point } from "./hands"
import { performTwoHands } from "./handModel"
import { DEMO_ASPECT, TWO_HAND_WORDS, WORD_SIGNS, demoSquare } from "./words"

/** How to make each still letter (one hand, palm toward the camera unless noted). */
export const LETTER_HOWTO: Record<string, string> = {
  A: "Fist with the thumb resting against the side of the index finger.",
  B: "Four fingers straight up and together, thumb folded across the palm.",
  C: "Curve all fingers and the thumb into a C shape, as if holding a cup.",
  D: "Index finger up; the other fingertips touch the thumb in a circle.",
  E: "Fingers bent down over the thumb, which is tucked across the palm.",
  F: "Thumb and index fingertips touch in a circle; the other three fingers stand up.",
  G: "Index finger and thumb point sideways, parallel, the rest closed.",
  H: "Index and middle fingers point sideways together, the rest closed.",
  I: "Little finger straight up, the rest in a fist.",
  K: "Index and middle up in a V with the thumb touching between them.",
  L: "Index finger up and thumb out to the side, making an L.",
  M: "Thumb tucked under the first three fingers, which fold over it.",
  N: "Thumb tucked under the index and middle fingers.",
  O: "All fingertips meet the thumb tip in a round O.",
  P: "Like K, but pointing down.",
  Q: "Like G, but pointing down.",
  R: "Index and middle fingers crossed.",
  S: "Fist with the thumb wrapped across the front of the fingers.",
  T: "Thumb tucked between the index and middle fingers of a fist.",
  U: "Index and middle fingers straight up and together.",
  V: "Index and middle fingers up and spread in a V.",
  W: "Index, middle and ring fingers up and spread.",
  X: "Index finger hooked like a hook, the rest closed.",
  Y: "Thumb and little finger out, the rest closed.",
  J: "Little finger up, then draw a J in the air (motion letter).",
  Z: "Index finger draws a Z in the air (motion letter).",
}

export const STILL_LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("")

export interface SignFigure {
  frames: Point[][]
  second?: Point[][]
}

function flipX(frame: Point[]): Point[] {
  return frame.map((p) => [-p[0], p[1], p[2]] as Point)
}

/**
 * Frames for a word's reference figure, as the signer sees themself in the mirrored camera view.
 * The built-in hand is a right hand seen by the camera, so it is flipped to the selfie view.
 */
export function wordFigure(id: string): SignFigure {
  if (TWO_HAND_WORDS.has(id)) {
    const { main, second } = performTwoHands(id as "HELP" | "SCHOOL", { rand: () => 0.5, jitter: 0, aspect: DEMO_ASPECT })
    return { frames: main.map((f) => flipX(squareFrame(f, DEMO_ASPECT))), second: second.map((f) => flipX(squareFrame(f, DEMO_ASPECT))) }
  }
  return { frames: demoSquare(id).map(flipX) }
}

/** The recorded row used as each letter's reference and example (checked in src/test/examples.test.ts). */
export function exampleRow(rows: number[][] | undefined): number[] | undefined {
  return rows?.[1] ?? rows?.[0]
}

/** A still letter from a recorded landmark row (stored as the model sees it, which is already the selfie view). */
export function letterFigure(row: number[]): SignFigure {
  const frame: Point[] = []
  const stride = row.length >= 63 ? 3 : 2
  for (let i = 0; i < 21; i++) frame.push([row[i * stride], row[i * stride + 1], 0])
  return { frames: [frame] }
}

const SYNONYMS: Record<string, string> = {
  I: "ME", ME: "ME", MY: "ME", MINE: "ME", MYSELF: "ME",
  YOU: "YOU", YOUR: "YOU", YOURS: "YOU", U: "YOU",
  WE: "WE", US: "WE", OUR: "WE",
  HI: "HELLO", HELLO: "HELLO", HEY: "HELLO",
  THANKS: "THANKS", THANK: "THANKS", THANKYOU: "THANKS",
  PLEASE: "PLEASE", SORRY: "SORRY", YES: "YES", NO: "NO",
  LOVE: "LOVE", STOP: "STOP", WATER: "WATER", MORE: "MORE", DRINK: "DRINK", EAT: "EAT", FOOD: "EAT",
  GO: "GO", COME: "COME", GOOD: "GOOD", BAD: "BAD", LIKE: "LIKE", WANT: "WANT", NEED: "NEED",
  WHERE: "WHERE", WHAT: "WHAT", NAME: "NAME", FINE: "FINE", OK: "FINE", OKAY: "FINE",
  HELP: "HELP", SCHOOL: "SCHOOL", HOME: "HOME", HOUSE: "HOME",
}

export interface SignCard {
  /** The text the card stands for. */
  text: string
  /** A word sign id, or a single letter. */
  sign: string
  kind: "word" | "letter"
}

/** Turn typed text into sign cards: known words become word signs, anything else is fingerspelled. */
export function textToSigns(text: string): SignCard[][] {
  const words = text
    .toUpperCase()
    .replace(/THANK\s+YOU/g, "THANKYOU")
    .split(/[^A-Z]+/)
    .filter(Boolean)
  const known = new Set(WORD_SIGNS.map((s) => s.id))
  return words.map((word) => {
    const id = SYNONYMS[word]
    if (id && known.has(id)) return [{ text: word === "THANKYOU" ? "THANK YOU" : word, sign: id, kind: "word" as const }]
    return word.split("").map((letter) => ({ text: letter, sign: letter, kind: "letter" as const }))
  })
}

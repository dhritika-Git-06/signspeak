/** Offline sentence builder: hold-to-confirm, fingerspelling, and gloss-to-English rules. */

export const DICTIONARY = [
  "WATER", "FOOD", "HELP", "NAME", "HOME", "SCHOOL", "GOOD", "BAD", "LOVE", "LIKE",
  "WANT", "NEED", "DRINK", "EAT", "COME", "STOP", "FINE", "HELLO", "THANKS", "PLEASE",
  "SORRY", "WHAT", "WHERE", "HOW", "MORE", "YOU", "HELLO", "YES", "NO", "GO", "ME",
  "CAT", "DOG", "MOM", "DAD", "HI", "OK", "BUS", "BOOK",
]

const GLOSS_TO_ENGLISH: Record<string, string> = {
  ME: "I",
  I: "I",
  YOU: "you",
  WE: "we",
  "HE/SHE": "they",
  WANT: "want",
  NEED: "need",
  HELP: "help",
  EAT: "eat",
  DRINK: "drink",
  WATER: "water",
  FOOD: "food",
  GO: "go",
  COME: "come",
  HOME: "home",
  SCHOOL: "school",
  GOOD: "good",
  BAD: "bad",
  LIKE: "like",
  LOVE: "love",
  NAME: "name",
  WHAT: "what",
  WHERE: "where",
  HOW: "how",
  MORE: "more",
  STOP: "stop",
  FINE: "fine",
  HELLO: "hello",
  YES: "yes",
  NO: "no",
  THANKS: "thank you",
  PLEASE: "please",
  SORRY: "sorry",
}

export function levenshtein(a: string, b: string): number {
  const row = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    let prev = i - 1
    row[0] = i
    for (let j = 1; j <= b.length; j++) {
      const next = a[i - 1] === b[j - 1] ? prev : Math.min(prev, row[j - 1], row[j]) + 1
      prev = row[j]
      row[j] = next
    }
  }
  return row[b.length]
}

export function autocorrect(spelling: string, dictionary: string[] = DICTIONARY): string {
  const upper = spelling.toUpperCase().replace(/[^A-Z]/g, "")
  if (!upper) return ""
  if (dictionary.includes(upper)) return upper
  if (upper.length < 3) return upper
  let best = upper
  let bestDistance = 2
  for (const word of dictionary) {
    if (Math.abs(word.length - upper.length) > 1) continue
    const distance = levenshtein(upper, word)
    if (distance < bestDistance) {
      bestDistance = distance
      best = word
    }
  }
  return bestDistance <= 1 ? best : upper
}

export function toEnglish(glosses: string[]): string {
  const g = glosses.map((item) => item.toUpperCase()).filter(Boolean)
  if (!g.length) return ""
  if (g[0] === "ME" && g[1] === "NAME" && g[2]) {
    return `My name is ${titleWord(g.slice(2).join(" "))}.`
  }
  if (g.includes("WHAT") && g.includes("NAME")) return "What is your name?"
  if (g.includes("WHERE") && (g.includes("HOME") || g.includes("YOU"))) return "Where is home?"
  if (g.includes("HOW") && g.includes("YOU")) return "How are you?"
  const words = g.map((gloss) => GLOSS_TO_ENGLISH[gloss] ?? gloss.toLowerCase())
  let sentence = words.join(" ")
  sentence = sentence.charAt(0).toUpperCase() + sentence.slice(1)
  const question = g.some((gloss) => gloss === "WHAT" || gloss === "WHERE" || gloss === "HOW")
  if (!/[.?!]$/.test(sentence)) sentence += question ? "?" : "."
  return sentence
}

function titleWord(value: string): string {
  return value
    .toLowerCase()
    .split(/\s+/)
    .map((part) => (part ? part.charAt(0).toUpperCase() + part.slice(1) : part))
    .join(" ")
}

export function suggestionsFor(glosses: string[]): string[] {
  const last = glosses.at(-1)?.toUpperCase()
  if (!last) return ["HELLO", "ME", "YOU"]
  if (last === "ME" || last === "I") return ["WANT", "NEED", "NAME"]
  if (last === "WANT" || last === "NEED") return ["WATER", "FOOD", "HELP"]
  if (last === "YOU") return ["NAME", "GOOD", "COME"]
  if (last === "HELLO") return ["ME", "PLEASE", "THANKS"]
  return ["PLEASE", "THANKS", "STOP"]
}

export class HoldConfirmer {
  private label: string | null = null
  private since = 0
  private count = 0
  private locked = false
  private holdMs: number
  private minConfidence: number
  private minFrames: number

  constructor(holdMs = 650, minConfidence = 0.55, minFrames = 8) {
    this.holdMs = holdMs
    this.minConfidence = minConfidence
    this.minFrames = minFrames
  }

  reset() {
    this.label = null
    this.since = 0
    this.count = 0
    this.locked = false
  }

  /** Confirms after the label holds for holdMs, or for minFrames in a row. */
  push(label: string | null, confidence: number, now: number): string | null {
    if (!label || confidence < this.minConfidence) {
      this.label = null
      this.count = 0
      this.locked = false
      return null
    }
    if (this.locked && label === this.label) return null
    if (label !== this.label) {
      this.label = label
      this.since = now
      this.count = 1
      this.locked = false
      return null
    }
    this.count += 1
    if (now - this.since >= this.holdMs || this.count >= this.minFrames) {
      this.locked = true
      return label
    }
    return null
  }

  /** 0 to 1 progress toward confirmation, for the hold ring. */
  progress(now: number): number {
    if (!this.label || this.locked) return this.locked ? 1 : 0
    return Math.max(0, Math.min(1, (now - this.since) / this.holdMs))
  }
}

export class SentenceSession {
  glosses: string[] = []
  spelling = ""
  history: string[] = []

  confirmLetter(letter: string) {
    const clean = letter.toUpperCase().replace(/[^A-Z]/g, "")
    if (clean.length === 1) this.spelling += clean
  }

  confirmWord(word: string) {
    this.flushSpelling()
    this.glosses.push(word.toUpperCase())
  }

  flushSpelling() {
    if (!this.spelling) return
    this.glosses.push(autocorrect(this.spelling))
    this.spelling = ""
  }

  space() {
    this.flushSpelling()
  }

  delete() {
    if (this.spelling) this.spelling = this.spelling.slice(0, -1)
    else this.glosses.pop()
  }

  clear() {
    this.glosses = []
    this.spelling = ""
  }

  /** Chips already confirmed, plus the letters currently being spelled. */
  chips(): string[] {
    return this.spelling ? [...this.glosses, this.spelling] : [...this.glosses]
  }

  /** Confirmed glosses plus the word being spelled, autocorrected. */
  pendingGlosses(): string[] {
    return this.spelling ? [...this.glosses, autocorrect(this.spelling)] : [...this.glosses]
  }

  previewEnglish(): string {
    return toEnglish(this.pendingGlosses())
  }

  finish(): string {
    this.flushSpelling()
    const sentence = toEnglish(this.glosses)
    if (sentence) this.history = [sentence, ...this.history].slice(0, 8)
    this.glosses = []
    return sentence
  }

  suggestions(): string[] {
    return suggestionsFor(this.glosses)
  }
}

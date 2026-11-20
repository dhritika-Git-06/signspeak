import { describe, expect, it } from "vitest"
import { textToSigns, wordFigure } from "./signLibrary"

describe("text to sign", () => {
  it("uses word signs for known words and fingerspells the rest", () => {
    const cards = textToSigns("Hello, my name is Ria. Thank you!")
    expect(cards.map((w) => w.map((c) => c.sign).join(""))).toEqual(["HELLO", "ME", "NAME", "IS", "RIA", "THANKS"])
    expect(cards[3].every((c) => c.kind === "letter")).toBe(true)
    expect(cards[0][0].kind).toBe("word")
  })
  it("has reference frames for one- and two-hand words", () => {
    expect(wordFigure("WATER").frames.length).toBeGreaterThan(10)
    const help = wordFigure("HELP")
    expect(help.second?.length).toBe(help.frames.length)
  })
})

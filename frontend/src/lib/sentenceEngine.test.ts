import { describe, expect, it } from "vitest"
import { HoldConfirmer, SentenceSession, autocorrect, toEnglish } from "./sentenceEngine"

describe("sentence engine", () => {
  it("turns a sign gloss into a spoken English sentence", () => {
    expect(toEnglish(["ME", "WANT", "WATER"])).toBe("I want water.")
    expect(toEnglish(["ME", "NAME", "CAT"])).toBe("My name is Cat.")
    expect(toEnglish(["WHAT", "YOU", "NAME"])).toBe("What is your name?")
  })

  it("corrects a near-miss fingerspelling against the dictionary", () => {
    expect(autocorrect("WATR")).toBe("WATER")
    expect(autocorrect("CAT")).toBe("CAT")
    expect(autocorrect("ZZ")).toBe("ZZ")
  })

  it("holds a prediction until the timer elapses, then locks it", () => {
    const hold = new HoldConfirmer(600, 0.5)
    expect(hold.push("ME", 0.9, 0)).toBeNull()
    expect(hold.push("ME", 0.9, 400)).toBeNull()
    expect(hold.push("ME", 0.91, 600)).toBe("ME")
    expect(hold.push("ME", 0.95, 900)).toBeNull()
    expect(hold.push(null, 0, 1200)).toBeNull()
    expect(hold.push("ME", 0.9, 1300)).toBeNull()
    expect(hold.push("ME", 0.9, 1900)).toBe("ME")
  })

  it("merges letters, then speaks the corrected sentence", () => {
    const session = new SentenceSession()
    session.confirmWord("ME")
    session.confirmWord("WANT")
    for (const letter of "WATR") session.confirmLetter(letter)
    session.flushSpelling()
    expect(session.glosses).toEqual(["ME", "WANT", "WATER"])
    expect(session.previewEnglish()).toBe("I want water.")
    expect(session.suggestions()).toEqual(["PLEASE", "THANKS", "STOP"])
    const spoken = session.finish()
    expect(spoken).toBe("I want water.")
    expect(session.history[0]).toBe("I want water.")
    expect(session.glosses).toEqual([])
  })
})

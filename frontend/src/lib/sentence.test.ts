import { describe, expect, it } from "vitest"
import { addSpace, applyCommit, deleteLast, render, type Token } from "./sentence"

describe("sentence builder", () => {
  it("appends letters and inserts words with spaces", () => {
    let tokens: Token[] = []
    tokens = applyCommit(tokens, "C", "letter")
    tokens = applyCommit(tokens, "A", "letter")
    tokens = applyCommit(tokens, "T", "letter")
    expect(render(tokens)).toBe("CAT")
    tokens = applyCommit(tokens, "HELLO", "word")
    expect(render(tokens)).toBe("CAT HELLO ")
  })

  it("deletes a word token, then a space, then a letter", () => {
    let tokens: Token[] = []
    for (const letter of ["C", "A", "T"]) tokens = applyCommit(tokens, letter, "letter")
    tokens = addSpace(tokens)
    tokens = applyCommit(tokens, "HELLO", "word")
    tokens = deleteLast(tokens)
    expect(render(tokens)).toBe("CAT ")
    tokens = deleteLast(tokens)
    expect(render(tokens)).toBe("CAT")
    tokens = deleteLast(tokens)
    expect(render(tokens)).toBe("CA")
  })

  it("does not stack spaces", () => {
    let tokens = applyCommit([], "C", "letter")
    tokens = addSpace(tokens)
    tokens = addSpace(tokens)
    expect(render(tokens)).toBe("C ")
  })
})

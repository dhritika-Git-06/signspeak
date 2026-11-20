import { describe, expect, it } from "vitest"
import { PRESENTATION_SCRIPT, demoSquare, recognizeWord } from "./words"

describe("rule-based word signs", () => {
  it("recognises the presentation sequence me, want, water", () => {
    for (const id of PRESENTATION_SCRIPT) {
      expect(recognizeWord(demoSquare(id))?.id, id).toBe(id)
    }
  })

  it("recognises a wave as hello and a fist nod as yes", () => {
    expect(recognizeWord(demoSquare("HELLO"))?.id).toBe("HELLO")
    expect(recognizeWord(demoSquare("YES"))?.id).toBe("YES")
    expect(recognizeWord(demoSquare("LOVE"))?.id).toBe("LOVE")
    expect(recognizeWord(demoSquare("FINE"))?.id).toBe("FINE")
  })
})

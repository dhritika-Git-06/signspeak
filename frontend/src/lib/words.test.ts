import { describe, expect, it } from "vitest"
import { PRESENTATION_SCRIPT, demoSequence, recognizeWord } from "./words"

describe("rule-based word signs", () => {
  it("recognises the presentation sequence me, want, water", () => {
    for (const id of PRESENTATION_SCRIPT) {
      expect(recognizeWord(demoSequence(id))?.id, id).toBe(id)
    }
  })

  it("recognises a wave as hello and a fist nod as yes", () => {
    expect(recognizeWord(demoSequence("HELLO"))?.id).toBe("HELLO")
    expect(recognizeWord(demoSequence("YES"))?.id).toBe("YES")
    expect(recognizeWord(demoSequence("LOVE"))?.id).toBe("LOVE")
    expect(recognizeWord(demoSequence("FINE"))?.id).toBe("FINE")
  })
})

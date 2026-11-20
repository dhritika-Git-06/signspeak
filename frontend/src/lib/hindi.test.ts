import { describe, expect, it } from "vitest"
import { toHindi } from "./hindi"

describe("offline Hindi", () => {
  it("handles the presentation sentence and common patterns", () => {
    expect(toHindi(["ME", "WANT", "WATER"])).toBe("मुझे पानी चाहिए।")
    expect(toHindi(["ME", "NAME", "DHRITIKA"])).toBe("मेरा नाम Dhritika है।")
    expect(toHindi(["WHAT", "NAME"])).toBe("आपका नाम क्या है?")
    expect(toHindi(["HELLO"])).toBe("नमस्ते।")
    expect(toHindi(["HELLO", "ME", "NEED", "HELP"])).toBe("नमस्ते। मुझे मदद की ज़रूरत है।")
    expect(toHindi(["ME", "FINE"])).toBe("मैं ठीक हूँ।")
    expect(toHindi(["ME", "GO", "HOME"])).toBe("मुझे घर जाना है।")
    expect(toHindi(["THANKS"])).toBe("धन्यवाद।")
    expect(toHindi([])).toBe("")
  })
})

import type { Point } from "./hands"

export interface Framing {
  tone: "ok" | "warn"
  text: string
}

/**
 * Live framing advice from the raw hand landmarks (fractions of the image) and the mean image
 * brightness (0..255), plus how long (ms) no hand has been seen. Returns null when there is nothing useful to say.
 */
export function framingHint(raw: Point[] | null, brightness: number | null, missingMs: number): Framing | null {
  if (brightness != null && brightness < 55) return { tone: "warn", text: "Too dark. Face a window or turn on a light." }
  if (!raw) {
    return missingMs > 2000 ? { tone: "warn", text: "No hand found. Show one hand, palm toward the camera." } : null
  }
  const xs = raw.map((p) => p[0])
  const ys = raw.map((p) => p[1])
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  if (minX < 0.015 || maxX > 0.985 || minY < 0.015 || maxY > 0.985) return { tone: "warn", text: "Keep your whole hand inside the frame." }
  const height = maxY - minY
  const width = maxX - minX
  const size = Math.max(height, width)
  if (size < 0.17) return { tone: "warn", text: "Too far. Move your hand closer to the camera." }
  if (size > 0.8) return { tone: "warn", text: "Too close. Move your hand back a little." }
  if (brightness != null && brightness > 235) return { tone: "warn", text: "Very bright. Avoid a window or lamp right behind you." }
  return { tone: "ok", text: "Good framing." }
}

export const ACCURACY_TIPS = [
  "Light your hand from the front. A window behind you makes the hand a dark shape.",
  "Keep your hand about 40–70 cm from the camera, so it fills roughly a third of the picture.",
  "Use a plain background if you can, and wear sleeves that contrast with your skin.",
  "Hold each letter still for about a second. Hold a still word (like WATER) for half a second.",
  "Make motion words clearly: wave for HELLO, nod your fist for YES, pull your open hand toward you for WANT.",
  "Drop your hand out of view between repeated signs, so the same word can be added again.",
  "Use Letters mode to spell names. Auto reads W, O, F and X as the words WATER, MORE, FINE and NEED.",
  "For HELP and SCHOOL keep both hands in view, close together.",
]

let audio: HTMLAudioElement | null = null
// Chrome can drop an utterance that is garbage-collected before it finishes, so keep a reference.
let current: SpeechSynthesisUtterance | null = null

function hasSpeech(): boolean {
  return typeof window !== "undefined" && "speechSynthesis" in window && typeof SpeechSynthesisUtterance !== "undefined"
}

/** Voices often arrive a moment after page load. Wait briefly for them instead of speaking with none. */
function voicesReady(timeoutMs = 1200): Promise<SpeechSynthesisVoice[]> {
  const synth = window.speechSynthesis
  const now = synth.getVoices()
  if (now.length) return Promise.resolve(now)
  return new Promise((resolve) => {
    const done = () => {
      synth.removeEventListener?.("voiceschanged", done)
      window.clearTimeout(timer)
      resolve(synth.getVoices())
    }
    const timer = window.setTimeout(done, timeoutMs)
    synth.addEventListener?.("voiceschanged", done)
  })
}

function pickVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  return (
    voices.find((v) => v.lang === "en-US" && v.name.includes("Google")) ??
    voices.find((v) => v.lang?.replace("_", "-") === "en-US") ??
    voices.find((v) => v.lang?.startsWith("en") && v.localService) ??
    voices.find((v) => v.lang?.startsWith("en")) ??
    null
  )
}

export async function speakBrowser(text: string): Promise<boolean> {
  if (!text.trim() || !hasSpeech()) return false
  const synth = window.speechSynthesis
  const voices = await voicesReady()
  const wasBusy = synth.speaking || synth.pending
  synth.cancel()
  // Speaking straight after cancel() is sometimes ignored by Chrome.
  if (wasBusy) await new Promise((resolve) => window.setTimeout(resolve, 60))
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = 0.95
  const voice = pickVoice(voices)
  try {
    if (voice) utterance.voice = voice
  } catch {
    /* an odd voice object should not stop the line from being spoken */
  }
  utterance.lang = voice?.lang ?? "en-US"
  utterance.onend = () => {
    if (current === utterance) current = null
  }
  current = utterance
  synth.resume()
  synth.speak(utterance)
  return true
}

export async function speak(text: string, apiBase: string | null): Promise<"gtts" | "browser" | "silent"> {
  const cleaned = text.trim()
  if (!cleaned) return "silent"
  if (apiBase) {
    try {
      const res = await fetch(`${apiBase}/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: cleaned, lang: "en" }),
      })
      if (res.ok) {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        if (audio) {
          audio.pause()
          URL.revokeObjectURL(audio.src)
        }
        audio = new Audio(url)
        await audio.play()
        return "gtts"
      }
    } catch {
      /* fall through to the browser voice */
    }
  }
  return (await speakBrowser(cleaned)) ? "browser" : "silent"
}

export function stopSpeech() {
  if (audio) audio.pause()
  if (hasSpeech()) window.speechSynthesis.cancel()
}

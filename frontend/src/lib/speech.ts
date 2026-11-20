let audio: HTMLAudioElement | null = null

export function speakBrowser(text: string) {
  if (!text.trim() || typeof window === "undefined" || !window.speechSynthesis) return
  window.speechSynthesis.cancel()
  const utterance = new SpeechSynthesisUtterance(text)
  utterance.rate = 0.95
  utterance.lang = "en-US"
  window.speechSynthesis.speak(utterance)
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
  speakBrowser(cleaned)
  return "browser"
}

export function stopSpeech() {
  if (audio) audio.pause()
  window.speechSynthesis?.cancel()
}

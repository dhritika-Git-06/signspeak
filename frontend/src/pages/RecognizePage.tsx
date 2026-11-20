import { useEffect, useRef, useState } from "react"
import { Badge, Button, SliderControl, SwitchControl } from "@/components/ui"
import { useHandTracker } from "@/hooks/useHandTracker"
import type { Point } from "@/lib/hands"
import {
  embedDynamic,
  embedStatic,
  predictDynamic,
  predictHead,
  predictStatic,
  type Ranked,
} from "@/lib/model"
import {
  flattenSequence,
  motionEnergy,
  normalizeLandmarks,
  normalizeSequence,
} from "@/lib/normalize"
import { asset } from "@/lib/asset"
import { SentenceSession } from "@/lib/sentenceEngine"
import { hasHindiVoice, speak } from "@/lib/speech"
import { MOVING, Recognizer, recentMotion } from "@/lib/recognizer"
import { toHindi } from "@/lib/hindi"
import { ACCURACY_TIPS, framingHint, type Framing } from "@/lib/hints"
import { SegmentReader, encodeSegment, loadAslWords, predictAslWords, type AslWordsModel } from "@/lib/aslWords"
import { transcriptText, downloadText } from "@/lib/transcript"
import { PRESENTATION_SCRIPT, WORD_SIGNS, demoSequence, demoSquare, recognizeWord } from "@/lib/words"
import { useModels } from "@/models-context"

interface Live {
  label: string
  confidence: number
  top: Ranked[]
  kind: "letter" | "word"
}

const LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("")

type Mode = "auto" | "letters" | "words" | "asl"

const MODE_LABELS: Record<Mode, string> = { auto: "Auto", letters: "Letters", words: "Words", asl: "ASL beta" }

const MODE_NOTES: Record<Mode, string> = {
  asl: "Beta: 14 real ASL words (HELLO, YES, NO, THANKS, PLEASE, WATER, DRINK, GO, HOME, LIKE, FINE, BAD, WHERE, WE) from a model trained on real signers. Make one sign, then hold still or lower your hand. Letters are off.",
  auto: "Words and letters. W, O, F and X look the same as WATER, MORE, FINE and NEED, so Auto reads them as words. Use Letters to spell with them.",
  letters: "All 24 still letters (plus J and Z by motion). Word signs are ignored.",
  words: "Word signs only. Letters are ignored.",
}

export function RecognizePage() {
  const models = useModels()
  const tracker = useHandTracker()
  const [live, setLive] = useState<Live | null>(null)
  const [sentenceRev, setSentenceRev] = useState(0)
  const session = useRef(new SentenceSession())
  const [hand, setHand] = useState<string | null>(null)
  const [mode, setMode] = useState<Mode>("auto")
  const [lang, setLang] = useState<"en" | "hi">("en")
  const [tipsOpen, setTipsOpen] = useState(false)
  const [framing, setFraming] = useState<Framing | null>(null)
  const [aslModel, setAslModel] = useState<AslWordsModel | null>(null)
  const [transcript, setTranscript] = useState<Array<{ at: Date; english: string; hindi: string }>>([])
  const segment = useRef(new SegmentReader())
  const aslFrames = useRef<Point[][]>([])
  const handSeenAt = useRef(0)
  const [mirror, setMirror] = useState<"auto" | "on" | "off">("auto")
  const [threshold, setThreshold] = useState(0.62)
  const [autoSpeak, setAutoSpeak] = useState(false)
  const [voice, setVoice] = useState<"browser" | "gtts" | "silent" | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [showingSample, setShowingSample] = useState(false)
  const recognizer = useRef(
    new Recognizer({
      letter: () => ({ label: "", confidence: 0, top: [] }),
      dynamic: () => null,
      motion: () => 0,
      motionThreshold: 1,
    }),
  )
  const [presenting, setPresenting] = useState(false)
  const replayTimer = useRef(0)

  const runtime = useRef({
    models,
    mode,
    mirror,
    threshold,
    autoSpeak,
    sentenceRev,
  })
  useEffect(() => {
    runtime.current = { models, mode, mirror, threshold, autoSpeak, sentenceRev }
  })

  function bump() {
    setSentenceRev((value) => value + 1)
  }

  function commit(label: string, kind: "letter" | "word") {
    if (kind === "word") session.current.confirmWord(label)
    else session.current.confirmLetter(label)
    bump()
    if (runtime.current.autoSpeak) {
      void speak(session.current.previewEnglish(), runtime.current.models.apiBase).then(setVoice)
    }
  }

  /** ASL beta mode: read one finished movement at a time with the PopSign-trained model. */
  function classifyAsl(frameLandmarks: Point[] | null, shouldMirror: boolean) {
    const model = aslModelRef.current
    if (frameLandmarks) {
      aslFrames.current.push(frameLandmarks)
      if (aslFrames.current.length > 20) aslFrames.current.shift()
    } else aslFrames.current = []
    const moving = frameLandmarks ? recentMotion(aslFrames.current) > MOVING : false
    const done = segment.current.push(frameLandmarks, moving)
    if (!done || !model) {
      if (moving) setLive({ label: "…", confidence: 0, top: [], kind: "word" })
      return
    }
    const reading = predictAslWords(model, encodeSegment(done, shouldMirror, null, model.steps))
    setLive({ ...reading, kind: "word" })
    if (reading.confidence >= Math.min(runtime.current.threshold, 0.6)) commit(reading.label, "word")
    else setNotice(`Not sure (${reading.label} ${Math.round(reading.confidence * 100)}%). Try the sign again.`)
  }

  function classify(frameLandmarks: Point[] | null, handedness: string | null, sampleKind?: "letter" | "word", second?: Point[] | null) {
    const { models: bundle, mode: currentMode, mirror: mirrorMode, threshold: minConf } = runtime.current
    if (currentMode === "asl" && !sampleKind) {
      setHand(frameLandmarks ? handedness ?? "Hand" : null)
      classifyAsl(frameLandmarks, mirrorMode === "on" || (mirrorMode === "auto" && handedness === "Left"))
      return
    }
    if (!bundle.meta || !bundle.classifier || !bundle.dynamic) {
      return
    }
    const meta = bundle.meta
    const shouldMirror = mirrorMode === "on" || (mirrorMode === "auto" && handedness === "Left")
    const zeroZ = meta.z_mode === "zero"
    setHand(frameLandmarks ? handedness ?? "Hand" : null)
    const engine = recognizer.current
    engine.setThreshold(minConf)
    engine.deps.motionThreshold = meta.motion_threshold
    engine.deps.letter = (frame, mirror) => {
      const norm = normalizeLandmarks(frame, { mirror, zeroZ })
      const base = predictStatic(bundle.classifier!, norm, meta.static_labels)
      return overrideCustom("static", () => embedStatic(bundle.embed!, norm), base, bundle)
    }
    engine.deps.dynamic = (frames, mirror, kind) => {
      const seq =
        kind === "word"
          ? normalizeSequence(frames.slice(-meta.sequence_length), meta.sequence_length, { mirror, zeroZ })
          : normalizeSequence(frames, meta.sequence_length, { mirror, zeroZ })
      const flat = flattenSequence(seq)
      const base = predictDynamic(bundle.dynamic!, flat, meta.dynamic_labels)
      return overrideCustom("dynamic", () => embedDynamic(bundle.dynamicEmbed!, flat), base, bundle)
    }
    engine.deps.motion = (frames, mirror) => motionEnergy(normalizeSequence(frames, meta.sequence_length, { mirror, zeroZ }))
    const engineMode = currentMode === "asl" ? "auto" : currentMode
    const result = engine.push(frameLandmarks, { mode: engineMode, mirror: shouldMirror, sampleKind, second })
    setLive(result.live)
    if (result.commit) commit(result.commit.label, result.commit.kind)
  }

  const aslModelRef = useRef<AslWordsModel | null>(null)
  useEffect(() => {
    aslModelRef.current = aslModel
  }, [aslModel])
  useEffect(() => {
    if (mode !== "asl" || aslModel) return
    loadAslWords(asset("models/asl_words.json"))
      .then(setAslModel)
      .catch(() => setNotice("Could not load the ASL word model. Auto mode still works."))
  }, [mode, aslModel])
  useEffect(() => {
    segment.current.reset()
    aslFrames.current = []
  }, [mode])

  const classifyRef = useRef(classify)
  useEffect(() => {
    classifyRef.current = classify
    tracker.setHandler((frame) => {
      const now = performance.now()
      // Time based, so slow machines (a few frames a second) still get the "no hand" advice after two seconds.
      if (frame.raw === undefined) handSeenAt.current = 0
      else if (frame.landmarks || handSeenAt.current === 0) handSeenAt.current = now
      const next = frame.raw === undefined ? null : framingHint(frame.raw ?? null, frame.brightness ?? null, now - handSeenAt.current)
      setFraming((prev) => (prev?.text === next?.text ? prev : next))
      classifyRef.current(frame.landmarks, frame.handedness, undefined, frame.second)
    })
  })

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return
      if (event.key === "Backspace") {
        event.preventDefault()
        session.current.delete()
        setSentenceRev((value) => value + 1)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  async function playSample(kind: "static" | "dynamic", label: string) {
    runtime.current = { models, mode, mirror, threshold, autoSpeak, sentenceRev }
    setShowingSample(true)
    window.clearInterval(replayTimer.current)
    recognizer.current.reset()
    try {
    if (kind === "static") {
      const data = await fetch(asset("samples/static_samples.json")).then((res) => res.json())
      const flat = data.samples[label][0] as number[]
      const pts: Point[] = []
      for (let i = 0; i < 21; i++) pts.push([flat[i * 2], flat[i * 2 + 1], 0])
      tracker.showSample(pts, "image")
      for (let i = 0; i < 30; i++) classify(pts, "Right", "letter")
      return
    }
    const data = await fetch(asset("samples/dynamic_samples.json")).then((res) => res.json())
    const seq = (data.samples[label][0] as number[][][]).map(
      (frame) => frame.map((p) => [p[0], p[1], p[2] ?? 0] as Point),
    )
    tracker.showSample(seq[Math.min(10, seq.length - 1)], "normalized")
    const { dynamic, dynamicEmbed, meta, apiBase } = runtime.current.models
    if (!dynamic || !dynamicEmbed || !meta) return
    const flat = flattenSequence(seq)
    // The recorded sample is one fixed clip, so run the motion model once and replay that reading.
    const base = predictDynamic(dynamic, flat, meta.dynamic_labels)
    const reading = overrideCustom("dynamic", () => embedDynamic(dynamicEmbed, flat), base, runtime.current.models)
    for (let i = 0; i < 20; i++) {
      const prediction = { ...reading, kind: "word" as const }
      setLive(prediction)
      setHand("Sample")
      const result = recognizer.current.pushReading(prediction)
      if (result.commit) {
        if (result.commit.kind === "letter") session.current.confirmLetter(result.commit.label)
        else session.current.confirmWord(result.commit.label)
        bump()
        if (runtime.current.autoSpeak) void speak(result.commit.label, apiBase).then(setVoice)
      }
    }
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Could not play that sample")
    }
  }

  const customLetters = models.signs.filter((sign) => sign.kind === "static").map((sign) => sign.name)
  const english = session.current.previewEnglish()
  const hindi = toHindi(session.current.pendingGlosses())
  const chips = session.current.chips()

  /** Speak the current line in the chosen language, falling back to English when there is no Hindi voice. */
  async function speakLine(line: string, hindiLine: string) {
    if (lang === "hi" && hindiLine) {
      if (models.apiBase || (await hasHindiVoice())) {
        const used = await speak(hindiLine, models.apiBase, "hi")
        if (used !== "silent") return { used, note: used === "gtts" ? "Played the Hindi server recording." : "Spoke in Hindi with the browser voice." }
      }
      const used = await speak(line, models.apiBase, "en")
      return { used, note: "This device has no Hindi voice, so the English line was spoken. The Hindi text is shown above." }
    }
    const used = await speak(line, models.apiBase, "en")
    return { used, note: used === "gtts" ? "Played the server recording." : used === "browser" ? "Spoke with the browser voice." : "This browser has no speech voice, so the sentence is shown on screen only." }
  }

  function currentText(): string {
    return transcriptText(transcript, english ? { english, hindi } : null, lang === "hi")
  }

  async function copyLine() {
    if (!english) return setNotice("Nothing to copy yet.")
    const text = lang === "hi" && hindi ? `${english}\n${hindi}` : english
    try {
      await navigator.clipboard.writeText(text)
      setNotice("Copied the sentence.")
    } catch {
      setNotice("Copy is blocked in this browser. Select the sentence and press Ctrl+C.")
    }
  }

  async function shareLine() {
    if (!english) return setNotice("Nothing to share yet.")
    const text = lang === "hi" && hindi ? `${english}\n${hindi}` : english
    if (typeof navigator.share === "function") {
      try {
        await navigator.share({ title: "SignSpeak AI", text })
        setNotice("Shared.")
        return
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return setNotice("Share cancelled.")
      }
    }
    await copyLine()
    setNotice("Sharing is not available here, so the sentence was copied instead.")
  }

  function downloadTranscript() {
    const text = currentText()
    if (!text) return setNotice("Nothing to download yet. Sign a sentence first.")
    downloadText(text, `signspeak-transcript-${new Date().toISOString().slice(0, 10)}.txt`)
    setNotice("Downloaded the transcript as a .txt file.")
  }

  async function runPresentation() {
    if (presenting) return
    setPresenting(true)
    setShowingSample(true)
    setNotice("Presentation demo. These poses are built-in examples of the word rules, for when the camera fails.")
    session.current.clear()
    recognizer.current.reset()
    bump()
    try {
    for (const id of PRESENTATION_SCRIPT) {
      const frames = demoSequence(id)
      for (const frame of frames) {
        tracker.showSample(frame, "image")
        setLive({ label: id, confidence: 0.95, top: [{ label: id, confidence: 0.95 }], kind: "word" })
        await new Promise((resolve) => window.setTimeout(resolve, 45))
      }
      const hit = recognizeWord(demoSquare(id))
      session.current.confirmWord(hit?.id ?? id)
      bump()
    }
    const spoken = session.current.previewEnglish()
    setNotice(`Demo sentence: ${spoken} Press Speak to say it again, or Clear to start your own.`)
    await speak(spoken, models.apiBase).then(setVoice)
    // Leave the sentence on screen for the audience; Speak or Clear moves on from here.
    bump()
    } finally {
      setPresenting(false)
    }
  }

  return (
    <div className="space-y-8">
      {tipsOpen && <TipsDialog onClose={() => setTipsOpen(false)} />}
      <header className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr] lg:items-end">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">SignSpeak AI</p>
          <h1 className="serif mt-2 text-4xl leading-tight sm:text-5xl">Read a hand. Build a sentence. Hear it.</h1>
          <p className="mt-3 max-w-xl text-[var(--muted)]">
            Letters come from a model trained on public ASL hand landmarks. Common words use documented handshape rules, then a small offline grammar turns the glosses into English.
          </p>
          <p className="mt-3 text-sm font-medium">Built by Dhritika Sahu</p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Button onClick={() => document.getElementById("demo")?.scrollIntoView({ behavior: "smooth" })}>Open the demo</Button>
            <Button variant="quiet" disabled={presenting} onClick={() => void runPresentation()}>{presenting ? "Playing demo…" : "Play presentation demo"}</Button>
          </div>
        </div>
        <aside className="panel p-4" aria-label="Presentation guide">
          <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">Presentation mode</p>
          <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm">
            <li>Point down with one finger — I / me</li>
            <li>Open hand, pull it toward you — want</li>
            <li>Index, middle and ring up together — water</li>
          </ol>
          <p className="mt-2 text-sm text-[var(--muted)]">The line should become “I want water.” If the room is dark, use Play presentation demo.</p>
        </aside>
      </header>

    <div id="demo" className="grid gap-5 lg:grid-cols-[minmax(0,1.3fr)_minmax(280px,0.7fr)]">
      <section className="panel overflow-hidden">
        <div className="relative aspect-video bg-[#1a1713]">
          <video
            ref={tracker.videoRef}
            className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]"
            playsInline
            muted
            aria-label="Webcam preview"
          />
          <canvas
            ref={tracker.canvasRef}
            className="pointer-events-none absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]"
          />
          {tracker.running && framing && (
            <p
              data-testid="camera-hint"
              role="status"
              className={`absolute bottom-3 left-1/2 max-w-[90%] -translate-x-1/2 rounded-full px-3 py-1 text-center text-sm ${framing.tone === "ok" ? "bg-[color-mix(in_srgb,#1a1713_70%,transparent)] text-[#d9f2d0]" : "bg-[#f4f0e6] text-[#1a1713]"}`}
            >
              {framing.text}
            </p>
          )}
          {!tracker.running && !showingSample && (
            <div className="absolute inset-0 grid place-items-center bg-[color-mix(in_srgb,#1a1713_55%,transparent)] p-6 text-center text-[#f4f0e6]">
              <div>
                <p className="serif text-3xl">Show one hand</p>
                <p className="mx-auto mt-2 max-w-sm text-sm text-[#d9d0c2]">
                  Start the camera for live recognition, or play a recorded landmark sample if this machine has no webcam.
                </p>
              </div>
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2 p-4">
          {!tracker.running ? (
            <Button onClick={() => void tracker.start()}>Start camera</Button>
          ) : (
            <Button variant="quiet" onClick={tracker.stop}>Stop camera</Button>
          )}
          <Button variant="quiet" onClick={() => setTipsOpen(true)}>Tips for best accuracy</Button>
          <Button variant="quiet" onClick={() => void playSample("static", "B")}>Play sample B</Button>
          <Button variant="quiet" onClick={() => void playSample("dynamic", "HELLO")}>Play sample HELLO</Button>
          <span className="text-xs text-[var(--muted)]">
            {tracker.running ? `${tracker.fps} fps` : "camera off"}
            {hand ? ` · ${hand} hand` : " · no hand"}
          </span>
          {tracker.error && <p className="w-full text-sm text-[var(--accent)]">{tracker.error}</p>}
          {models.error && <p className="w-full text-sm text-[var(--accent)]">{models.error}</p>}
        </div>
      </section>

      <section className="flex flex-col gap-4">
        <div className="panel p-5" aria-live="polite">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">Live prediction</p>
              <p className="serif text-7xl leading-none">{live?.label ?? "–"}</p>
            </div>
            <Badge tone={live && live.confidence >= threshold ? "good" : "neutral"}>
              {live ? live.kind : "waiting"}
            </Badge>
          </div>
          <div className="mt-4 h-2 overflow-hidden rounded-full bg-[var(--line)]">
            <div
              className="h-full rounded-full bg-[var(--accent)]"
              style={{ width: `${Math.round((live?.confidence ?? 0) * 100)}%` }}
            />
          </div>
          <p className="mt-2 text-sm text-[var(--muted)]">
            Confidence {live ? `${Math.round(live.confidence * 100)}%` : "—"}
            {live?.top?.length ? ` · also ${live.top.slice(1).map((item) => `${item.label} ${Math.round(item.confidence * 100)}%`).join(", ")}` : ""}
          </p>
        </div>

        <div className="panel space-y-4 p-4">
          <div className="flex gap-1 rounded-full bg-[var(--bg)] p-1" role="group" aria-label="Recognition mode">
            {(["auto", "letters", "words", "asl"] as const).map((item) => (
              <button
                key={item}
                className={`flex-1 whitespace-nowrap rounded-full px-2 py-1.5 text-sm ${mode === item ? "bg-[var(--ink)] text-[var(--bg)]" : ""}`}
                onClick={() => setMode(item)}
                aria-pressed={mode === item}
              >
                {MODE_LABELS[item]}
              </button>
            ))}
          </div>
          <p className="text-xs text-[var(--muted)]" data-testid="mode-note">{MODE_NOTES[mode]}</p>
          <SliderControl id="confidence" label="Confidence to commit" min={0.4} max={0.95} step={0.01} value={threshold} onChange={setThreshold} />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <SwitchControl id="autospeak" label="Auto-speak" checked={autoSpeak} onCheckedChange={setAutoSpeak} />
            <label className="text-sm">
              Mirror
              <select
                className="ml-2 rounded-lg border border-[var(--line)] bg-[var(--bg)] px-2 py-1"
                value={mirror}
                onChange={(event) => setMirror(event.target.value as typeof mirror)}
              >
                <option value="auto">Auto</option>
                <option value="on">Force</option>
                <option value="off">Off</option>
              </select>
            </label>
          </div>
        </div>
      </section>

      <section className="panel p-4 lg:col-span-2">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">Sentence</p>
            <p className="serif min-h-12 text-3xl" data-testid="sentence">
              {english || <span className="text-[var(--muted)]">Letters and words will gather here.</span>}
            </p>
            {lang === "hi" && (
              <p className="min-h-6 text-xl" lang="hi" data-testid="sentence-hindi">
                {hindi || <span className="text-sm text-[var(--muted)]">हिन्दी वाक्य यहाँ दिखेगा।</span>}
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="quiet" onClick={() => { session.current.space(); bump() }}>Space</Button>
            <Button variant="quiet" onClick={() => { session.current.delete(); bump() }}>Delete</Button>
            <Button variant="quiet" onClick={() => { session.current.clear(); bump() }}>Clear</Button>
            <label className="flex items-center gap-1 text-sm">
              <span className="sr-only sm:not-sr-only">Voice</span>
              <select
                aria-label="Speech language"
                className="rounded-lg border border-[var(--line)] bg-[var(--bg)] px-2 py-1"
                value={lang}
                onChange={(event) => setLang(event.target.value as "en" | "hi")}
              >
                <option value="en">English</option>
                <option value="hi">हिन्दी (Hindi)</option>
              </select>
            </label>
            <Button
              onClick={() => {
                const line = session.current.previewEnglish()
                const hindiLine = toHindi(session.current.pendingGlosses())
                if (!line) {
                  setNotice("Nothing to say yet.")
                  return
                }
                void speakLine(line, hindiLine).then(({ used, note }) => {
                  setVoice(used)
                  setNotice(note)
                  setTranscript((list) => [...list, { at: new Date(), english: line, hindi: hindiLine }])
                  session.current.finish()
                  bump()
                })
              }}
            >
              Speak
            </Button>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label="Sentence tools">
          <Button variant="quiet" onClick={() => void copyLine()}>Copy</Button>
          <Button variant="quiet" onClick={() => void shareLine()}>Share</Button>
          <Button variant="quiet" onClick={downloadTranscript}>Download .txt</Button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label="Confirmed signs">
          {chips.length === 0 && <span className="text-sm text-[var(--muted)]">No signs confirmed yet.</span>}
          {chips.map((chip, index) => (
            <span key={`${chip}-${index}`} className="rounded-full bg-[var(--bg)] px-3 py-1 text-sm">{chip}</span>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label="Suggestions">
          {session.current.suggestions().map((item) => (
            <button key={item} className="rounded-full border border-[var(--line)] px-3 py-1 text-xs" onClick={() => { session.current.confirmWord(item); bump() }}>
              {item}
            </button>
          ))}
        </div>
        {session.current.history.length > 0 && (
          <ul className="mt-3 text-sm text-[var(--muted)]" aria-label="Sentence history">
            {session.current.history.map((line, index) => <li key={`${index}-${line}`}>{line}</li>)}
          </ul>
        )}
        {notice && (
          <p className="mt-2 text-sm text-[var(--muted)]">
            {notice}
            {voice === "browser" && !models.apiBase ? " The gTTS server is offline, so this tab is using speechSynthesis." : ""}
          </p>
        )}
        <div className="mt-4 flex flex-wrap gap-1.5" aria-label="Alphabet">
          {LETTERS.map((letter) => (
            <span
              key={letter}
              className={`grid h-8 w-8 place-items-center rounded-lg text-sm ${live?.label === letter ? "bg-[var(--ink)] text-[var(--bg)]" : "bg-[var(--bg)] text-[var(--muted)]"}`}
            >
              {letter}
            </span>
          ))}
          {["J", "Z"].map((letter) => (
            <span key={letter} className="grid h-8 w-8 place-items-center rounded-lg border border-dashed border-[var(--line)] text-xs text-[var(--muted)]" title="Motion letter, word mode">
              {letter}
            </span>
          ))}
          {customLetters.map((name) => (
            <span key={name} className="rounded-lg bg-[color-mix(in_srgb,var(--good)_18%,transparent)] px-2 py-1 text-xs text-[var(--good)]">
              {name}
            </span>
          ))}
        </div>
      </section>
    </div>

      <section id="signs" aria-label="Supported signs">
        <h2 className="serif text-3xl">Signs you can show</h2>
        <p className="mt-2 max-w-2xl text-sm text-[var(--muted)]">
          The 24 still letters are the trained model. The words below are handshape rules; their measured scores are on the Evaluation page, labelled real data or synthetic test.
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {WORD_SIGNS.map((sign) => (
            <article key={sign.id} className="panel p-4">
              <div className="flex items-center justify-between gap-2">
                <h3 className="font-medium">{sign.title}</h3>
                <Badge tone={sign.id === "HELP" || sign.id === "SCHOOL" ? "good" : "neutral"}>{sign.id === "HELP" || sign.id === "SCHOOL" ? "Two hands" : "Rule"}</Badge>
              </div>
              <p className="mt-1 text-sm">{sign.show}</p>
              <p className="mt-1 text-xs text-[var(--muted)]">{sign.detail}</p>
            </article>
          ))}
        </div>
      </section>

      <section id="how" className="grid gap-3 md:grid-cols-3">
        {[
          ["1. See the hand", "MediaPipe returns 21 landmarks from the webcam. Nothing is uploaded."],
          ["2. Read a sign", "A 1D CNN, trained on the public ASL Alphabet landmarks, reads still letters. Word rules read shape and motion."],
          ["3. Say the sentence", "Glosses such as ME WANT WATER become “I want water.” The browser speaks the line."],
        ].map(([title, body]) => (
          <article key={title} className="panel p-4">
            <h2 className="serif text-2xl">{title}</h2>
            <p className="mt-2 text-sm text-[var(--muted)]">{body}</p>
          </article>
        ))}
      </section>

      <section id="about" className="panel p-5">
        <h2 className="serif text-3xl">About</h2>
        <p className="mt-2">Built by Dhritika Sahu.</p>
        <p className="mt-2 text-sm text-[var(--muted)]">
          Letter landmarks: ASL Alphabet photographs by Akash Nagaraj, GPL-2.0, converted to MediaPipe Hands points in the MinorvaFalk/KNN_Alphabet CSV. Held-out letter accuracy is 94.19% (macro F1 94.23%). Word signs on this page are geometric rules, not that test set. MediaPipe runs in the browser under its own licence.
        </p>
      </section>
    </div>
  )
}

function overrideCustom(
  kind: "static" | "dynamic",
  vector: () => number[],
  base: { label: string; confidence: number; top: Ranked[] },
  bundle: ReturnType<typeof useModels>,
): { label: string; confidence: number; top: Ranked[] } {
  const head = bundle.heads[kind]
  if (!head) return base
  const custom = new Set(bundle.signs.filter((sign) => sign.kind === kind).map((sign) => sign.name))
  if (!custom.size) return base
  // Only embed the hand when the user has trained their own signs.
  const alt = predictHead(head.model, vector(), head.labels)
  if (custom.has(alt.label) && alt.confidence >= 0.55) return alt
  return base
}

function TipsDialog({ onClose }: { onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-[color-mix(in_srgb,#1a1713_60%,transparent)] p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="tips-title"
        className="panel max-h-[90vh] w-full max-w-lg overflow-auto p-5"
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id="tips-title" className="serif text-3xl">Tips for best accuracy</h2>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-sm">
          {ACCURACY_TIPS.map((tip) => (
            <li key={tip}>{tip}</li>
          ))}
        </ul>
        <p className="mt-3 text-sm text-[var(--muted)]">While the camera runs, a hint under the picture says if your hand is too far, too close, out of frame or too dark.</p>
        <div className="mt-4 flex justify-end">
          <Button ref={closeRef} onClick={onClose}>Got it</Button>
        </div>
      </div>
    </div>
  )
}

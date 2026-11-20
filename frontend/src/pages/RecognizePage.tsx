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
import { speak } from "@/lib/speech"
import { Stabilizer } from "@/lib/stabilizer"
import { PRESENTATION_SCRIPT, WORD_SIGNS, demoSequence, recognizeWord } from "@/lib/words"
import { useModels } from "@/models-context"

interface Live {
  label: string
  confidence: number
  top: Ranked[]
  kind: "letter" | "word"
}

const LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY".split("")

export function RecognizePage() {
  const models = useModels()
  const tracker = useHandTracker()
  const [live, setLive] = useState<Live | null>(null)
  const [sentenceRev, setSentenceRev] = useState(0)
  const session = useRef(new SentenceSession())
  const [hand, setHand] = useState<string | null>(null)
  const [mode, setMode] = useState<"auto" | "letters" | "words">("auto")
  const [mirror, setMirror] = useState<"auto" | "on" | "off">("auto")
  const [threshold, setThreshold] = useState(0.62)
  const [autoSpeak, setAutoSpeak] = useState(false)
  const [voice, setVoice] = useState<"browser" | "gtts" | "silent" | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [showingSample, setShowingSample] = useState(false)
  const stabilizer = useRef(new Stabilizer({ window: 14, majority: 0.8, minConfidence: 0.62, releaseFrames: 6, cooldownFrames: 30 }))
  const buffer = useRef<Point[][]>([])
  const wordVotes = useRef(0)
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

  function classify(frameLandmarks: Point[] | null, handedness: string | null, sampleKind?: "letter" | "word") {
    const { models: bundle, mode: currentMode, mirror: mirrorMode, threshold: minConf } = runtime.current
    stabilizer.current.setMinConfidence(minConf)
    if (!bundle.meta || !bundle.classifier || !bundle.dynamic) {
      return
    }
    const meta = bundle.meta
    const shouldMirror = mirrorMode === "on" || (mirrorMode === "auto" && handedness === "Left")
    const zeroZ = meta.z_mode === "zero"
    setHand(frameLandmarks ? handedness ?? "Hand" : null)
    if (!frameLandmarks) {
      wordVotes.current = 0
      buffer.current = []
      setLive(null)
      stabilizer.current.push(null, 0)
      return
    }
    buffer.current.push(frameLandmarks)
    if (buffer.current.length > 48) buffer.current.shift()

    let motion = 0
    if (buffer.current.length >= 8 && sampleKind !== "word") {
      try {
        motion = motionEnergy(normalizeSequence(buffer.current, meta.sequence_length, { mirror: shouldMirror, zeroZ }))
      } catch {
        motion = 0
      }
    }
    if (sampleKind === "word") wordVotes.current = 12
    else if (motion > meta.motion_threshold) wordVotes.current = Math.min(12, wordVotes.current + 1)
    else wordVotes.current = Math.max(0, wordVotes.current - 1)

    const useWords =
      sampleKind === "word" ||
      currentMode === "words" ||
      (currentMode === "auto" && sampleKind !== "letter" && wordVotes.current >= 6 && buffer.current.length >= 12)

    let prediction: Live
    if (useWords && buffer.current.length >= 8) {
      const seq =
        sampleKind === "word"
          ? normalizeSequence(buffer.current.slice(-meta.sequence_length), meta.sequence_length, { mirror: shouldMirror, zeroZ })
          : normalizeSequence(buffer.current, meta.sequence_length, { mirror: shouldMirror, zeroZ })
      const flat = flattenSequence(seq)
      const base = predictDynamic(bundle.dynamic, flat, meta.dynamic_labels)
      prediction = { ...overrideCustom("dynamic", embedDynamic(bundle.dynamicEmbed!, flat), base, bundle), kind: "word" }
    } else {
      const norm = normalizeLandmarks(frameLandmarks, { mirror: shouldMirror, zeroZ })
      const base = predictStatic(bundle.classifier, norm, meta.static_labels)
      prediction = { ...overrideCustom("static", embedStatic(bundle.embed!, norm), base, bundle), kind: "letter" }
    }
    if (mode !== "letters") {
      const hit = recognizeWord(buffer.current)
      if (hit) {
        prediction = {
          label: hit.id,
          confidence: hit.score,
          top: [{ label: hit.id, confidence: hit.score }],
          kind: "word",
        }
      }
    }
    setLive(prediction)
    const stable = stabilizer.current.push(prediction.label, prediction.confidence)
    if (stable) commit(stable, prediction.kind)
  }

  const classifyRef = useRef(classify)
  useEffect(() => {
    classifyRef.current = classify
    tracker.setHandler((frame) => classifyRef.current(frame.landmarks, frame.handedness))
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
    stabilizer.current.reset()
    buffer.current = []
    try {
    if (kind === "static") {
      const data = await fetch(asset("samples/static_samples.json")).then((res) => res.json())
      const flat = data.samples[label][0] as number[]
      const pts: Point[] = []
      for (let i = 0; i < 21; i++) pts.push([flat[i * 2], flat[i * 2 + 1], 0])
      tracker.showSample(pts, "image")
      for (let i = 0; i < 20; i++) classify(pts, "Right", "letter")
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
    for (let i = 0; i < 20; i++) {
      const base = predictDynamic(dynamic, flat, meta.dynamic_labels)
      const prediction = {
        ...overrideCustom("dynamic", embedDynamic(dynamicEmbed, flat), base, runtime.current.models),
        kind: "word" as const,
      }
      setLive(prediction)
      setHand("Sample")
      const stable = stabilizer.current.push(prediction.label, prediction.confidence)
      if (stable) {
        {
          session.current.confirmWord(stable)
          bump()
        }
        if (runtime.current.autoSpeak) void speak(stable, apiBase).then(setVoice)
      }
    }
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Could not play that sample")
    }
  }

  const customLetters = models.signs.filter((sign) => sign.kind === "static").map((sign) => sign.name)
  const english = session.current.previewEnglish()
  const chips = session.current.chips()

  async function runPresentation() {
    setShowingSample(true)
    setNotice("Presentation demo. These poses are built-in examples of the word rules, for when the camera fails.")
    session.current.clear()
    stabilizer.current.reset()
    bump()
    for (const id of PRESENTATION_SCRIPT) {
      const frames = demoSequence(id)
      for (const frame of frames) {
        tracker.showSample(frame, "image")
        setLive({ label: id, confidence: 0.95, top: [{ label: id, confidence: 0.95 }], kind: "word" })
        await new Promise((resolve) => window.setTimeout(resolve, 45))
      }
      const hit = recognizeWord(frames)
      session.current.confirmWord(hit?.id ?? id)
      bump()
    }
    const spoken = session.current.previewEnglish()
    setNotice(spoken)
    await speak(spoken, models.apiBase).then(setVoice)
    session.current.finish()
    bump()
  }

  return (
    <div className="space-y-8">
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
            <Button variant="quiet" onClick={() => void runPresentation()}>Play presentation demo</Button>
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
            {(["auto", "letters", "words"] as const).map((item) => (
              <button
                key={item}
                className={`flex-1 rounded-full px-2 py-1.5 text-sm capitalize ${mode === item ? "bg-[var(--ink)] text-[var(--bg)]" : ""}`}
                onClick={() => setMode(item)}
                aria-pressed={mode === item}
              >
                {item}
              </button>
            ))}
          </div>
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
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="quiet" onClick={() => { session.current.space(); bump() }}>Space</Button>
            <Button variant="quiet" onClick={() => { session.current.delete(); bump() }}>Delete</Button>
            <Button variant="quiet" onClick={() => { session.current.clear(); bump() }}>Clear</Button>
            <Button
              onClick={() => {
                const line = session.current.previewEnglish()
                void speak(line, models.apiBase).then((used) => {
                  setVoice(used)
                  setNotice(used === "gtts" ? "Played the server recording." : used === "browser" ? "Spoke with the browser voice." : "Nothing to say yet.")
                  if (line) session.current.finish()
                  bump()
                })
              }}
            >
              Speak
            </Button>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-2" aria-label="Confirmed signs">
          {chips.length === 0 && <span className="text-sm text-[var(--muted)]">No signs confirmed yet.</span>}
          {chips.map((chip, index) => (
            <span key={`${chip}-${index}`} className="rounded-full bg-[var(--bg)] px-3 py-1 text-sm">{chip}</span>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-2" aria-label="Suggestions">
          {session.current.suggestions().map((item) => (
            <button key={item} className="rounded-full border border-[var(--line)] px-3 py-1 text-xs" onClick={() => { session.current.confirmWord(item); bump() }}>
              {item}
            </button>
          ))}
        </div>
        {session.current.history.length > 0 && (
          <ul className="mt-3 text-sm text-[var(--muted)]" aria-label="Sentence history">
            {session.current.history.map((line) => <li key={line}>{line}</li>)}
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
          The 24 still letters are the trained model. The words below are handshape rules, so a viva answer should call them rules, not a second accuracy score. HELP and SCHOOL need two hands and are listed only so you can say they are out of scope.
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {WORD_SIGNS.map((sign) => (
            <article key={sign.id} className="panel p-4">
              <div className="flex items-center justify-between gap-2">
                <h3 className="font-medium">{sign.title}</h3>
                <Badge tone={sign.id === "HELP" || sign.id === "SCHOOL" ? "warn" : "neutral"}>{sign.id === "HELP" || sign.id === "SCHOOL" ? "Not detected" : "Rule"}</Badge>
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
  vector: number[],
  base: { label: string; confidence: number; top: Ranked[] },
  bundle: ReturnType<typeof useModels>,
): { label: string; confidence: number; top: Ranked[] } {
  const head = bundle.heads[kind]
  if (!head) return base
  const custom = new Set(bundle.signs.filter((sign) => sign.kind === kind).map((sign) => sign.name))
  if (!custom.size) return base
  const alt = predictHead(head.model, vector, head.labels)
  if (custom.has(alt.label) && alt.confidence >= 0.55) return alt
  return base
}

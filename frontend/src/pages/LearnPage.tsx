import { useEffect, useMemo, useRef, useState } from "react"
import { HandFigure } from "@/components/HandFigure"
import { Badge, Button } from "@/components/ui"
import { useHandTracker } from "@/hooks/useHandTracker"
import { asset } from "@/lib/asset"
import { squareFrame, type Point } from "@/lib/hands"
import { performTwoHands } from "@/lib/handModel"
import { liveDeps } from "@/lib/liveDeps"
import { Recognizer } from "@/lib/recognizer"
import { LETTER_HOWTO, STILL_LETTERS, exampleRow, letterFigure, textToSigns, wordFigure, type SignCard, type SignFigure } from "@/lib/signLibrary"
import { DEMO_ASPECT, TWO_HAND_WORDS, WORD_SIGNS, demoSquare } from "@/lib/words"
import { useModels } from "@/models-context"

type Samples = Record<string, number[][]>

const WORD_IDS = WORD_SIGNS.map((s) => s.id)
const ALL_SIGNS = [...WORD_IDS, ...STILL_LETTERS]
const QUIZ_LENGTH = 5

function isLetter(id: string) {
  return /^[A-Z]$/.test(id)
}

function howTo(id: string): { title: string; show: string; detail: string } {
  if (isLetter(id)) return { title: `Letter ${id}`, show: LETTER_HOWTO[id] ?? "", detail: "Hold the shape still for about a second. Practise letters in Letters mode." }
  const sign = WORD_SIGNS.find((s) => s.id === id)!
  return { title: sign.title, show: sign.show, detail: sign.detail }
}

function figureFor(id: string, samples: Samples | null): SignFigure | null {
  if (!isLetter(id)) return wordFigure(id)
  const row = exampleRow(samples?.[id])
  return row ? letterFigure(row) : null
}

function shuffle<T>(items: T[]): T[] {
  const out = [...items]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

interface Quiz {
  items: string[]
  at: number
  right: number
  done: boolean
}

export function LearnPage() {
  const models = useModels()
  const tracker = useHandTracker()
  const [samples, setSamples] = useState<Samples | null>(null)
  const [target, setTarget] = useState("HELLO")
  const [play, setPlay] = useState(0)
  const [feedback, setFeedback] = useState<{ tone: "good" | "warn" | "neutral"; text: string } | null>(null)
  const [score, setScore] = useState({ correct: 0, tries: 0 })
  const [live, setLive] = useState<string | null>(null)
  const [quiz, setQuiz] = useState<Quiz | null>(null)
  const [hint, setHint] = useState(false)
  const [typed, setTyped] = useState("")
  const [cards, setCards] = useState<SignCard[][] | null>(null)
  const engine = useRef<Recognizer | null>(null)
  const advance = useRef(0)

  useEffect(() => {
    fetch(asset("samples/static_samples.json"))
      .then((res) => res.json())
      .then((data: { samples: Samples }) => setSamples(data.samples))
      .catch(() => setSamples({}))
    return () => window.clearTimeout(advance.current)
  }, [])

  const current = quiz && !quiz.done ? quiz.items[quiz.at] : target
  const info = howTo(current)
  const figure = useMemo(() => figureFor(current, samples), [current, samples])
  const ready = Boolean(models.meta && models.classifier && models.dynamic)

  function getEngine() {
    if (!models.meta || !models.classifier || !models.dynamic) return null
    engine.current ??= new Recognizer(liveDeps(models.meta, models.classifier, models.dynamic))
    return engine.current
  }

  // Keep the grading logic current for the camera handler.
  const grade = useRef<(label: string) => void>(() => {})
  grade.current = (label: string) => {
    const goal = quiz && !quiz.done ? quiz.items[quiz.at] : target
    if (label === goal) {
      setFeedback({ tone: "good", text: `Correct! That was ${goal}.` })
      setScore((s) => ({ correct: s.correct + 1, tries: s.tries + 1 }))
      if (quiz && !quiz.done) {
        window.clearTimeout(advance.current)
        advance.current = window.setTimeout(() => nextQuestion(true), 900)
      }
    } else {
      setFeedback({ tone: "warn", text: `That looked like ${label}. Try again.` })
      setScore((s) => ({ ...s, tries: s.tries + 1 }))
    }
  }

  const modeFor = (id: string) => (isLetter(id) ? ("letters" as const) : ("auto" as const))
  const currentRef = useRef(current)
  currentRef.current = current

  useEffect(() => {
    tracker.setHandler((frame) => {
      const e = getEngine()
      if (!e) return
      const result = e.push(frame.landmarks, { mode: modeFor(currentRef.current), mirror: frame.handedness === "Left", second: frame.second })
      setLive(result.live?.label ?? null)
      if (result.commit) grade.current(result.commit.label)
    })
  })

  function select(id: string) {
    setTarget(id)
    setFeedback(null)
    setPlay((p) => p + 1)
    engine.current?.reset()
  }

  function step(delta: number) {
    const i = ALL_SIGNS.indexOf(target)
    select(ALL_SIGNS[(i + delta + ALL_SIGNS.length) % ALL_SIGNS.length])
  }

  /** Run the built-in example of the current sign through the same recognizer, as if signed. */
  function playExample() {
    const e = getEngine()
    if (!e) return setFeedback({ tone: "warn", text: "The models are still loading." })
    const id = currentRef.current
    e.reset()
    setPlay((p) => p + 1)
    const feed = (frame: Point[] | null, mirror: boolean, second?: Point[] | null) => {
      const r = e.push(frame, { mode: modeFor(id), mirror, second })
      if (r.live) setLive(r.live.label)
      return r.commit?.label ?? null
    }
    const commits: string[] = []
    if (isLetter(id)) {
      const fig = figureFor(id, samples)
      if (!fig) return setFeedback({ tone: "warn", text: "Letter examples are still loading." })
      for (let i = 0; i < 30; i++) {
        const c = feed(fig.frames[0], false)
        if (c) commits.push(c)
      }
    } else if (TWO_HAND_WORDS.has(id)) {
      const { main, second } = performTwoHands(id as "HELP" | "SCHOOL", { rand: () => 0.5, jitter: 0, aspect: DEMO_ASPECT })
      const hold = 10
      for (let i = 0; i < main.length + hold; i++) {
        const k = Math.min(i, main.length - 1)
        const c = feed(squareFrame(main[k], DEMO_ASPECT), true, squareFrame(second[k], DEMO_ASPECT))
        if (c) commits.push(c)
      }
    } else {
      for (const frame of demoSquare(id)) {
        const c = feed(frame, true)
        if (c) commits.push(c)
      }
    }
    for (let i = 0; i < 12; i++) feed(null, false)
    if (commits.length) commits.forEach((c) => grade.current(c))
    else setFeedback({ tone: "warn", text: "The example was not recognised this time." })
  }

  function startQuiz() {
    const pool = shuffle(ALL_SIGNS.filter((id) => !TWO_HAND_WORDS.has(id)))
    setQuiz({ items: pool.slice(0, QUIZ_LENGTH), at: 0, right: 0, done: false })
    setScore({ correct: 0, tries: 0 })
    setFeedback(null)
    setHint(false)
    engine.current?.reset()
  }

  function nextQuestion(correct: boolean) {
    setHint(false)
    engine.current?.reset()
    setQuiz((q) => {
      if (!q || q.done) return q
      const right = q.right + (correct ? 1 : 0)
      const at = q.at + 1
      return at >= q.items.length ? { ...q, right, at: q.items.length - 1, done: true } : { ...q, right, at }
    })
    if (!correct) setFeedback({ tone: "neutral", text: "Skipped." })
  }

  return (
    <div className="space-y-8">
      <header>
        <p className="text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Learn</p>
        <h1 className="serif text-4xl">Learn and practise signs</h1>
        <p className="mt-2 max-w-2xl text-sm text-[var(--muted)]">
          Pick a sign, watch the reference hand, then sign it to the camera. The app checks it with the same recognizer as the main page.
          No camera? Press Play example to see how a correct sign is graded.
        </p>
      </header>

      <div className="grid gap-5 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <section className="panel min-w-0 space-y-4 p-5" aria-label="Reference">
          {!quiz || quiz.done ? (
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="quiet" onClick={() => step(-1)} aria-label="Previous sign">←</Button>
              <label className="text-sm">
                <span className="sr-only">Sign to practise</span>
                <select
                  aria-label="Sign to practise"
                  className="rounded-lg border border-[var(--line)] bg-[var(--bg)] px-2 py-1.5"
                  value={target}
                  onChange={(event) => select(event.target.value)}
                >
                  <optgroup label="Words">
                    {WORD_SIGNS.map((s) => (
                      <option key={s.id} value={s.id}>{s.title}</option>
                    ))}
                  </optgroup>
                  <optgroup label="Letters">
                    {STILL_LETTERS.map((l) => (
                      <option key={l} value={l}>Letter {l}</option>
                    ))}
                  </optgroup>
                </select>
              </label>
              <Button variant="quiet" onClick={() => step(1)} aria-label="Next sign">→</Button>
            </div>
          ) : (
            <p className="text-sm" data-testid="quiz-progress">
              Question {quiz.at + 1} of {quiz.items.length}
            </p>
          )}
          <div>
            <h2 className="serif text-3xl" data-testid="learn-title">{quiz && !quiz.done ? `Sign: ${info.title}` : info.title}</h2>
            {TWO_HAND_WORDS.has(current) && <Badge tone="good">Two hands</Badge>}
          </div>
          {(!quiz || quiz.done || hint) && (
            <div className="flex flex-wrap items-start gap-4">
              {figure ? (
                <HandFigure frames={figure.frames} second={figure.second} play={play} label={`Reference hand for ${info.title}`} />
              ) : (
                <div className="grid h-[180px] w-[180px] place-items-center rounded-xl bg-[var(--bg)] text-xs text-[var(--muted)]">Loading…</div>
              )}
              <div className="min-w-0 flex-1 space-y-2 text-sm">
                <p className="font-medium" data-testid="learn-show">{info.show}</p>
                <p className="text-[var(--muted)]">{info.detail}</p>
                <p className="text-xs text-[var(--muted)]">The figure is shown as in a mirror, like your camera preview.</p>
              </div>
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            {(!quiz || quiz.done || hint) && figure && figure.frames.length > 1 && (
              <Button variant="quiet" onClick={() => setPlay((p) => p + 1)}>Show me</Button>
            )}
            {quiz && !quiz.done && !hint && <Button variant="quiet" onClick={() => setHint(true)}>Show hint</Button>}
            <Button variant="quiet" disabled={!ready} onClick={playExample}>Play example</Button>
            {quiz && !quiz.done && <Button variant="quiet" onClick={() => nextQuestion(false)}>Skip</Button>}
          </div>
        </section>

        <section className="panel min-w-0 overflow-hidden" aria-label="Practice">
          <div className="relative aspect-video bg-[#1a1713]">
            <video ref={tracker.videoRef} className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]" playsInline muted aria-label="Practice camera" />
            <canvas ref={tracker.canvasRef} className="pointer-events-none absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]" />
            {!tracker.running && (
              <div className="absolute inset-0 grid place-items-center p-6 text-center text-sm text-[#d9d0c2]">Start the camera to practise.</div>
            )}
          </div>
          <div className="space-y-3 p-4">
            <div className="flex flex-wrap items-center gap-2">
              {!tracker.running ? (
                <Button onClick={() => void tracker.start()}>Start camera</Button>
              ) : (
                <Button variant="quiet" onClick={tracker.stop}>Stop camera</Button>
              )}
              <span className="text-xs text-[var(--muted)]">Seeing: <strong data-testid="learn-live">{live ?? "—"}</strong></span>
            </div>
            {tracker.error && <p className="text-sm text-[var(--accent)]">{tracker.error}</p>}
            <p
              aria-live="polite"
              data-testid="learn-feedback"
              className={`min-h-6 text-lg ${feedback?.tone === "good" ? "text-[var(--good)]" : feedback?.tone === "warn" ? "text-[var(--accent)]" : ""}`}
            >
              {feedback?.text ?? ""}
            </p>
            <p className="text-sm" data-testid="learn-score">
              Score {score.correct} / {score.tries}
            </p>
            {quiz?.done ? (
              <div className="rounded-xl bg-[var(--bg)] p-3" data-testid="quiz-result">
                <p className="serif text-2xl">Quiz done: {quiz.right} of {quiz.items.length} right</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  <Button onClick={startQuiz}>Play again</Button>
                  <Button variant="quiet" onClick={() => setQuiz(null)}>Back to practice</Button>
                </div>
              </div>
            ) : quiz ? (
              <Button variant="quiet" onClick={() => setQuiz(null)}>End quiz</Button>
            ) : (
              <Button onClick={startQuiz}>Start quiz ({QUIZ_LENGTH} signs)</Button>
            )}
          </div>
        </section>
      </div>

      <section className="panel space-y-4 p-5" aria-label="Text to sign">
        <div>
          <h2 className="serif text-3xl">Text to sign</h2>
          <p className="text-sm text-[var(--muted)]">Type a word or sentence. Known words show their sign; other words are fingerspelled letter by letter.</p>
        </div>
        <form
          className="flex flex-wrap gap-2"
          onSubmit={(event) => {
            event.preventDefault()
            setCards(textToSigns(typed))
          }}
        >
          <label className="min-w-0 flex-1">
            <span className="sr-only">Type a word or sentence</span>
            <input
              aria-label="Type a word or sentence"
              className="w-full rounded-lg border border-[var(--line)] bg-[var(--bg)] px-3 py-2"
              placeholder="e.g. Hello, my name is Ria"
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
            />
          </label>
          <Button type="submit">Show signs</Button>
          {cards && (
            <Button type="button" variant="quiet" onClick={() => { setCards(null); setTyped("") }}>Clear</Button>
          )}
        </form>
        {cards && cards.length === 0 && <p className="text-sm text-[var(--muted)]">Type some letters or words first.</p>}
        {cards && cards.length > 0 && (
          <div className="space-y-4" data-testid="sign-cards">
            {cards.map((group, gi) => (
              <div key={gi} className="flex flex-wrap gap-3">
                {group.map((card, ci) => (
                  <SignCardView key={`${gi}-${ci}`} card={card} samples={samples} />
                ))}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

function SignCardView({ card, samples }: { card: SignCard; samples: Samples | null }) {
  const [play, setPlay] = useState(0)
  const motionLetter = card.kind === "letter" && (card.sign === "J" || card.sign === "Z")
  const figure = motionLetter ? null : card.kind === "word" ? wordFigure(card.sign) : exampleRow(samples?.[card.sign]) ? letterFigure(exampleRow(samples?.[card.sign])!) : null
  const caption = card.kind === "word" ? WORD_SIGNS.find((s) => s.id === card.sign)?.show ?? "" : LETTER_HOWTO[card.sign] ?? ""
  return (
    <article className="w-[150px] rounded-xl border border-[var(--line)] p-2 text-center" aria-label={`Sign card ${card.text}`}>
      {figure ? (
        <button type="button" className="block" onClick={() => setPlay((p) => p + 1)} aria-label={`Replay ${card.text}`}>
          <HandFigure frames={figure.frames} second={figure.second} play={play} size={130} label={`${card.text} sign`} />
        </button>
      ) : (
        <div className="grid h-[130px] place-items-center rounded-xl bg-[var(--bg)] text-xs text-[var(--muted)]">{motionLetter ? "Motion letter" : "—"}</div>
      )}
      <p className="mt-1 font-medium">{card.text}</p>
      <p className="text-xs text-[var(--muted)]">{caption}</p>
    </article>
  )
}

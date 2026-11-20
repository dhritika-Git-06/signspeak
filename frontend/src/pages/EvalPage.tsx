import { useEffect, useState } from "react"
import { asset } from "@/lib/asset"

interface ClassRow {
  precision: number
  recall: number
  f1: number
  support: number
}

interface Block {
  accuracy: number
  macro_f1: number
  weighted_f1: number
  test_size: number
  labels: string[]
  per_class: Record<string, ClassRow>
  report: string
  top_confusions: Array<{ actual: string; predicted: string; count: number }>
  z_mode?: string
  z_noise_accuracy?: number
  data?: string
}

interface Metrics {
  generated_at: string
  static: Block
  dynamic: Block
  motion_threshold: number
  datasets: {
    static: { name: string; images: string; images_license: string; images_url: string; landmarks_url: string; notes: string }
    dynamic: { name: string; license: string; notes: string; classes: string[]; handshapes: Record<string, string> }
  }
}

interface AslWords {
  source: string
  labels: string[]
  accuracy: number
  macro_f1: number
  top3_accuracy: number
  train_clips: number
  test_clips: number
  train_signers: number
  test_signers: number
  per_class: Record<string, ClassRow>
  app_check?: {
    clips: number
    model_accuracy: number
    rules_accuracy: number
    rules_no_word: number
    per_class: Record<string, { clips: number; model: number; rules: number; rules_no_word: number }>
    note: string
  }
}

interface SignRow {
  n: number
  exactly_once: number
  extra: number
  missed: number
  wrong: Record<string, number>
}

interface Synthetic {
  what: string
  score: string
  samples_per_sign: number
  words_auto: { overall: number; per_sign: Record<string, SignRow> }
  letters_letters_mode: { overall: number; per_sign: Record<string, SignRow> }
  letters_auto_mode: { overall: number; per_sign: Record<string, SignRow>; note: string }
}

function getJson<T>(path: string): Promise<T | null> {
  return fetch(asset(path))
    .then((res) => (res.ok ? (res.json() as Promise<T>) : null))
    .catch(() => null)
}

export function EvalPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [asl, setAsl] = useState<AslWords | null>(null)
  const [synthetic, setSynthetic] = useState<Synthetic | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(asset("evaluation/metrics.json"))
      .then((res) => {
        if (!res.ok) throw new Error("Metrics file missing")
        return res.json()
      })
      .then(setMetrics)
      .catch((err: Error) => setError(err.message))
    void getJson<AslWords>("evaluation/asl_words.json").then(setAsl)
    void getJson<Synthetic>("evaluation/synthetic_signs.json").then(setSynthetic)
  }, [])

  if (error) return <p className="text-[var(--accent)]">{error}</p>
  if (!metrics) return <p className="text-[var(--muted)]">Loading the evaluation report…</p>

  return (
    <article className="space-y-6">
      <header>
        <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">Evaluation</p>
        <h1 className="serif text-4xl">Measured accuracy</h1>
        <p className="mt-2 max-w-3xl text-sm text-[var(--muted)]">
          Every number here was measured, not estimated. “Real data” means held-out recordings of real people.
          “Synthetic test” means simulated hands run through the app's own pipeline. None of these is a live webcam study.
          Letter figures generated {new Date(metrics.generated_at).toLocaleString()} by <code>python evaluate.py</code>.
        </p>
      </header>
      <section aria-label="Summary" className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Score tag="Real data" title="Still letters (24), trained model" accuracy={metrics.static.accuracy} f1={metrics.static.macro_f1} detail={`${metrics.static.test_size} held-out photos of real hands.`} />
        {asl && (
          <Score tag="Real data" title={`Real ASL words (${asl.labels.length}), trained model`} accuracy={asl.accuracy} f1={asl.macro_f1} detail={`${asl.test_clips} clips from ${asl.test_signers} signers never seen in training. Top-3 ${(asl.top3_accuracy * 100).toFixed(1)}%.`} />
        )}
        {asl?.app_check && (
          <Percent tag="Real data" title="Hand-written word rules on the same clips" value={asl.app_check.rules_accuracy} detail={`No word at all on ${asl.app_check.rules_no_word}% of clips. The rules use this app's simplified one-hand signs, so real ASL forms often do not match them.`} />
        )}
        {synthetic && (
          <Percent tag="Synthetic test" title="Word rules, full live pipeline" value={synthetic.words_auto.overall} detail={`Simulated hands, ${synthetic.samples_per_sign} per word. Passes only if the word is added exactly once.`} />
        )}
      </section>
      {asl && (
        <section className="panel min-w-0 space-y-3 p-5">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="serif text-2xl">Real ASL words: trained model vs rules</h2>
            <Tag>Real data</Tag>
          </div>
          <p className="text-sm text-[var(--muted)]">
            {asl.train_clips} training clips from {asl.train_signers} signers, {asl.test_clips} test clips from {asl.test_signers} other signers. Recall per word on the test signers. Chance is {(100 / asl.labels.length).toFixed(1)}%.
            {asl.app_check ? ` Checked again with the app's own TypeScript code: ${asl.app_check.model_accuracy}%.` : ""}
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[420px] text-sm">
              <thead>
                <tr className="text-left text-[var(--muted)]">
                  <th className="py-1 pr-3 font-normal">Word</th>
                  <th className="py-1 pr-3 font-normal">Test clips</th>
                  <th className="py-1 pr-3 font-normal">Trained model</th>
                  <th className="py-1 pr-3 font-normal">Rules</th>
                </tr>
              </thead>
              <tbody>
                {asl.labels.map((label) => (
                  <tr key={label} className="border-t border-[var(--line)]">
                    <td className="py-1 pr-3">{label}</td>
                    <td className="py-1 pr-3">{asl.per_class[label]?.support ?? "—"}</td>
                    <td className="py-1 pr-3">{((asl.per_class[label]?.recall ?? 0) * 100).toFixed(1)}%</td>
                    <td className="py-1 pr-3">{asl.app_check?.per_class[label] ? `${asl.app_check.per_class[label].rules}%` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {asl.app_check && <p className="text-xs text-[var(--muted)]">{asl.app_check.note}</p>}
          <p className="text-xs text-[var(--muted)]">
            Data: PopSign ASL v1.0 (Starner et al., NeurIPS 2023 Datasets and Benchmarks), licensed{" "}
            <a className="underline" target="_blank" rel="noreferrer" href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>, frames from{" "}
            <a className="underline" target="_blank" rel="noreferrer" href="https://huggingface.co/datasets/sign/popsign-images">huggingface.co/datasets/sign/popsign-images</a>. Hand landmarks were extracted with MediaPipe; no video is shipped with the app.
          </p>
        </section>
      )}
      {synthetic && (
        <section className="panel min-w-0 space-y-3 p-5">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="serif text-2xl">Per-sign tests through the live pipeline</h2>
            <Tag tone="muted">Synthetic test</Tag>
          </div>
          <p className="text-sm text-[var(--muted)]">{synthetic.what} {synthetic.score}</p>
          <div className="grid gap-4 lg:grid-cols-3">
            <SignTable title={`Words, Auto mode · ${synthetic.words_auto.overall}%`} rows={synthetic.words_auto.per_sign} />
            <SignTable title={`Letters, Letters mode · ${synthetic.letters_letters_mode.overall}%`} rows={synthetic.letters_letters_mode.per_sign} />
            <SignTable title={`Letters, Auto mode · ${synthetic.letters_auto_mode.overall}%`} rows={synthetic.letters_auto_mode.per_sign} note={synthetic.letters_auto_mode.note} />
          </div>
        </section>
      )}
      <div className="grid gap-4 md:grid-cols-2">
        <Score tag="Synthetic test" title="Retired kinematic test, not used live" accuracy={metrics.dynamic.accuracy} f1={metrics.dynamic.macro_f1} detail="Synthetic motions on real handshapes. The live demo uses handshape rules instead, so do not quote this number as webcam accuracy." />
      </div>
      <section className="panel grid gap-4 p-5 md:grid-cols-2">
        <figure className="min-w-0">
          <img src={asset("evaluation/static_confusion.png")} alt="Confusion matrix for the 24 static ASL letters" className="w-full rounded-xl" />
          <figcaption className="mt-2 text-xs text-[var(--muted)]">Static test set. The bright diagonal is correct letters. M and N are the usual mix-up.</figcaption>
        </figure>
        <figure className="min-w-0">
          <img src={asset("evaluation/dynamic_confusion.png")} alt="Confusion matrix for the retired kinematic word test" className="w-full rounded-xl" />
          <figcaption className="mt-2 text-xs text-[var(--muted)]">Dynamic held-out synthetic test. High accuracy is expected: the motions were generated, not filmed.</figcaption>
        </figure>
      </section>
      <section className="grid gap-4 lg:grid-cols-2">
        <Report title="Static classification report" text={metrics.static.report} confusions={metrics.static.top_confusions} />
        <Report title="Dynamic classification report" text={metrics.dynamic.report} confusions={metrics.dynamic.top_confusions} />
      </section>
      <section className="panel space-y-3 p-5 text-sm text-[var(--muted)]">
        <h2 className="serif text-2xl text-[var(--ink)]">Datasets</h2>
        <p>
          <strong className="text-[var(--ink)]">{metrics.datasets.static.name}.</strong> {metrics.datasets.static.notes} Images: {metrics.datasets.static.images} ({metrics.datasets.static.images_license}).{" "}
          <a className="underline" target="_blank" rel="noreferrer" href={metrics.datasets.static.images_url}>Kaggle record</a>. Landmark file:{" "}
          <a className="underline" target="_blank" rel="noreferrer" href={metrics.datasets.static.landmarks_url}>hand_dataset_1000_24.csv</a>.
        </p>
        <p>
          <strong className="text-[var(--ink)]">{metrics.datasets.dynamic.name}.</strong> {metrics.datasets.dynamic.notes} Classes: {metrics.datasets.dynamic.classes.join(", ")}.
          Handshape donors: {Object.entries(metrics.datasets.dynamic.handshapes).map(([word, letter]) => `${word}←${letter}`).join(", ")}.
        </p>
        <p>Motion threshold used by Auto mode: {metrics.motion_threshold.toFixed(3)} in palm-size units per frame.</p>
        {metrics.static.z_noise_accuracy != null && (
          <p>Static accuracy with random depth noise (σ = 0.1): {(metrics.static.z_noise_accuracy * 100).toFixed(2)}%.</p>
        )}
      </section>
    </article>
  )
}

function Tag({ children, tone = "good" }: { children: string; tone?: "good" | "muted" }) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs ${tone === "good" ? "bg-[color-mix(in_srgb,var(--good)_18%,transparent)] text-[var(--good)]" : "bg-[var(--bg)] text-[var(--muted)]"}`}>
      {children}
    </span>
  )
}

function Percent({ tag, title, value, detail }: { tag: string; title: string; value: number; detail: string }) {
  return (
    <section className="panel min-w-0 p-5">
      <Tag tone={tag === "Real data" ? "good" : "muted"}>{tag}</Tag>
      <h2 className="mt-2 text-sm text-[var(--muted)]">{title}</h2>
      <p className="serif text-5xl">{value.toFixed(1)}%</p>
      <p className="mt-2 text-xs text-[var(--muted)]">{detail}</p>
    </section>
  )
}

function SignTable({ title, rows, note }: { title: string; rows: Record<string, SignRow>; note?: string }) {
  return (
    <div className="min-w-0">
      <h3 className="font-medium">{title}</h3>
      {note && <p className="mt-1 text-xs text-[var(--muted)]">{note}</p>}
      <table className="mt-2 w-full text-sm">
        <tbody>
          {Object.entries(rows).map(([label, row]) => {
            const mistakes = Object.entries(row.wrong).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([k, v]) => `${k}×${v}`)
            if (row.extra) mistakes.push(`extra×${row.extra}`)
            if (row.missed) mistakes.push(`none×${row.missed}`)
            return (
              <tr key={label} className="border-t border-[var(--line)]">
                <td className="py-0.5 pr-2">{label}</td>
                <td className="py-0.5 pr-2 tabular-nums">{row.exactly_once}%</td>
                <td className="py-0.5 text-xs text-[var(--muted)]">{mistakes.join(", ")}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Score({ tag, title, accuracy, f1, detail }: { tag?: string; title: string; accuracy: number; f1: number; detail: string }) {
  return (
    <section className="panel min-w-0 p-5">
      {tag && <Tag tone={tag === "Real data" ? "good" : "muted"}>{tag}</Tag>}
      <h2 className="mt-2 text-sm text-[var(--muted)]">{title}</h2>
      <p className="serif text-5xl">{(accuracy * 100).toFixed(2)}%</p>
      <p className="text-sm">Macro F1 {(f1 * 100).toFixed(2)}%</p>
      <p className="mt-2 text-xs text-[var(--muted)]">{detail}</p>
    </section>
  )
}

function Report({
  title,
  text,
  confusions,
}: {
  title: string
  text: string
  confusions: Array<{ actual: string; predicted: string; count: number }>
}) {
  return (
    <section className="panel min-w-0 p-5">
      <h2 className="serif text-2xl">{title}</h2>
      {confusions.length > 0 && (
        <ul className="mt-2 text-sm text-[var(--muted)]">
          {confusions.slice(0, 4).map((item) => (
            <li key={`${item.actual}-${item.predicted}`}>
              {item.actual} predicted as {item.predicted}: {item.count}
            </li>
          ))}
        </ul>
      )}
      <pre className="mt-3 max-h-80 overflow-auto text-xs leading-relaxed">{text}</pre>
    </section>
  )
}

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

export function EvalPage() {
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(asset("evaluation/metrics.json"))
      .then((res) => {
        if (!res.ok) throw new Error("Metrics file missing")
        return res.json()
      })
      .then(setMetrics)
      .catch((err: Error) => setError(err.message))
  }, [])

  if (error) return <p className="text-[var(--accent)]">{error}</p>
  if (!metrics) return <p className="text-[var(--muted)]">Loading the evaluation report…</p>

  return (
    <article className="space-y-6">
      <header>
        <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">Evaluation</p>
        <h1 className="serif text-4xl">Numbers from the held-out sets</h1>
        <p className="mt-2 max-w-3xl text-sm text-[var(--muted)]">
          Generated {new Date(metrics.generated_at).toLocaleString()}. These are the figures written by <code>python evaluate.py</code>.
          They are not rounded up, and the dynamic figure is not a webcam study.
        </p>
      </header>
      <div className="grid gap-4 md:grid-cols-2">
        <Score title="Static letters" accuracy={metrics.static.accuracy} f1={metrics.static.macro_f1} detail={`${metrics.static.test_size} stratified test rows · z mode ${metrics.static.z_mode}`} />
        <Score title="Retired kinematic test, not used live" accuracy={metrics.dynamic.accuracy} f1={metrics.dynamic.macro_f1} detail="Synthetic motions on real handshapes. The live demo uses handshape rules instead, so do not quote this number as webcam accuracy." />
      </div>
      <section className="panel grid gap-4 p-5 md:grid-cols-2">
        <figure>
          <img src={asset("evaluation/static_confusion.png")} alt="Confusion matrix for the 24 static ASL letters" className="w-full rounded-xl" />
          <figcaption className="mt-2 text-xs text-[var(--muted)]">Static test set. The bright diagonal is correct letters. M and N are the usual mix-up.</figcaption>
        </figure>
        <figure>
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
          <a className="underline" href={metrics.datasets.static.images_url}>Kaggle record</a>. Landmark file:{" "}
          <a className="underline" href={metrics.datasets.static.landmarks_url}>hand_dataset_1000_24.csv</a>.
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

function Score({ title, accuracy, f1, detail }: { title: string; accuracy: number; f1: number; detail: string }) {
  return (
    <section className="panel p-5">
      <h2 className="text-sm text-[var(--muted)]">{title}</h2>
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
    <section className="panel p-5">
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

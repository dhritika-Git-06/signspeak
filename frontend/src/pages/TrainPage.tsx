import { useRef, useState } from "react"
import { Badge, Button, Input, Label } from "@/components/ui"
import { useHandTracker } from "@/hooks/useHandTracker"
import { serverTrain } from "@/lib/backend"
import type { Point } from "@/lib/hands"
import { embedDynamic, embedStatic, loadSupport, trainLinearHead } from "@/lib/model"
import { flattenSequence, normalizeLandmarks, normalizeSequence } from "@/lib/normalize"
import {
  clearHead,
  deleteSign,
  exportBundle,
  importBundle,
  listSigns,
  saveHead,
  saveSign,
  type StoredSign,
} from "@/lib/storage"
import { useModels } from "@/models-context"

export function TrainPage() {
  const models = useModels()
  const tracker = useHandTracker()
  const [kind, setKind] = useState<"static" | "dynamic">("static")
  const [name, setName] = useState("")
  const [status, setStatus] = useState("Name a sign, hold it in frame, and record a few bursts.")
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const recording = useRef<Point[][] | null>(null)
  const bursts = useRef<Point[][][]>([])

  tracker.setHandler((frame) => {
    if (recording.current && frame.landmarks) recording.current.push(frame.landmarks)
  })

  const needed = kind === "static" ? 1 : 6

  async function record() {
    if (!tracker.running) {
      setStatus("Start the camera first, then hold the sign steady.")
      return
    }
    const clean = name.trim().toUpperCase().replace(/\s+/g, "_")
    if (!/^[A-Z][A-Z0-9_-]{0,23}$/.test(clean)) {
      setStatus("Use a short name that starts with a letter, such as MOM or J.")
      return
    }
    setBusy(true)
    setStatus(kind === "static" ? "Hold still — capturing a burst…" : "Perform the gesture once, through the whole motion…")
    const bucket: Point[][] = []
    recording.current = bucket
    await new Promise((resolve) => window.setTimeout(resolve, kind === "static" ? 1400 : 1800))
    recording.current = null
    if (bucket.length < 8) {
      setStatus("The camera did not see a hand. Step into the light and try again.")
      setBusy(false)
      return
    }
    bursts.current = [...bursts.current, bucket]
    setProgress(bursts.current.length)
    setStatus(
      kind === "static"
        ? `Captured ${bucket.length} frames. You can record another angle, or train now.`
        : `Saved take ${bursts.current.length} of ${needed}. Vary the speed a little between takes.`,
    )
    setBusy(false)
  }

  async function train() {
    const clean = name.trim().toUpperCase().replace(/\s+/g, "_")
    if (!bursts.current.length) {
      setStatus("Record at least one burst before training.")
      return
    }
    if (kind === "dynamic" && bursts.current.length < needed) {
      setStatus(`Dynamic signs need ${needed} separate takes. You have ${bursts.current.length}.`)
      return
    }
    if (!models.embed || !models.dynamicEmbed || !models.meta) {
      setStatus("The base models are still loading.")
      return
    }
    setBusy(true)
    setStatus("Training a small head in the browser…")
    try {
      const zeroZ = models.meta.z_mode === "zero"
      let vectors: number[][] = []
      let stored: StoredSign
      if (kind === "static") {
        const frames = bursts.current.flat()
        vectors = frames.map((frame) => embedStatic(models.embed!, normalizeLandmarks(frame, { zeroZ })))
        stored = { id: `static:${clean}`, name: clean, kind: "static", samples: frames, createdAt: Date.now() }
      } else {
        vectors = bursts.current.map((take) =>
          embedDynamic(
            models.dynamicEmbed!,
            flattenSequence(normalizeSequence(take, models.meta!.sequence_length, { zeroZ })),
          ),
        )
        stored = { id: `dynamic:${clean}`, name: clean, kind: "dynamic", samples: bursts.current, createdAt: Date.now() }
      }
      await saveSign(stored)
      const support = await loadSupport(kind)
      const all = (await listSigns()).filter((sign) => sign.kind === kind)
      const custom = all.map((sign) => ({
        name: sign.name,
        vectors:
          sign.id === stored.id
            ? vectors
            : embedStored(sign, models, zeroZ),
      }))
      const head = await trainLinearHead(support, custom)
      await saveHead(kind, { labels: head.labels, weights: head.weights, trainedAt: Date.now() })
      await models.refresh()
      setStatus(`“${clean}” is in the recognizer. Training fit ${Math.round(head.trainAccuracy * 100)}% of the head’s examples.`)
      bursts.current = []
      setProgress(0)
      if (models.apiBase) {
        try {
          await serverTrain(models.apiBase, {
            name: clean,
            kind,
            samples: kind === "static" ? (stored.samples as Point[][]) : stored.samples,
          })
        } catch {
          setStatus((current) => `${current} The browser copy is saved. The server copy could not be updated.`)
        }
      }
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Training failed")
    } finally {
      setBusy(false)
    }
  }

  async function remove(sign: StoredSign) {
    await deleteSign(sign.id)
    const remaining = models.signs.filter((item) => item.id !== sign.id && item.kind === sign.kind)
    if (!remaining.length) await clearHead(sign.kind)
    else {
      const support = await loadSupport(sign.kind)
      const zeroZ = models.meta?.z_mode === "zero"
      const custom = remaining.map((item) => ({
        name: item.name,
        vectors: embedStored(item, models, !!zeroZ),
      }))
      const head = await trainLinearHead(support, custom)
      await saveHead(sign.kind, { labels: head.labels, weights: head.weights, trainedAt: Date.now() })
    }
    await models.refresh()
    setStatus(`Removed ${sign.name}.`)
  }

  async function onExport() {
    const bundle = await exportBundle()
    const blob = new Blob([JSON.stringify(bundle)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = "signspeak-signs.json"
    link.click()
    URL.revokeObjectURL(url)
  }

  async function onImport(file: File) {
    const payload = JSON.parse(await file.text())
    await importBundle(payload)
    await models.refresh()
    setStatus("Imported signs into this browser.")
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1.1fr)_minmax(280px,0.9fr)]">
      <section className="panel overflow-hidden">
        <div className="relative aspect-video bg-[#1a1713]">
          <video ref={tracker.videoRef} className="absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]" playsInline muted />
          <canvas ref={tracker.canvasRef} className="pointer-events-none absolute inset-0 h-full w-full object-cover [transform:scaleX(-1)]" />
          {!tracker.running && (
            <div className="absolute inset-0 grid place-items-center text-center text-[#f4f0e6]">
              <p className="serif text-3xl">Camera for new signs</p>
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-2 p-4">
          {!tracker.running ? (
            <Button onClick={() => void tracker.start()}>Start camera</Button>
          ) : (
            <Button variant="quiet" onClick={tracker.stop}>Stop camera</Button>
          )}
          {tracker.error && <p className="w-full text-sm text-[var(--accent)]">{tracker.error}</p>}
        </div>
      </section>

      <section className="panel space-y-4 p-5">
        <div>
          <h1 className="serif text-3xl">Teach a sign</h1>
          <p className="mt-1 text-sm text-[var(--muted)]">
            The base network stays frozen. A small softmax head learns your samples in this browser and is kept in IndexedDB.
          </p>
        </div>
        <div className="flex gap-1 rounded-full bg-[var(--bg)] p-1" role="group" aria-label="Sign type">
          {(["static", "dynamic"] as const).map((item) => (
            <button
              key={item}
              className={`flex-1 rounded-full px-3 py-1.5 text-sm capitalize ${kind === item ? "bg-[var(--ink)] text-[var(--bg)]" : ""}`}
              onClick={() => {
                setKind(item)
                bursts.current = []
                setProgress(0)
              }}
            >
              {item === "static" ? "Letter / pose" : "Word / motion"}
            </button>
          ))}
        </div>
        <div>
          <Label htmlFor="sign-name">Name</Label>
          <Input id="sign-name" value={name} onChange={(event) => setName(event.target.value)} placeholder={kind === "static" ? "MOM" : "HELLO"} className="mt-1" />
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={() => void record()} disabled={busy}>Record burst</Button>
          <Button variant="quiet" onClick={() => void train()} disabled={busy}>Train head</Button>
        </div>
        <p className="text-sm" role="status">{status}</p>
        <p className="text-xs text-[var(--muted)]">
          Takes this session: {progress}. {kind === "static" ? "One burst is enough; two or three angles help." : `Aim for ${needed} takes.`}
        </p>
        <div className="flex flex-wrap gap-2">
          <Button variant="quiet" size="sm" onClick={() => void onExport()}>Export</Button>
          <label className="inline-flex h-8 cursor-pointer items-center rounded-full border border-[var(--line)] px-3 text-xs">
            Import
            <input
              type="file"
              accept="application/json"
              className="sr-only"
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) void onImport(file)
              }}
            />
          </label>
        </div>
      </section>

      <section className="panel p-5 lg:col-span-2">
        <h2 className="text-sm font-medium">Signs saved in this browser</h2>
        {models.signs.length === 0 ? (
          <p className="mt-2 text-sm text-[var(--muted)]">None yet. Built-in letters and words still run without any custom signs.</p>
        ) : (
          <ul className="mt-3 divide-y divide-[var(--line)]">
            {models.signs.map((sign) => (
              <li key={sign.id} className="flex items-center justify-between gap-3 py-2">
                <span>
                  <span className="font-medium">{sign.name}</span>{" "}
                  <Badge>{sign.kind}</Badge>
                </span>
                <Button variant="danger" size="sm" onClick={() => void remove(sign)}>Remove</Button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  )
}

function embedStored(
  sign: StoredSign,
  models: ReturnType<typeof useModels>,
  zeroZ: boolean,
): number[][] {
  if (!models.embed || !models.dynamicEmbed || !models.meta) return []
  if (sign.kind === "static") {
    const frames = sign.samples as Point[][]
    return frames.map((frame) => embedStatic(models.embed!, normalizeLandmarks(frame, { zeroZ })))
  }
  const takes = sign.samples as Point[][][]
  return takes.map((take) =>
    embedDynamic(
      models.dynamicEmbed!,
      flattenSequence(normalizeSequence(take, models.meta!.sequence_length, { zeroZ })),
    ),
  )
}

// Node-only helpers for tests and reports: load the shipped TF.js weights from public/models.
import { readFileSync } from "node:fs"
import path from "node:path"
import { buildDynamic, buildStatic, predictDynamic, predictStatic, type Meta } from "@/lib/model"
import { flattenSequence, motionEnergy, normalizeLandmarks, normalizeSequence } from "@/lib/normalize"
import type { Point } from "@/lib/hands"
import type { RecognizerDeps } from "@/lib/recognizer"

const root = path.resolve(__dirname, "../../public/models")

function loadBin(stem: "static" | "dynamic") {
  const manifest = JSON.parse(readFileSync(path.join(root, `${stem}.json`), "utf8")) as {
    tensors: Array<{ shape: number[]; offset: number; length: number }>
  }
  const data = new Float32Array(readFileSync(path.join(root, `${stem}.bin`)).buffer)
  return manifest.tensors.map((tensor) => Array.from(data.slice(tensor.offset, tensor.offset + tensor.length)))
}

export async function loadShippedModels() {
  const tf = await import("@tensorflow/tfjs")
  await tf.setBackend("cpu")
  const meta = JSON.parse(readFileSync(path.join(root, "meta.json"), "utf8")) as Meta
  const classifier = buildStatic(meta.static_labels, meta.static_filters)
  classifier.setWeights(loadBin("static").map((row, i) => tf.tensor(row, classifier.getWeights()[i].shape)))
  const dynamic = buildDynamic(meta.dynamic_labels, meta.dynamic_lstm)
  dynamic.setWeights(loadBin("dynamic").map((row, i) => tf.tensor(row, dynamic.getWeights()[i].shape)))
  const zeroZ = meta.z_mode === "zero"
  const letter = (frame: Point[], mirror: boolean) => predictStatic(classifier, normalizeLandmarks(frame, { mirror, zeroZ }), meta.static_labels)
  const deps: RecognizerDeps = {
    letter,
    dynamic: (frames, mirror, kind) => {
      const seq = kind === "word"
        ? normalizeSequence(frames.slice(-meta.sequence_length), meta.sequence_length, { mirror, zeroZ })
        : normalizeSequence(frames, meta.sequence_length, { mirror, zeroZ })
      return predictDynamic(dynamic, flattenSequence(seq), meta.dynamic_labels)
    },
    motion: (frames, mirror) => motionEnergy(normalizeSequence(frames, meta.sequence_length, { mirror, zeroZ })),
    motionThreshold: meta.motion_threshold,
  }
  return { tf, meta, classifier, dynamic, letter, deps }
}

import * as tf from "@tensorflow/tfjs"
import { asset } from "./asset"

export interface Meta {
  version: string
  static_labels: string[]
  dynamic_labels: string[]
  z_mode: "zero" | "passthrough"
  sequence_length: number
  static_embed_dim: number
  dynamic_embed_dim: number
  static_filters: number[]
  dynamic_lstm: number[]
  motion_threshold: number
  static_accuracy: number
  dynamic_accuracy: number
  static_macro_f1?: number
  dynamic_macro_f1?: number
}

interface Manifest {
  file: string
  tensors: Array<{ name: string; shape: number[]; offset: number; length: number }>
}

async function loadTensors(manifestUrl: string): Promise<tf.Tensor[]> {
  const manifest = (await fetch(manifestUrl).then((res) => {
    if (!res.ok) throw new Error(`Missing ${manifestUrl}`)
    return res.json()
  })) as Manifest
  const binUrl = new URL(manifest.file, new URL(manifestUrl, window.location.href)).toString()
  const buffer = await fetch(binUrl).then((res) => {
    if (!res.ok) throw new Error(`Missing ${binUrl}`)
    return res.arrayBuffer()
  })
  const data = new Float32Array(buffer)
  return manifest.tensors.map((tensor) =>
    tf.tensor(Array.from(data.slice(tensor.offset, tensor.offset + tensor.length)), tensor.shape),
  )
}

export function buildStatic(labels: string[], filters: number[]) {
  const model = tf.sequential()
  model.add(
    tf.layers.conv1d({
      filters: filters[0],
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      inputShape: [21, 3],
    }),
  )
  model.add(tf.layers.conv1d({ filters: filters[1], kernelSize: 3, padding: "same", activation: "relu" }))
  model.add(tf.layers.conv1d({ filters: filters[2], kernelSize: 3, padding: "same", activation: "relu" }))
  model.add(tf.layers.globalAveragePooling1d({}))
  model.add(tf.layers.dense({ units: 128, activation: "relu" }))
  model.add(tf.layers.dense({ units: labels.length, activation: "softmax" }))
  return model
}

export function buildStaticEmbed(filters: number[]) {
  const model = tf.sequential()
  model.add(
    tf.layers.conv1d({
      filters: filters[0],
      kernelSize: 3,
      padding: "same",
      activation: "relu",
      inputShape: [21, 3],
    }),
  )
  model.add(tf.layers.conv1d({ filters: filters[1], kernelSize: 3, padding: "same", activation: "relu" }))
  model.add(tf.layers.conv1d({ filters: filters[2], kernelSize: 3, padding: "same", activation: "relu" }))
  model.add(tf.layers.globalAveragePooling1d({}))
  model.add(tf.layers.dense({ units: 128, activation: "relu" }))
  return model
}

export function buildDynamic(labels: string[], units: number[]) {
  const model = tf.sequential()
  model.add(tf.layers.lstm({ units: units[0], returnSequences: true, inputShape: [30, 63] }))
  model.add(tf.layers.lstm({ units: units[1], returnSequences: false }))
  model.add(tf.layers.dense({ units: 32, activation: "relu" }))
  model.add(tf.layers.dense({ units: labels.length, activation: "softmax" }))
  return model
}

export function buildDynamicEmbed(units: number[]) {
  const model = tf.sequential()
  model.add(tf.layers.lstm({ units: units[0], returnSequences: true, inputShape: [30, 63] }))
  model.add(tf.layers.lstm({ units: units[1], returnSequences: false }))
  model.add(tf.layers.dense({ units: 32, activation: "relu" }))
  return model
}

export interface Ranked {
  label: string
  confidence: number
}

function rank(data: ArrayLike<number>, labels: string[]): { label: string; confidence: number; top: Ranked[] } {
  let best = 0
  for (let i = 1; i < data.length; i++) if (data[i] > data[best]) best = i
  const top = Array.from(data)
    .map((confidence, index) => ({ label: labels[index], confidence }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3)
  return { label: labels[best], confidence: data[best], top }
}

export function predictStatic(model: tf.LayersModel, input: number[][], labels: string[]) {
  const values = tf.tidy(() => {
    const probs = model.predict(tf.tensor(input.flat(), [1, 21, 3])) as tf.Tensor
    return Array.from(probs.dataSync())
  }) as number[]
  return rank(values, labels)
}

export function embedStatic(model: tf.LayersModel, input: number[][]): number[] {
  return tf.tidy(() => {
    const emb = model.predict(tf.tensor(input.flat(), [1, 21, 3])) as tf.Tensor
    return Array.from(emb.dataSync())
  })
}

export function predictDynamic(model: tf.LayersModel, flat: number[][], labels: string[]) {
  const values = tf.tidy(() => {
    const probs = model.predict(tf.tensor(flat.flat(), [1, 30, 63])) as tf.Tensor
    return Array.from(probs.dataSync())
  }) as number[]
  return rank(values, labels)
}

export function embedDynamic(model: tf.LayersModel, flat: number[][]): number[] {
  return tf.tidy(() => {
    const emb = model.predict(tf.tensor(flat.flat(), [1, 30, 63])) as tf.Tensor
    return Array.from(emb.dataSync())
  })
}

export async function loadModels(meta: Meta) {
  const [staticTensors, dynamicTensors] = await Promise.all([
    loadTensors(asset("models/static.json")),
    loadTensors(asset("models/dynamic.json")),
  ])
  const classifier = buildStatic(meta.static_labels, meta.static_filters)
  const embed = buildStaticEmbed(meta.static_filters)
  const dynamic = buildDynamic(meta.dynamic_labels, meta.dynamic_lstm)
  const dynamicEmbed = buildDynamicEmbed(meta.dynamic_lstm)
  const own = (tensors: tf.Tensor[]) => tensors.map((tensor) => tensor.clone())
  classifier.setWeights(own(staticTensors))
  embed.setWeights(own(staticTensors.slice(0, 8)))
  dynamic.setWeights(own(dynamicTensors))
  dynamicEmbed.setWeights(own(dynamicTensors.slice(0, 8)))
  staticTensors.forEach((tensor) => tensor.dispose())
  dynamicTensors.forEach((tensor) => tensor.dispose())
  return { classifier, embed, dynamic, dynamicEmbed }
}

export interface SupportSet {
  dim: number
  labels: string[]
  vectors: number[][]
}

export async function loadSupport(kind: "static" | "dynamic"): Promise<SupportSet> {
  const res = await fetch(asset(`models/support_${kind}.json`))
  if (!res.ok) throw new Error("Support set missing")
  return res.json()
}

export async function trainLinearHead(
  support: SupportSet,
  custom: Array<{ name: string; vectors: number[][] }>,
) {
  const names = Array.from(new Set(support.labels))
  const index = new Map(names.map((name, i) => [name, i]))
  const xs: number[][] = support.vectors.map((row) => row.slice())
  const ys: number[] = support.labels.map((name) => index.get(name) ?? 0)
  for (const sign of custom) {
    if (!sign.vectors.length) continue
    if (!index.has(sign.name)) {
      index.set(sign.name, names.length)
      names.push(sign.name)
    }
    const reps = Math.max(1, Math.round(support.vectors.length / names.length / sign.vectors.length))
    for (let r = 0; r < reps; r++) {
      for (const vector of sign.vectors) {
        xs.push(vector)
        ys.push(index.get(sign.name) ?? 0)
      }
    }
  }
  const model = tf.sequential()
  model.add(tf.layers.dense({ units: names.length, activation: "softmax", inputShape: [xs[0].length] }))
  model.compile({ optimizer: tf.train.adam(0.08), loss: "categoricalCrossentropy" })
  const yOne = ys.map((slot) => {
    const row = new Array(names.length).fill(0)
    row[slot] = 1
    return row
  })
  const xTensor = tf.tensor2d(xs)
  const yTensor = tf.tensor2d(yOne)
  const history = await model.fit(xTensor, yTensor, { epochs: 36, batchSize: 64, shuffle: true })
  const accTensor = model.predict(xTensor) as tf.Tensor
  const predicted = accTensor.argMax(1).dataSync()
  let hits = 0
  predicted.forEach((value, i) => {
    if (value === ys[i]) hits += 1
  })
  xTensor.dispose()
  yTensor.dispose()
  accTensor.dispose()
  const weights = model.getWeights().map((weight) => ({
    data: Array.from(weight.dataSync()),
    shape: weight.shape,
  }))
  const loss = history.history.loss?.at(-1)
  return {
    model,
    labels: names,
    weights,
    trainAccuracy: hits / ys.length,
    loss: typeof loss === "number" ? loss : null,
  }
}

export function headFromWeights(weights: Array<{ data: number[]; shape: number[] }>, inputDim: number, outputs: number) {
  const model = tf.sequential()
  model.add(tf.layers.dense({ units: outputs, activation: "softmax", inputShape: [inputDim] }))
  model.setWeights(weights.map((weight) => tf.tensor(weight.data, weight.shape)))
  return model
}

export function predictHead(model: tf.LayersModel, vector: number[], labels: string[]) {
  const values = tf.tidy(() => {
    const probs = model.predict(tf.tensor2d([vector])) as tf.Tensor
    return Array.from(probs.dataSync())
  }) as number[]
  return rank(values, labels)
}

import { readFileSync } from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"
import { buildDynamic, buildStatic, predictDynamic, predictStatic } from "./model"

function loadBin(stem: "static" | "dynamic") {
  const root = path.resolve(__dirname, "../../public/models")
  const manifest = JSON.parse(readFileSync(path.join(root, `${stem}.json`), "utf8")) as {
    tensors: Array<{ shape: number[]; offset: number; length: number }>
  }
  const data = new Float32Array(readFileSync(path.join(root, `${stem}.bin`)).buffer)
  return manifest.tensors.map((tensor) => Array.from(data.slice(tensor.offset, tensor.offset + tensor.length)))
}

describe("TensorFlow.js weights match Keras", () => {
  it("reproduces a static letter probability vector", async () => {
    const tf = await import("@tensorflow/tfjs")
    const meta = JSON.parse(readFileSync(path.resolve(__dirname, "../../public/models/meta.json"), "utf8"))
    const fixture = JSON.parse(readFileSync(path.resolve(__dirname, "../../../models/parity_static.json"), "utf8"))
    const model = buildStatic(meta.static_labels, meta.static_filters)
    const values = loadBin("static")
    model.setWeights(values.map((row, index) => tf.tensor(row, model.getWeights()[index].shape)))
    const input: number[][] = []
    for (let i = 0; i < 21; i++) input.push(fixture.normalized.slice(i * 3, i * 3 + 3))
    const pred = predictStatic(model, input, meta.static_labels)
    expect(pred.label).toBe(fixture.label)
    const again = tf.tidy(() => {
      const probs = model.predict(tf.tensor(input.flat(), [1, 21, 3])) as tf.Tensor
      return Array.from(probs.dataSync())
    }) as number[]
    again.forEach((value, index) => expect(value).toBeCloseTo(fixture.probs[index], 3))
  })

  it("reproduces a dynamic sequence probability vector", async () => {
    const tf = await import("@tensorflow/tfjs")
    const meta = JSON.parse(readFileSync(path.resolve(__dirname, "../../public/models/meta.json"), "utf8"))
    const fixture = JSON.parse(readFileSync(path.resolve(__dirname, "../../../models/parity_dynamic.json"), "utf8"))
    const model = buildDynamic(meta.dynamic_labels, meta.dynamic_lstm)
    const values = loadBin("dynamic")
    model.setWeights(values.map((row, index) => tf.tensor(row, model.getWeights()[index].shape)))
    const source = fixture.input as number[][]
    const frames: number[][] = source[0]?.length === 63 ? source : []
    const pred = predictDynamic(model, frames, meta.dynamic_labels)
    expect(pred.label).toBe(fixture.label)
    const again = tf.tidy(() => {
      const probs = model.predict(tf.tensor([frames])) as tf.Tensor
      return Array.from(probs.dataSync())
    }) as number[]
    again.forEach((value, index) => expect(value).toBeCloseTo(fixture.probs[index], 2))
  })
})

import type * as tf from "@tensorflow/tfjs"
import { predictDynamic, predictStatic, type Meta } from "./model"
import { flattenSequence, motionEnergy, normalizeLandmarks, normalizeSequence } from "./normalize"
import type { RecognizerDeps } from "./recognizer"

/** Recognizer inputs from the loaded models, without the custom-sign heads (used by Learn). */
export function liveDeps(meta: Meta, classifier: tf.LayersModel, dynamic: tf.LayersModel): RecognizerDeps {
  const zeroZ = meta.z_mode === "zero"
  return {
    letter: (frame, mirror) => predictStatic(classifier, normalizeLandmarks(frame, { mirror, zeroZ }), meta.static_labels),
    dynamic: (frames, mirror, kind) => {
      const seq = kind === "word"
        ? normalizeSequence(frames.slice(-meta.sequence_length), meta.sequence_length, { mirror, zeroZ })
        : normalizeSequence(frames, meta.sequence_length, { mirror, zeroZ })
      return predictDynamic(dynamic, flattenSequence(seq), meta.dynamic_labels)
    },
    motion: (frames, mirror) => motionEnergy(normalizeSequence(frames, meta.sequence_length, { mirror, zeroZ })),
    motionThreshold: meta.motion_threshold,
  }
}

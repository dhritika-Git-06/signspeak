import type * as tf from "@tensorflow/tfjs"
import { createContext, useContext, useEffect, useState, type ReactNode } from "react"
import { asset } from "@/lib/asset"
import { detectBackend } from "@/lib/backend"
import {
  headFromWeights,
  loadModels,
  type Meta,
} from "@/lib/model"
import { listSigns, loadHead, type StoredSign } from "@/lib/storage"

export interface LiveHead {
  model: tf.LayersModel
  labels: string[]
}

interface ModelsState {
  status: "loading" | "ready" | "error"
  error: string | null
  meta: Meta | null
  classifier: tf.LayersModel | null
  embed: tf.LayersModel | null
  dynamic: tf.LayersModel | null
  dynamicEmbed: tf.LayersModel | null
  signs: StoredSign[]
  heads: { static: LiveHead | null; dynamic: LiveHead | null }
  apiBase: string | null
  refresh: () => Promise<void>
}

const Ctx = createContext<ModelsState | null>(null)

export function ModelsProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ModelsState["status"]>("loading")
  const [error, setError] = useState<string | null>(null)
  const [meta, setMeta] = useState<Meta | null>(null)
  const [nets, setNets] = useState<{
    classifier: tf.LayersModel
    embed: tf.LayersModel
    dynamic: tf.LayersModel
    dynamicEmbed: tf.LayersModel
  } | null>(null)
  const [signs, setSigns] = useState<StoredSign[]>([])
  const [heads, setHeads] = useState<ModelsState["heads"]>({ static: null, dynamic: null })
  const [apiBase, setApiBase] = useState<string | null>(null)

  async function refresh() {
    const stored = await listSigns()
    setSigns(stored)
    const next: ModelsState["heads"] = { static: null, dynamic: null }
    for (const kind of ["static", "dynamic"] as const) {
      const saved = await loadHead(kind)
      if (!saved) continue
      const dim = saved.weights[0]?.shape[0]
      if (!dim) continue
      next[kind] = {
        labels: saved.labels,
        model: headFromWeights(saved.weights, dim, saved.labels.length),
      }
    }
    setHeads(next)
  }

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const metaRes = await fetch(asset("models/meta.json"))
        if (!metaRes.ok) throw new Error("Classifier metadata is missing")
        const loadedMeta = (await metaRes.json()) as Meta
        const loaded = await loadModels(loadedMeta)
        if (cancelled) return
        setMeta(loadedMeta)
        setNets(loaded)
        await refresh()
        setStatus("ready")
        const base = await detectBackend()
        if (!cancelled) setApiBase(base)
      } catch (err) {
        if (cancelled) return
        setStatus("error")
        setError(err instanceof Error ? err.message : "Could not load the models")
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const value: ModelsState = {
    status,
    error,
    meta,
    classifier: nets?.classifier ?? null,
    embed: nets?.embed ?? null,
    dynamic: nets?.dynamic ?? null,
    dynamicEmbed: nets?.dynamicEmbed ?? null,
    signs,
    heads,
    apiBase,
    refresh,
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useModels() {
  const value = useContext(Ctx)
  if (!value) throw new Error("useModels must be used inside ModelsProvider")
  return value
}

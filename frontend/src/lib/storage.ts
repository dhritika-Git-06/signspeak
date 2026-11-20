import { openDB, type DBSchema } from "idb"

export interface StoredSign {
  id: string
  name: string
  kind: "static" | "dynamic"
  samples: number[][][] | number[][][][]
  createdAt: number
}

export interface StoredHead {
  labels: string[]
  weights: Array<{ data: number[]; shape: number[] }>
  trainedAt: number
}

interface SignDB extends DBSchema {
  signs: { key: string; value: StoredSign }
  kv: { key: string; value: StoredHead | string }
}

const dbPromise = openDB<SignDB>("signspeak", 1, {
  upgrade(db) {
    db.createObjectStore("signs", { keyPath: "id" })
    db.createObjectStore("kv")
  },
})

export async function listSigns(): Promise<StoredSign[]> {
  const db = await dbPromise
  const all = await db.getAll("signs")
  return all.sort((a, b) => a.createdAt - b.createdAt)
}

export async function saveSign(sign: StoredSign) {
  const db = await dbPromise
  await db.put("signs", sign)
}

export async function deleteSign(id: string) {
  const db = await dbPromise
  await db.delete("signs", id)
}

export async function saveHead(kind: "static" | "dynamic", head: StoredHead) {
  const db = await dbPromise
  await db.put("kv", head, `head-${kind}`)
}

export async function loadHead(kind: "static" | "dynamic"): Promise<StoredHead | null> {
  const db = await dbPromise
  const value = await db.get("kv", `head-${kind}`)
  if (!value || typeof value === "string") return null
  return value
}

export async function clearHead(kind: "static" | "dynamic") {
  const db = await dbPromise
  await db.delete("kv", `head-${kind}`)
}

export async function exportBundle() {
  const signs = await listSigns()
  const heads = {
    static: await loadHead("static"),
    dynamic: await loadHead("dynamic"),
  }
  return { version: 1, signs, heads }
}

export async function importBundle(payload: {
  version?: number
  signs?: StoredSign[]
  heads?: { static?: StoredHead | null; dynamic?: StoredHead | null }
}) {
  if (!payload || !Array.isArray(payload.signs)) throw new Error("That file is not a SignSpeak export")
  for (const sign of payload.signs) {
    if (!sign.id || !sign.name || !sign.kind) continue
    await saveSign(sign)
  }
  if (payload.heads?.static) await saveHead("static", payload.heads.static)
  if (payload.heads?.dynamic) await saveHead("dynamic", payload.heads.dynamic)
}

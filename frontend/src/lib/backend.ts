export interface Health {
  status: string
  static_labels: string[]
  dynamic_labels: string[]
  custom_static: string[]
  custom_dynamic: string[]
}

export async function detectBackend(): Promise<string | null> {
  // The GitHub Pages build is static only: no server voice and no server training, so do not probe.
  if (import.meta.env.VITE_STATIC_ONLY === "1") return null
  const candidates: string[] = []
  const configured = import.meta.env.VITE_API_URL as string | undefined
  if (configured) candidates.push(configured.replace(/\/$/, ""))
  const host = window.location.hostname
  // GitHub Pages is static hosting, so there is no /api there; skip the probe and its 404.
  if (!host.endsWith("github.io")) candidates.push("/api")
  if (import.meta.env.DEV && (host === "localhost" || host === "127.0.0.1")) candidates.push("http://127.0.0.1:8765")
  for (const base of candidates) {
    try {
      const res = await fetch(`${base}/health`, { signal: AbortSignal.timeout(1500) })
      if (!res.ok) continue
      const type = res.headers.get("content-type") ?? ""
      if (!type.includes("json")) continue
      const body = (await res.json()) as { status?: string }
      if (body?.status) return base
    } catch {
      /* try the next candidate */
    }
  }
  return null
}

export async function serverTrain(
  apiBase: string,
  body: { name: string; kind: "static" | "dynamic"; samples: unknown },
) {
  const res = await fetch(`${apiBase}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Train failed (${res.status})`)
  }
  return res.json()
}

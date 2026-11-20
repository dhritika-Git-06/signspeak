export interface Line {
  at: Date
  english: string
  hindi: string
}

/** Plain-text transcript: every spoken line with its time, then the line still being built. */
export function transcriptText(lines: Line[], current: { english: string; hindi: string } | null, withHindi: boolean): string {
  if (!lines.length && !current) return ""
  const out = ["SignSpeak AI transcript", `Saved ${new Date().toLocaleString()}`, ""]
  for (const line of lines) {
    out.push(`[${line.at.toLocaleTimeString()}] ${line.english}`)
    if (withHindi && line.hindi) out.push(`    ${line.hindi}`)
  }
  if (current) {
    out.push(`[not yet spoken] ${current.english}`)
    if (withHindi && current.hindi) out.push(`    ${current.hindi}`)
  }
  return out.join("\n") + "\n"
}

export function downloadText(text: string, filename: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }))
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

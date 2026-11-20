import { useEffect, useMemo, useState } from "react"
import { HAND_EDGES, type Point } from "@/lib/hands"

/**
 * A hand drawn from landmarks. With several frames it plays the motion once when `play` changes
 * (and loops while `loop` is set). Coordinates can be in any units; the drawing fits the box.
 */
export function HandFigure({
  frames,
  second,
  play = 0,
  loop = false,
  size = 180,
  label,
}: {
  frames: Point[][]
  second?: Point[][]
  play?: number
  loop?: boolean
  size?: number
  label: string
}) {
  const [index, setIndex] = useState(frames.length - 1)
  const box = useMemo(() => {
    const all = [...frames, ...(second ?? [])].flat()
    const xs = all.map((p) => p[0])
    const ys = all.map((p) => p[1])
    const minX = Math.min(...xs)
    const minY = Math.min(...ys)
    const span = Math.max(Math.max(...xs) - minX, Math.max(...ys) - minY, 1e-3)
    return { minX, minY, span, w: Math.max(...xs) - minX, h: Math.max(...ys) - minY }
  }, [frames, second])

  useEffect(() => {
    if (frames.length < 2) return
    let i = 0
    setIndex(0)
    const timer = window.setInterval(() => {
      i += 1
      if (i >= frames.length) {
        if (!loop) {
          window.clearInterval(timer)
          setIndex(frames.length - 1)
          return
        }
        i = 0
      }
      setIndex(i)
    }, 1000 / 30)
    return () => window.clearInterval(timer)
  }, [frames, play, loop])

  const pad = 14
  const scale = (size - pad * 2) / box.span
  const offX = pad + (size - pad * 2 - box.w * scale) / 2
  const offY = pad + (size - pad * 2 - box.h * scale) / 2
  const at = (p: Point) => [offX + (p[0] - box.minX) * scale, offY + (p[1] - box.minY) * scale] as const
  const draw = (pts: Point[] | undefined, key: string) =>
    pts ? (
      <g key={key}>
        {HAND_EDGES.map(([a, b]) => {
          const [x1, y1] = at(pts[a])
          const [x2, y2] = at(pts[b])
          return <line key={`${a}-${b}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#ef8d55" strokeWidth={3} strokeLinecap="round" />
        })}
        {pts.map((p, i) => {
          const [x, y] = at(p)
          return <circle key={i} cx={x} cy={y} r={i === 4 || i === 8 ? 4 : 2.6} fill={i === 4 ? "#2f6db3" : "var(--ink)"} />
        })}
      </g>
    ) : null
  const i = Math.min(index, frames.length - 1)
  return (
    <svg role="img" aria-label={label} width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="rounded-xl bg-[var(--bg)]">
      {draw(second?.[Math.min(i, (second?.length ?? 1) - 1)], "second")}
      {draw(frames[i], "main")}
    </svg>
  )
}

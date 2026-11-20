export const HAND_EDGES: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
]

export type Point = [number, number, number]

/**
 * MediaPipe x and y are fractions of the frame's width and height. Multiplying x by width / height
 * puts both in the same units, so a hand has the same shape on a 16:9 webcam as in the square
 * training photos.
 */
export function squareFrame(points: Point[], aspect: number): Point[] {
  return points.map((p) => [p[0] * aspect, p[1], p[2]] as Point)
}

export function drawLandmarks(
  canvas: HTMLCanvasElement,
  points: Point[],
  space: "image" | "normalized",
) {
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  const project = (p: Point): [number, number] => {
    if (space === "image") {
      return [p[0] * canvas.width, p[1] * canvas.height]
    }
    const xs = points.map((q) => q[0])
    const ys = points.map((q) => q[1])
    const minX = Math.min(...xs)
    const maxX = Math.max(...xs)
    const minY = Math.min(...ys)
    const maxY = Math.max(...ys)
    const span = Math.max(maxX - minX, maxY - minY, 1e-3)
    const pad = 36
    const scale = Math.min(canvas.width - pad * 2, canvas.height - pad * 2) / span
    return [(p[0] - minX) * scale + pad, (p[1] - minY) * scale + pad]
  }
  ctx.lineWidth = 3
  ctx.strokeStyle = "#ef8d55"
  ctx.lineCap = "round"
  for (const [a, b] of HAND_EDGES) {
    const pa = project(points[a])
    const pb = project(points[b])
    ctx.beginPath()
    ctx.moveTo(pa[0], pa[1])
    ctx.lineTo(pb[0], pb[1])
    ctx.stroke()
  }
  ctx.fillStyle = "#1c1914"
  for (const p of points) {
    const [x, y] = project(p)
    ctx.beginPath()
    ctx.arc(x, y, 4.5, 0, Math.PI * 2)
    ctx.fill()
  }
}

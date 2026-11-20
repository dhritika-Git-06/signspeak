import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision"
import { useCallback, useEffect, useRef, useState } from "react"
import { asset } from "@/lib/asset"
import { drawLandmarks, squareFrame, type Point } from "@/lib/hands"

export interface HandFrame {
  /** Main hand in square units (x scaled by width / height), ready for recognition. */
  landmarks: Point[] | null
  handedness: string | null
  /** A second hand, when one is in view, in the same units. */
  second?: Point[] | null
  /** Main hand in raw image fractions (0..1), for framing hints. */
  raw?: Point[] | null
  /** Mean brightness of the camera image, 0 (black) to 255, sampled about once a second. */
  brightness?: number | null
}

function toPoints(hand: Array<{ x: number; y: number; z?: number }>): Point[] {
  return hand.map((p) => [p.x, p.y, p.z ?? 0] as Point)
}

/**
 * Keep following the same hand when two are in view: the one whose wrist is nearest the last
 * main hand. MediaPipe may list the two hands in either order from frame to frame.
 */
export function pickMain(hands: Point[][], previous: Point[] | null): number {
  if (hands.length < 2 || !previous) return 0
  const d = (h: Point[]) => Math.hypot(h[0][0] - previous[0][0], h[0][1] - previous[0][1])
  return d(hands[1]) < d(hands[0]) ? 1 : 0
}

export function useHandTracker() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const markerRef = useRef<HandLandmarker | null>(null)
  const onFrame = useRef<(frame: HandFrame) => void>(() => {})
  const raf = useRef(0)
  const active = useRef(false)
  const starting = useRef(false)
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle")
  const [error, setError] = useState<string | null>(null)
  const [fps, setFps] = useState(0)

  async function ensure() {
    if (markerRef.current) return markerRef.current
    setStatus("loading")
    const vision = await FilesetResolver.forVisionTasks(asset("wasm"))
    const options = {
      runningMode: "VIDEO" as const,
      numHands: 2,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.45,
    }
    try {
      markerRef.current = await HandLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetPath: asset("models/hand_landmarker.task"), delegate: "GPU" },
      })
    } catch {
      markerRef.current = await HandLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetPath: asset("models/hand_landmarker.task"), delegate: "CPU" },
      })
    }
    setStatus("ready")
    return markerRef.current
  }

  function paint(hands: Point[][]) {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas) return
    if (video && video.videoWidth) {
      if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth
      if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight
    } else if (canvas.width < 32) {
      canvas.width = 640
      canvas.height = 480
    }
    if (!hands.length) {
      const ctx = canvas.getContext("2d")
      ctx?.clearRect(0, 0, canvas.width, canvas.height)
      return
    }
    hands.forEach((points, index) => drawLandmarks(canvas, points, "image", index === 0))
  }

  const lastMain = useRef<Point[] | null>(null)
  const probe = useRef<HTMLCanvasElement | null>(null)
  const brightness = useRef<number | null>(null)

  /** Average brightness of a tiny copy of the frame. */
  function sampleBrightness(video: HTMLVideoElement) {
    try {
      probe.current ??= document.createElement("canvas")
      const c = probe.current
      c.width = 32
      c.height = 18
      const ctx = c.getContext("2d", { willReadFrequently: true })
      if (!ctx) return
      ctx.drawImage(video, 0, 0, 32, 18)
      const data = ctx.getImageData(0, 0, 32, 18).data
      let sum = 0
      for (let i = 0; i < data.length; i += 4) sum += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]
      brightness.current = sum / (data.length / 4)
    } catch {
      brightness.current = null
    }
  }

  async function start() {
    if (active.current || starting.current) return
    starting.current = true
    setError(null)
    let stream: MediaStream | null = null
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw Object.assign(new Error("Camera needs a secure (https) page in a modern browser."), { name: "InsecureContext" })
      }
      // Ask for the camera first so the permission prompt appears straight away.
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      const marker = await ensure()
      const video = videoRef.current
      if (!video) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }
      video.srcObject = stream
      await video.play()
      active.current = true
      setRunning(true)
      let last = performance.now()
      let frames = 0
      let stamp = performance.now()
      const loop = () => {
        if (!active.current) return
        raf.current = requestAnimationFrame(loop)
        if (!videoRef.current || videoRef.current.readyState < 2 || !markerRef.current) return
        const now = performance.now()
        if (now - last < 30) return
        last = now
        frames += 1
        if (now - stamp > 1000) {
          setFps(frames)
          frames = 0
          stamp = now
        }
        const video = videoRef.current
        const result = marker.detectForVideo(video, now)
        if (frames % 30 === 1) sampleBrightness(video)
        const hands = (result.landmarks ?? []).map(toPoints)
        const main = pickMain(hands, lastMain.current)
        const ordered = hands.length > 1 && main === 1 ? [hands[1], hands[0]] : hands
        paint(ordered)
        const raw = ordered[0] ?? null
        lastMain.current = raw
        const handedness = result.handedness?.[hands.length > 1 ? main : 0]?.[0]?.categoryName ?? null
        // Recognition works in square units so a 16:9 webcam sees the same hand shape as the training photos.
        const aspect = video.videoWidth && video.videoHeight ? video.videoWidth / video.videoHeight : 4 / 3
        const landmarks = raw ? squareFrame(raw, aspect) : null
        const second = ordered[1] ? squareFrame(ordered[1], aspect) : null
        onFrame.current({ landmarks, handedness, second, raw, brightness: brightness.current })
      }
      loop()
    } catch (err) {
      stream?.getTracks().forEach((track) => track.stop())
      active.current = false
      setStatus("error")
      setRunning(false)
      setError(cameraMessage(err))
    } finally {
      starting.current = false
    }
  }

  function stop() {
    active.current = false
    cancelAnimationFrame(raf.current)
    const video = videoRef.current
    const stream = video?.srcObject
    if (stream instanceof MediaStream) stream.getTracks().forEach((track) => track.stop())
    if (video) video.srcObject = null
    setRunning(false)
    onFrame.current({ landmarks: null, handedness: null })
  }

  useEffect(() => () => stop(), [])

  return {
    videoRef,
    canvasRef,
    start,
    stop,
    running,
    status,
    error,
    fps,
    setHandler: useCallback((fn: (frame: HandFrame) => void) => {
      onFrame.current = fn
    }, []),
    showSample(points: Point[], space: "image" | "normalized", second?: Point[] | null) {
      const canvas = canvasRef.current
      if (!canvas) return
      // Built-in demo frames are drawn for a 16:9 view, like the camera.
      canvas.width = space === "image" ? 1280 : 640
      canvas.height = space === "image" ? 720 : 480
      drawLandmarks(canvas, points, space)
      if (second) drawLandmarks(canvas, second, space, false)
    },
  }
}

/** Plain-language camera errors, with the no-camera fallback named. */
export function cameraMessage(err: unknown): string {
  const name = err instanceof Error || (typeof DOMException !== "undefined" && err instanceof DOMException) ? (err as Error).name : ""
  const message = err instanceof Error ? err.message : String(err ?? "")
  const fallback = " You can still use Play presentation demo or the recorded samples."
  if (name === "NotAllowedError" || name === "SecurityError" || /denied|notallowed|permission/i.test(message)) {
    return "Camera permission was blocked. Click the camera icon in the address bar, allow it, then press Start camera again." + fallback
  }
  if (name === "NotFoundError" || name === "OverconstrainedError" || /not found|no device/i.test(message)) {
    return "No camera was found on this device." + fallback
  }
  if (name === "NotReadableError" || name === "AbortError" || /could not start|in use/i.test(message)) {
    return "The camera is busy in another app or tab (Zoom, Meet, Camera). Close it and press Start camera again." + fallback
  }
  if (name === "InsecureContext") return message + fallback
  return `Could not start the camera or hand tracker: ${message || "unknown error"}.` + fallback
}

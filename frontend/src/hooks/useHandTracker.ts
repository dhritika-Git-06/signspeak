import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from "@mediapipe/tasks-vision"
import { useCallback, useEffect, useRef, useState } from "react"
import { asset } from "@/lib/asset"
import { drawLandmarks, type Point } from "@/lib/hands"

export interface HandFrame {
  landmarks: Point[] | null
  handedness: string | null
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
      numHands: 1,
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

  function paint(result: HandLandmarkerResult | null, fallback: Point[] | null) {
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
    const points = result?.landmarks?.[0]?.map((p) => [p.x, p.y, p.z ?? 0] as Point) ?? fallback
    if (!points) {
      const ctx = canvas.getContext("2d")
      ctx?.clearRect(0, 0, canvas.width, canvas.height)
      return
    }
    const space = result?.landmarks?.[0] ? "image" : "normalized"
    drawLandmarks(canvas, points, space)
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
        const result = marker.detectForVideo(videoRef.current, now)
        paint(result, null)
        const landmarks = result.landmarks?.[0]?.map((p) => [p.x, p.y, p.z ?? 0] as Point) ?? null
        const handedness = result.handedness?.[0]?.[0]?.categoryName ?? null
        onFrame.current({ landmarks, handedness })
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
    showSample(points: Point[], space: "image" | "normalized") {
      const canvas = canvasRef.current
      if (!canvas) return
      canvas.width = 640
      canvas.height = 480
      drawLandmarks(canvas, points, space)
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

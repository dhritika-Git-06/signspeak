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
    if (active.current) return
    setError(null)
    try {
      const marker = await ensure()
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      const video = videoRef.current
      if (!video) return
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
      setStatus("error")
      setRunning(false)
      const message = err instanceof Error ? err.message : "Camera unavailable"
      setError(
        message.toLowerCase().includes("denied") || message.toLowerCase().includes("notallowed")
          ? "Camera permission was blocked. Allow the camera, or use Play sample."
          : message,
      )
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

import type { Point } from "./hands"

/**
 * A small forward-kinematics hand that produces MediaPipe-style landmarks (21 points, image
 * coordinates 0..1, z relative to the wrist). It is used for the per-sign test harness, the
 * reference cards on the Learn page, and the presentation demo. Lengths are in palm units
 * (wrist to middle knuckle = 1). Angles are degrees.
 */

type V3 = [number, number, number]

export interface FingerPose {
  /** Sideways spread in the palm plane, positive toward the thumb. */
  splay: number
  mcp: number
  pip: number
  dip: number
}

export interface ThumbPose {
  /** Angle from the palm's long axis toward the thumb side (0 = along the index, 90 = straight out). */
  abduct: number
  /** Rotation out of the palm plane toward the palm side (positive = across the front of the palm). */
  oppose: number
  mcp: number
  ip: number
}

export interface Handshape {
  thumb: ThumbPose
  index: FingerPose
  middle: FingerPose
  ring: FingerPose
  pinky: FingerPose
}

export interface Placement {
  /** Wrist position in the image (0..1). */
  x: number
  y: number
  /** Palm length as a fraction of image height. */
  size: number
  /** In-plane rotation; 0 = fingers up, 180 = fingers down, positive turns the fingers toward image right. */
  roll: number
  /** Turn about the vertical axis; 90 shows the thumb edge to the camera. */
  yaw: number
  /** Tilt about the horizontal axis; positive tips the fingers toward the camera. */
  pitch: number
  /** Image width / height; MediaPipe normalises x and y separately. */
  aspect: number
  hand: "Right" | "Left"
}

const MCP: Record<"index" | "middle" | "ring" | "pinky", V3> = {
  index: [0.36, -0.94, 0],
  middle: [0.1, -1, 0],
  ring: [-0.13, -0.94, 0],
  pinky: [-0.33, -0.83, 0],
}
const LENGTHS = {
  index: [0.46, 0.27, 0.2],
  middle: [0.5, 0.31, 0.21],
  ring: [0.47, 0.29, 0.2],
  pinky: [0.37, 0.22, 0.18],
}
const D = Math.PI / 180

function add(a: V3, b: V3): V3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
function mul(a: V3, k: number): V3 {
  return [a[0] * k, a[1] * k, a[2] * k]
}

function finger(base: V3, lengths: number[], pose: FingerPose): V3[] {
  // Up the palm with a sideways spread, then flex each joint toward the palm side.
  const s = pose.splay * D
  const dir: V3 = [Math.sin(s), -Math.cos(s), 0]
  const pts: V3[] = []
  let p = base
  const flex = [pose.mcp, pose.pip, pose.dip]
  let total = 0
  for (let i = 0; i < 3; i++) {
    total += flex[i]
    p = add(p, mul(flexDir(dir, total), lengths[i]))
    pts.push(p)
  }
  return pts
}

/** Bend a direction (unit, in the palm plane) toward the palm normal (-z) by `deg`. */
function flexDir(dir: V3, deg: number): V3 {
  const a = deg * D
  return [dir[0] * Math.cos(a), dir[1] * Math.cos(a), -Math.sin(a)]
}

function thumb(pose: ThumbPose): V3[] {
  const cmc: V3 = [0.22, -0.16, -0.02]
  // Base direction: from along the index (-y) rotated toward +x by abduct, then toward -z by oppose.
  const ab = pose.abduct * D
  const op = pose.oppose * D
  let dir: V3 = [Math.sin(ab) * Math.cos(op), -Math.cos(ab) * Math.cos(op), -Math.sin(op)]
  const mcpPt = add(cmc, mul(dir, 0.42))
  // Thumb flexion curls it toward the palm centre (-x) and the camera side.
  const curl = (d: V3, deg: number): V3 => {
    const a = deg * D
    const n = Math.hypot(d[0], d[1], d[2]) || 1
    const u: V3 = [d[0] / n, d[1] / n, d[2] / n]
    const toward: V3 = [-0.8, -0.2, -0.55]
    return [u[0] * Math.cos(a) + toward[0] * Math.sin(a), u[1] * Math.cos(a) + toward[1] * Math.sin(a), u[2] * Math.cos(a) + toward[2] * Math.sin(a)]
  }
  dir = curl(dir, pose.mcp)
  const ipPt = add(mcpPt, mul(dir, 0.33))
  dir = curl(dir, pose.ip)
  const tip = add(ipPt, mul(dir, 0.27))
  return [cmc, mcpPt, ipPt, tip]
}

/** Local 3D landmarks of a right hand, palm facing the camera, fingers up. */
export function localHand(shape: Handshape): V3[] {
  const pts: V3[] = new Array(21)
  pts[0] = [0, 0, 0]
  const t = thumb(shape.thumb)
  for (let i = 0; i < 4; i++) pts[1 + i] = t[i]
  const order = ["index", "middle", "ring", "pinky"] as const
  order.forEach((name, k) => {
    const base = MCP[name]
    pts[5 + k * 4] = base
    const f = finger(base, LENGTHS[name], shape[name])
    for (let i = 0; i < 3; i++) pts[6 + k * 4 + i] = f[i]
  })
  return pts
}

function rotate(p: V3, roll: number, yaw: number, pitch: number): V3 {
  let [x, y, z] = p
  // pitch about x
  const cp = Math.cos(pitch * D), sp = Math.sin(pitch * D)
  ;[y, z] = [y * cp + z * sp, -y * sp + z * cp]
  // yaw about y
  const cy = Math.cos(yaw * D), sy = Math.sin(yaw * D)
  ;[x, z] = [x * cy + z * sy, -x * sy + z * cy]
  // roll about z (image plane)
  const cr = Math.cos(roll * D), sr = Math.sin(roll * D)
  ;[x, y] = [x * cr - y * sr, x * sr + y * cr]
  return [x, y, z]
}

/** Landmarks in MediaPipe image coordinates. */
export function placeHand(shape: Handshape, at: Placement): Point[] {
  const local = localHand(shape).map((p) => (at.hand === "Left" ? ([-p[0], p[1], p[2]] as V3) : p))
  // A left hand is the mirror image, so its roll and yaw turn the other way.
  const roll = at.hand === "Left" ? -at.roll : at.roll
  const yaw = at.hand === "Left" ? -at.yaw : at.yaw
  return local.map((p) => {
    const r = rotate(p, roll, yaw, at.pitch)
    return [at.x + (r[0] * at.size) / at.aspect, at.y + r[1] * at.size, r[2] * at.size] as Point
  })
}

const straight = (splay = 0): FingerPose => ({ splay, mcp: 0, pip: 0, dip: 0 })
const curled = (splay = 0): FingerPose => ({ splay, mcp: 85, pip: 100, dip: 60 })
const hooked = (splay = 0): FingerPose => ({ splay, mcp: 5, pip: 90, dip: 65 })

/** Handshapes used by the word rules and the reference cards. */
export const SHAPES: Record<string, Handshape> = {
  open5: { thumb: { abduct: 55, oppose: 5, mcp: 0, ip: 0 }, index: straight(9), middle: straight(2), ring: straight(-6), pinky: straight(-14) },
  flatB: { thumb: { abduct: 15, oppose: 25, mcp: 20, ip: 10 }, index: straight(2), middle: straight(0), ring: straight(-2), pinky: straight(-4) },
  fistS: { thumb: { abduct: 25, oppose: 45, mcp: 50, ip: 30 }, index: curled(), middle: curled(), ring: curled(), pinky: curled() },
  fistA: { thumb: { abduct: 2, oppose: 12, mcp: 28, ip: 18 }, index: curled(), middle: curled(), ring: curled(), pinky: curled() },
  thumbUp: { thumb: { abduct: 80, oppose: -10, mcp: 0, ip: 0 }, index: curled(), middle: curled(), ring: curled(), pinky: curled() },
  index1: { thumb: { abduct: 25, oppose: 45, mcp: 50, ip: 30 }, index: straight(0), middle: curled(), ring: curled(), pinky: curled() },
  W: { thumb: { abduct: 25, oppose: 45, mcp: 55, ip: 35 }, index: straight(8), middle: straight(0), ring: straight(-8), pinky: curled() },
  V: { thumb: { abduct: 25, oppose: 45, mcp: 55, ip: 35 }, index: straight(10), middle: straight(-6), ring: curled(), pinky: curled() },
  U: { thumb: { abduct: 25, oppose: 45, mcp: 55, ip: 35 }, index: straight(-2), middle: straight(3), ring: curled(), pinky: curled() },
  ILY: { thumb: { abduct: 70, oppose: 0, mcp: 0, ip: 0 }, index: straight(4), middle: curled(), ring: curled(), pinky: straight(-10) },
  OK: { thumb: { abduct: 30, oppose: 40, mcp: 30, ip: 20 }, index: { splay: 0, mcp: 45, pip: 60, dip: 30 }, middle: straight(4), ring: straight(-3), pinky: straight(-10) },
  flatO: { thumb: { abduct: 15, oppose: 30, mcp: 25, ip: 15 }, index: { splay: 0, mcp: 40, pip: 60, dip: 20 }, middle: { splay: 0, mcp: 40, pip: 60, dip: 20 }, ring: { splay: 0, mcp: 40, pip: 60, dip: 20 }, pinky: { splay: 0, mcp: 40, pip: 60, dip: 20 } },
  C: { thumb: { abduct: 60, oppose: 60, mcp: 15, ip: 10 }, index: { splay: 2, mcp: 40, pip: 30, dip: 15 }, middle: { splay: 0, mcp: 40, pip: 30, dip: 15 }, ring: { splay: -2, mcp: 40, pip: 30, dip: 15 }, pinky: { splay: -4, mcp: 40, pip: 30, dip: 15 } },
  X: { thumb: { abduct: 25, oppose: 45, mcp: 50, ip: 30 }, index: hooked(), middle: curled(), ring: curled(), pinky: curled() },
  claw: { thumb: { abduct: 45, oppose: 15, mcp: 15, ip: 25 }, index: hooked(6), middle: hooked(0), ring: hooked(-5), pinky: hooked(-10) },
}

export interface Variation {
  rand: () => number
  hand?: "Right" | "Left"
  /** Width / height of the camera frame. */
  aspect?: number
  /** Per-landmark jitter in image units. */
  jitter?: number
}

export function seededRandom(seed: number) {
  let t = seed >>> 0
  return () => {
    t += 0x6d2b79f5
    let r = Math.imul(t ^ (t >>> 15), 1 | t)
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r)
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296
  }
}

function between(rand: () => number, lo: number, hi: number) {
  return lo + (hi - lo) * rand()
}

/** Blend two handshapes (0 = a, 1 = b). */
export function blendShape(a: Handshape, b: Handshape, t: number): Handshape {
  const mix = (x: number, y: number) => x + (y - x) * t
  const f = (p: FingerPose, q: FingerPose): FingerPose => ({ splay: mix(p.splay, q.splay), mcp: mix(p.mcp, q.mcp), pip: mix(p.pip, q.pip), dip: mix(p.dip, q.dip) })
  return {
    thumb: { abduct: mix(a.thumb.abduct, b.thumb.abduct), oppose: mix(a.thumb.oppose, b.thumb.oppose), mcp: mix(a.thumb.mcp, b.thumb.mcp), ip: mix(a.thumb.ip, b.thumb.ip) },
    index: f(a.index, b.index), middle: f(a.middle, b.middle), ring: f(a.ring, b.ring), pinky: f(a.pinky, b.pinky),
  }
}

/** Small random changes to each joint, like a real person's version of the shape. */
export function wobbleShape(s: Handshape, rand: () => number, amount = 8): Handshape {
  const w = () => (rand() - 0.5) * 2 * amount
  const f = (p: FingerPose): FingerPose => ({ splay: p.splay + w() * 0.5, mcp: Math.max(0, p.mcp + w()), pip: Math.max(0, p.pip + w()), dip: Math.max(0, p.dip + w() * 0.6) })
  return { thumb: { abduct: s.thumb.abduct + w(), oppose: s.thumb.oppose + w(), mcp: Math.max(0, s.thumb.mcp + w()), ip: Math.max(0, s.thumb.ip + w()) }, index: f(s.index), middle: f(s.middle), ring: f(s.ring), pinky: f(s.pinky) }
}

export interface Motion {
  /** Handshape at time t (0..1). */
  shape: (t: number) => Handshape
  /** Wrist offset in palm units and extra rotation at time t. */
  path?: (t: number) => { dx?: number; dy?: number; grow?: number; roll?: number; pitch?: number }
  /** Base orientation for this sign. */
  roll?: number
  yaw?: number
  pitch?: number
  seconds?: number
}

const still = (shape: Handshape, extra: Partial<Motion> = {}): Motion => ({ shape: () => shape, ...extra })
const cycles = (t: number, n: number) => Math.sin(t * Math.PI * 2 * n)

/** How each word is performed in this app (see WORD_SIGNS). One-hand signs only. */
export const WORD_MOTIONS: Record<string, Motion> = {
  HELLO: { shape: () => SHAPES.open5, path: (t) => ({ dx: cycles(t, 2) * 0.7 }) },
  YES: { shape: () => SHAPES.fistS, path: (t) => ({ dy: cycles(t, 2) * 0.45, pitch: cycles(t, 2) * 15 }) },
  NO: { shape: () => SHAPES.V, path: (t) => ({ dx: cycles(t, 2) * 0.45 }) },
  THANKS: { shape: () => SHAPES.flatB, path: (t) => ({ dy: t * 0.9 }), pitch: 10 },
  PLEASE: { shape: () => SHAPES.flatB, path: (t) => ({ dx: Math.cos(t * Math.PI * 2) * 0.45, dy: Math.sin(t * Math.PI * 2) * 0.45 }) },
  SORRY: { shape: () => SHAPES.fistS, path: (t) => ({ dx: Math.cos(t * Math.PI * 2) * 0.45, dy: Math.sin(t * Math.PI * 2) * 0.45 }) },
  ME: still(SHAPES.index1, { roll: 180, pitch: -10 }),
  YOU: still(SHAPES.index1, { pitch: 10 }),
  WE: { shape: () => SHAPES.index1, path: (t) => ({ dx: -0.6 + t * 1.3 }), roll: 70 },
  LOVE: still(SHAPES.ILY),
  STOP: { shape: () => SHAPES.flatB, path: (t) => ({ grow: 1 + 0.4 * Math.min(1, t * 1.6) }) },
  WATER: still(SHAPES.W),
  MORE: still(SHAPES.flatO, { yaw: 40 }),
  DRINK: { shape: () => SHAPES.C, path: (t) => ({ dy: -t * 0.8 }), yaw: 60 },
  EAT: { shape: () => SHAPES.flatO, path: (t) => ({ dy: -t * 0.8 }), yaw: 40 },
  GO: { shape: () => SHAPES.index1, path: (t) => ({ grow: 1 + 0.4 * Math.min(1, t * 1.6) }), pitch: 25 },
  COME: { shape: () => SHAPES.C, path: (t) => ({ grow: 1 - 0.3 * Math.min(1, t * 1.6) }) },
  GOOD: still(SHAPES.thumbUp, { roll: -90 }),
  BAD: { shape: () => SHAPES.flatB, path: (t) => ({ roll: Math.min(1, t * 1.5) * 150, dy: t * 0.3 }) },
  LIKE: { shape: (t) => blendShape(SHAPES.open5, SHAPES.flatO, Math.min(1, Math.max(0, (t - 0.2) * 2))) },
  WANT: { shape: () => SHAPES.open5, path: (t) => ({ grow: 1 - 0.3 * Math.min(1, t * 1.6) }) },
  NEED: still(SHAPES.X),
  WHERE: { shape: () => SHAPES.index1, path: (t) => ({ dx: cycles(t, 2) * 0.35 }) },
  WHAT: { shape: () => SHAPES.open5, path: (t) => ({ dx: cycles(t, 3) * 0.22 }), pitch: -25 },
  NAME: { shape: () => SHAPES.U, path: (t) => ({ dy: cycles(t, 2) * 0.35 }), roll: 60 },
  FINE: still(SHAPES.OK),
  HOME: { shape: (t) => blendShape(SHAPES.flatO, SHAPES.open5, Math.min(1, Math.max(0, (t - 0.2) * 2))) },
}

/** Render a motion as frames at `fps`, with natural variation. */
export function performMotion(motion: Motion, v: Variation, fps = 30): Point[][] {
  const { rand } = v
  const seconds = (motion.seconds ?? 1.4) * between(rand, 0.8, 1.2)
  const n = Math.max(8, Math.round(seconds * fps))
  const size = between(rand, 0.16, 0.32)
  const base = {
    x: between(rand, 0.35, 0.65),
    y: between(rand, 0.45, 0.7),
    roll: (motion.roll ?? 0) + between(rand, -18, 18),
    yaw: (motion.yaw ?? 0) + between(rand, -20, 20),
    pitch: (motion.pitch ?? 0) + between(rand, -15, 15),
  }
  const shapeNoise = seededRandom(Math.floor(rand() * 1e9))
  const wobbleSeed = Math.floor(rand() * 1e9)
  const amp = between(rand, 0.75, 1.25)
  const frames: Point[][] = []
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1)
    const p = motion.path?.(t) ?? {}
    const grow = 1 + ((p.grow ?? 1) - 1) * amp
    const shape = wobbleShape(motion.shape(t), seededRandom(wobbleSeed), 7)
    const pts = placeHand(shape, {
      x: base.x + ((p.dx ?? 0) * size * amp) / (v.aspect ?? 16 / 9),
      y: base.y + (p.dy ?? 0) * size * amp,
      size: size * grow,
      roll: base.roll + (p.roll ?? 0),
      yaw: base.yaw,
      pitch: base.pitch + (p.pitch ?? 0),
      aspect: v.aspect ?? 16 / 9,
      hand: v.hand ?? "Right",
    })
    const j = v.jitter ?? 0.003
    frames.push(pts.map((q) => [q[0] + (shapeNoise() - 0.5) * 2 * j, q[1] + (shapeNoise() - 0.5) * 2 * j, q[2]] as Point))
  }
  return frames
}

/** Two-hand words: the main (moving or fist) hand and the other hand, in image coordinates. */
export function performTwoHands(id: "HELP" | "SCHOOL", v: Variation, fps = 30): { main: Point[][]; second: Point[][] } {
  const { rand } = v
  const aspect = v.aspect ?? 16 / 9
  const seconds = 1.4 * between(rand, 0.8, 1.2)
  const n = Math.max(10, Math.round(seconds * fps))
  const size = between(rand, 0.15, 0.26)
  const right = v.hand !== "Left"
  const mainHand = right ? "Right" : "Left"
  const otherHand = right ? "Left" : "Right"
  const cx = between(rand, 0.4, 0.6)
  const cy = between(rand, 0.55, 0.7)
  const tilt = between(rand, -12, 12)
  const j = v.jitter ?? 0.003
  const noise = seededRandom(Math.floor(rand() * 1e9))
  const shake = (pts: Point[]) => pts.map((q) => [q[0] + (noise() - 0.5) * 2 * j, q[1] + (noise() - 0.5) * 2 * j, q[2]] as Point)
  const main: Point[][] = []
  const second: Point[][] = []
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1)
    // The flat hand lies palm up across the body, fingers pointing to the side.
    const flatRoll = (right ? -1 : 1) * (80 + tilt)
    const lift = id === "HELP" ? -0.4 * t * size : 0
    const flat = placeHand(SHAPES.flatB, { x: cx, y: cy + lift, size, roll: flatRoll, yaw: 0, pitch: 55, aspect, hand: otherHand })
    let top: Point[]
    if (id === "HELP") {
      top = placeHand(SHAPES.thumbUp, { x: cx + (right ? 0.1 : -0.1) * size / aspect, y: cy - 0.75 * size + lift, size, roll: (right ? -1 : 1) * 90, yaw: 10, pitch: 0, aspect, hand: mainHand })
    } else {
      // The top hand claps down onto the flat hand twice.
      const gap = 0.35 + 0.9 * (0.5 + 0.5 * Math.cos(t * Math.PI * 4))
      top = placeHand(SHAPES.flatB, { x: cx, y: cy - gap * size, size, roll: (right ? 1 : -1) * (80 - tilt), yaw: 0, pitch: -55, aspect, hand: mainHand })
    }
    main.push(shake(top))
    second.push(shake(flat))
  }
  return { main, second }
}

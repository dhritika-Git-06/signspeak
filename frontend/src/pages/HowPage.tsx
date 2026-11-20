const steps = [
  {
    title: "Camera",
    body: "The page asks for one webcam and draws MediaPipe’s 21 hand landmarks over the mirrored preview. Nothing is uploaded unless you explicitly train a sign while the API is reachable.",
  },
  {
    title: "Landmarks",
    body: "Each hand becomes 21 points. Static letters are wrist-centred and divided by the palm size, so the letter does not depend on where the hand sits in the frame. Moving signs keep the path of the wrist relative to the first frame.",
  },
  {
    title: "Letters and word rules",
    body: "A 1D CNN, trained on public ASL Alphabet landmarks, reads a still letter. Common words such as ME, WANT and WATER are geometric rules on the same 21 points. They are not a second accuracy number.",
  },
  {
    title: "A steady sentence",
    body: "A sign is confirmed after it holds steady. Fingerspelled letters can be corrected against a small dictionary. ME WANT WATER becomes “I want water.” The browser speaks the line.",
  },
  {
    title: "Your own signs",
    body: "Train records bursts, embeds them with the frozen network, and fits a small softmax head in TensorFlow.js. The head lives in IndexedDB and can be exported. The API exposes the same idea at POST /train.",
  },
]

export function HowPage() {
  return (
    <article className="space-y-6">
      <header className="max-w-3xl">
        <p className="text-xs uppercase tracking-[0.16em] text-[var(--muted)]">How it works</p>
        <h1 className="serif text-4xl">From a hand shape to a spoken sentence</h1>
        <p className="mt-3 text-[var(--muted)]">
          Built by Dhritika Sahu. It is a fingerspelling and
          isolated-gesture tool: it does not translate full American Sign Language grammar, facial expression, or two-handed conversation.
        </p>
      </header>
      <ol className="grid gap-3 md:grid-cols-2">
        {steps.map((step, index) => (
          <li key={step.title} className="panel p-5">
            <p className="text-xs text-[var(--muted)]">0{index + 1}</p>
            <h2 className="serif text-2xl">{step.title}</h2>
            <p className="mt-2 text-sm text-[var(--muted)]">{step.body}</p>
          </li>
        ))}
      </ol>
      <section className="panel p-5">
        <h2 className="serif text-2xl">What is actually running</h2>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-[var(--muted)]">
          <li>Browser: React, MediaPipe Tasks Vision HandLandmarker, TensorFlow.js, speechSynthesis.</li>
          <li>Optional API: FastAPI, the same weights in NumPy, gTTS, and a server-side head at /train.</li>
          <li>Static letters A–Y except J and Z, trained on 24,000 public MediaPipe rows from the ASL Alphabet photographs.</li>
          <li>J, Z, HELLO, YES, NO, THANKS, PLEASE, and SORRY on an LSTM. Those sequences are motion templates applied to real handshapes from the same photographs, then tested on different photographs.</li>
        </ul>
      </section>
    </article>
  )
}

import { readFileSync } from "node:fs"
import { expect, openApp, sentence, spoken, test } from "./helpers"

async function sayIWantWater(page: import("@playwright/test").Page) {
  const s = page.getByRole("group", { name: "Suggestions" })
  for (const w of ["ME", "WANT", "WATER"]) await s.getByRole("button", { name: w, exact: true }).click()
  await expect(sentence(page)).toHaveText("I want water.")
}

test("sentence tools: Copy, Share and Download .txt", async ({ page, errors }) => {
  void errors
  await page.addInitScript(() => {
    // Headless Chrome has no share sheet; record what would be shared.
    ;(window as unknown as { __shared: unknown[] }).__shared = []
    Object.defineProperty(navigator, "share", {
      configurable: true,
      value: async (data: unknown) => {
        ;(window as unknown as { __shared: unknown[] }).__shared.push(data)
      },
    })
  })
  await openApp(page)
  const tools = page.getByRole("group", { name: "Sentence tools" })
  await tools.getByRole("button", { name: "Copy" }).click()
  await expect(page.getByText("Nothing to copy yet.")).toBeVisible()
  await tools.getByRole("button", { name: "Share" }).click()
  await expect(page.getByText("Nothing to share yet.")).toBeVisible()
  await tools.getByRole("button", { name: "Download .txt" }).click()
  await expect(page.getByText("Nothing to download yet. Sign a sentence first.")).toBeVisible()

  await sayIWantWater(page)
  await tools.getByRole("button", { name: "Copy" }).click()
  await expect(page.getByText("Copied the sentence.")).toBeVisible()
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe("I want water.")
  await tools.getByRole("button", { name: "Share" }).click()
  await expect(page.getByText("Shared.")).toBeVisible()
  expect(await page.evaluate(() => (window as unknown as { __shared: Array<{ text: string }> }).__shared[0].text)).toBe("I want water.")

  await page.getByRole("button", { name: "Speak", exact: true }).click()
  await expect(page.getByText("Spoke with the browser voice.")).toBeVisible()
  await page.getByRole("group", { name: "Suggestions" }).getByRole("button", { name: "HELLO", exact: true }).click()
  const [download] = await Promise.all([page.waitForEvent("download"), tools.getByRole("button", { name: "Download .txt" }).click()])
  expect(download.suggestedFilename()).toMatch(/^signspeak-transcript-\d{4}-\d{2}-\d{2}\.txt$/)
  const text = readFileSync((await download.path())!, "utf8")
  expect(text).toContain("I want water.")
  expect(text).toContain("[not yet spoken] Hello.")
  await expect(page.getByText("Downloaded the transcript as a .txt file.")).toBeVisible()
})

test("Share falls back to copying when the browser cannot share", async ({ page, errors }) => {
  void errors
  await page.addInitScript(() => Object.defineProperty(navigator, "share", { configurable: true, value: undefined }))
  await openApp(page)
  await sayIWantWater(page)
  await page.getByRole("group", { name: "Sentence tools" }).getByRole("button", { name: "Share" }).click()
  await expect(page.getByText("Sharing is not available here, so the sentence was copied instead.")).toBeVisible()
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe("I want water.")
})

test("Hindi: text appears, and speech falls back to English without a Hindi voice", async ({ page, errors }) => {
  void errors
  await page.addInitScript(() => {
    speechSynthesis.getVoices = () => [{ name: "English", lang: "en-US", localService: true, default: true, voiceURI: "en" } as SpeechSynthesisVoice]
  })
  await openApp(page)
  await sayIWantWater(page)
  await page.getByLabel("Speech language").selectOption("hi")
  await expect(page.getByTestId("sentence-hindi")).toHaveText("मुझे पानी चाहिए।")
  await page.getByRole("button", { name: "Speak", exact: true }).click()
  await expect(page.getByText(/no Hindi voice, so the English line was spoken/)).toBeVisible()
  await expect.poll(async () => (await spoken(page)).map((s) => s.text)).toContain("I want water.")
  await page.getByLabel("Speech language").selectOption("en")
  await expect(page.getByTestId("sentence-hindi")).toHaveCount(0)
})

test("Hindi: speaks the Hindi line with a hi-IN voice when the device has one", async ({ page, errors }) => {
  void errors
  await page.addInitScript(() => {
    const voices = [
      { name: "English", lang: "en-US", localService: true, default: true, voiceURI: "en" },
      { name: "Hindi", lang: "hi-IN", localService: true, default: false, voiceURI: "hi" },
    ] as SpeechSynthesisVoice[]
    speechSynthesis.getVoices = () => voices
    // A real voice object is required by the setter; keep the language from the fake one.
    Object.defineProperty(SpeechSynthesisUtterance.prototype, "voice", { configurable: true, set() {}, get() { return null } })
  })
  await openApp(page)
  await sayIWantWater(page)
  await page.getByLabel("Speech language").selectOption("hi")
  await page.getByRole("button", { name: "Speak", exact: true }).click()
  await expect(page.getByText("Spoke in Hindi with the browser voice.")).toBeVisible()
  await expect.poll(async () => await spoken(page)).toContainEqual({ text: "मुझे पानी चाहिए।", lang: "hi-IN" })
})

test("tips overlay opens and closes three ways", async ({ page, errors }) => {
  void errors
  await openApp(page)
  const open = page.getByRole("button", { name: "Tips for best accuracy" })
  const dialog = page.getByRole("dialog", { name: "Tips for best accuracy" })
  await open.click()
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole("listitem")).toHaveCount(8)
  await page.keyboard.press("Escape")
  await expect(dialog).toHaveCount(0)
  await open.click()
  await dialog.getByRole("button", { name: "Got it" }).click()
  await expect(dialog).toHaveCount(0)
  await open.click()
  await page.mouse.click(5, 5)
  await expect(dialog).toHaveCount(0)
})

test("live framing hint appears while the camera runs", async ({ page, errors }) => {
  void errors
  await openApp(page)
  await page.getByRole("button", { name: "Start camera" }).click()
  await expect(page.getByRole("button", { name: "Stop camera" })).toBeVisible({ timeout: 30_000 })
  // The fake webcam shows no hand, so after about two seconds the hint asks for one (or for light).
  await expect(page.getByTestId("camera-hint")).toHaveText(/No hand found|Too dark|Very bright/, { timeout: 15_000 })
  await page.getByRole("button", { name: "Stop camera" }).click()
  await expect(page.getByTestId("camera-hint")).toHaveCount(0)
})

test("ASL beta mode loads the real-signer word model", async ({ page, errors }) => {
  void errors
  await openApp(page)
  const modelLoad = page.waitForResponse((r) => r.url().endsWith("/models/asl_words.json") && r.status() === 200)
  const modes = page.getByRole("group", { name: "Recognition mode" })
  await modes.getByRole("button", { name: "ASL beta" }).click()
  await modelLoad
  await expect(modes.getByRole("button", { name: "ASL beta" })).toHaveAttribute("aria-pressed", "true")
  await expect(page.getByTestId("mode-note")).toContainText("14 real ASL words")
})

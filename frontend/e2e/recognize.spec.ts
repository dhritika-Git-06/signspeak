import { chips, expect, openApp, sentence, spoken, test } from "./helpers"

test("presentation demo builds and speaks I want water", async ({ page, errors }) => {
  void errors
  await openApp(page)
  const demo = page.getByRole("button", { name: "Play presentation demo" })
  await demo.click()
  await expect(page.getByRole("button", { name: "Playing demo…" })).toBeDisabled()
  await expect(sentence(page)).toHaveText("I want water.")
  await expect(chips(page)).toHaveText(["ME", "WANT", "WATER"])
  await expect.poll(async () => (await spoken(page)).map((s) => s.text)).toContain("I want water.")
  await expect(demo).toBeEnabled()
})

test("Open the demo scrolls to the camera panel", async ({ page, errors }) => {
  void errors
  await openApp(page)
  await page.getByRole("button", { name: "Open the demo" }).click()
  await expect.poll(() => page.evaluate(() => Math.round(document.getElementById("demo")!.getBoundingClientRect().top))).toBeLessThan(200)
})

test("sentence controls: suggestions, Space, Delete, Backspace, Clear, Speak and history", async ({ page, errors }) => {
  void errors
  await openApp(page)
  const suggestions = page.getByRole("group", { name: "Suggestions" })
  await expect(suggestions.getByRole("button")).toHaveText(["HELLO", "ME", "YOU"])
  await suggestions.getByRole("button", { name: "ME", exact: true }).click()
  await suggestions.getByRole("button", { name: "WANT", exact: true }).click()
  await suggestions.getByRole("button", { name: "WATER", exact: true }).click()
  await expect(sentence(page)).toHaveText("I want water.")
  await page.getByRole("button", { name: "Delete", exact: true }).click()
  await expect(sentence(page)).toHaveText("I want.")
  await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur())
  await page.keyboard.press("Backspace")
  await expect(sentence(page)).toHaveText("I.")
  await page.getByRole("button", { name: "Clear", exact: true }).click()
  await expect(sentence(page)).toContainText("Letters and words will gather here")

  // A fingerspelled letter from the recorded sample, then Space turns it into a word.
  await page.getByRole("button", { name: "Play sample B" }).click()
  await expect(chips(page)).toHaveText(["B"])
  await expect(suggestions.getByRole("button")).toHaveText(["HELLO", "ME", "YOU"])
  await page.getByRole("button", { name: "Space", exact: true }).click()
  await expect(suggestions.getByRole("button")).toHaveText(["PLEASE", "THANKS", "STOP"])
  await page.getByRole("button", { name: "Play sample HELLO" }).click()
  await expect(chips(page)).toHaveText(["B", "HELLO"])
  await page.getByRole("button", { name: "Clear", exact: true }).click()

  for (let round = 0; round < 2; round++) {
    for (const w of ["HELLO"]) await suggestions.getByRole("button", { name: w, exact: true }).click()
    await page.getByRole("button", { name: "Speak", exact: true }).click()
    await expect(page.getByText("Spoke with the browser voice.")).toBeVisible()
  }
  await expect(page.getByRole("list", { name: "Sentence history" }).getByRole("listitem")).toHaveText(["Hello.", "Hello."])
  await expect.poll(async () => (await spoken(page)).filter((s) => s.text === "Hello.").length).toBe(2)
  await page.getByRole("button", { name: "Speak", exact: true }).click()
  await expect(page.getByText("Nothing to say yet.")).toBeVisible()
})

test("recognition settings: mode buttons, confidence slider, auto-speak and mirror", async ({ page, errors }) => {
  void errors
  await openApp(page)
  const modes = page.getByRole("group", { name: "Recognition mode" })
  const notes: Record<string, RegExp> = { letters: /All 24 still letters/, words: /Word signs only/, auto: /Use Letters to spell/ }
  for (const m of ["letters", "words", "auto"]) {
    await modes.getByRole("button", { name: m }).click()
    await expect(modes.getByRole("button", { name: m })).toHaveAttribute("aria-pressed", "true")
    await expect(page.getByTestId("mode-note")).toHaveText(notes[m])
  }
  const slider = page.getByRole("slider")
  await slider.focus()
  await page.keyboard.press("ArrowRight")
  await page.keyboard.press("ArrowRight")
  await expect(page.getByText("64%")).toBeVisible()
  await page.keyboard.press("ArrowLeft")
  await page.keyboard.press("ArrowLeft")
  await expect(page.getByText("62%")).toBeVisible()

  const auto = page.getByRole("switch", { name: "Auto-speak" })
  await auto.click()
  await expect(auto).toHaveAttribute("aria-checked", "true")
  await page.getByRole("button", { name: "Play sample HELLO" }).click()
  await expect(chips(page)).toHaveText(["HELLO"])
  await expect.poll(async () => (await spoken(page)).map((s) => s.text)).toContain("HELLO")
  await auto.click()
  await expect(auto).toHaveAttribute("aria-checked", "false")

  const mirror = page.getByLabel("Mirror")
  for (const v of ["on", "off", "auto"]) {
    await mirror.selectOption(v)
    await expect(mirror).toHaveValue(v)
  }
})

test("camera start and stop with a fake webcam", async ({ page, errors }) => {
  void errors
  await openApp(page)
  await page.getByRole("button", { name: "Start camera" }).click()
  await expect(page.getByRole("button", { name: "Stop camera" })).toBeVisible({ timeout: 60_000 })
  await expect(page.getByText(/\d+ fps/)).toBeVisible()
  await page.getByRole("button", { name: "Stop camera" }).click()
  await expect(page.getByRole("button", { name: "Start camera" })).toBeVisible()
  await expect(page.getByText(/camera off/)).toBeVisible()
})

test("no camera: a clear message and the demo still works", async ({ page, errors }) => {
  void errors
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException("Requested device not found", "NotFoundError"))
  })
  await openApp(page)
  await page.getByRole("button", { name: "Start camera" }).click()
  await expect(page.getByText(/No camera was found on this device/)).toBeVisible()
  await page.getByRole("button", { name: "Play presentation demo" }).click()
  await expect(sentence(page)).toHaveText("I want water.")
})

test("camera permission denied shows how to allow it", async ({ page, errors }) => {
  void errors
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException("Permission denied", "NotAllowedError"))
  })
  await openApp(page)
  await page.getByRole("button", { name: "Start camera" }).click()
  await expect(page.getByText(/Camera permission was blocked/)).toBeVisible()
})

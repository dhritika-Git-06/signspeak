import { expect, type Page, test as base } from "@playwright/test"

declare global {
  interface Window {
    __spoken: Array<{ text: string; lang: string }>
    __copied: string[]
  }
}

/** Records what the page would say, so speech can be asserted without audio. */
export function installSpeechSpy() {
  window.__spoken = []
  const synth = window.speechSynthesis
  if (!synth) return
  synth.speak = (u: SpeechSynthesisUtterance) => {
    window.__spoken.push({ text: u.text, lang: u.lang })
  }
}

export const test = base.extend<{ errors: string[] }>({
  errors: async ({ page }, use) => {
    const errors: string[] = []
    page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`))
    page.on("console", (m) => {
      if (m.type() !== "error") return
      const text = m.text()
      // The deep-link document itself is served with a 404 status by GitHub Pages; that is expected.
      if (/Failed to load resource: the server responded with a status of 404/.test(text) && /\/signspeak\/(train|how|evaluation|learn)/.test(m.location()?.url ?? "")) return
      errors.push(`console: ${text} @ ${m.location()?.url ?? ""}`)
    })
    page.on("response", (r) => {
      if (r.status() >= 400 && r.request().resourceType() !== "document") errors.push(`${r.status()} ${r.url()}`)
    })
    await page.addInitScript(installSpeechSpy)
    await use(errors)
    expect(errors, "no console errors or failed requests").toEqual([])
  },
})

export { expect }

export async function openApp(page: Page, path = "") {
  await page.goto(path)
  await expect(page.getByText("Models ready")).toBeVisible({ timeout: 60_000 })
}

export const sentence = (page: Page) => page.getByTestId("sentence")
export const chips = (page: Page) => page.getByRole("group", { name: "Confirmed signs" }).locator("span")
export const spoken = (page: Page) => page.evaluate(() => window.__spoken)

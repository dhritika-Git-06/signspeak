import path from "node:path"
import { mkdirSync, writeFileSync } from "node:fs"
import { expect, openApp, test } from "./helpers"

test("Train page: every control does something visible", async ({ page, errors }, info) => {
  void errors
  await openApp(page, "train")
  const status = page.getByRole("status")
  // Sign type toggle
  const type = page.getByRole("group", { name: "Sign type" })
  await type.getByRole("button", { name: "Word / motion" }).click()
  await expect(type.getByRole("button", { name: "Word / motion" })).toHaveAttribute("aria-pressed", "true")
  await expect(page.getByText("Aim for 6 takes.", { exact: false })).toBeVisible()
  await type.getByRole("button", { name: "Letter / pose" }).click()
  await expect(type.getByRole("button", { name: "Letter / pose" })).toHaveAttribute("aria-pressed", "true")

  // Record without a camera, then with a bad name, then with the fake camera (no hand in it).
  await page.getByRole("button", { name: "Record burst" }).click()
  await expect(status).toHaveText(/Start the camera first/)
  await page.getByRole("button", { name: "Train head" }).click()
  await expect(status).toHaveText(/Record at least one burst/)
  await page.getByRole("button", { name: "Start camera" }).click()
  await expect(page.getByRole("button", { name: "Stop camera" })).toBeVisible({ timeout: 60_000 })
  await page.getByLabel("Name").fill("1bad")
  await page.getByRole("button", { name: "Record burst" }).click()
  await expect(status).toHaveText(/short name that starts with a letter/)
  await page.getByLabel("Name").fill("MOM")
  await page.getByRole("button", { name: "Record burst" }).click()
  await expect(status).toHaveText(/did not see a hand/, { timeout: 15_000 })
  await page.getByRole("button", { name: "Stop camera" }).click()
  await expect(page.getByRole("button", { name: "Start camera" })).toBeVisible()

  // Import a saved sign, see it listed, export it, then remove it.
  const pts = Array.from({ length: 21 }, (_, i) => [0.5 + (i % 5) * 0.02, 0.6 - Math.floor(i / 5) * 0.04, 0])
  const bundle = { version: 1, signs: [{ id: "static:MOM", name: "MOM", kind: "static", samples: Array.from({ length: 10 }, () => pts), createdAt: 1 }], heads: {} }
  mkdirSync(info.outputDir, { recursive: true })
  const file = path.join(info.outputDir, "signs.json")
  writeFileSync(file, JSON.stringify(bundle))
  await page.locator('input[type="file"]').setInputFiles(file)
  await expect(status).toHaveText(/Imported 1 sign/)
  await expect(page.getByRole("listitem").filter({ hasText: "MOM" })).toBeVisible()
  const download = page.waitForEvent("download")
  await page.getByRole("button", { name: "Export" }).click()
  expect((await download).suggestedFilename()).toBe("signspeak-signs.json")
  await expect(status).toHaveText(/Exported 1 sign/)
  await page.getByRole("listitem").filter({ hasText: "MOM" }).getByRole("button", { name: "Remove" }).click()
  await expect(status).toHaveText(/Removed MOM/)
  await expect(page.getByText("None yet.", { exact: false })).toBeVisible()

  // A broken file gives a clear message.
  const bad = path.join(info.outputDir, "bad.json")
  writeFileSync(bad, "not json")
  await page.locator('input[type="file"]').setInputFiles(bad)
  await expect(status).toHaveText(/not valid JSON/)
})

test("How page renders all steps", async ({ page, errors }) => {
  void errors
  await openApp(page, "how")
  await expect(page.getByRole("heading", { name: "From a hand shape to a spoken sentence" })).toBeVisible()
  await expect(page.locator("main")).toContainText("Built by Dhritika Sahu")
})

test("Evaluation page: numbers, images and dataset links", async ({ page, errors }) => {
  void errors
  await openApp(page, "evaluation")
  await expect(page.getByText("94.19%").first()).toBeVisible()
  for (const img of await page.locator("main img").all()) {
    await expect(img).toBeVisible()
    expect(await img.evaluate((el: HTMLImageElement) => el.complete && el.naturalWidth > 0)).toBe(true)
  }
  for (const link of await page.locator("main a[href^='http']").all()) {
    await expect(link).toHaveAttribute("target", "_blank")
    expect(await link.getAttribute("href")).toMatch(/^https:\/\//)
  }
})

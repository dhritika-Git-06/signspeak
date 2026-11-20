import { expect, openApp, test } from "./helpers"

test("navigation links, logo, theme toggle, skip link and footer", async ({ page, errors }) => {
  void errors
  await openApp(page)
  const nav = page.getByRole("navigation", { name: "Primary" })
  for (const [name, heading] of [
    ["Train", "Teach a sign"],
    ["How it works", "From a hand shape to a spoken sentence"],
    ["Evaluation", "Measured accuracy"],
    ["Recognize", "Read a hand. Build a sentence. Hear it."],
  ] as const) {
    await nav.getByRole("link", { name, exact: true }).click()
    await expect(page.getByRole("heading", { name: heading })).toBeVisible()
    await expect(nav.getByRole("link", { name, exact: true })).toHaveAttribute("aria-current", "page")
  }
  // Logo goes home from any page.
  await nav.getByRole("link", { name: "Train", exact: true }).click()
  await page.getByRole("link", { name: /SignSpeak AI/ }).first().click()
  await expect(page).toHaveURL(/\/signspeak\/?$/)

  // Theme toggle flips the page colours and survives a reload.
  const toggle = page.getByRole("button", { name: /Switch to (dark|light) theme/ })
  const wasDark = await page.evaluate(() => document.documentElement.classList.contains("dark"))
  const bgBefore = await page.evaluate(() => getComputedStyle(document.body).backgroundColor)
  await toggle.click()
  await expect.poll(() => page.evaluate(() => document.documentElement.classList.contains("dark"))).toBe(!wasDark)
  expect(await page.evaluate(() => getComputedStyle(document.body).backgroundColor)).not.toBe(bgBefore)
  await page.reload()
  await expect.poll(() => page.evaluate(() => document.documentElement.classList.contains("dark"))).toBe(!wasDark)
  await page.getByRole("button", { name: /Switch to (dark|light) theme/ }).click()
  await expect.poll(() => page.evaluate(() => document.documentElement.classList.contains("dark"))).toBe(wasDark)

  // Skip link moves focus to the main content.
  await openApp(page)
  await page.keyboard.press("Tab")
  const skip = page.getByRole("link", { name: "Skip to content" })
  await expect(skip).toBeFocused()
  await skip.press("Enter")
  await expect(page).toHaveURL(/#main$/)

  await expect(page.locator("footer")).toContainText("Built by Dhritika Sahu")
  await expect(page.locator("body")).not.toContainText(/medi-?caps|university|indore/i)
})

for (const sub of ["train", "how", "evaluation"]) {
  test(`refreshing /${sub} still loads the app`, async ({ page, errors }) => {
    void errors
    await page.goto(sub)
    await expect(page.getByRole("navigation", { name: "Primary" })).toBeVisible()
    await expect(page.locator("main h1").first()).toBeVisible()
  })
}

test("phone width has no sideways scroll", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true })
  const page = await ctx.newPage()
  for (const sub of ["", "train", "how", "evaluation"]) {
    await page.goto(sub)
    await expect(page.locator("main h1").first()).toBeVisible()
    const { scroll, width } = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, width: document.documentElement.clientWidth }))
    expect(scroll, `/${sub}`).toBeLessThanOrEqual(width)
  }
  await ctx.close()
})

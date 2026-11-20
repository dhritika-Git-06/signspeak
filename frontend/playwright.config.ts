import { defineConfig } from "@playwright/test"

/**
 * End-to-end tests against the production build served under /signspeak/, the same path
 * GitHub Pages uses. Chrome gets a fake camera so every camera button can be pressed.
 * Run: npm run build:pages && npm run e2e
 */
export default defineConfig({
  testDir: "e2e",
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:4317/signspeak/",
    channel: process.env.PW_CHROME_PATH ? undefined : "chrome",
    launchOptions: {
      executablePath: process.env.PW_CHROME_PATH || undefined,
      args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
    },
    permissions: ["camera", "clipboard-read", "clipboard-write"],
    viewport: { width: 1366, height: 900 },
    acceptDownloads: true,
  },
  webServer: {
    command: "node e2e/serve.mjs 4317",
    url: "http://127.0.0.1:4317/signspeak/",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
})

import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import fs from "node:fs"
import path from "node:path"
import { defineConfig, type Plugin } from "vite"

const outDir = path.resolve(import.meta.dirname, "../dist")

/** GitHub Pages serves 404.html for unknown paths; a copy of index.html lets /signspeak/train etc. load on refresh. */
function spaFallback(): Plugin {
  return {
    name: "spa-404-fallback",
    apply: "build",
    closeBundle() {
      const index = path.join(outDir, "index.html")
      if (fs.existsSync(index)) fs.copyFileSync(index, path.join(outDir, "404.html"))
    },
  }
}

export default defineConfig({
  base: process.env.VITE_BASE || "/",
  plugins: [react(), tailwindcss(), spaFallback()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "src") },
  },
  server: { host: "0.0.0.0", port: 43123, strictPort: true },
  preview: { host: "0.0.0.0", port: 43123, strictPort: true, allowedHosts: true },
  build: {
    outDir,
    emptyOutDir: true,
  },
})

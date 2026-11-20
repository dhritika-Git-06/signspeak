import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import path from "node:path"
import { defineConfig } from "vite"

export default defineConfig({
  base: process.env.VITE_BASE || "/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "src") },
  },
  server: { host: "0.0.0.0", port: 43123, strictPort: true },
  preview: { host: "0.0.0.0", port: 43123, strictPort: true, allowedHosts: true },
  build: {
    outDir: path.resolve(import.meta.dirname, "../dist"),
    emptyOutDir: true,
  },
})

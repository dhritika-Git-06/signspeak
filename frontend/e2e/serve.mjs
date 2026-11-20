// Static server that behaves like GitHub Pages for /signspeak/: real files are served,
// unknown paths get 404.html (the SPA copy of index.html) with a 404 status.
import { createReadStream, existsSync, statSync } from "node:fs"
import { createServer } from "node:http"
import path from "node:path"

const root = path.resolve(import.meta.dirname, "../../dist")
const port = Number(process.argv[2] || 4317)
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".wasm": "application/wasm", ".svg": "image/svg+xml", ".png": "image/png", ".task": "application/octet-stream", ".bin": "application/octet-stream", ".txt": "text/plain" }

createServer((req, res) => {
  const url = decodeURIComponent((req.url || "/").split("?")[0])
  const send = (file, status = 200) => {
    res.writeHead(status, { "Content-Type": types[path.extname(file)] || "application/octet-stream" })
    createReadStream(file).pipe(res)
  }
  if (url === "/signspeak") {
    res.writeHead(301, { Location: "/signspeak/" })
    return res.end()
  }
  if (!url.startsWith("/signspeak/")) return send(path.join(root, "404.html"), 404)
  let file = path.join(root, url.slice("/signspeak/".length))
  if (!file.startsWith(root)) return send(path.join(root, "404.html"), 404)
  if (existsSync(file) && statSync(file).isDirectory()) file = path.join(file, "index.html")
  if (existsSync(file)) return send(file)
  return send(path.join(root, "404.html"), 404)
}).listen(port, "127.0.0.1", () => console.log(`serving ${root} at http://127.0.0.1:${port}/signspeak/`))

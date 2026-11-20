/** Prefix public files with Vite's base, so GitHub Pages under /signspeak/ still finds them. */
export function asset(path: string): string {
  const base = import.meta.env.BASE_URL ?? "/"
  return `${base}${path.replace(/^\/+/, "")}`
}

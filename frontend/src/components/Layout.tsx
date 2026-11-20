import { Moon, Sun } from "lucide-react"
import { NavLink, Outlet } from "react-router-dom"
import { Badge, Button } from "@/components/ui"
import { useTheme } from "@/hooks/useTheme"
import { useModels } from "@/models-context"

const links = [
  { to: "/", label: "Recognize", end: true },
  { to: "/train", label: "Train" },
  { to: "/how", label: "How it works" },
  { to: "/evaluation", label: "Evaluation" },
]

export function Layout() {
  const { theme, toggle } = useTheme()
  const { apiBase, status } = useModels()
  return (
    <div className="min-h-screen">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-full focus:bg-[var(--accent)] focus:px-3 focus:py-2">
        Skip to content
      </a>
      <header className="sticky top-0 z-20 border-b border-[var(--line)] bg-[color-mix(in_srgb,var(--bg)_88%,transparent)] backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-3 px-4 py-3">
          <NavLink to="/" className="mr-2 flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-2xl bg-[var(--ink)] text-[var(--bg)] serif text-xl">S</span>
            <span>
              <span className="block text-sm font-semibold tracking-tight">SignSpeak AI</span>
              <span className="block text-xs text-[var(--muted)]">Dhritika Sahu · Medi-Caps University</span>
            </span>
          </NavLink>
          <nav className="flex flex-1 flex-wrap gap-1" aria-label="Primary">
            {links.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                end={link.end}
                className={({ isActive }) =>
                  `rounded-full px-3 py-1.5 text-sm ${isActive ? "bg-[var(--ink)] text-[var(--bg)]" : "text-[var(--muted)] hover:text-[var(--ink)]"}`
                }
              >
                {link.label}
              </NavLink>
            ))}
          </nav>
          <Badge tone={status === "ready" ? "good" : status === "error" ? "warn" : "neutral"}>
            {status === "ready" ? "Models ready" : status === "error" ? "Models failed" : "Loading models"}
          </Badge>
          <Badge tone={apiBase ? "good" : "neutral"}>{apiBase ? "Server voice" : "Browser voice"}</Badge>
          <Button variant="quiet" size="sm" onClick={toggle} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            {theme === "dark" ? "Light" : "Dark"}
          </Button>
        </div>
      </header>
      <main id="main" className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
      <footer className="mx-auto max-w-6xl px-4 pb-10 text-xs text-[var(--muted)]">
        Built by Dhritika Sahu · Final-year B.Tech CSE, Medi-Caps University, Indore.
        Letter landmarks from the ASL Alphabet set (Akash Nagaraj, GPL-2.0) via MediaPipe Hands.
        Live words are geometric rules, not a second trained accuracy.
      </footer>
    </div>
  )
}

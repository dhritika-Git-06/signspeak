import { useEffect, useState } from "react"

export function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() =>
    document.documentElement.classList.contains("dark") ? "dark" : "light",
  )
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark")
    localStorage.setItem("signspeak-theme", theme)
  }, [theme])
  return {
    theme,
    toggle: () => setTheme((current) => (current === "dark" ? "light" : "dark")),
  }
}

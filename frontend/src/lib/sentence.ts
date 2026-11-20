export type TokenKind = "letter" | "word"

export interface Token {
  kind: TokenKind
  text: string
}

export function render(tokens: Token[]): string {
  let out = ""
  for (const token of tokens) {
    if (token.kind === "word") {
      if (out.length > 0 && !out.endsWith(" ")) out += " "
      out += `${token.text} `
    } else {
      out += token.text
    }
  }
  return out
}

export function applyCommit(tokens: Token[], label: string, kind: TokenKind): Token[] {
  if (label === "DELETE") return tokens.slice(0, -1)
  if (label === "SPACE") return addSpace(tokens)
  if (kind === "word") return [...tokens, { kind: "word", text: label }]
  return [...tokens, { kind: "letter", text: label }]
}

export function deleteLast(tokens: Token[]): Token[] {
  return tokens.slice(0, -1)
}

export function addSpace(tokens: Token[]): Token[] {
  if (!tokens.length) return [{ kind: "letter", text: " " }]
  const last = tokens[tokens.length - 1]
  if (last.kind === "letter" && last.text === " ") return tokens
  if (last.kind === "word") return tokens
  return [...tokens, { kind: "letter", text: " " }]
}

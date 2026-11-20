/**
 * Offline Hindi for the sentence bar: a few whole-sentence patterns, then a word table in
 * Hindi word order (subject, object, verb). Gender-neutral forms are used where Hindi allows,
 * e.g. "मुझे पानी चाहिए" rather than "मैं पानी चाहता/चाहती हूँ".
 */
export const HINDI_WORDS: Record<string, string> = {
  ME: "मैं",
  I: "मैं",
  YOU: "आप",
  WE: "हम",
  WANT: "चाहिए",
  NEED: "ज़रूरत है",
  HELP: "मदद",
  EAT: "खाना",
  DRINK: "पीना",
  WATER: "पानी",
  FOOD: "खाना",
  GO: "जाना",
  COME: "आना",
  HOME: "घर",
  SCHOOL: "स्कूल",
  GOOD: "अच्छा",
  BAD: "बुरा",
  LIKE: "पसंद",
  LOVE: "प्यार",
  NAME: "नाम",
  WHAT: "क्या",
  WHERE: "कहाँ",
  HOW: "कैसे",
  MORE: "और",
  STOP: "रुकिए",
  FINE: "ठीक",
  HELLO: "नमस्ते",
  YES: "हाँ",
  NO: "नहीं",
  THANKS: "धन्यवाद",
  PLEASE: "कृपया",
  SORRY: "माफ़ कीजिए",
  MOM: "माँ",
  DAD: "पापा",
  DOG: "कुत्ता",
  CAT: "बिल्ली",
  BOOK: "किताब",
  BUS: "बस",
  OK: "ठीक है",
  HI: "नमस्ते",
}

const DATIVE: Record<string, string> = { ME: "मुझे", I: "मुझे", YOU: "आपको", WE: "हमें" }
const GREETINGS = new Set(["HELLO", "HI", "THANKS", "PLEASE", "SORRY", "YES", "NO"])

function word(gloss: string): string {
  return HINDI_WORDS[gloss] ?? gloss.charAt(0) + gloss.slice(1).toLowerCase()
}

function core(g: string[]): string {
  if (!g.length) return ""
  if (g[0] === "ME" && g[1] === "NAME" && g[2]) return `मेरा नाम ${g.slice(2).map(word).join(" ")} है।`
  if (g.includes("WHAT") && g.includes("NAME")) return "आपका नाम क्या है?"
  if (g.includes("WHERE") && g.includes("HOME")) return "घर कहाँ है?"
  if (g.includes("WHERE") && g.includes("SCHOOL")) return "स्कूल कहाँ है?"
  if (g.includes("HOW") && g.includes("YOU")) return "आप कैसे हैं?"
  const subject = DATIVE[g[0]] ? g[0] : null
  const rest = subject ? g.slice(1) : g
  const verb = rest.find((x) => ["WANT", "NEED", "LIKE", "LOVE"].includes(x))
  if (verb) {
    const objects = rest.filter((x) => x !== verb).map(word).join(" ")
    const who = DATIVE[subject ?? "ME"]
    if (verb === "WANT") return `${who} ${objects ? objects + " " : ""}चाहिए।`.replace("  ", " ")
    if (verb === "NEED") return `${who} ${objects ? objects + " की " : ""}ज़रूरत है।`
    if (verb === "LIKE") return `${who} ${objects ? objects + " " : ""}पसंद है।`
    return `${who} ${objects ? objects + " से " : ""}प्यार है।`
  }
  if (subject && rest.length === 1 && ["FINE", "GOOD", "BAD"].includes(rest[0])) {
    const be = subject === "YOU" ? "हैं" : subject === "WE" ? "हैं" : "हूँ"
    return `${word(subject)} ${word(rest[0])} ${be}।`
  }
  if (subject && rest[0] === "GO" && rest[1]) return `${DATIVE[subject]} ${word(rest[1])} जाना है।`
  if (subject && rest[0] === "COME" && rest[1]) return `${DATIVE[subject]} ${word(rest[1])} आना है।`
  const question = g.some((x) => x === "WHAT" || x === "WHERE" || x === "HOW")
  return `${g.map(word).join(" ")}${question ? "?" : "।"}`
}

/** Hindi for a list of glosses (the same list the English line is built from). */
export function toHindi(glosses: string[]): string {
  const g = glosses.map((x) => x.toUpperCase()).filter(Boolean)
  if (!g.length) return ""
  // Greetings and courtesy words at the start become their own short sentence.
  const lead: string[] = []
  while (g.length && GREETINGS.has(g[0]) && (g.length === 1 || !["WANT", "NEED", "LIKE", "LOVE"].includes(g[1]))) {
    lead.push(`${word(g.shift()!)}।`)
  }
  return [...lead, core(g)].filter(Boolean).join(" ")
}

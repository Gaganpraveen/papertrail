/**
 * Text primitives: normalisation, sentence splitting and tokenisation.
 *
 * Sentence splitting is rule-based, tuned for scientific prose. A candidate
 * boundary is terminal punctuation followed by whitespace and something that
 * can open a sentence (a capital, a digit, an opening bracket or quote). Each
 * candidate is then vetted: abbreviations that never end a sentence ("e.g.",
 * "Fig.", "Eq.", "vs."), reference abbreviations followed by a label
 * ("No. 5", "p. 12"), author initials ("A. Vaswani"), "et al." followed by a
 * year or citation, and list markers ("1.") are rejected; ambiguous cases
 * ("et al.", "etc.", "U.S.") break only when the next word is a typical
 * sentence opener such as "The" or "We". Decimals and URLs never produce a
 * candidate because the period is not followed by whitespace.
 *
 * Tokenisation lowercases, keeps hyphenated compounds ("key-value", "gpt-4")
 * as one token, drops stopwords and applies a light, readable stemmer
 * (plurals, plus "-ing"/"-ed" only when a confident stem remains), so
 * "embeddings", "embedding" and "embedded" all become "embed" while tokens
 * stay human-readable for labels.
 */

// ───────────────────────────────── Stopwords ─────────────────────────────────

export const STOPWORDS: Set<string> = new Set(
  `a about above across after afterwards again against all almost alone along already also although always am among
  amongst an and another any anyhow anyone anything anyway anywhere are around as at be became because become becomes
  becoming been before beforehand behind being below beside besides between beyond both but by can cannot could did do
  does doing done down during each either else elsewhere enough even ever every everyone everything everywhere except
  few for former formerly from further furthermore had has have having he hence her here hereafter hereby herein
  hers herself him himself his how however i if in indeed into is it its itself just last latter latterly least less
  many may me meanwhile might mine more moreover most mostly much must my myself namely neither never nevertheless
  next no nobody none noone nor not nothing now nowhere of off often on once one only onto or other others otherwise
  our ours ourselves out over own per perhaps please rather same seem seemed seeming seems several she should since
  so some somehow someone something sometime sometimes somewhere still such than that the their theirs them
  themselves then thence there thereafter thereby therefore therein thereupon these they this those though through
  throughout thus to together too toward towards under until up upon us very via was we well were what whatever when
  whence whenever where whereafter whereas whereby wherein whereupon wherever whether which while whither who whoever
  whole whom whose why will with within without would yet you your yours yourself yourselves
  can't cannot couldn't didn't doesn't don't hadn't hasn't haven't isn't it's let's shouldn't wasn't weren't won't
  wouldn't i.e e.g etc et al vs cf`
    .split(/\s+/)
    .filter(Boolean),
);

// ─────────────────────────────── Normalisation ───────────────────────────────

/** Zero-width characters and soft hyphens that PDF extraction leaves behind. */
const INVISIBLE_RE = /[­​-‍⁠﻿]/g;

/**
 * NFKC (which also expands typographic ligatures such as "ﬁ" -> "fi"),
 * removes invisible characters and collapses all whitespace to single spaces.
 */
export function normalizeText(s: string): string {
  return s.normalize("NFKC").replace(INVISIBLE_RE, "").replace(/\s+/g, " ").trim();
}

// ───────────────────────────── Sentence splitting ────────────────────────────

/** Never end a sentence, whatever follows ("e.g. The Pile", "Fig. 2a", "vs. GPT-4"). */
const NEVER_FINAL_ABBREVIATIONS = new Set([
  "e.g", "i.e", "cf", "vs", "viz", "w.r.t", "a.k.a", "approx", "resp", "incl", "esp", "ca",
  "fig", "figs", "eq", "eqs", "eqn", "eqns", "sec", "secs", "sect", "tab", "tbl", "tabs",
  "thm", "thms", "lem", "alg", "algo", "appx", "chap", "dr", "mr", "mrs", "ms", "prof", "jr",
]);

/**
 * Reference abbreviations that are also ordinary words ("no.", "p.", "app."):
 * they only continue the sentence when a label follows ("No. 5", "p. 12", "App. B").
 */
const LABEL_ABBREVIATIONS = new Set([
  "no", "nos", "p", "pp", "vol", "vols", "ch", "app", "ref", "refs", "ex", "exp", "expt",
  "prop", "cor", "def", "rem", "obs", "art", "pt", "fn", "suppl", "ed", "eds",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
]);

/** End a sentence only when a typical sentence opener follows ("et al. We ..."). */
const AMBIGUOUS_ABBREVIATIONS = new Set(["etc", "inc", "ltd", "co", "corp", "univ", "dept", "st"]);

/** Words after which a single capital is a label, not an initial ("Appendix B."). */
const LABEL_NOUNS = new Set([
  "appendix", "section", "table", "figure", "part", "phase", "stage", "type", "case", "setting",
  "model", "method", "variant", "step", "task", "group", "option", "level", "class", "condition",
  "category", "version", "track", "split", "subset", "set", "experiment", "algorithm",
]);

/** Discourse markers that commonly open a sentence in papers (beyond stopwords). */
const SENTENCE_OPENERS = new Set([
  "additionally", "consequently", "conversely", "finally", "first", "second", "third", "specifically",
  "notably", "overall", "note", "similarly", "instead", "importantly", "interestingly", "recently",
  "previously", "table", "figure", "section", "results", "experiments", "unlike", "despite", "given",
  "following", "using", "based", "compared", "existing", "recent", "prior", "let", "consider",
]);

const OPENS_SENTENCE_RE = /^["'“‘(\[]*[\p{Lu}\d]/u;
const BULLET_RE = /\s*[•●▪◦‣∙]\s*/u;
const LIST_MARKER_RE = /^(?:\(?\d{1,2}[.)]|\(?[ivx]{1,4}\)|[-–*])\s+/u;
const MIN_SENTENCE_CHARS = 25;

/**
 * Candidate boundaries: terminal punctuation, optional closing quotes or
 * brackets, an optional footnote marker glued to the period ("available.2"),
 * then whitespace.
 */
const BOUNDARY_RE = /([.!?]+)(["'”’)\]]*)(\d{1,2}(?:,\d{1,2})*)?(\s+)/gu;

interface BoundaryContext {
  /** The word right before the punctuation, opening brackets stripped ("e.g", "al", "2"). */
  token: string;
  /** The word before `token` ("et" in "et al."). */
  previous: string;
  punctuation: string;
  closers: string;
  footnote: string;
  /** Character right before the punctuation. */
  charBefore: string;
  /** The next whitespace-delimited word. */
  next: string;
}

function isTypicalOpener(word: string): boolean {
  const bare = word.replace(/^["'“‘(\[]+/u, "");
  if (!/^\p{Lu}/u.test(bare)) return false;
  const lower = bare.toLowerCase().replace(/[^\p{L}]+$/u, "");
  return STOPWORDS.has(lower) || SENTENCE_OPENERS.has(lower);
}

/** "5", "(3)", "[12]", "IV", "B", "B.2", "A1" — what follows "No.", "p.", "App.". */
function looksLikeLabel(word: string): boolean {
  return /^[([]?(?:\d|[IVX]+\b|[A-Z](?:$|[\d.)\],;:]))/u.test(word);
}

function isDottedAbbreviation(token: string): boolean {
  return /^(?:\p{L}{1,3}\.)+\p{L}{1,3}$/u.test(token);
}

/** True for a word that ends in a period only because it is abbreviated ("Eq.", "e.g.", "A."). */
function isAbbreviation(word: string): boolean {
  const bare = word.replace(/^["'“‘(\[]+/u, "").replace(/\.$/u, "");
  const lower = bare.toLowerCase();
  return (
    NEVER_FINAL_ABBREVIATIONS.has(lower) ||
    LABEL_ABBREVIATIONS.has(lower) ||
    AMBIGUOUS_ABBREVIATIONS.has(lower) ||
    lower === "al" ||
    isDottedAbbreviation(bare) ||
    /^\p{Lu}$/u.test(bare)
  );
}

/** "follows: 1. We", "first claim. 2. We": a bare small number that opens a list item. */
function isListMarker(token: string, previous: string): boolean {
  if (!/^\d{1,2}$/u.test(token)) return false;
  if (previous === "" || /[:;]$/u.test(previous)) return true;
  return previous.endsWith(".") && !isAbbreviation(previous); // but "Eq. 3. We" ends a sentence
}

function isSentenceBoundary(ctx: BoundaryContext): boolean {
  const { token, previous, punctuation, closers, footnote, next } = ctx;
  if (!OPENS_SENTENCE_RE.test(next)) return false;
  if (footnote && !/[\p{L})\]]/u.test(ctx.charBefore)) return false; // "0.12 We": a decimal, not a footnote
  if (!punctuation.includes(".")) return true; // "?" and "!" are never abbreviations
  if (closers) return true; // "(see Table 2.) We"

  const lower = token.toLowerCase();
  if (NEVER_FINAL_ABBREVIATIONS.has(lower)) return false;
  if (LABEL_ABBREVIATIONS.has(lower)) return !looksLikeLabel(next);
  if (lower === "al" && previous.toLowerCase() === "et") return isTypicalOpener(next);
  if (AMBIGUOUS_ABBREVIATIONS.has(lower)) return isTypicalOpener(next);
  if (isDottedAbbreviation(token)) return isTypicalOpener(next); // "U.S. Army" vs "U.S. The"
  if (/^\p{Lu}$/u.test(token)) {
    // A lone capital is an initial ("A. Vaswani") unless it labels something ("Appendix B. We").
    return LABEL_NOUNS.has(previous.toLowerCase()) || isTypicalOpener(next);
  }
  return !isListMarker(token, previous);
}

function lastWord(text: string): string {
  const match = /(\S+)\s*$/u.exec(text);
  return match ? match[1] : "";
}

/** Strips leading list markers and bullets left at the start of a sentence. */
function cleanSentence(sentence: string): string {
  return sentence.replace(LIST_MARKER_RE, "").trim();
}

function splitBlock(block: string): string[] {
  const sentences: string[] = [];
  let start = 0;
  for (const match of block.matchAll(BOUNDARY_RE)) {
    const at = match.index ?? 0;
    const [, punctuation, closers, footnote = "", space] = match;
    const head = block.slice(start, at);
    const tokenWithBrackets = lastWord(head);
    const token = tokenWithBrackets.replace(/^["'“‘(\[]+/u, "");
    const previous = lastWord(head.slice(0, head.length - tokenWithBrackets.length));
    const afterAt = at + match[0].length;
    const next = /^\S+/u.exec(block.slice(afterAt))?.[0] ?? "";
    const ctx: BoundaryContext = {
      token,
      previous,
      punctuation,
      closers,
      footnote,
      charBefore: block.charAt(at - 1),
      next,
    };
    if (!next || !isSentenceBoundary(ctx)) continue;
    // A glued footnote marker belongs to no sentence: keep the punctuation, drop the digits.
    const end = at + punctuation.length + closers.length;
    sentences.push(block.slice(start, end));
    start = end + footnote.length + space.length;
  }
  sentences.push(block.slice(start));
  return sentences;
}

/**
 * Splits text into sentences. Blank lines and bullets are hard boundaries;
 * fragments shorter than 25 characters (stray labels, headings) are dropped.
 */
export function splitSentences(text: string): string[] {
  const out: string[] = [];
  for (const paragraph of text.split(/\n\s*\n/)) {
    for (const block of normalizeText(paragraph).split(BULLET_RE)) {
      if (!block) continue;
      for (const raw of splitBlock(block)) {
        const sentence = cleanSentence(raw);
        if (sentence.length >= MIN_SENTENCE_CHARS) out.push(sentence);
      }
    }
  }
  return out;
}

// ──────────────────────────────── Tokenisation ───────────────────────────────

const WORD_RE = /[\p{L}\p{N}]+(?:['’-][\p{L}\p{N}]+)*/gu;

/** Raw word tokens in their original case, stopwords included. */
export function words(text: string): string[] {
  return text.normalize("NFKC").match(WORD_RE) ?? [];
}

/** Words whose trailing "s" is not a plural. */
const NOT_PLURAL = new Set([
  "bias", "alias", "atlas", "canvas", "lens", "news", "series", "species", "always", "perhaps", "whereas",
  "yes", "gas", "chaos", "ethos", "pathos", "thanks", "physics", "mathematics", "statistics", "economics",
  "linguistics", "robotics", "genomics", "dynamics", "analytics", "semantics", "pragmatics", "diagnostics",
]);

/** Irregular plurals common in papers. */
const IRREGULAR_SINGULAR = new Map<string, string>([
  ["biases", "bias"], ["aliases", "alias"], ["analyses", "analysis"], ["hypotheses", "hypothesis"],
  ["theses", "thesis"], ["syntheses", "synthesis"], ["indices", "index"], ["matrices", "matrix"],
  ["vertices", "vertex"], ["appendices", "appendix"], ["criteria", "criterion"], ["phenomena", "phenomenon"],
  ["corpora", "corpus"], ["schemata", "schema"], ["caches", "cache"], ["niches", "niche"],
]);

/** Words ending in "-ing"/"-ed" that are not inflections. */
const NOT_INFLECTED = new Set([
  "string", "thing", "something", "anything", "nothing", "everything", "king", "ring", "spring", "during",
  "morning", "evening", "ceiling", "hundred", "kindred", "sacred", "naked", "wicked", "rugged", "ragged",
]);

/**
 * After "-ing"/"-ed" is removed, these stem endings had a silent "e"
 * ("generat" -> "generate", "encod" -> "encode", "quantiz" -> "quantize").
 * Each pattern excludes the common words that end the same way without one
 * ("treat", "clear", "pair", "avoid", "train").
 */
const RESTORE_E = [
  /(?:[^eo]|cre)at$/u, /[iy]z$/u, /ys$/u, /[^aeioul]l$/u, /ut$/u, /uc$/u, /(?<!o)od$/u, /os$/u, /ov$/u,
  /iev$/u, /(?<![ao])ir$/u, /(?<!e)ar$/u, /ag$/u, /[rl]g$/u, /[rl]v$/u, /[nr]c$/u, /dg$/u,
  /(?<![aeiou])ur$/u, /(?<![aeiou])in$/u, /(?<![aeo])id$/u, /(?<![aeiou])un$/u, /[ae]ng$/u, /eas$/u,
  /is$/u, /ib$/u, /ud$/u, /u$/u,
];

function singular(word: string): string {
  if (word.length <= 3 || NOT_PLURAL.has(word)) return word;
  const irregular = IRREGULAR_SINGULAR.get(word);
  if (irregular) return irregular;
  if (word.endsWith("ies") && word.length > 4) return `${word.slice(0, -3)}y`;
  if (/(?:ss|x|ch|sh)es$/u.test(word)) return word.slice(0, -2);
  if (/(?:ss|us|is|ous)$/u.test(word)) return word;
  if (word.endsWith("s") && !word.endsWith("'s")) return word.slice(0, -1);
  return word;
}

function removeInflection(word: string): string {
  if (NOT_INFLECTED.has(word)) return word;
  let stem: string;
  if (word.endsWith("ied") && word.length > 5) return `${word.slice(0, -3)}y`;
  if (word.endsWith("ing")) stem = word.slice(0, -3);
  else if (word.endsWith("ed") && !word.endsWith("eed")) stem = word.slice(0, -2);
  else return word;
  // Conservative: require a stem with a vowel and at least four letters ("used", "based" stay intact).
  if (stem.length < 4 || !/[aeiouy]/u.test(stem)) return word;
  if (/([b-df-hj-kmnp-rtv-y])\1$/u.test(stem)) return stem.slice(0, -1); // "embedd" -> "embed"
  return RESTORE_E.some((re) => re.test(stem)) ? `${stem}e` : stem;
}

/** Light, readable stemmer: plurals, then "-ing"/"-ed" when a confident stem remains. */
export function stem(word: string): string {
  return removeInflection(singular(word));
}

/** Lowercase content tokens: hyphenated compounds kept, stopwords and 1-char tokens dropped, light stemming. */
export function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const raw of words(text)) {
    const word = raw.toLowerCase().replace(/’/gu, "'").replace(/'s$/u, "");
    if (word.length < 2 || STOPWORDS.has(word)) continue;
    out.push(stem(word));
  }
  return out;
}

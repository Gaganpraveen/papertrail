/**
 * Input understanding: decides what the user pasted, without any network call.
 *
 * Rule-based recognition, most specific pattern first:
 *   full URLs (arXiv-family hosts, doi.org, publisher "/doi/" paths, PDF links)
 *   -> "arXiv:" / "doi:" prefixes -> bare DOIs -> bare arXiv IDs
 *   -> scheme-less URLs -> otherwise a free-text topic search.
 * arXiv IDs are validated structurally (real month, 4 vs 5 digit serials by
 * era, known pre-2007 archive names), so "2023.12345" or "foo/1234567" are
 * not mistaken for papers.
 *
 * The file also hosts the identifier and text normalisers that every source
 * mapper shares, so DataCite, OpenAlex and Crossref records are cleaned the
 * same way.
 */
import type { ParsedInput } from "@/types";

// ───────────────────────────── arXiv identifiers ─────────────────────────────

export interface ArxivRef {
  id: string; // "2402.02750" or "hep-th/9901001", never with a version
  version?: string; // "v2"
}

/** Archive names of the pre-2007 identifier scheme ("hep-th/9901001"). */
// prettier-ignore
const OLD_ARXIV_ARCHIVES = new Set([
  "acc-phys", "adap-org", "alg-geom", "ao-sci", "astro-ph", "atom-ph", "bayes-an", "chao-dyn",
  "chem-ph", "cmp-lg", "comp-gas", "cond-mat", "cs", "dg-ga", "funct-an", "gr-qc", "hep-ex",
  "hep-lat", "hep-ph", "hep-th", "math", "math-ph", "mtrl-th", "nlin", "nucl-ex", "nucl-th",
  "patt-sol", "physics", "plasm-ph", "q-alg", "q-bio", "quant-ph", "solv-int", "supr-con",
]);

const NEW_ARXIV_ID = /^(\d{2})(\d{2})\.(\d{4,5})(v\d+)?$/i;
const OLD_ARXIV_ID = /^([a-z]+(?:-[a-z]+)?)(?:\.[a-z]{2})?\/(\d{2})(\d{2})(\d{3})(v\d+)?$/i;

function isMonth(mm: string): boolean {
  const month = Number(mm);
  return month >= 1 && month <= 12;
}

/**
 * Parses one arXiv identifier (optionally prefixed with "arXiv:" and/or
 * suffixed with a version). Old-style subject classes are dropped because
 * "math.GT/0309136" and "math/0309136" name the same paper.
 */
export function parseArxivId(text: string): ArxivRef | null {
  const candidate = text.trim().replace(/^arxiv\s*:\s*/i, "");

  const modern = NEW_ARXIV_ID.exec(candidate);
  if (modern) {
    const [, yy, mm, serial, version] = modern;
    const yymm = Number(yy + mm);
    // New-style IDs started in April 2007; serials grew from 4 to 5 digits in January 2015.
    const expectedSerialLength = yymm >= 1501 ? 5 : 4;
    if (!isMonth(mm) || yymm < 704 || serial.length !== expectedSerialLength) return null;
    return withVersion(`${yy}${mm}.${serial}`, version);
  }

  const legacy = OLD_ARXIV_ID.exec(candidate);
  if (legacy) {
    const [, archive, yy, mm, serial, version] = legacy;
    const name = archive.toLowerCase();
    const year = Number(yy);
    // The archive/YYMMNNN scheme ran from August 1991 to March 2007.
    const inLegacyEra = year >= 91 || year <= 7;
    if (!OLD_ARXIV_ARCHIVES.has(name) || !isMonth(mm) || !inLegacyEra) return null;
    return withVersion(`${name}/${yy}${mm}${serial}`, version);
  }
  return null;
}

function withVersion(id: string, version: string | undefined): ArxivRef {
  return version ? { id, version: version.toLowerCase() } : { id };
}

// prettier-ignore
const ARXIV_HOSTS = new Set([
  "arxiv.org", "www.arxiv.org", "export.arxiv.org", "browse.arxiv.org",
  "ar5iv.org", "ar5iv.labs.arxiv.org", "alphaxiv.org", "www.alphaxiv.org",
]);
const ARXIV_PATH_KINDS = new Set(["abs", "pdf", "html", "format", "ps", "src", "overview"]);

/** Extracts the arXiv ID from arXiv (and mirror) URLs, e.g. /abs/<id>, /pdf/<id>v2.pdf. */
export function arxivIdFromUrl(rawUrl: string): ArxivRef | null {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    return null;
  }
  const host = url.hostname.toLowerCase();
  const segments = url.pathname.split("/").filter(Boolean).map(safeDecode);
  const isHfPaper = (host === "huggingface.co" || host === "www.huggingface.co") && segments[0] === "papers";
  const isArxivPath = ARXIV_HOSTS.has(host) && ARXIV_PATH_KINDS.has(segments[0] ?? "");
  if (!isHfPaper && !isArxivPath) return null;
  // URLs pasted from prose often carry the sentence's "." or ")"; trim before dropping ".pdf".
  const idPart = trimTrailingPunctuation(segments.slice(1).join("/")).replace(/\.pdf$/i, "");
  return parseArxivId(idPart);
}

// ────────────────────────────────── DOIs ──────────────────────────────────

const DOI_PATTERN = /^10\.\d{4,9}\/\S+$/;
const DOI_URL_PREFIX = /^(?:https?:\/\/)?(?:dx\.)?doi\.org\//i;

function safeDecode(text: string): string {
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}

/** Drops sentence punctuation pasted along with an identifier; keeps balanced ")" (e.g. "...(20)30183-5"). */
function trimTrailingPunctuation(text: string): string {
  let out = text;
  for (;;) {
    const last = out.at(-1);
    if (last === undefined) return out;
    if (".,;:'\"]}>".includes(last)) {
      out = out.slice(0, -1);
    } else if (last === ")" && countChar(out, "(") < countChar(out, ")")) {
      out = out.slice(0, -1);
    } else {
      return out;
    }
  }
}

function countChar(text: string, char: string): number {
  let count = 0;
  for (const c of text) if (c === char) count++;
  return count;
}

/** "https://doi.org/10.1145/X", "doi:10.1145/X" or "10.1145/X" -> "10.1145/x"; null if not a DOI. */
export function normalizeDoi(text: string): string | null {
  const stripped = text
    .trim()
    .replace(DOI_URL_PREFIX, "")
    .replace(/^doi\s*:\s*/i, "");
  const doi = trimTrailingPunctuation(safeDecode(stripped).trim()).toLowerCase();
  return DOI_PATTERN.test(doi) ? doi : null;
}

/** Percent-encodes a DOI for a URL path while keeping its "/" separators readable. */
export function encodeDoiPath(doi: string): string {
  return encodeURIComponent(doi).replace(/%2F/gi, "/");
}

/** arXiv registers DOIs of the form 10.48550/arXiv.<id> through DataCite. */
export function arxivIdFromDoi(doi: string): ArxivRef | null {
  const match = /^10\.48550\/arxiv\.(.+)$/i.exec(doi.trim());
  return match ? parseArxivId(match[1]) : null;
}

/** Publisher landing pages that embed the DOI in the path, e.g. dl.acm.org/doi/10.1145/... */
function doiFromPublisherUrl(url: URL): string | null {
  const match = /\/doi\/(?:abs\/|full\/|pdf\/|epdf\/|book\/)?(10\.\d{4,9}\/.+)$/i.exec(url.pathname);
  return match ? normalizeDoi(match[1]) : null;
}

// ─────────────────────────────── parseInput ───────────────────────────────

function looksLikePdfUrl(url: URL): boolean {
  if (/\.pdf$/i.test(url.pathname)) return true;
  // OpenReview serves PDFs from /pdf?id=<forum id>.
  return url.hostname.endsWith("openreview.net") && url.pathname === "/pdf" && url.searchParams.has("id");
}

/** "arxiv.org/abs/x" or "www.example.com" without a scheme; a TLD made of letters avoids matching DOIs. */
const SCHEMELESS_URL = /^(?:www\.[^\s/]+|[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}(?::\d+)?\/\S*)$/i;

function stripWrapping(text: string): string {
  return text
    .trim()
    .replace(/^[<"'“‘`]+/, "")
    .replace(/[>"'”’`]+$/, "")
    .trim();
}

function fromDoi(raw: string, doi: string, url?: string): ParsedInput {
  const arxiv = arxivIdFromDoi(doi);
  const parsed: ParsedInput = arxiv ? { ...fromArxiv(raw, arxiv, url), doi } : { kind: "doi", raw, doi };
  if (url) parsed.url = url;
  return parsed;
}

function fromArxiv(raw: string, ref: ArxivRef, url?: string): ParsedInput {
  const parsed: ParsedInput = { kind: "arxiv", raw, arxivId: ref.id };
  if (ref.version) parsed.version = ref.version;
  if (url) parsed.url = url;
  return parsed;
}

function fromUrl(raw: string, href: string): ParsedInput | null {
  let url: URL;
  try {
    url = new URL(href);
  } catch {
    return null;
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") return null;
  const normalized = url.toString();

  if (/^(?:dx\.)?doi\.org$/i.test(url.hostname)) {
    const doi = normalizeDoi(url.pathname.slice(1));
    return doi ? fromDoi(raw, doi, normalized) : { kind: "url", raw, url: normalized };
  }
  const arxiv = arxivIdFromUrl(normalized);
  if (arxiv) return fromArxiv(raw, arxiv, normalized);

  const publisherDoi = doiFromPublisherUrl(url);
  if (publisherDoi) return fromDoi(raw, publisherDoi, normalized);

  if (looksLikePdfUrl(url)) return { kind: "pdf-url", raw, url: normalized };
  return { kind: "url", raw, url: normalized };
}

/**
 * Classifies raw user input. Never throws: anything unrecognised is treated
 * as a topic query (an empty string yields an empty topic).
 */
export function parseInput(raw: string): ParsedInput {
  const text = stripWrapping(raw);

  if (/^https?:\/\//i.test(text)) {
    const parsed = fromUrl(raw, text);
    if (parsed) return parsed;
  }

  // Citation style "arXiv:2402.02750 [cs.CL]" carries the primary category after the ID.
  const arxivCandidate = trimTrailingPunctuation(text.replace(/\s*\[[a-z-]+(?:\.[a-z-]+)?\]$/i, ""));
  if (/^arxiv\s*:/i.test(text)) {
    const ref = parseArxivId(arxivCandidate);
    if (ref) return fromArxiv(raw, ref);
  }

  if (/^doi\s*:/i.test(text) || /^10\.\d{4,9}\//.test(text)) {
    const doi = normalizeDoi(text);
    if (doi) return fromDoi(raw, doi);
  }

  const bareArxiv = parseArxivId(arxivCandidate);
  if (bareArxiv) return fromArxiv(raw, bareArxiv);

  if (SCHEMELESS_URL.test(text)) {
    const parsed = fromUrl(raw, `https://${text}`);
    if (parsed) return parsed;
  }

  return { kind: "topic", raw, query: text.replace(/\s+/g, " ") };
}

// ───────────────────────── Metadata text normalisation ─────────────────────────

// prettier-ignore
const HTML_ENTITIES: Record<string, string> = {
  amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", ndash: "–", mdash: "—",
};

function decodeEntities(text: string): string {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (match, body: string) => {
    if (body[0] === "#") {
      const code =
        body[1] === "x" || body[1] === "X" ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
      return Number.isFinite(code) && code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : match;
    }
    return HTML_ENTITIES[body.toLowerCase()] ?? match;
  });
}

/** Only real tags (a letter right after "<"), so "a < b and c > d" survives. */
const MARKUP_TAG = /<\/?[a-z][\w:.-]*(?:\s[^<>]*)?\/?>/gi;

// prettier-ignore
const LATEX_SYMBOLS: Record<string, string> = {
  times: "×", sim: "~", approx: "≈", pm: "±", mp: "∓", le: "≤", leq: "≤", ge: "≥", geq: "≥",
  ll: "≪", gg: "≫", ne: "≠", neq: "≠", cdot: "·", to: "→", rightarrow: "→", leftarrow: "←",
  infty: "∞", ldots: "…", cdots: "…", dots: "…", ell: "ℓ", circ: "°",
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", Delta: "Δ", epsilon: "ε", varepsilon: "ε",
  eta: "η", theta: "θ", kappa: "κ", lambda: "λ", mu: "μ", pi: "π", rho: "ρ", sigma: "σ",
  Sigma: "Σ", tau: "τ", phi: "φ", varphi: "φ", omega: "ω", Omega: "Ω",
};

/** Commands whose single braced argument is simply shown: \textbf{X}, \mathbf{X}, \emph{X}, ... */
// prettier-ignore
const LATEX_FORMAT_NAMES = [
  "textbf", "textit", "textrm", "textsf", "texttt", "textsc", "text", "emph", "underline",
  "mathbf", "mathrm", "mathit", "mathsf", "mathtt", "mathcal", "mathbb", "boldsymbol", "operatorname",
];
const LATEX_FORMAT_COMMAND = new RegExp(`\\\\(?:${LATEX_FORMAT_NAMES.join("|")})\\s*\\{([^{}]*)\\}`, "g");

const SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉";

function renderScript(operator: string, body: string): string {
  if (!/^\d+$/.test(body)) return `${operator}${body}`;
  const digits = operator === "^" ? SUPERSCRIPT_DIGITS : SUBSCRIPT_DIGITS;
  return [...body].map((d) => digits[Number(d)]).join("");
}

function renderInlineMath(math: string): string {
  return math
    .replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "$1/$2")
    .replace(/\\sqrt\s*\{([^{}]*)\}/g, "√$1")
    .replace(/\\(?:left|right)\b/g, "")
    .replace(/\\[,;:!]/g, " ")
    .replace(/([_^])(?:\{([^{}]*)\}|([A-Za-z0-9]))/g, (_m, op: string, group?: string, single?: string) =>
      renderScript(op, group ?? single ?? ""),
    )
    .replace(/[{}]/g, "");
}

/**
 * Light LaTeX-to-Unicode for arXiv abstracts ("$\mathbf{2.6\times}$" -> "2.6×",
 * "KV$^2$" -> "KV²"). Not a TeX renderer: it only covers the inline markup that
 * routinely appears in titles and abstracts, and leaves anything else readable.
 */
export function cleanLatex(text: string): string {
  if (!/[\\$]/.test(text)) return text;
  let out = text;
  // Unwrap innermost formatting first; three passes cover realistic nesting.
  for (let pass = 0; pass < 3; pass++) out = out.replace(LATEX_FORMAT_COMMAND, "$1");
  out = out.replace(/(?<!\\)\$([^$]{1,300}?)(?<!\\)\$/g, (_m, math: string) => renderInlineMath(math));
  out = out.replace(/\\([a-zA-Z]+)/g, (match, name: string) => LATEX_SYMBOLS[name] ?? match);
  return out.replace(/\\([%&#_$])/g, "$1");
}

/** Normalises any title/abstract string coming from an API: markup, entities, LaTeX, whitespace. */
export function cleanMetaText(text: string | null | undefined): string {
  if (!text) return "";
  const withoutTags = decodeEntities(text.replace(MARKUP_TAG, " "));
  return cleanLatex(withoutTags)
    .replace(/(\w)---(\w)/g, "$1—$2")
    .replace(/(\w)--(\w)/g, "$1–$2")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:!?)\]])(?=\s|$)/g, "$1")
    .replace(/([(\[])\s+/g, "$1")
    .trim();
}

/** "Liu, Zirui" -> "Zirui Liu"; names without exactly one comma are kept as written. */
export function normalizePersonName(name: string): string {
  const clean = cleanMetaText(name);
  const parts = clean.split(",");
  if (parts.length !== 2) return clean;
  const [family, given] = parts.map((p) => p.trim());
  return family && given ? `${given} ${family}` : clean;
}

/** Anything shorter is a fragment or a placeholder, not an abstract. */
const MIN_ABSTRACT_WORDS = 20;

function surname(name: string): string {
  return (name.trim().split(/\s+/).at(-1) ?? "").toLowerCase();
}

/**
 * Indexes sometimes store a citation string in the abstract field (OpenAlex
 * has "Jacob Devlin, Ming-Wei Chang, ... Proceedings of ..." for BERT). Real
 * abstracts do not open with their own author list.
 */
function looksLikeCitation(text: string, authors: string[]): boolean {
  const lower = text.toLowerCase();
  if (authors.length === 1) return lower.startsWith(authors[0].toLowerCase());
  const surnames = authors.map(surname).filter((s) => s.length > 1);
  if (surnames.length < 2) return false;
  const head = lower.slice(0, 200);
  const named = surnames.filter((s) => head.includes(s)).length;
  // Opens with the first author and names at least one co-author right after.
  return lower.slice(0, 40).includes(surnames[0]) && named >= 2;
}

/**
 * Cleans an abstract and rejects values that are not one: too short, or a
 * citation string. A leading "Abstract" label is dropped only when a capital
 * letter follows, so "Abstract interpretation is ..." survives.
 */
export function normalizeAbstract(
  raw: string | null | undefined,
  authors: string[] = [],
): string | undefined {
  const text = cleanMetaText(raw).replace(
    /^(?:Abstract|ABSTRACT|Summary|SUMMARY)\s*[:.\-–—]?\s+(?=[A-Z])/,
    "",
  );
  if (text.split(" ").length < MIN_ABSTRACT_WORDS) return undefined;
  if (looksLikeCitation(text, authors)) return undefined;
  return text;
}

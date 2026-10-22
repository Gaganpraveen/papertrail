/**
 * Document structure recovery: from visual lines to sections, references,
 * captions and links.
 *
 * Papers follow strong typographic conventions, so the structure is inferred
 * from layout statistics rather than learned:
 *   - The body font is the most common font size; headings are short lines
 *     that are bold, larger, or ALL CAPS, and either numbered ("3", "3.2",
 *     "A.1", "IV") or a well-known section name. Numbered headings must follow
 *     the numbering sequence, which rejects bold body lines such as "2 times".
 *   - Running headers/footers are lines at the top/bottom edge whose text
 *     (digits ignored) repeats across pages; bare page numbers are dropped too.
 *   - Captions start with "Figure 3:" / "Table 2." and continue while the font
 *     and line spacing stay the same. Small trailing lines of a column are
 *     footnotes. Both are kept out of the paragraph flow.
 *   - Line breaks are dehyphenated using the document's own vocabulary
 *     ("compres-" + "sion" becomes "compression" but "state-of-the-" + "art"
 *     keeps its hyphen).
 *   - References are split on labels ("[12]"), hanging indents, or vertical
 *     gaps, whichever the bibliography style uses, then parsed for year,
 *     arXiv id, DOI, authors and title.
 */
import type { Caption, ParsedDocument, Reference, Section, SectionKind, TextLine } from "@/types";

// ──────────────────────────────── Statistics ───────────────────────────────

interface DocStats {
  body: number;
  /** Typical full line width per column type (0 = full width, 1/2 = columns). */
  width: Record<0 | 1 | 2, number>;
  pageTop: Map<number, number>;
  pageBottom: Map<number, number>;
}

function modalFontSize(lines: TextLine[]): number {
  const weights = new Map<number, number>();
  for (const line of lines) {
    const key = Math.round(line.fontSize * 2) / 2;
    weights.set(key, (weights.get(key) ?? 0) + line.text.length);
  }
  let best = 10;
  let bestWeight = -1;
  for (const [size, weight] of weights) {
    if (weight > bestWeight) {
      best = size;
      bestWeight = weight;
    }
  }
  return best;
}

function percentile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

function computeStats(lines: TextLine[]): DocStats {
  const body = modalFontSize(lines);
  const widths: Record<0 | 1 | 2, number[]> = { 0: [], 1: [], 2: [] };
  const pageTop = new Map<number, number>();
  const pageBottom = new Map<number, number>();
  for (const line of lines) {
    if (Math.abs(line.fontSize - body) <= 0.6) widths[line.column].push(line.width);
    pageTop.set(line.page, Math.min(pageTop.get(line.page) ?? Infinity, line.y));
    pageBottom.set(line.page, Math.max(pageBottom.get(line.page) ?? -Infinity, line.y));
  }
  const fallback = percentile([...widths[0], ...widths[1], ...widths[2]], 0.8);
  return {
    body,
    width: {
      0: percentile(widths[0], 0.8) || fallback,
      1: percentile(widths[1], 0.8) || fallback,
      2: percentile(widths[2], 0.8) || fallback,
    },
    pageTop,
    pageBottom,
  };
}

const letterRatio = (text: string): number => {
  const visible = text.replace(/\s/g, "");
  if (!visible) return 0;
  return (visible.match(/\p{L}/gu)?.length ?? 0) / visible.length;
};

const isAllCaps = (text: string): boolean => /\p{Lu}/u.test(text) && text === text.toUpperCase();

const wordCount = (text: string): number => text.split(/\s+/).filter(Boolean).length;

// ─────────────────────────── Headers and footers ──────────────────────────

const PAGE_NUMBER_RE = /^(?:page\s*)?[-–—]?\s*\d{1,4}\s*[-–—]?(?:\s*(?:of|\/)\s*\d{1,4})?$/i;
const ROMAN_PAGE_RE = /^[ivxlc]{1,6}$/i;
const ARXIV_STAMP_RE = /^arXiv:\s*\S+\s+\[[\w.-]+\]\s+\d{1,2}\s+\w{3}\s+\d{4}$/i;
/** Only the outermost rows of a page, within this distance of its edge, can be furniture. */
const EDGE_BAND = 40;

const furnitureKey = (text: string): string =>
  text.toLowerCase().replace(/\d+/g, "#").replace(/\s+/g, " ").trim();

/** Indices of running headers, footers, page numbers and arXiv stamps. */
function findFurniture(lines: TextLine[], stats: DocStats, pagesParsed: number): Set<number> {
  const candidates: number[] = [];
  lines.forEach((line, i) => {
    const top = stats.pageTop.get(line.page) ?? 0;
    const bottom = stats.pageBottom.get(line.page) ?? 0;
    if (line.y - top <= EDGE_BAND || bottom - line.y <= EDGE_BAND) candidates.push(i);
  });

  const pagesByKey = new Map<string, Set<number>>();
  for (const i of candidates) {
    const key = furnitureKey(lines[i].text);
    if (!pagesByKey.has(key)) pagesByKey.set(key, new Set());
    pagesByKey.get(key)?.add(lines[i].page);
  }
  // A line repeated on a quarter of the pages (at least 2, at most 4) is a running header.
  const minRepeats = Math.max(2, Math.min(4, Math.round(pagesParsed / 4)));

  const furniture = new Set<number>();
  for (const i of candidates) {
    const text = lines[i].text.trim();
    const repeats = pagesByKey.get(furnitureKey(text))?.size ?? 0;
    // Running heads are set small; the page-1 title they often repeat is not.
    const repeated = repeats >= minRepeats && lines[i].fontSize <= stats.body + 1;
    if (repeated || PAGE_NUMBER_RE.test(text) || ROMAN_PAGE_RE.test(text)) furniture.add(i);
  }
  lines.forEach((line, i) => {
    if (ARXIV_STAMP_RE.test(line.text.trim())) furniture.add(i);
  });
  return furniture;
}

// ───────────────────────────── Text joining ───────────────────────────────

const URL_TOKEN_RE = /(?:https?:\/\/|www\.|github\.com\/|huggingface\.co\/)\S*$/i;

/** Words seen inside lines (never at a line end), used to judge line-break hyphens. */
export function buildVocabulary(texts: string[]): Set<string> {
  const vocab = new Set<string>();
  for (const text of texts) {
    const tokens = text.split(/\s+/);
    tokens.pop(); // the last token may be a hyphenated fragment
    for (const token of tokens) {
      const word = token.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "").toLowerCase();
      if (word.length > 1) vocab.add(word);
    }
  }
  return vocab;
}

/** A wrapped URL continues with a path-like token, not with an ordinary word. */
function continuesUrl(next: string, vocab: Set<string>): boolean {
  const token = next.split(/\s/, 1)[0].replace(/[.,;:)]+$/, "");
  if (!/^[\p{Ll}\d/_\-.%#?=&~]/u.test(token)) return false; // a capital starts a new sentence or heading
  return /[/._\-=?&#%~\d]/.test(token) || !vocab.has(token.toLowerCase());
}

/**
 * Appends `next` to `text` across a line break: removes hyphenation
 * ("compres-" + "sion"), keeps compound hyphens ("state-of-the-" + "art"),
 * and glues URLs that wrapped onto the next line.
 */
export function joinLineText(text: string, next: string, vocab: Set<string>): string {
  if (!text) return next;
  if (!next) return text;
  const lastToken = text.slice(text.lastIndexOf(" ") + 1);
  const isUrl = URL_TOKEN_RE.test(lastToken);
  if (isUrl && /[/\-_=?&#~%:]$/.test(lastToken) && continuesUrl(next, vocab)) return text + next;
  if (/\p{L}[-\u2010]$/u.test(text)) {
    if (!/^\p{Ll}/u.test(next)) return text + next; // "GPT-" + "4", "Llama-" + "2"
    const stem = lastToken.replace(/^[^\p{L}\p{N}]+/u, "").slice(0, -1);
    const firstWord = next.match(/^[\p{L}]+/u)?.[0] ?? "";
    if (isUrl || stem.includes("-")) return text + next;
    const joined = (stem + firstWord).toLowerCase();
    const hyphenated = `${stem}-${firstWord}`.toLowerCase();
    if (vocab.has(joined)) return text.slice(0, -1) + next;
    if (vocab.has(hyphenated)) return text + next;
    // Two real words ("commonly-" + "used", "in-" + "depth") form a compound.
    if (stem.length >= 2 && firstWord.length >= 3 && vocab.has(stem.toLowerCase()) && vocab.has(firstWord.toLowerCase())) {
      return text + next;
    }
    return text.slice(0, -1) + next;
  }
  if (/[–—]$/.test(text)) return text + next; // page ranges and dashes broken across lines
  return `${text} ${next}`;
}

function joinLines(texts: string[], vocab: Set<string>): string {
  let out = "";
  for (const t of texts) out = joinLineText(out, t.trim(), vocab);
  return out.trim();
}

// ──────────────────────────────── Captions ────────────────────────────────

const CAPTION_RE =
  /^(figure|fig\.?|table|tab\.?)\s*([A-Z]?\d{1,3}(?:\.\d{1,2})?|[IVX]{1,5})\s*(:|\.|\||[—–-](?=\s))\s*/i;
const MAX_CAPTION_LINES = 15;

interface CaptionBlock {
  caption: Caption;
  lineIndices: number[];
}

/** Overlap of two lines' x-extents, as a share of the narrower one. */
function horizontalOverlap(a: TextLine, b: TextLine): number {
  const overlap = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
  return Math.max(0, overlap) / Math.max(1, Math.min(a.width, b.width));
}

function captionKind(word: string): "figure" | "table" {
  return word.toLowerCase().startsWith("fig") ? "figure" : "table";
}

/** A line that opens a caption, rather than a sentence that happens to start with "Table 2.". */
function opensCaption(lines: TextLine[], i: number, stats: DocStats, match: RegExpMatchArray): boolean {
  if (match[3] !== ".") return true;
  const line = lines[i];
  const prev = lines[i - 1];
  if (line.bold || Math.abs(line.fontSize - stats.body) >= 0.5) return true;
  if (!prev || prev.page !== line.page || prev.column !== line.column) return true;
  return /[.:!?]$/.test(prev.text) || line.y - prev.y > stats.body * 1.6;
}

function findCaptions(
  lines: TextLine[],
  skip: Set<number>,
  stats: DocStats,
  vocab: Set<string>,
): CaptionBlock[] {
  const blocks: CaptionBlock[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < lines.length; i++) {
    if (skip.has(i)) continue;
    const match = lines[i].text.match(CAPTION_RE);
    if (!match || !opensCaption(lines, i, stats, match)) continue;

    const start = lines[i];
    const indices = [i];
    let prev = start;
    // The label may be set larger/bolder than the caption text ("Table 1:" 10pt, text 9pt).
    let size = start.fontSize;
    for (let j = i + 1; j < lines.length && indices.length < MAX_CAPTION_LINES; j++) {
      if (skip.has(j)) continue;
      const next = lines[j];
      const sizeFits =
        Math.abs(next.fontSize - size) <= 0.6 ||
        (indices.length === 1 && next.fontSize < size && next.fontSize >= size - 1.5);
      const sameBlock =
        next.page === start.page &&
        next.column === start.column &&
        sizeFits &&
        next.y > prev.y &&
        next.y - prev.y <= start.fontSize * 1.8 &&
        horizontalOverlap(next, start) >= 0.5 &&
        !CAPTION_RE.test(next.text) &&
        !NUMBERED_RE.test(next.text.trim());
      if (!sameBlock) break;
      indices.push(j);
      prev = next;
      size = next.fontSize;
    }

    const kind = captionKind(match[1]);
    const label = `${kind === "figure" ? "Figure" : "Table"} ${match[2]}`;
    const body = joinLines(
      indices.map((k) => lines[k].text),
      vocab,
    ).slice(match[0].length);
    for (const k of indices) skip.add(k);
    i = indices[indices.length - 1];
    if (seen.has(label)) continue; // a repeated label is a continuation ("Table 3 (cont.)")
    seen.add(label);
    blocks.push({ caption: { kind, label, text: body.trim(), page: start.page }, lineIndices: indices });
  }
  return blocks;
}

// ─────────────────────────────── Footnotes ────────────────────────────────

/** Footnotes live in the lower part of the page… */
const FOOTNOTE_ZONE = 0.55;
/** …and are set at least this many points smaller than the body. */
const FOOTNOTE_SHRINK = 0.6;

/** Trailing runs of small-font lines at the bottom of each page column. */
function findFootnotes(lines: TextLine[], skip: Set<number>, stats: DocStats): Set<number> {
  const footnotes = new Set<number>();
  const groups = new Map<string, number[]>();
  lines.forEach((line, i) => {
    if (skip.has(i)) return;
    const key = `${line.page}:${line.column}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)?.push(i);
  });
  for (const indices of groups.values()) {
    const page = lines[indices[0]].page;
    const top = stats.pageTop.get(page) ?? 0;
    const bottom = stats.pageBottom.get(page) ?? 0;
    const zoneStart = top + (bottom - top) * FOOTNOTE_ZONE;
    for (let k = indices.length - 1; k >= 0; k--) {
      const line = lines[indices[k]];
      if (line.fontSize > stats.body - FOOTNOTE_SHRINK || line.y < zoneStart) break;
      footnotes.add(indices[k]);
    }
  }
  return footnotes;
}

// ──────────────────────────────── Headings ────────────────────────────────

const KNOWN_HEADING_RE =
  /^(abstract|introduction|related works?|background|preliminaries|methods?|methodology|approach|our approach|experiments?|experimental setup|experimental results|experiments and results|results|evaluation|discussion|limitations?|limitations and future work|conclusions?|conclusions? and future work|summary|future work|references|bibliography|acknowledge?ments?|appendix|appendices|supplementary materials?|broader impacts?|impact statement|ethics statement|ethical considerations|ethical concerns|reproducibility statement|author contributions)$/i;

const NUMBERED_RE = /^(\d{1,2}(?:\.\d{1,2}){0,3}|[A-Z](?:\.\d{1,2}){0,3}|[IVX]{1,5})\.?\s+(\S.*)$/;
const APPENDIX_PREFIX_RE = /^appendix\s+([A-Z](?:\.\d{1,2})*)\s*[:.—–-]?\s+(\S.*)$/i;
const RUN_IN_ABSTRACT_RE = /^abstract\s*[.:—–-]\s*(\S.*)$/i;

interface HeadingCandidate {
  number?: string;
  title: string;
  /** Body text that shares the heading line ("Abstract— We study…"). */
  rest?: string;
  known: boolean;
}

function stripHeadingPunctuation(title: string): string {
  return title.replace(/[\s:.]+$/, "").trim();
}

/** Purely textual match; style and numbering order are checked separately. */
function matchHeadingText(text: string): HeadingCandidate | null {
  const trimmed = text.trim();
  const runIn = trimmed.match(RUN_IN_ABSTRACT_RE);
  if (runIn && wordCount(runIn[1]) >= 4) return { title: "Abstract", rest: runIn[1], known: true };

  const bare = stripHeadingPunctuation(trimmed);
  if (KNOWN_HEADING_RE.test(bare)) return { title: bare, known: true };

  const appendix = trimmed.match(APPENDIX_PREFIX_RE);
  if (appendix) return { number: appendix[1].toUpperCase(), title: stripHeadingPunctuation(appendix[2]), known: true };

  const numbered = trimmed.match(NUMBERED_RE);
  if (!numbered) return null;
  const title = stripHeadingPunctuation(numbered[2]);
  if (!/^[\p{L}\p{N}"“(]/u.test(title) || letterRatio(title) < 0.6) return null;
  if ((title.match(/\p{L}/gu)?.length ?? 0) < 3) return null;
  if (/[,;]$/.test(numbered[2].trim())) return null;
  // Long titles that end like a sentence are bold body text, not headings.
  if (/\.$/.test(numbered[2].trim()) && wordCount(title) > 6) return null;
  return { number: numbered[1], title, known: KNOWN_HEADING_RE.test(title) };
}

/** Heading lines are short and typographically emphasised. */
function hasHeadingStyle(line: TextLine, candidate: HeadingCandidate, body: number): boolean {
  if (candidate.rest) return line.bold || /^ABSTRACT/.test(line.text) || line.fontSize >= body + 0.8;
  if (line.text.length > 120 || wordCount(candidate.title) > 16) return false;
  const larger = line.fontSize >= body + 0.8;
  const caps = isAllCaps(candidate.title) && candidate.title.length > 3;
  if (candidate.number === undefined) return wordCount(candidate.title) <= 6 && (line.bold || larger || caps);
  if (line.fontSize < body - 0.3) return false; // table headers and figure labels are smaller
  return line.bold || larger || (caps && line.fontSize >= body - 0.2);
}

type Scheme = "arabic" | "letter" | "roman";

const ROMAN_VALUES: Record<string, number> = { I: 1, V: 5, X: 10 };
function romanToInt(roman: string): number {
  let total = 0;
  for (let i = 0; i < roman.length; i++) {
    const value = ROMAN_VALUES[roman[i]];
    const next = ROMAN_VALUES[roman[i + 1]] ?? 0;
    total += value < next ? -value : value;
  }
  return total;
}
const isRoman = (s: string): boolean => /^[IVX]{1,5}$/.test(s) && romanToInt(s) > 0;

/**
 * Accepts numbered headings only when they continue the numbering sequence
 * (1, 2, 2.1, 3, ... then A, A.1, B, ...), allowing one missed heading.
 */
class HeadingSequencer {
  private arabic = 0;
  private letter = 0;
  private roman = 0;
  private subLetter = 0;

  /** Returns the heading level and numbering scheme, or null when the number is out of sequence. */
  accept(number: string, appendixAllowed: boolean): { level: number; scheme: Scheme } | null {
    const parts = number.split(".");
    const head = parts[0];
    const depth = parts.length;

    if (/^\d+$/.test(head)) {
      const n = Number(head);
      const ok =
        depth === 1
          ? (this.arabic === 0 && n <= 2) || (n > this.arabic && n <= this.arabic + 2)
          : n === this.arabic || n === this.arabic + 1 || (this.arabic === 0 && n <= 2);
      if (!ok || parts.slice(1).some((p) => Number(p) > 30)) return null;
      this.arabic = Math.max(this.arabic, n);
      return { level: depth, scheme: "arabic" };
    }

    const scheme = this.schemeOf(head, depth);
    if (scheme === "roman") {
      const n = romanToInt(head);
      if (!(n === this.roman + 1 || (n === this.roman + 2 && this.roman > 0))) return null;
      this.roman = n;
      this.subLetter = 0;
      return { level: 1, scheme: "roman" };
    }
    const code = head.charCodeAt(0) - 64;
    if (this.roman > 0) {
      // IEEE style: lettered subsections under roman sections.
      if (depth !== 1 || code > this.subLetter + 2) return null;
      this.subLetter = Math.max(this.subLetter, code);
      return { level: 2, scheme: "roman" };
    }
    if (!appendixAllowed && this.letter === 0) return null;
    const ok =
      depth === 1
        ? (this.letter === 0 && code <= 2) || (code > this.letter && code <= this.letter + 2)
        : code === this.letter || code === this.letter + 1 || (this.letter === 0 && code <= 2);
    if (!ok) return null;
    this.letter = Math.max(this.letter, code);
    return { level: depth, scheme: "letter" };
  }

  private schemeOf(head: string, depth: number): Scheme {
    if (depth === 1 && isRoman(head) && this.letter === 0 && this.arabic === 0) {
      const n = romanToInt(head);
      if (n === this.roman + 1 || (this.roman > 0 && n === this.roman + 2)) return "roman";
    }
    return "letter";
  }
}

interface DetectedHeading {
  index: number;
  /** Extra line indices merged into the heading (wrapped titles, TOC page numbers). */
  consumed: number[];
  number?: string;
  title: string;
  level: number;
  /** Lettered appendix numbering ("A", "B.2"), as opposed to IEEE lettered subsections. */
  appendixLetter: boolean;
  rest?: string;
}

const sameRow = (a: TextLine, b: TextLine): boolean =>
  a.page === b.page && a.column === b.column && Math.abs(a.y - b.y) < 2;

function hasRowmate(lines: TextLine[], i: number): boolean {
  return [lines[i - 1], lines[i + 1]].some((other) => other !== undefined && sameRow(other, lines[i]));
}

const rowKey = (page: number, y: number): string => `${page}:${Math.round(y)}`;

/** Rows (page + baseline) that hold a bare page number, as table-of-contents entries do. */
function pageNumberRows(lines: TextLine[]): Set<string> {
  return new Set(lines.filter((l) => /^\d{1,3}$/.test(l.text.trim())).map((l) => rowKey(l.page, l.y)));
}

/** Table-of-contents entries: dot leaders, or a page number on the same row ("B Experimental Details 19"). */
function isTocEntry(line: TextLine, numberRows: Set<string>): boolean {
  if (/\.{4,}|(?:\.\s){4,}/.test(line.text)) return true;
  return [-1, 0, 1].some((dy) => numberRows.has(rowKey(line.page, line.y + dy)));
}

/** Size/bold signature of top-level numbered headings, used to accept unnumbered ones. */
function learnSectionStyle(lines: TextLine[], skip: Set<number>, body: number): { size: number; bold: boolean } | null {
  const counts = new Map<string, number>();
  lines.forEach((line, i) => {
    if (skip.has(i)) return;
    const candidate = matchHeadingText(line.text);
    if (!candidate?.number || candidate.number.includes(".") || !hasHeadingStyle(line, candidate, body)) return;
    const key = `${Math.round(line.fontSize * 2) / 2}|${line.bold}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  });
  let best: string | null = null;
  let bestCount = 1; // need at least two numbered headings to trust the style
  for (const [key, count] of counts) {
    if (count > bestCount) {
      best = key;
      bestCount = count;
    }
  }
  if (!best) return null;
  const [size, bold] = best.split("|");
  return { size: Number(size), bold: bold === "true" };
}

function matchesSectionStyle(line: TextLine, style: { size: number; bold: boolean } | null): boolean {
  if (!style || line.bold !== style.bold || Math.abs(line.fontSize - style.size) > 0.3) return false;
  const text = line.text.trim();
  return (
    wordCount(text) <= 8 &&
    /^\p{Lu}/u.test(text) &&
    !/[.,;:]$/.test(text) &&
    letterRatio(text) >= 0.8 &&
    !/\d/.test(text)
  );
}

function detectHeadings(
  lines: TextLine[],
  skip: Set<number>,
  stats: DocStats,
  frontMatterEnd: (i: number) => boolean,
): DetectedHeading[] {
  const sequencer = new HeadingSequencer();
  const style = learnSectionStyle(lines, skip, stats.body);
  const headings: DetectedHeading[] = [];
  const numberRows = pageNumberRows(lines);
  let appendixAllowed = false;

  for (let i = 0; i < lines.length; i++) {
    if (skip.has(i)) continue;
    const line = lines[i];
    let candidate = matchHeadingText(line.text);
    if (candidate && !hasHeadingStyle(line, candidate, stats.body)) candidate = null;
    // Unnumbered names ("Summary", "Results") also appear as table header cells.
    if (candidate && candidate.number === undefined && hasRowmate(lines, i)) candidate = null;
    if (!candidate && frontMatterEnd(i) && matchesSectionStyle(line, style)) {
      candidate = { title: line.text.trim(), known: false };
    }
    if (!candidate || isTocEntry(line, numberRows)) continue;

    let level = 1;
    let appendixLetter = false;
    if (candidate.number !== undefined) {
      const accepted = sequencer.accept(candidate.number, appendixAllowed);
      if (accepted === null) continue;
      level = accepted.level;
      appendixLetter = accepted.scheme === "letter";
    }
    if (/^(references|bibliography|appendix|appendices|supplementary)/i.test(candidate.title)) appendixAllowed = true;

    const consumed = mergeWrappedTitle(lines, i, skip, stats);
    const title = [candidate.title, ...consumed.map((k) => lines[k].text.trim())].join(" ");
    headings.push({ index: i, consumed, number: candidate.number, title, level, appendixLetter, rest: candidate.rest });
    i = consumed.length ? consumed[consumed.length - 1] : i;
  }
  return headings;
}

/** Long headings wrap; the continuation shares the heading's font and sits right below it. */
function mergeWrappedTitle(lines: TextLine[], i: number, skip: Set<number>, stats: DocStats): number[] {
  const line = lines[i];
  const next = lines[i + 1];
  if (!next || skip.has(i + 1)) return [];
  // A heading wraps when it fills its line, or (beside a wrapped figure) when both halves are ALL CAPS.
  const fillsLine = line.width >= stats.width[line.column] * 0.6;
  if (!fillsLine && !(isAllCaps(line.text) && isAllCaps(next.text))) return [];
  const continues =
    next.page === line.page &&
    next.bold === line.bold &&
    Math.abs(next.fontSize - line.fontSize) <= 0.3 &&
    next.y > line.y &&
    next.y - line.y <= line.fontSize * 1.7 &&
    wordCount(next.text) <= 8 &&
    !matchHeadingText(next.text);
  return continues ? [i + 1] : [];
}

// ──────────────────────────── Section kinds ───────────────────────────────

const KIND_RULES: [SectionKind, RegExp][] = [
  ["abstract", /\babstract\b/i],
  ["references", /\b(references|bibliography|works cited)\b/i],
  ["acknowledgments", /\backnowledge?ments?\b|\bfunding\b/i],
  ["appendix", /\b(appendix|appendices|supplementary)\b/i],
  ["limitations", /\b(limitations?|weakness(es)?|failure (cases|modes))\b/i],
  ["conclusion", /\b(conclusions?|concluding|future work)\b/i],
  ["related", /\b(related work|prior work|literature)\b/i],
  ["background", /\b(background|preliminar(y|ies)|problem (setup|setting|formulation|statement)|notation)\b/i],
  ["introduction", /\b(introduction|motivation)\b/i],
  ["discussion", /\b(discussion|broader impacts?|ethic(s|al)|societal|impact statement)\b/i],
  ["method", /\b(theor(y|etical)|proofs?)\b/i],
  ["results", /\b(results?|ablations?|analys[ie]s|findings|case stud(y|ies)|comparisons?)\b/i],
  ["experiments", /\b(experiments?|experimental|evaluations?|benchmarks?|empirical|setup)\b/i],
  [
    "method",
    /\b(methods?|methodology|approach|architectur(e|al)|framework|algorithms?|design|proposed|training|pre-?training|fine-?tuning|formulation|implementation|technical|mechanism|objective|model details)\b/i,
  ],
];

function keywordKind(title: string): SectionKind | null {
  for (const [kind, re] of KIND_RULES) if (re.test(title)) return kind;
  return null;
}

/** Subsection titles strong enough to override the kind inherited from the parent. */
const SUBSECTION_OVERRIDES = new Set<SectionKind>(["results", "experiments", "limitations", "discussion", "conclusion", "related"]);
const EVALUATION_KINDS = new Set<SectionKind>(["experiments", "results"]);

interface KindInput {
  title: string;
  level: number;
  numbered: boolean;
  appendixLetter: boolean;
}

/** Whether a top-level section has an experiments/results subsection before the next top-level one. */
function hasEvaluationChild(items: KindInput[], index: number): boolean {
  for (let k = index + 1; k < items.length && items[k].level > 1; k++) {
    const kind = keywordKind(items[k].title);
    if (kind && EVALUATION_KINDS.has(kind)) return true;
  }
  return false;
}

/**
 * Keywords first. Subsections inherit their parent's kind unless their own
 * title names a stronger one ("3.2 Results"). Unlabelled numbered top-level
 * sections are placed by position: with experimental subsections they are
 * experiments; between the introduction and the first evaluation they are
 * the method; right after an evaluation section they continue it.
 */
function assignKinds(items: KindInput[]): SectionKind[] {
  const kinds: SectionKind[] = [];
  const byLevel: SectionKind[] = [];
  let previousTop: SectionKind = "other";
  let seenIntro = false;
  let seenEvaluation = false;
  let inAppendix = false;
  let afterReferences = false;

  items.forEach((item, index) => {
    const keyword = keywordKind(item.title);
    let kind: SectionKind;
    if (item.level > 1) {
      const parent = byLevel[item.level - 2] ?? previousTop;
      kind = parent !== "appendix" && keyword && SUBSECTION_OVERRIDES.has(keyword) ? keyword : parent;
    } else {
      const mainBody = item.numbered && !afterReferences;
      if (item.appendixLetter && keyword !== "references" && keyword !== "acknowledgments") kind = "appendix";
      else if (keyword) kind = keyword;
      else if (inAppendix) kind = "appendix";
      else if (mainBody && hasEvaluationChild(items, index)) kind = "experiments";
      else if (mainBody && seenIntro && !seenEvaluation) kind = "method";
      else if (mainBody && EVALUATION_KINDS.has(previousTop)) kind = "experiments";
      else kind = "other";
      previousTop = kind;
    }
    byLevel[item.level - 1] = kind;
    byLevel.length = item.level;
    kinds.push(kind);
    if (kind === "introduction") seenIntro = true;
    if (EVALUATION_KINDS.has(kind) || kind === "conclusion") seenEvaluation = true;
    if (kind === "appendix") inAppendix = true;
    if (kind === "references") afterReferences = true;
  });
  return kinds;
}

// ─────────────────────────────── References ───────────────────────────────

const LABEL_RE = /^\s*\[[^\]]{1,24}\]\s+/;
const NUMBERED_REF_RE = /^\s*\d{1,3}\.\s+\p{Lu}/u;

const groupKey = (line: TextLine): string => `${line.page}:${line.column}`;

/** Splits bibliography lines into entries using the cue the style provides. */
export function splitReferenceEntries(lines: TextLine[]): TextLine[][] {
  if (lines.length === 0) return [];
  const labelled = lines.filter((l) => LABEL_RE.test(l.text)).length;
  const numbered = lines.filter((l) => NUMBERED_REF_RE.test(l.text)).length;

  let isStart: (line: TextLine, i: number) => boolean;
  if (labelled >= 2 && labelled >= lines.length * 0.12) {
    isStart = (line) => LABEL_RE.test(line.text);
  } else if (numbered >= 2 && numbered >= lines.length * 0.12) {
    isStart = (line) => NUMBERED_REF_RE.test(line.text);
  } else {
    isStart = layoutStartDetector(lines);
  }

  const entries: TextLine[][] = [];
  lines.forEach((line, i) => {
    if (entries.length === 0 || isStart(line, i)) entries.push([line]);
    else entries[entries.length - 1].push(line);
  });
  return entries;
}

/** Hanging indent if the style uses one, otherwise extra vertical space between entries. */
function layoutStartDetector(lines: TextLine[]): (line: TextLine, i: number) => boolean {
  const minX = new Map<string, number>();
  for (const line of lines) minX.set(groupKey(line), Math.min(minX.get(groupKey(line)) ?? Infinity, line.x));
  const indented = lines.filter((l) => l.x > (minX.get(groupKey(l)) ?? 0) + 3).length;
  if (indented >= lines.length * 0.2) {
    return (line) => line.x <= (minX.get(groupKey(line)) ?? 0) + 2.5;
  }
  const gaps: number[] = [];
  for (let i = 1; i < lines.length; i++) {
    if (groupKey(lines[i]) === groupKey(lines[i - 1])) gaps.push(lines[i].y - lines[i - 1].y);
  }
  const typicalGap = percentile(gaps, 0.3);
  return (line, i) => {
    const prev = lines[i - 1];
    if (!prev || groupKey(prev) !== groupKey(line)) return /\.$/.test(prev?.text ?? ".");
    return line.y - prev.y > typicalGap * 1.3;
  };
}

const YEAR_RE = /(?<![\d–\-:/.])((?:19[5-9]|20[0-4])\d)(?![\d–\-:/])[a-z]?/g;
const ARXIV_RES = [
  /arxiv\.org\/(?:abs|pdf)\/(\d{4}\.\d{4,5})/i,
  /arXiv(?:\s*preprint)?(?:\s*arXiv)?\s*[:\s]\s*(\d{4}\.\d{4,5})/i,
  /\babs\/(\d{4}\.\d{4,5})/i,
  /10\.48550\/arXiv\.(\d{4}\.\d{4,5})/i,
  /arXiv[:\s]+([a-z-]+(?:\.[A-Z]{2})?\/\d{7})/i,
];
const DOI_RE = /\b(10\.\d{4,9}\/[^\s,;"'<>]+)/;
const LINK_RE = /\b(?:https?:\/\/|www\.)\S+/gi;
/** "J." or "J.-P." or common abbreviations do not end an author list or title. */
const NON_TERMINAL_RE = /^(?:\p{Lu}|\p{Lu}\.-?\p{Lu}|Jr|Sr|St|vs|eds?|Proc|Vol|No|pp|Inc|Dept|Univ)$/u;
const VENUE_START_RE = /^(in\b|in:|proceedings|proc\.|arxiv|advances in|journal|url\b|http|retrieved|available|transactions|technical report)/i;

function pickYear(text: string): number | undefined {
  const cleaned = text.replace(LINK_RE, " ").replace(/arXiv:\S+/gi, " ").replace(DOI_RE, " ");
  let year: number | undefined;
  for (const match of cleaned.matchAll(YEAR_RE)) year = Number(match[1]);
  return year;
}

function pickArxivId(text: string): string | undefined {
  for (const re of ARXIV_RES) {
    const match = text.match(re);
    if (match) return match[1];
  }
  return undefined;
}

function pickDoi(text: string): string | undefined {
  const match = text.match(DOI_RE);
  return match ? match[1].replace(/[.)\]]+$/, "").toLowerCase() : undefined;
}

/** "Touvron, H., Lavril, T., and Lample, G." — surname first, initials after. */
const SURNAME_FIRST_RE = /^\p{Lu}[\p{L}'’-]+(?:\s[\p{L}'’-]+)?,\s(?:\p{Lu}\.\s?-?)+/u;
const INITIAL_OR_JOINER_RE = /^(?:\p{Lu}\.(?:-\p{Lu}\.)?,?|and|&)$/u;

/**
 * In surname-first lists the author block ends with an initial, so an
 * initial followed by anything but another name part closes the block.
 */
function endsSurnameFirstAuthors(text: string, start: number, nextAt: number): boolean {
  if (!SURNAME_FIRST_RE.test(text.slice(start))) return false;
  const nextToken = text.slice(nextAt).split(/\s/, 1)[0];
  return !!nextToken && !INITIAL_OR_JOINER_RE.test(nextToken) && !nextToken.endsWith(",");
}

/** Splits at sentence ends that are not initials ("J. Smith") or abbreviations. */
function referenceSegments(text: string): string[] {
  const segments: string[] = [];
  let start = 0;
  for (const match of text.matchAll(/([.?!])\s+/g)) {
    const end = match.index ?? 0;
    const before = text.slice(start, end);
    const lastWord = before.slice(before.lastIndexOf(" ") + 1).replace(/^[(“"]+/, "");
    const nextAt = end + match[0].length;
    if (match[1] === "." && NON_TERMINAL_RE.test(lastWord) && !endsSurnameFirstAuthors(text, start, nextAt)) continue;
    segments.push(text.slice(start, end + 1).trim());
    start = nextAt;
  }
  if (start < text.length) segments.push(text.slice(start).trim());
  return segments.filter(Boolean);
}

const YEAR_ONLY_RE = /^\(?\s*(?:19|20)\d{2}[a-z]?\s*\)?[.,]?$/;

function cleanTitle(title: string): string | undefined {
  const cleaned = title
    .replace(/,?\s*\(?(?:19|20)\d{2}[a-z]?\)?\.?$/, "")
    .replace(/^[\s.,:;]+|[\s.,:;]+$/g, "")
    .trim();
  if (cleaned.length < 6 || letterRatio(cleaned) < 0.6 || VENUE_START_RE.test(cleaned)) return undefined;
  return cleaned;
}

/** Drops the separator after an author list, keeping the period of a final initial ("Parmar, N.") or "et al.". */
function tidyAuthors(text: string): string {
  const trimmed = text.replace(/[\s.,:;]+$/, "").trim();
  const abbreviated = /(?:^|[\s.-])(?:\p{Lu}|al)$/u.test(trimmed) && /\.\s*$/.test(text);
  return abbreviated ? `${trimmed}.` : trimmed;
}

function splitAuthorsAndTitle(text: string): { authors?: string; title?: string } {
  const quoted = text.match(/[“"]([^”"]{6,300})[”"]/);
  if (quoted && quoted.index !== undefined) {
    const authors = tidyAuthors(text.slice(0, quoted.index));
    return { authors: authors || undefined, title: cleanTitle(quoted[1]) };
  }
  const segments = referenceSegments(text);
  if (segments.length < 2) return {};
  const authors = tidyAuthors(segments[0].replace(/\s*\(?(?:19|20)\d{2}[a-z]?\)?\.?$/, ""));
  let titleIndex = 1;
  if (YEAR_ONLY_RE.test(segments[1]) && segments.length > 2) titleIndex = 2;
  return {
    authors: authors && letterRatio(authors) > 0.5 && !LINK_RE.test(authors) ? authors : undefined,
    title: cleanTitle(segments[titleIndex]),
  };
}

/** Best-effort field extraction from one bibliography entry. */
export function parseReference(raw: string): Reference {
  const text = raw.replace(LABEL_RE, "").replace(/^\s*\d{1,3}\.\s+/, "").trim();
  LINK_RE.lastIndex = 0;
  const { authors, title } = splitAuthorsAndTitle(text);
  const reference: Reference = { raw };
  if (title) reference.title = title;
  if (authors) reference.authors = authors;
  const year = pickYear(text);
  if (year) reference.year = year;
  const arxivId = pickArxivId(text);
  if (arxivId) reference.arxivId = arxivId;
  const doi = pickDoi(text);
  if (doi) reference.doi = doi;
  return reference;
}

// ───────────────────────────────── Links ──────────────────────────────────

const URL_RE = /\b(?:https?:\/\/|www\.)[^\s<>"'`{}|\\^]+|\b(?:github\.com|huggingface\.co|gitlab\.com)\/[^\s<>"'`{}|\\^]+/gi;

function tidyUrl(url: string): string {
  let out = url;
  for (;;) {
    const trimmed = out
      .replace(/[.,;:!?'"]+$/, "")
      .replace(/\)$/, (m) => (out.includes("(") ? m : ""))
      .replace(/\]$/, (m) => (out.includes("[") ? m : ""));
    if (trimmed === out) break;
    out = trimmed;
  }
  if (!/^https?:\/\//i.test(out)) out = `https://${out}`;
  return out;
}

export function extractLinks(texts: string[]): string[] {
  const links = new Map<string, string>();
  for (const text of texts) {
    for (const match of text.matchAll(URL_RE)) {
      const url = tidyUrl(match[0]);
      if (!/^https?:\/\/[^/\s]+\.[a-z]{2,}/i.test(url)) continue;
      const key = url.toLowerCase().replace(/\/$/, "");
      if (!links.has(key)) links.set(key, url);
    }
  }
  return [...links.values()];
}

// ──────────────────────────── Paragraph flow ──────────────────────────────

interface FlowLine {
  line: TextLine;
  text: string;
}

/** Joins flow lines into paragraphs: vertical gaps and first-line indents start new ones. */
function flowToText(flow: FlowLine[], stats: DocStats, vocab: Set<string>): string {
  const paragraphs: string[][] = [];
  let prev: TextLine | null = null;
  for (const { line, text } of flow) {
    const sameColumn = prev && prev.page === line.page && prev.column === line.column && line.y > prev.y;
    const gap = sameColumn && prev ? line.y - prev.y : 0;
    const indented =
      sameColumn &&
      prev &&
      line.x > prev.x + stats.body * 0.8 &&
      prev.width < stats.width[prev.column] * 0.85 &&
      /[.:!?]$/.test(prev.text);
    const newParagraph = !prev || gap > Math.max(line.fontSize, stats.body) * 1.55 || indented;
    if (newParagraph || paragraphs.length === 0) paragraphs.push([]);
    paragraphs[paragraphs.length - 1].push(text);
    prev = line;
  }
  return paragraphs
    .map((p) => joinLines(p, vocab))
    .filter((p) => wordCount(p) >= 3 || /[.:!?]$/.test(p)) // stray figure labels ("exi", "(a)")
    .join("\n\n");
}

// ──────────────────────────── Title & abstract ────────────────────────────

const NOT_TITLE_RE = /^(arxiv|preprint|published|accepted|under review|proceedings|technical report|workshop)/i;

function findTitle(lines: TextLine[], skip: Set<number>, limit: number): { text?: string; indices: number[] } {
  const candidates: number[] = [];
  for (let i = 0; i < Math.min(lines.length, limit); i++) {
    if (lines[i].page !== 1) break;
    if (!skip.has(i) && letterRatio(lines[i].text) > 0.5 && !NOT_TITLE_RE.test(lines[i].text)) candidates.push(i);
  }
  if (candidates.length === 0) return { indices: [] };
  const maxSize = Math.max(...candidates.map((i) => lines[i].fontSize));
  const indices: number[] = [];
  for (const i of candidates) {
    if (Math.abs(lines[i].fontSize - maxSize) <= 0.5) indices.push(i);
    else if (indices.length > 0) break;
  }
  const text = indices
    .map((i) => lines[i].text.trim())
    .join(" ")
    .replace(/\s+/g, " ");
  return { text: text.length >= 4 ? text : undefined, indices };
}

/** Without an "Abstract" heading, the abstract is the first substantial paragraph. */
function abstractFromFrontMatter(paragraphText: string): string | undefined {
  return paragraphText.split("\n\n").find((p) => p.length >= 400 && letterRatio(p) > 0.75);
}

// ──────────────────────────────── Assembly ────────────────────────────────

/** Lines smaller than body by this much are figure labels, not prose. */
const FIGURE_TEXT_SHRINK = 1.4;

interface Draft {
  heading: DetectedHeading;
  flow: FlowLine[];
  captions: string[];
  pageEnd: number;
}

/** Figure labels, equations and table cells, judged from the line alone and its row. */
function isCellLike(lines: TextLine[], i: number, stats: DocStats): boolean {
  const line = lines[i];
  if (line.fontSize < stats.body - FIGURE_TEXT_SHRINK) return true;
  if (letterRatio(line.text) < 0.45) return true;
  // Short cells sharing a row with other cells are table content.
  const narrow = line.width < stats.width[line.column] * 0.4;
  const rowmate = (k: number) => lines[k] && lines[k] !== line && sameRow(lines[k], line);
  return narrow && (rowmate(i - 1) || rowmate(i + 1));
}

/** How far (in lines of reading order) to look for the rows around a table header. */
const TABLE_SCAN = 12;

/**
 * A table's header row ("Sampling prop. Epochs Disk size") is wordy enough to
 * pass as prose on its own; it gives itself away by opening a block (no prose
 * line just above it) whose next two rows are table cells.
 */
function isTableHeader(lines: TextLine[], i: number, stats: DocStats): boolean {
  const line = lines[i];
  if (wordCount(line.text) > 8 || /[.,;:!?]$/.test(line.text.trim())) return false;
  const inFlow = (other: TextLine) => other.page === line.page && other.column === line.column;
  for (let k = i - 1; k >= Math.max(0, i - TABLE_SCAN); k--) {
    const above = lines[k];
    if (!inFlow(above) || sameRow(above, line)) continue;
    if (line.y - above.y < line.fontSize * 1.6) return false; // continues a paragraph
    break;
  }
  let lastY = line.y;
  let rows = 0;
  for (let k = i + 1; k < Math.min(lines.length, i + TABLE_SCAN); k++) {
    const below = lines[k];
    if (!inFlow(below) || Math.abs(below.y - line.y) < 2) continue;
    // Below a header, a row split into several cells is table content even when one cell is long.
    if (!isCellLike(lines, k, stats) && !hasRowmate(lines, k)) return false;
    if (Math.abs(below.y - lastY) < 2) continue; // another cell of the same row
    if (below.y - lastY > below.fontSize * 2.2) return false;
    lastY = below.y;
    if (++rows === 2) return true;
  }
  return false;
}

function isNonProse(lines: TextLine[], i: number, stats: DocStats, inReferences: boolean): boolean {
  if (inReferences) return lines[i].fontSize < stats.body - FIGURE_TEXT_SHRINK; // references are number-heavy by nature
  return isCellLike(lines, i, stats) || isTableHeader(lines, i, stats);
}

function qualityOf(avgChars: number): ParsedDocument["quality"] {
  if (avgChars < 80) return "image-only";
  if (avgChars < 500) return "degraded";
  return "good";
}

/** Above this many pages the reader is warned that analysis covers a long document. */
const LONG_PAPER_PAGES = 50;

/** Recovers sections, references, captions and links from ordered layout lines. */
export function buildDocument(lines: TextLine[], pageCount: number, pagesParsed: number): ParsedDocument {
  const stats = computeStats(lines);
  const vocab = buildVocabulary(lines.map((l) => l.text));
  const furniture = findFurniture(lines, stats, pagesParsed);

  const skip = new Set<number>(furniture);
  const title = findTitle(lines, skip, 40);
  const captionBlocks = findCaptions(lines, new Set(skip), stats, vocab);
  const captionLines = new Set(captionBlocks.flatMap((b) => b.lineIndices));
  const footnotes = findFootnotes(lines, new Set([...skip, ...captionLines]), stats);

  const headingSkip = new Set([...skip, ...captionLines, ...footnotes, ...title.indices]);
  const firstHeadingGuess = lines.findIndex((l, i) => !headingSkip.has(i) && matchHeadingText(l.text) !== null);
  const pastFrontMatter = (i: number) => i > Math.max(firstHeadingGuess, title.indices.at(-1) ?? 0);
  const headings = detectHeadings(lines, headingSkip, stats, pastFrontMatter);

  // Walk the lines once, routing each to the front matter or its section.
  const headingAt = new Map(headings.map((h) => [h.index, h]));
  const consumed = new Set(headings.flatMap((h) => h.consumed));
  const captionByStart = new Map(captionBlocks.map((b) => [b.lineIndices[0], b]));
  const front: FlowLine[] = [];
  const drafts: Draft[] = [];
  const referenceLines: TextLine[] = [];
  const linkTexts: string[] = [];

  lines.forEach((line, i) => {
    if (furniture.has(i) || consumed.has(i) || title.indices.includes(i)) return;
    const heading = headingAt.get(i);
    if (heading) {
      drafts.push({ heading, flow: [], captions: [], pageEnd: line.page });
      if (heading.rest) drafts[drafts.length - 1].flow.push({ line, text: heading.rest });
      return;
    }
    const current = drafts[drafts.length - 1];
    const inReferences = !!current && /\b(references|bibliography)\b/i.test(current.heading.title);
    if (current) current.pageEnd = line.page;

    const caption = captionByStart.get(i);
    if (caption) {
      if (!inReferences) current?.captions.push(`${caption.caption.label}: ${caption.caption.text}`);
      linkTexts.push(caption.caption.text);
      return;
    }
    if (captionLines.has(i)) return;
    if (!inReferences) linkTexts.push(line.text);
    if (footnotes.has(i) || isNonProse(lines, i, stats, inReferences)) return;
    if (inReferences) referenceLines.push(line);
    else if (current) current.flow.push({ line, text: line.text });
    else front.push({ line, text: line.text });
  });

  const kinds = assignKinds(
    drafts.map(({ heading }) => ({
      title: heading.title,
      level: heading.level,
      numbered: heading.number !== undefined,
      appendixLetter: heading.appendixLetter,
    })),
  );
  const sections: Section[] = drafts.map((draft, n) => ({
    id: `s${n}`,
    title: draft.heading.title,
    ...(draft.heading.number !== undefined ? { number: draft.heading.number } : {}),
    level: draft.heading.level,
    kind: kinds[n],
    pageStart: lines[draft.heading.index].page,
    pageEnd: draft.pageEnd,
    text: [flowToText(draft.flow, stats, vocab), ...draft.captions].filter(Boolean).join("\n\n"),
  }));

  const frontText = flowToText(front, stats, vocab);
  let abstract = sections.find((s) => s.kind === "abstract")?.text;
  if (!abstract) {
    abstract = abstractFromFrontMatter(frontText);
    if (abstract && sections.length > 0) {
      sections.unshift({
        id: "s0",
        title: "Abstract",
        level: 1,
        kind: "abstract",
        pageStart: 1,
        pageEnd: 1,
        text: abstract,
      });
      sections.forEach((s, n) => (s.id = `s${n}`));
    }
  }

  const references = splitReferenceEntries(referenceLines)
    .map((entry) => joinLines(entry.map((l) => l.text), vocab))
    .filter((raw) => raw.length >= 15 && letterRatio(raw) > 0.4)
    .map(parseReference);

  const pages = buildPages(lines, furniture, pagesParsed, stats, vocab);
  const totalChars = pages.reduce((sum, p) => sum + p.charCount, 0);
  const quality = qualityOf(totalChars / Math.max(1, pagesParsed));

  return {
    pageCount,
    pagesParsed,
    pages,
    ...(title.text ? { title: title.text } : {}),
    ...(abstract ? { abstract } : {}),
    sections,
    references,
    captions: captionBlocks.map((b) => b.caption),
    links: extractLinks(linkTexts.length ? [joinLines(linkTexts, vocab)] : []),
    warnings: buildWarnings({ pageCount, pagesParsed, quality, sections, references }),
    quality,
  };
}

function buildPages(
  lines: TextLine[],
  furniture: Set<number>,
  pagesParsed: number,
  stats: DocStats,
  vocab: Set<string>,
): ParsedDocument["pages"] {
  const byPage = new Map<number, FlowLine[]>();
  lines.forEach((line, i) => {
    if (furniture.has(i)) return;
    if (!byPage.has(line.page)) byPage.set(line.page, []);
    byPage.get(line.page)?.push({ line, text: line.text });
  });
  return Array.from({ length: pagesParsed }, (_, k) => {
    const text = flowToText(byPage.get(k + 1) ?? [], stats, vocab);
    return { page: k + 1, text, charCount: text.length };
  });
}

function buildWarnings(doc: {
  pageCount: number;
  pagesParsed: number;
  quality: ParsedDocument["quality"];
  sections: Section[];
  references: Reference[];
}): string[] {
  const warnings: string[] = [];
  if (doc.pagesParsed < doc.pageCount) {
    warnings.push(`Only the first ${doc.pagesParsed} of ${doc.pageCount} pages were parsed.`);
  }
  if (doc.quality === "image-only") {
    warnings.push("Almost no text could be extracted; the PDF looks scanned or image-only.");
  } else if (doc.quality === "degraded") {
    warnings.push("Little text could be extracted per page; some content may be missing.");
  }
  if (doc.sections.length === 0 && doc.quality !== "image-only") {
    warnings.push("No section headings were detected, so the paper could not be split into sections.");
  } else if (doc.references.length === 0 && doc.sections.length > 0) {
    warnings.push("No reference list was found.");
  }
  if (doc.pageCount >= LONG_PAPER_PAGES) {
    warnings.push(`Very long document (${doc.pageCount} pages); analysis may take longer.`);
  }
  return warnings;
}

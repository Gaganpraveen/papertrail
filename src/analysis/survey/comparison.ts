/**
 * Side-by-side comparison matrix: one row per paper with the facts a reader
 * scans first (problem, approach, benchmarks, metrics, headline result,
 * number of limitations, rigor score, code availability).
 *
 * Every cell is read off the paper: benchmarks and metrics come from the
 * curated lexicons matched against the paper's own sentences (outside related
 * work, where other people's benchmarks are merely mentioned), the problem
 * and headline result are verbatim sentences picked by zone and evidence
 * density, and rigor and limitations come from the single-paper analysis.
 * Also home to the quoting helpers the survey uses to cite sentences cleanly.
 */
import type { Evidence, Sentence, SectionKind, SurveyPaperInput, SurveyResult, ZonedSentence } from "@/types";
import { CODE_HOST_RE, CODE_STATEMENT_RE, DATASET_MATCHERS, METRICS } from "@/analysis/lexicons";
import { extractMetrics } from "@/nlp/numbers";
import { splitSentences } from "@/nlp/text";

export type ComparisonRow = SurveyResult["comparison"][number];

// ────────────────────────────────── Quoting ─────────────────────────────────

/** "[12]", "[3, 7]", "[4–6]": the source paper's own citation markers. */
const NUMERIC_CITATION_RE = /\s*\[\d+(?:\s*[-–,]\s*\d+)*\]/g;
/** "(Vaswani et al., 2017)", "(B. Peng et al. 2023)", "(Liu and Chen 2023; Wu et al., 2024a, inter alia)". */
const AUTHOR_YEAR_CITATION_RE =
  /\s*\((?:see |e\.g\.,? |cf\. )?[A-Z][^()]{0,80}?,? (?:19|20)\d{2}[a-z]?(?:; ?[^;()]{0,80}?(?:19|20)\d{2}[a-z]?)*(?:,? inter alia)?\)/gu;
/** Discourse connectives and context phrases that only make sense in the sentence's original place. */
const LEADING_CONNECTIVE_RE =
  /^(?:however|moreover|furthermore|in addition|additionally|thus|therefore|hence|finally|specifically|notably|overall|to this end|in particular|concretely|importantly|consequently|instead|meanwhile|firstly|secondly|thirdly|first|second|third|lastly|then|also|further|interestingly|as a result|in summary|in conclusion|to summarize|to sum up|in short|crucially|surprisingly|similarly|in practice|for (?:example|instance)|with this in mind|(?:based on|motivated by|inspired by|building on|guided by) (?:these|this|the above|our|such) \w+(?: \w+)?|to (?:address|tackle|solve|mitigate|overcome) (?:this|these|that|those|such)(?: \w+)?|as (?:shown|seen|reported|illustrated|discussed) (?:in|above|below)(?: (?:tables?|figs?\.?|figures?|sections?|appendix) [\w.]+(?: \([^)]*\))?)?),?\s+/i;
/** Pointers into the source paper's own figures and tables, which the survey does not have. */
const FIGURE_POINTER_RE =
  /\s*\((?:see |cf\. )?(?:tables?|tab\.|figs?\.|figures?|sections?|sec\.|appendix|eqs?\.|equations?)\s*[\w.]+(?:\s*(?:and|,|&)\s*[\w.]+)*(?:\s*\([^)]*\))?\)/gi;
const INLINE_POINTER_RE =
  /,?\s*as (?:shown|seen|reported|illustrated) in (?:tables?|figs?\.?|figures?|sections?|appendix) [\w.]+(?: \([^)]*\))?,?/gi;

/**
 * Cleans a sentence for quoting in another document: removes the source's
 * citation markers (they would collide with the survey's own [n]) and
 * context-dependent connectives, and ensures terminal punctuation.
 */
export function cleanQuote(text: string): string {
  let s = text.replace(/\s+/g, " ").trim();
  s = s.replace(NUMERIC_CITATION_RE, "").replace(AUTHOR_YEAR_CITATION_RE, "").replace(FIGURE_POINTER_RE, "");
  s = s.replace(INLINE_POINTER_RE, ",").replace(/^,\s*/, "");
  // Connectives can stack ("However, to address this, we ..."), so strip until none is left.
  for (let previous = ""; previous !== s; ) {
    previous = s;
    s = s.replace(LEADING_CONNECTIVE_RE, "");
  }
  s = s.replace(/\s+([.,;:!?])/g, "$1").replace(/\(\s*\)/g, "").trim();
  if (s.length > 0) s = s.charAt(0).toUpperCase() + s.slice(1);
  if (s && !/[.!?]["'”’)]?$/.test(s)) s += ".";
  return s;
}

/** Cuts at a word boundary and marks the cut, so a shortened quote never looks complete. */
export function truncate(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const cut = text.slice(0, maxChars);
  const lastSpace = cut.lastIndexOf(" ");
  return `${(lastSpace > maxChars * 0.6 ? cut.slice(0, lastSpace) : cut).replace(/[,;:\s]+$/, "")}…`;
}

export function toEvidence(paperId: string, sentence: Sentence): Evidence {
  return {
    paperId,
    page: sentence.page,
    section: sentence.sectionTitle,
    quote: sentence.text,
    sentenceId: sentence.id,
  };
}

/**
 * Sentences opening with a pointer to what came before ("This allows ...",
 * "Here we ...") cannot stand alone in another document. "This paper/work"
 * is not matched: the survey rewrites it instead.
 */
const ANAPHORIC_START_RE = /^(?:this|these|that|those|it|its|they|their|such|here|thus|then)\b(?! (?:paper|work|study|article|report)\b)/i;

export function startsWithAnaphor(text: string): boolean {
  return ANAPHORIC_START_RE.test(text.trim());
}

/**
 * Sections whose sentences are not this paper's own claims: related work and
 * background describe other papers, appendices hold checklists and details.
 */
const NOT_OWN_CLAIMS = new Set<SectionKind>(["related", "background", "appendix", "references", "acknowledgments"]);
/** Statements required by venues (ethics, broader impact, checklists) rather than written about the research. */
const BOILERPLATE_TITLE_RE = /ethic|broader impact|societal|impact statement|reproducibility statement|checklist|author contributions/i;

export function isOwnClaim(s: Sentence): boolean {
  return !NOT_OWN_CLAIMS.has(s.sectionKind) && !BOILERPLATE_TITLE_RE.test(s.sectionTitle);
}

/** Reproducibility checklists ("[Yes] (b) Did you ...") are boilerplate, not findings. */
const CHECKLIST_RE = /\[(?:yes|no|n\/a)\]|^\(?[a-z]\)\s|did you\b/i;

/** Sentences worth quoting: long enough to say something, short enough to read, not an equation or a question. */
export function isQuotable(text: string): boolean {
  if (text.length < 40 || text.length > 420) return false;
  if (text.includes("?") || CHECKLIST_RE.test(text)) return false;
  return mathSymbols(text) <= 1;
}

/** Operators and Greek letters: a sentence full of them is notation, not prose. */
export function mathSymbols(text: string): number {
  return (text.match(/[=∈∑∏≤≥←→∼≈]|\p{Script=Greek}/gu) ?? []).length;
}

/**
 * How much a sentence leans on context the survey reader does not have:
 * pointers to the source's figures ("the prompt on the left", "Table 3
 * shows"), variable names ("tK is added to XKr"), table text that leaked into
 * the sentence, and inline list items. Subtracted from selection scores.
 */
export function clutterPenalty(text: string): number {
  let penalty = 0;
  if (/\b(?:fig(?:ure)?s?\.?|tables?|appendix|eq(?:uation)?s?\.?|sections?|sec\.)\s*\d|\bon the (?:left|right)\b|\b(?:above|below)\b|\((?:left|right|top|bottom|middle)\)/i.test(text)) penalty += 0.3;
  const tokens = text.split(/\s+/);
  if (tokens.filter(looksLikeVariable).length >= 2) penalty += 0.3;
  if (longestRun(tokens, (w) => /^[(\p{Lu}\p{N}]/u.test(w)) >= TABLE_RUN) penalty += 0.3;
  if (/(?:^|\s)[A-Da-d1-4]\)\s/.test(text)) penalty += 0.3; // inline list items: "A) Intuitive and ..."
  return penalty;
}

/**
 * Six or more capitalised or numeric tokens in a row ("SELF-RAG (50k) 45.5
 * Training No Retriever R 43.6") is table text that leaked into a sentence;
 * prose and even benchmark lists are broken up by lower-case words.
 */
const TABLE_RUN = 6;

function longestRun(tokens: string[], test: (token: string) => boolean): number {
  let best = 0;
  let current = 0;
  for (const token of tokens) {
    current = test(token) ? current + 1 : 0;
    best = Math.max(best, current);
  }
  return best;
}

/** "tK", "XKr", "Q(XKG)", a lone "R": math symbols written as words. */
function looksLikeVariable(word: string): boolean {
  const bare = word.replace(/^[(]+|[)\],.;:]+$/g, "");
  if (/^[B-HJ-Z]$/.test(bare)) return true; // lone capitals other than the words "A" and "I"
  if (/^[A-Z]\([A-Za-z]+$/.test(bare)) return true; // "Q(XKG)" once the closing bracket is stripped
  if (/^[A-Z]{2,}s$/.test(bare)) return false; // acronym plurals: "LLMs", "GPUs"
  // Short mixed-case tokens such as "tK" or "XKr"; all-caps acronyms ("KV", "LLM") are fine.
  return bare.length <= 4 && /[a-z]/.test(bare) && /[A-Z]/.test(bare.slice(1));
}

// ─────────────────────────────── Signals ────────────────────────────────────

/** Sections where a benchmark name is someone else's, not this paper's evaluation. */
const NON_EVALUATION_SECTIONS = new Set<SectionKind>(["related", "background", "references", "acknowledgments"]);

const MAX_DATASETS = 6;
const MAX_METRICS = 5;

function evaluationText(input: SurveyPaperInput): string[] {
  const sentences = input.sentences.filter((s) => !NON_EVALUATION_SECTIONS.has(s.sectionKind)).map((s) => s.text);
  if (sentences.length > 0) return sentences;
  return input.meta.abstract ? splitSentences(input.meta.abstract) : [];
}

function rankByMentions(counts: Map<string, number>, limit: number): string[] {
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([name]) => name);
}

/** Evaluation benchmarks named in the paper, most mentioned first (training corpora excluded). */
export function detectDatasets(texts: string[]): string[] {
  const counts = new Map<string, number>();
  for (const { entry, re } of DATASET_MATCHERS) {
    if (entry.corpus) continue;
    for (const text of texts) if (re.test(text)) counts.set(entry.name, (counts.get(entry.name) ?? 0) + 1);
  }
  return rankByMentions(counts, MAX_DATASETS);
}

export function detectMetrics(texts: string[]): string[] {
  const counts = new Map<string, number>();
  for (const { name, re } of METRICS) {
    for (const text of texts) if (re.test(text)) counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return rankByMentions(counts, MAX_METRICS);
}

/** A code-hosting link or an explicit release statement anywhere in the text, or a critique strength saying so. */
export function hasCode(input: SurveyPaperInput): boolean {
  const texts = [...input.sentences.map((s) => s.text), ...input.chunks.map((c) => c.text)];
  if (texts.some((t) => CODE_HOST_RE.test(t) || CODE_STATEMENT_RE.test(t))) return true;
  return input.analysis.critique.findings.some((f) => f.severity === "strength" && /\bcode\b/i.test(f.title));
}

// ─────────────────────────── Representative sentences ───────────────────────

/** Abstract, introduction and conclusion carry the paper's own framing; they quote best. */
const FRAMING_SECTIONS = new Set<SectionKind>(["abstract", "introduction", "conclusion"]);

export function bestOfZone(
  sentences: ZonedSentence[],
  zone: ZonedSentence["zone"],
  bonus: (s: ZonedSentence) => number = () => 0,
): ZonedSentence | undefined {
  let best: ZonedSentence | undefined;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const s of sentences) {
    if (s.zone !== zone || !isQuotable(s.text)) continue;
    const score = s.confidence + (FRAMING_SECTIONS.has(s.sectionKind) ? 0.15 : 0) + bonus(s) - clutterPenalty(s.text);
    if (score > bestScore) {
      bestScore = score;
      best = s;
    }
  }
  return best;
}

/** First sentence of a possibly multi-sentence briefing text. */
export function firstSentence(text: string | undefined): string | undefined {
  if (!text) return undefined;
  return splitSentences(text)[0] ?? text;
}

/** Quotable and free of pointers into the source's figures or notation. */
export function readsStandalone(text: string): boolean {
  return isQuotable(text) && clutterPenalty(text) === 0;
}

/** The first briefing item that reads well out of context (the analysis has already ranked them). */
function firstReadable(items: { text: string }[]): string | undefined {
  return items.map((item) => firstSentence(item.text)).find((t): t is string => Boolean(t && readsStandalone(t)));
}

/** The sentence that states the problem the paper tackles. */
export function pickProblem(input: SurveyPaperInput): string {
  const briefed = firstReadable(input.analysis.briefing.problem);
  if (briefed) return cleanQuote(briefed);
  const zoned = bestOfZone(input.sentences.filter(isOwnClaim), "problem");
  if (zoned) return cleanQuote(zoned.text);
  const fallback = firstSentence(input.analysis.briefing.tldr.text) ?? firstSentence(input.meta.abstract);
  return fallback ? cleanQuote(fallback) : "";
}

/** The headline result: the analysis' top readable result, else the most number-dense result sentence. */
export function pickKeyResult(input: SurveyPaperInput): string {
  const briefed = firstReadable(input.analysis.briefing.results);
  if (briefed) return cleanQuote(briefed);
  // Up to +0.3 for sentences that carry measured numbers: "results" without numbers are usually claims.
  const zoned = bestOfZone(input.sentences.filter(isOwnClaim), "result", (s) => 0.1 * Math.min(3, extractMetrics(s.text).length));
  return zoned ? cleanQuote(zoned.text) : "";
}

// ──────────────────────────────── Rows ──────────────────────────────────────

const APPROACH_TERMS = 4;
const CELL_CHARS = 280;

export function paperYear(input: SurveyPaperInput): number | undefined {
  if (input.meta.year) return input.meta.year;
  const year = Number.parseInt(input.meta.published?.slice(0, 4) ?? "", 10);
  return Number.isFinite(year) ? year : undefined;
}

export function compareRow(input: SurveyPaperInput): ComparisonRow {
  const texts = evaluationText(input);
  const { briefing, rigor } = input.analysis;
  return {
    paperId: input.meta.id,
    year: paperYear(input),
    problem: truncate(pickProblem(input), CELL_CHARS),
    approach: briefing.keyphrases.slice(0, APPROACH_TERMS).map((k) => k.phrase),
    datasets: detectDatasets(texts),
    metrics: detectMetrics(texts),
    keyResult: truncate(pickKeyResult(input), CELL_CHARS),
    // Author-stated plus detected limitations: both are concrete, evidenced items in the briefing.
    limitations: briefing.limitations.authorStated.length + briefing.limitations.detected.length,
    rigor: Math.round(rigor.overall),
    code: hasCode(input),
  };
}

export function buildComparison(papers: SurveyPaperInput[]): ComparisonRow[] {
  return papers.map(compareRow);
}


/**
 * Quantitative claim extraction ("93.2%", "2.6× less memory", "+4.1 BLEU",
 * "F1 of 88.5").
 *
 * A number only counts as a metric when something marks it as a measurement:
 * a unit (%, pp, points, ×, ms, s, GB, MB, B, M, K) or a metric name right
 * next to it (BLEU, accuracy, pass@1, "on GSM8K"). Bare numbers are never
 * reported, which already rules out most noise. Before scanning, spans that
 * look numeric but are not measurements are blanked out (same length, so
 * offsets stay valid): bracket and author-year citations, references to
 * tables, figures, sections, equations and appendices, URLs, arXiv ids,
 * DOIs and dates. Numbers glued to identifiers ("GPT-4", "pass@1", "8x7B",
 * "Llama-2-7B") and configuration values ("4-bit", "5-shot") are skipped,
 * and so are year-like integers that carry no unit.
 *
 * Direction comes from an explicit sign, a comparative right after the
 * value ("less memory"), or the nearest change verb in the same clause
 * ("reduces memory by 2.6×", "outperforms ... by 4.1 points").
 */
import { normalizeText } from "./text";

export interface Metric {
  raw: string;
  value: number;
  unit?: "%" | "x" | "pp" | "points" | "ms" | "s" | "GB" | "MB" | "B" | "M" | "K";
  metric?: string;
  direction?: "up" | "down";
}

type Unit = NonNullable<Metric["unit"]>;

// ───────────────────────────────── Masking ─────────────────────────────────

const REFERENCE_LABEL =
  "(?:tables?|tab\\.|figures?|figs?\\.|sections?|secs?\\.|sect\\.|eqs?\\.|eqn\\.|equations?|appendix|appendices|app\\.|" +
  "algorithms?|alg\\.|theorems?|thm\\.|lemmas?|corollary|definition|proposition|prop\\.|steps?|lines?|chapters?|" +
  "pages?|pp?\\.|examples?|assumptions?|remarks?|rows?|columns?|layers?|stages?|phases?|footnote|listing|box|panel)";
/** "3", "3.2", "2a", "(3)", "A", "A.1", "B2". */
const LABEL_VALUE = "\\(?(?:[A-Z](?:\\.?\\d+)*|\\d+(?:\\.\\d+)*)[a-z]?\\)?";
const REFERENCE_RE = new RegExp(
  `\\b${REFERENCE_LABEL}\\s*~?${LABEL_VALUE}(?:\\s*(?:,|and|&|–|-|to)\\s*${LABEL_VALUE})*`,
  "gi",
);
const MONTHS =
  "(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)";

const MASKS: RegExp[] = [
  /https?:\/\/\S+|www\.\S+/g, // URLs
  /\b10\.\d{4,9}\/\S+/g, // DOIs
  /\barXiv:\s*\S+|\b\d{4}\.\d{4,5}(?:v\d+)?\b/gi, // arXiv identifiers
  /\[[\d\s,;–-]+\]/g, // numeric citations [12], [3, 4], [12–15]
  /\bet al\.?,?\s*\(?(?:19|20)\d{2}[a-z]?\)?/g, // "et al. (2023)", "et al., 2023"
  /§\s*\d+(?:\.\d+)*/g, // §3.2
  REFERENCE_RE, // Table 2, Fig. 3a, Sec. 4.2, Eq. (3), Appendix B.1
  /\(\d{1,3}\)/g, // bare equation numbers "(3)"
  /\b(?:19|20)\d{2}-\d{2}(?:-\d{2})?\b/g, // ISO dates
  new RegExp(`\\b${MONTHS}\\.?\\s+\\d{1,2}(?:,\\s*(?:19|20)\\d{2})?\\b`, "g"), // "Oct. 12, 2023"
];

/** Author-year citations: "(Brown et al., 2020; Taylor, 2022)". */
const PAREN_RE = /\([^()]*\)/g;
const AUTHOR_YEAR_RE = /(?:et al\.|\p{Lu}[\p{L}'’-]+),?\s+(?:19|20)\d{2}[a-z]?\b/u;

function blank(match: string): string {
  return " ".repeat(match.length);
}

function maskNonMeasurements(text: string): string {
  let masked = text.replace(PAREN_RE, (m) => (AUTHOR_YEAR_RE.test(m) ? blank(m) : m));
  for (const re of MASKS) masked = masked.replace(re, blank);
  return masked;
}

// ───────────────────────────────── Lexicons ────────────────────────────────

/**
 * "score" names evaluation metrics, which may anchor a unit-less number
 * ("28.4 BLEU", "F1 of 88.5"); "quantity" names only label numbers that
 * already carry a unit ("2.6× less memory"), because "16 parameters" alone
 * is a setting, not a result.
 */
interface MetricName {
  re: RegExp;
  kind: "score" | "quantity";
  label: (m: RegExpExecArray) => string;
}

const asWritten = (m: RegExpExecArray): string => m[0];
const lower = (m: RegExpExecArray): string => m[0].toLowerCase();
const fixed =
  (label: string) =>
  (): string =>
    label;

const METRIC_NAMES: MetricName[] = [
  { re: /^top-?[15]\s+acc(?:uracy)?\b/i, kind: "score", label: (m) => `top-${/[15]/.exec(m[0])?.[0]} accuracy` },
  { re: /^exact[- ]match\b/i, kind: "score", label: fixed("exact match") },
  { re: /^(?:win|success|error|pass|hit|attack success|acceptance) rates?\b/i, kind: "score", label: lower },
  { re: /^ROUGE(?:-(?:L(?:sum)?|1|2))?\b/i, kind: "score", label: asWritten },
  { re: /^(?:Sacre)?BLEU\b/i, kind: "score", label: fixed("BLEU") },
  { re: /^pass@\d+\b/i, kind: "score", label: lower },
  { re: /^(?:NDCG|recall|precision|hits?)@\d+\b/i, kind: "score", label: asWritten },
  { re: /^acc(?:uracy|uracies)?\b/i, kind: "score", label: fixed("accuracy") },
  { re: /^F1(?:-score)?\b/i, kind: "score", label: fixed("F1") },
  { re: /^(?:perplexity|perplexities|PPL)\b/i, kind: "score", label: fixed("perplexity") },
  {
    re: /^(?:METEOR|CIDEr|SPICE|chrF\+*|BERTScore|BLEURT|COMET|EM|MRR|NDCG|mAP|AP|AUC|AUROC|AUPRC|WER|CER|FID|Elo|MSE|MAE|RMSE)\b/,
    kind: "score",
    label: asWritten,
  },
  { re: /^(?:precision|recall|reward|win rate)\b/i, kind: "score", label: lower },
  { re: /^compression ratios?\b/i, kind: "quantity", label: fixed("compression ratio") },
  { re: /^(?:batch|model|cache|window|vocabulary|context) sizes?\b|^(?:context|sequence) lengths?\b/i, kind: "quantity", label: lower },
  { re: /^GPU[- ]hours?\b/i, kind: "quantity", label: fixed("GPU hours") },
  {
    re: /^(?:latency|throughput|memory|speedup|speed-up|runtime|wall-clock time|cost|energy|FLOPs|parameters|params|tokens|bandwidth|storage|footprint|correlation|loss|score|inference|training|time|size)\b/i,
    kind: "quantity",
    label: (m) => (m[0] === "FLOPs" ? m[0] : m[0].toLowerCase()),
  },
];

function lookupMetric(text: string, kinds: MetricName["kind"][]): { label: string; length: number } | null {
  for (const name of METRIC_NAMES) {
    if (!kinds.includes(name.kind)) continue;
    const m = name.re.exec(text);
    if (m) return { label: name.label(m), length: m[0].length };
  }
  return null;
}

const ANY_KIND: MetricName["kind"][] = ["score", "quantity"];
const SCORES_ONLY: MetricName["kind"][] = ["score"];

/** Benchmark-like names after "on": two or more capitals or digits ("MMLU", "GSM8K", "HumanEval"). */
const BENCHMARK_RE = /^\s+on\s+(?:the\s+)?([\p{Lu}\d][\p{L}\d-]*)/u;
const isBenchmarkName = (name: string): boolean => /[\p{Lu}\d].*[\p{Lu}\d]/u.test(name);

const UP_CUE =
  /^(?:improv|increas|outperform|boost|rais|rose|rise|grow|grew|gain|exceed|surpass|beat|accelerat|enhanc|speed|higher|better|faster|larger|greater|more|longer|stronger)/i;
const DOWN_CUE =
  /^(?:reduc|decreas|cut|lower|drop|declin|degrad|shrink|sav|less|fewer|smaller|slower|worse|shorter|cheaper|compress|lose|lost|fall|fell|diminish|slash)/i;
/** Bounds, not changes: "more than 90%", "up to 4×". */
const BOUND_RE = /\b(?:more|less|fewer|greater|higher|lower|larger|smaller) than\b|\bup to\b/gi;
const CLAUSE_BREAK_RE = /[;:,]|\s(?:and|while|whereas|but|which|compared|versus|vs\.)\s/giu;

// ─────────────────────────────── Unit parsing ──────────────────────────────

interface UnitMatch {
  unit: Unit;
  /** Characters consumed after the number. */
  length: number;
}

const COMPARATIVE_RE =
  /^\s+(?:faster|slower|less|more|fewer|smaller|larger|higher|lower|speedup|speed-up|reduction|improvement|increase|decrease|compression|cheaper|longer|shorter|throughput|memory)\b/i;

function matchUnit(after: string, hasDecimal: boolean, signed: boolean, before: string): UnitMatch | null {
  let m: RegExpExecArray | null;
  if ((m = /^\s?%/.exec(after)) || (m = /^\s+percent\b/i.exec(after))) return { unit: "%", length: m[0].length };
  if ((m = /^\s*(?:pp\b|p\.p\.|percentage[- ]points?\b)/i.exec(after))) return { unit: "pp", length: m[0].length };
  if ((m = /^\s*(?:absolute\s+)?(?:points?|pts)\b/i.exec(after))) {
    // "1000 points" in a point cloud is a count; a gain is fractional, signed or "by N points".
    if (hasDecimal || signed || /\bby\s*$/i.test(before)) return { unit: "points", length: m[0].length };
    return null;
  }
  if ((m = /^\s?×(?![\p{L}\d])/u.exec(after)) || (m = /^x(?![\p{L}\d])/u.exec(after))) {
    if (/^\s*\d/u.test(after.slice(m[0].length))) return null; // "3 × 3" is a shape
    return { unit: "x", length: m[0].length };
  }
  if ((m = /^\s(?:x|times)(?![\p{L}\d])/iu.exec(after)) && COMPARATIVE_RE.test(after.slice(m[0].length))) {
    return { unit: "x", length: m[0].length };
  }
  if ((m = /^\s?(?:ms|msec|milliseconds?)\b/.exec(after))) return { unit: "ms", length: m[0].length };
  if ((m = /^\s?(?:s|secs?|seconds?)\b(?![/-])/.exec(after))) return { unit: "s", length: m[0].length };
  if ((m = /^\s?(?:GB|GiB)\b/.exec(after))) return { unit: "GB", length: m[0].length };
  if ((m = /^\s?(?:MB|MiB)\b/.exec(after))) return { unit: "MB", length: m[0].length };
  if ((m = /^(?:B\b|\s?(?:billion|bn)\b|\sB\b(?=\s+(?:param|token)))/.exec(after))) return { unit: "B", length: m[0].length };
  if ((m = /^(?:M\b|\s?million\b|\sM\b(?=\s+(?:param|token)))/.exec(after))) return { unit: "M", length: m[0].length };
  if ((m = /^(?:[kK]\b|\s?thousand\b)/.exec(after))) return { unit: "K", length: m[0].length };
  return null;
}

/** An evaluation metric directly after the number: "28.4 BLEU", "67.0 pass@1". */
function metricAfter(after: string): { metric: string; length: number } | null {
  const lead = /^\s+/.exec(after)?.[0];
  if (!lead) return null;
  const name = lookupMetric(after.slice(lead.length), SCORES_ONLY);
  return name ? { metric: name.label, length: lead.length + name.length } : null;
}

const CHANGE_VERB = "(?:improv|increas|decreas|drop|rise|rises|rose|fall|falls|fell|declin|grow|grew|jump|climb|go|goes|went)\\w*";
const CONNECTOR_TAIL_RE = new RegExp(
  `\\s*(?:${CHANGE_VERB}\\s+)?(?:of|is|was|were|=|:|to|from|by|reaches|reached|achieves|achieved)?\\s*` +
    "(?:about|approximately|around|roughly|nearly|over|only|just)?\\s*$",
  "i",
);

/** An evaluation metric right before the number: "accuracy of", "F1:", "BLEU score of", "perplexity improves from". */
function metricBefore(before: string): string | undefined {
  const tail = before.replace(CONNECTOR_TAIL_RE, "").split(/\s+/).slice(-3);
  for (let i = 0; i < tail.length; i++) {
    const phrase = tail.slice(i).join(" ");
    for (const candidate of [phrase, phrase.replace(/\s+(?:scores?|values?)$/i, "")]) {
      const found = lookupMetric(candidate, SCORES_ONLY);
      if (found && found.length === candidate.length) return found.label;
    }
  }
  return undefined;
}

// ───────────────────────────── Context inference ───────────────────────────

function clauseBefore(text: string): string {
  let cut = 0;
  for (const m of text.matchAll(CLAUSE_BREAK_RE)) cut = (m.index ?? 0) + m[0].length;
  return text.slice(cut);
}

function clauseAfter(text: string): string {
  const m = /[;:,]|\s(?:and|while|whereas|but|which)\s/iu.exec(text);
  return m ? text.slice(0, m.index) : text;
}

function wordsOf(text: string): string[] {
  return text.match(/\p{L}[\p{L}\d@-]*/gu) ?? [];
}

function inferDirection(sign: string, before: string, after: string): Metric["direction"] {
  if (sign === "+") return "up";
  if (sign === "-" || sign === "−") return "down";
  const following = wordsOf(clauseAfter(after).replace(BOUND_RE, " ")).slice(0, 2);
  for (const word of following) {
    if (DOWN_CUE.test(word)) return "down";
    if (UP_CUE.test(word)) return "up";
  }
  const preceding = wordsOf(clauseBefore(before).replace(BOUND_RE, " ")).slice(-8).reverse();
  for (const word of preceding) {
    if (DOWN_CUE.test(word)) return "down";
    if (UP_CUE.test(word)) return "up";
  }
  return undefined;
}

/** Label for a number with a unit: the first metric word after it, a benchmark, or one in the clause before. */
function inferMetric(before: string, after: string): string | undefined {
  const clause = clauseAfter(after);
  const following = wordsOf(clause).slice(0, 3);
  for (let i = 0; i < following.length; i++) {
    const found = lookupMetric(following.slice(i).join(" "), ANY_KIND);
    if (found) return found.label;
  }
  const bench = BENCHMARK_RE.exec(clause);
  if (bench && isBenchmarkName(bench[1])) return bench[1];
  const preceding = wordsOf(clauseBefore(before)).slice(-6);
  for (let i = preceding.length - 1; i >= 0; i--) {
    const found = lookupMetric(preceding.slice(i).join(" "), ANY_KIND);
    if (found) return found.label;
  }
  return undefined;
}

// ───────────────────────────────── Scanning ────────────────────────────────

const NUMBER_RE = /([+\-−±]?)(\d{1,3}(?:,\d{3})+(?!\d)|\d+)(\.\d+)?/gu;
const IDENTIFIER_CHAR_RE = /[\p{L}\p{N}_@/.#^]/u;
/** A free-standing number right before a dash: the dash opens a range ("5-10%"), not "Llama-2-7B". */
const RANGE_START_RE = /(?:^|[\s(])\d+(?:\.\d+)?$/u;
/** "Mistral 7B", "Llama 2 13B": a size that is part of a model's name. */
const MODEL_NAME_BEFORE_RE = /(?:\b\p{Lu}[\p{L}\d-]*|\b\p{Lu}[\p{L}\d-]*\s+\d+(?:\.\d+)?)\s+$/u;
/** "2B − 1": part of a formula. */
const ARITHMETIC_AFTER_RE = /^\s*[−\-+*/=^]\s*\d/u;
/** "10-20%", "3 to 5×": the first number takes its unit from the second. */
const RANGE_RE = /^\s*(?:-|–|—|to)\s*(?=\d)/u;
/** "from 5.68 to 5.12", "72.1 vs. 75.3": the second value inherits the first one's metric. */
const CONTINUATION_RE = /^\s*(?:to|→|->|vs\.?|versus|and)\s*$/u;
/** Model and data sizes ("7B", "32K") carry no change direction of their own. */
const SIZE_UNITS = new Set<Unit>(["B", "M", "K"]);

interface Reading {
  unit?: Unit;
  metric?: string;
  /** Characters after the number that belong to the measurement ("%", " BLEU"). */
  consumed: number;
}

function isYearLike(integer: string, decimals: string | undefined): boolean {
  if (decimals) return false;
  const n = Number(integer);
  return n >= 1900 && n <= 2099;
}

/**
 * What marks a number as a measurement, or null when nothing does.
 * `inherited` is the measurement this number continues ("from 5.68 to 5.12"):
 * it outranks a benchmark name, which says where the value was measured, not what.
 */
function readMeasurement(
  after: string,
  before: string,
  integer: string,
  hasDecimal: boolean,
  signed: boolean,
  inherited: Metric | null,
): Reading | null {
  const unit = matchUnit(after, hasDecimal, signed, before);
  if (unit) return { unit: unit.unit, consumed: unit.length };
  const continued: Reading | null = inherited ? { unit: inherited.unit, metric: inherited.metric, consumed: 0 } : null;
  // Without a unit, small integers are almost always counts ("3 runs", "2 accuracy metrics").
  if (!hasDecimal && !signed && Number(integer) < 10) return continued;
  const named = metricAfter(after);
  if (named) return { metric: named.metric, consumed: named.length };
  const anchored = metricBefore(before);
  if (anchored) return { metric: anchored, consumed: 0 };
  if (continued) return continued;
  const bench = BENCHMARK_RE.exec(after);
  if (bench && hasDecimal && isBenchmarkName(bench[1])) return { metric: bench[1], consumed: 0 };
  return null;
}

/** Extracts measured quantities from one sentence, in reading order. */
export function extractMetrics(sentence: string): Metric[] {
  const text = normalizeText(sentence);
  const masked = maskNonMeasurements(text);
  const metrics: Metric[] = [];
  let consumedUntil = 0;
  let previous: { metric: Metric; end: number } | null = null;
  let range: { start: number; resumeAt: number } | null = null;

  for (const match of masked.matchAll(NUMBER_RE)) {
    const start = match.index ?? 0;
    if (start < consumedUntil) continue;
    const [whole, signChar, integer, decimals] = match;
    let sign = signChar;
    const charBefore = masked.charAt(start - 1);
    if (sign === "±") continue; // a standalone uncertainty is not a result
    if ((sign === "-" || sign === "−") && RANGE_START_RE.test(masked.slice(0, start))) sign = ""; // "5-10%"
    else if (charBefore && IDENTIFIER_CHAR_RE.test(charBefore)) continue; // "GPT-4", "pass@1", "Llama-2-7B"

    const numberStart = start + signChar.length;
    const numberEnd = start + whole.length;
    const after = masked.slice(numberEnd);
    if (/^-\p{L}/u.test(after)) continue; // "4-bit", "5-shot", "3-layer": a configuration
    const before = masked.slice(Math.max(0, start - 80), start);
    const hasDecimal = decimals !== undefined;

    // Absorb an uncertainty term: "85.3 ± 0.4".
    const spreadLength = /^\s*(?:±|\+\/?-)\s*\d+(?:\.\d+)?/u.exec(after)?.[0].length ?? 0;
    const inherited = previous && CONTINUATION_RE.test(masked.slice(previous.end, start)) ? previous.metric : null;
    const reading = readMeasurement(after.slice(spreadLength), before, integer, hasDecimal, sign !== "", inherited);
    if (!reading) {
      const connector = RANGE_RE.exec(after);
      range = connector ? { start: numberStart, resumeAt: numberEnd + connector[0].length } : null;
      continue;
    }
    if (isYearLike(integer, decimals) && (!reading.unit || reading.unit === "s" || reading.unit === "x")) continue;
    const afterReading = after.slice(spreadLength + reading.consumed);
    if (ARITHMETIC_AFTER_RE.test(afterReading)) continue;
    if ((reading.unit === "B" || reading.unit === "M") && !sign && MODEL_NAME_BEFORE_RE.test(before)) continue;

    const rawStart = sign ? start : range?.resumeAt === numberStart ? range.start : numberStart;
    const rawEnd: number = numberEnd + spreadLength + reading.consumed;
    const rest = masked.slice(rawEnd);
    const magnitude = Number(`${integer.replace(/,/g, "")}${decimals ?? ""}`);
    const direction = inferDirection(sign, reading.unit && SIZE_UNITS.has(reading.unit) ? "" : before, rest);
    const label = reading.metric ?? inferMetric(before, rest);
    const metric: Metric = {
      raw: text.slice(rawStart, rawEnd).trim(),
      value: sign === "-" || sign === "−" ? -magnitude : magnitude,
      ...(reading.unit ? { unit: reading.unit } : {}),
      ...(label ? { metric: label } : {}),
      ...(direction ? { direction } : {}),
    };
    metrics.push(metric);
    previous = { metric, end: rawEnd };
    consumedUntil = rawEnd;
    range = null;
  }
  return metrics;
}

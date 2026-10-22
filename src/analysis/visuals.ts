/**
 * Data behind the visual analysis tab.
 *
 * - Rhetorical map: every sentence placed at its relative position in the
 *   paper and coloured by argumentative zone, showing how the argument moves
 *   from problem to method to results.
 * - Keyphrase network: two keyphrases are linked when they occur in the same
 *   sentence more often than chance. The link weight is normalised pointwise
 *   mutual information,
 *     NPMI(a, b) = log(p(a,b) / (p(a) p(b))) / -log p(a,b),
 *   which is 1 for phrases that always co-occur, 0 for independent ones and
 *   negative for phrases that avoid each other. Unlike raw counts it does not
 *   simply connect the most frequent terms to everything.
 * - Section composition, zone distribution, citation ages and readability.
 */
import type { SectionKind, VisualData, Zone } from "@/types";
import { readability } from "@/nlp/readability";
import { tokenize } from "@/nlp/text";
import type { AnalyzeInput } from "./signals";

const ZONES: Zone[] = ["background", "problem", "contribution", "method", "result", "limitation", "future", "other"];
const MAX_NODES = 18;
const MAX_LINKS = 40;
/** Typical reading speed for dense technical prose, slower than the ~240 wpm of general text. */
const TECHNICAL_WPM = 200;

// ───────────────────────────── Keyphrase network ────────────────────────────

/** True when `needle` occurs as a contiguous run inside `haystack`. */
function containsRun(haystack: string[], needle: string[]): boolean {
  if (needle.length === 0 || needle.length > haystack.length) return false;
  outer: for (let i = 0; i + needle.length <= haystack.length; i++) {
    for (let j = 0; j < needle.length; j++) if (haystack[i + j] !== needle[j]) continue outer;
    return true;
  }
  return false;
}

export function keyphraseGraph(
  sentenceTexts: string[],
  keyphrases: { phrase: string; score: number }[],
): VisualData["keyphraseGraph"] {
  const top = keyphrases.slice(0, MAX_NODES);
  const phraseTokens = top.map((k) => tokenize(k.phrase));
  const sentenceTokens = sentenceTexts.map(tokenize);
  // occurrences[i] = indices of sentences containing phrase i (stemmed, so plurals match).
  const occurrences = phraseTokens.map((p) =>
    sentenceTokens.reduce<number[]>((acc, toks, s) => (containsRun(toks, p) ? [...acc, s] : acc), []),
  );
  const n = Math.max(1, sentenceTexts.length);
  // In short texts (an abstract) a single shared sentence is already informative.
  const minCooccurrence = sentenceTexts.length > 60 ? 2 : 1;

  const links: VisualData["keyphraseGraph"]["links"] = [];
  for (let a = 0; a < top.length; a++) {
    const inA = new Set(occurrences[a]);
    for (let b = a + 1; b < top.length; b++) {
      const both = occurrences[b].filter((s) => inA.has(s)).length;
      if (both < minCooccurrence) continue;
      const pab = both / n;
      const pa = occurrences[a].length / n;
      const pb = occurrences[b].length / n;
      const npmi = pab >= 1 ? 1 : Math.log(pab / (pa * pb)) / -Math.log(pab);
      if (npmi > 0) links.push({ source: top[a].phrase, target: top[b].phrase, weight: round(npmi, 3) });
    }
  }
  links.sort((x, y) => y.weight - x.weight);
  return {
    nodes: top.map((k) => ({ id: k.phrase, weight: round(k.score, 3) })),
    links: links.slice(0, MAX_LINKS),
  };
}

// ──────────────────────────────── Composition ───────────────────────────────

function sectionComposition(input: AnalyzeInput): VisualData["sectionComposition"] {
  const sections = input.doc?.sections ?? [];
  if (sections.length === 0) {
    const words = input.sentences.reduce((sum, s) => sum + countWords(s.text), 0);
    return words > 0 ? [{ kind: "abstract", title: "Abstract", words }] : [];
  }
  // Sub-sections are folded into their top-level parent so the bar stays readable.
  const out: { kind: SectionKind; title: string; words: number }[] = [];
  for (const section of sections) {
    if (section.kind === "references" || section.kind === "acknowledgments") continue;
    const words = countWords(section.text);
    const parent = out[out.length - 1];
    if (section.level > 1 && parent) parent.words += words;
    else out.push({ kind: section.kind, title: section.title, words });
  }
  return out.filter((s) => s.words > 0);
}

function countWords(text: string): number {
  return text.split(/\s+/u).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
}

function citationYears(input: AnalyzeInput): VisualData["citationYears"] {
  const latest = new Date().getFullYear() + 1;
  const counts = new Map<number, number>();
  for (const r of input.doc?.references ?? []) {
    if (r.year && r.year >= 1900 && r.year <= latest) counts.set(r.year, (counts.get(r.year) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => a[0] - b[0]).map(([year, count]) => ({ year, count }));
}

function readabilityStats(input: AnalyzeInput, keyphrases: { phrase: string }[]): VisualData["readability"] {
  const text = input.sentences.map((s) => s.text).join(" ");
  const stats = readability(text);
  const tokens = tokenize(text);
  const jargon = new Set(keyphrases.flatMap((k) => tokenize(k.phrase)));
  const jargonTokens = tokens.filter((t) => jargon.has(t)).length;
  return {
    fleschReadingEase: stats.fleschReadingEase,
    avgSentenceLength: stats.avgSentenceLength,
    readingMinutes: Math.max(1, Math.round(stats.words / TECHNICAL_WPM)),
    jargonDensity: tokens.length ? round(jargonTokens / tokens.length, 3) : 0,
  };
}

// ────────────────────────────────── Entry ───────────────────────────────────

export function buildVisuals(input: AnalyzeInput, keyphrases: { phrase: string; score: number }[]): VisualData {
  const sentences = input.sentences;
  const last = Math.max(1, sentences.length - 1);
  return {
    rhetoricalMap: sentences.map((s, i) => ({
      position: sentences.length === 1 ? 0 : round(i / last, 4),
      zone: s.zone,
      page: s.page,
      sentenceId: s.id,
    })),
    sectionComposition: sectionComposition(input),
    zoneDistribution: ZONES.map((zone) => ({ zone, count: sentences.filter((s) => s.zone === zone).length })),
    keyphraseGraph: keyphraseGraph(
      sentences.map((s) => s.text),
      keyphrases,
    ),
    citationYears: citationYears(input),
    readability: readabilityStats(input, keyphrases),
  };
}

function round(x: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(x * f) / f;
}

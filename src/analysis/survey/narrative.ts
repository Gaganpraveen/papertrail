/**
 * Writes the literature review as Markdown.
 *
 * Nothing here is generated free-form. The review is assembled from three
 * kinds of material, so every claim can be traced:
 *   1. Sentences from the papers, chosen by zone and by how well they match
 *      the theme's c-TF-IDF terms, lightly cleaned (source citation markers
 *      and context connectives removed) and attributed to their authors
 *      ("We propose X" becomes "Liu et al. [1] propose X").
 *   2. Facts computed by the survey: themes, similarities, shared benchmarks
 *      and metrics, bibliographic coupling, consensus groups and gaps.
 *   3. Connective prose generated from that structure: the transition
 *      between two papers depends on whether one cites the other, on their
 *      years and on their similarity relative to the theme.
 * The reference list is built from each paper's metadata.
 */
import type { PaperMeta, SectionKind, SurveyCluster, SurveyPaperInput, ZonedSentence } from "@/types";
import { STOPWORDS, tokenize, words } from "@/nlp/text";
import type { Representation } from "./clustering";
import {
  bestOfZone,
  cleanQuote,
  clutterPenalty,
  firstSentence,
  isOwnClaim,
  mathSymbols,
  pickKeyResult,
  startsWithAnaphor,
  type ComparisonRow,
} from "./comparison";
import type { ConsensusStatement } from "./consensus";
import type { CouplingEdge, DirectCitation } from "./coupling";
import type { GapItem } from "./gaps";

export interface NarrativeInput {
  papers: SurveyPaperInput[];
  clusters: SurveyCluster[];
  similarity: number[][];
  silhouette: number;
  representation: Representation;
  comparison: ComparisonRow[];
  coupling: CouplingEdge[];
  citations: DirectCitation[];
  consensus: ConsensusStatement[];
  /** Cosine bar the consensus search used (depends on whether sentences were embedded). */
  consensusThreshold: number;
  gaps: GapItem[];
  sharedVocabulary: string[];
}

// ───────────────────────────────── Citations ────────────────────────────────

const NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];

function countWord(n: number, capitalised = false): string {
  const word = NUMBER_WORDS[n] ?? String(n);
  return capitalised ? word.charAt(0).toUpperCase() + word.slice(1) : word;
}

/** "a", "a and b", "a, b and c". */
export function joinList(items: string[]): string {
  if (items.length <= 1) return items[0] ?? "";
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

function surname(author: string): string {
  const trimmed = author.trim();
  if (trimmed.includes(",")) return trimmed.split(",")[0].trim();
  const parts = trimmed.split(/\s+/).filter((p) => !/^(?:jr\.?|sr\.?|ii|iii|iv)$/i.test(p));
  return parts[parts.length - 1] ?? trimmed;
}

interface Attribution {
  /** "Liu et al. [1]", "Liu and Chen [1]", "Liu [1]", "The authors of [1]". */
  subject: string;
  /** "Liu et al.'s [1]" for possessive rewrites of "Our ...". */
  possessive: string;
  /** Whether the subject takes a plural verb ("propose" vs "proposes"). */
  plural: boolean;
}

function attributionFor(meta: PaperMeta, cite: string): Attribution {
  const names = meta.authors.map(surname).filter(Boolean);
  if (names.length === 0) return { subject: `The authors of ${cite}`, possessive: `The authors' ${cite}`, plural: true };
  const who = names.length === 1 ? names[0] : names.length === 2 ? `${names[0]} and ${names[1]}` : `${names[0]} et al.`;
  return { subject: `${who} ${cite}`, possessive: `${who}'s ${cite}`, plural: names.length > 1 };
}

// ─────────────────────────────── Attribution ────────────────────────────────

const LEAD_IN_RE = /^(?:in this (?:paper|work|study|article|report)|here|in this section)\s*,\s*(?=we\b)/i;
const SELF_REFERENCE_RE = /^(?:this|our|the present) (paper|work|study|article|report)\b/i;

/** First words that may be lower-cased safely when a sentence is embedded after "note that". */
const COMMON_OPENERS = new Set([
  "existing", "recent", "current", "most", "many", "prior", "standard", "despite", "although", "while", "modern",
  "previous", "large", "conventional", "traditional", "popular", "state-of-the-art", "experiments", "results",
  "extensive", "empirical", "unlike", "compared", "across", "using", "given", "training", "inference",
]);

/**
 * Lower-cases the first word when it is an ordinary word: a stopword, a
 * common opener, or a word the paper itself writes in lower case elsewhere
 * ("Quantizing" -> "quantizing", but "Mamba" and "KIVI" stay as written).
 */
function lowerFirst(sentence: string, vocabulary: Set<string>): string {
  const first = sentence.split(/\s+/)[0] ?? "";
  const bare = first.toLowerCase().replace(/[^\p{L}-]/gu, "");
  if (!/^\p{Lu}\p{Ll}*$/u.test(first.replace(/[^\p{L}]/gu, ""))) return sentence; // acronyms, names with inner capitals
  const ordinary = STOPWORDS.has(bare) || COMMON_OPENERS.has(bare) || vocabulary.has(bare);
  return ordinary ? sentence.charAt(0).toLowerCase() + sentence.slice(1) : sentence;
}

/**
 * Ordinary words of a paper: written in lower case more often than capitalised
 * mid-sentence. "quantizing" qualifies; "Mamba" does not, even if a URL
 * somewhere spells it "mamba".
 */
export function lowercaseVocabulary(sentences: { text: string }[]): Set<string> {
  const lower = new Map<string, number>();
  const capitalised = new Map<string, number>();
  for (const { text } of sentences) {
    words(text).forEach((word, i) => {
      if (/^\p{Ll}[\p{Ll}-]*$/u.test(word)) lower.set(word, (lower.get(word) ?? 0) + 1);
      else if (i > 0 && /^\p{Lu}\p{Ll}+$/u.test(word)) {
        const key = word.toLowerCase();
        capitalised.set(key, (capitalised.get(key) ?? 0) + 1);
      }
    });
  }
  return new Set([...lower].filter(([word, count]) => count > (capitalised.get(word) ?? 0)).map(([word]) => word));
}

/** First-person to third-person, for text that now speaks about the authors. */
function thirdPerson(text: string): string {
  return text
    .replace(/\bwe\b/g, "they")
    .replace(/\bWe\b/g, "They")
    .replace(/\bours\b/g, "theirs")
    .replace(/\bour\b/g, "their")
    .replace(/\bOur\b/g, "Their")
    .replace(/\bourselves\b/g, "themselves")
    .replace(/\bus\b/g, "them");
}

/** Inserts the citation before the final punctuation, unless the sentence already carries it. */
function withCitation(sentence: string, cite: string): string {
  if (sentence.includes(cite)) return sentence;
  const match = /([.!?]["'”’)]?)$/.exec(sentence);
  return match ? `${sentence.slice(0, match.index)} ${cite}${match[1]}` : `${sentence} ${cite}.`;
}

const REPORTING_VERB: Partial<Record<ZonedSentence["zone"], string>> = {
  problem: "argue",
  result: "report",
  limitation: "acknowledge",
  future: "suggest",
};

/**
 * Turns a sentence from paper [n] into a sentence about paper [n]:
 * "We propose X" -> "Liu et al. [1] propose X"; "Our method reaches Y" ->
 * "Liu et al.'s [1] method reaches Y"; "This paper studies Z" -> "The paper by
 * Liu et al. [1] studies Z"; anything else -> "Liu et al. [1] note that ...".
 */
export function attribute(
  text: string,
  meta: PaperMeta,
  cite: string,
  zone: ZonedSentence["zone"] = "other",
  vocabulary: Set<string> = new Set(),
): string {
  const sentence = cleanQuote(text).replace(LEAD_IN_RE, "");
  const who = attributionFor(meta, cite);

  if (/^we\b/i.test(sentence)) {
    // "Liu [1] propose" would be ungrammatical; for a single author, quote instead.
    if (!who.plural) return `As ${who.subject} put it, “${sentence}”`;
    return thirdPerson(`${who.subject}${sentence.slice(2)}`);
  }
  if (/^our\b/i.test(sentence)) return thirdPerson(`${who.possessive}${sentence.slice(3)}`);
  const self = SELF_REFERENCE_RE.exec(sentence);
  if (self) {
    const by = who.subject.startsWith("The authors of") ? cite : who.subject;
    return thirdPerson(`The ${self[1].toLowerCase()} by ${by}${sentence.slice(self[0].length)}`);
  }
  const verb = REPORTING_VERB[zone] ?? "note";
  return thirdPerson(`${who.subject} ${who.plural ? verb : `${verb}s`} that ${lowerFirst(sentence, vocabulary)}`);
}

/** A follow-up sentence about a paper already introduced: "They also ..." with the citation appended. */
function continueAbout(text: string, cite: string): string {
  const sentence = cleanQuote(text).replace(LEAD_IN_RE, "");
  return withCitation(thirdPerson(sentence), cite);
}

// ─────────────────────────────── Sentence choice ─────────────────────────────

/** Bonus for sentences that use the theme's distinctive vocabulary (up to +0.3). */
function themeAffinity(themeStems: Set<string>): (s: ZonedSentence) => number {
  return (s) => {
    if (themeStems.size === 0) return 0;
    const hits = new Set(tokenize(s.text).filter((t) => themeStems.has(t))).size;
    return 0.1 * Math.min(3, hits);
  };
}

/** Sentences opening with "We"/"Our" attribute cleanly, so they get a small preference. */
function firstPersonBonus(s: ZonedSentence): number {
  return /^(?:(?:in this (?:paper|work)|here),\s*)?(?:we|our)\b/i.test(s.text.trim()) ? 0.1 : 0;
}

interface PaperSummary {
  lead: { text: string; zone: ZonedSentence["zone"] } | null;
  method: string | null;
}

/** Sections where authors describe their method in prose rather than in notation or settings. */
const OVERVIEW_SECTIONS = new Set<SectionKind>(["abstract", "introduction", "conclusion"]);

/** A method sentence is optional: better none than one that needs the paper's notation or figures. */
function isReadableMethod(s: ZonedSentence): boolean {
  return OVERVIEW_SECTIONS.has(s.sectionKind) && clutterPenalty(s.text) === 0 && mathSymbols(s.text) === 0;
}

function summarisePaper(paper: SurveyPaperInput, themeStems: Set<string>): PaperSummary {
  const affinity = themeAffinity(themeStems);
  const own = paper.sentences.filter((s) => isOwnClaim(s) && !startsWithAnaphor(s.text));
  const bonus = (s: ZonedSentence) => affinity(s) + firstPersonBonus(s);
  // The single-paper analysis already picked its best method bullets; overview prose beats section detail.
  const briefed = new Set(paper.analysis.briefing.method.map((m) => m.text.trim()));
  const methodBonus = (s: ZonedSentence) => bonus(s) + (briefed.has(s.text.trim()) ? 0.2 : 0);
  const contribution = bestOfZone(own, "contribution", bonus);
  const problem = contribution ? undefined : bestOfZone(own, "problem", bonus);
  const leadSentence = contribution ?? problem;
  const fallback = firstSentence(paper.analysis.briefing.tldr.text) ?? firstSentence(paper.meta.abstract);
  const lead = leadSentence
    ? { text: leadSentence.text, zone: leadSentence.zone }
    : fallback
      ? { text: fallback, zone: "other" as const }
      : null;
  const method = bestOfZone(
    own.filter((s) => s.text !== leadSentence?.text && isReadableMethod(s)),
    "method",
    methodBonus,
  );
  return { lead, method: method?.text ?? null };
}

// ──────────────────────────────── Sections ──────────────────────────────────

interface Context extends NarrativeInput {
  number: Map<string, number>;
  byId: Map<string, SurveyPaperInput>;
  index: Map<string, number>;
  vocabulary: Map<string, Set<string>>;
}

function cite(ctx: Context, paperId: string): string {
  return `[${ctx.number.get(paperId) ?? "?"}]`;
}

function cites(ctx: Context, paperIds: string[]): string {
  const sorted = [...paperIds].sort((a, b) => (ctx.number.get(a) ?? 0) - (ctx.number.get(b) ?? 0));
  return joinList(sorted.map((id) => cite(ctx, id)));
}

function similarityOf(ctx: Context, a: string, b: string): number {
  const i = ctx.index.get(a);
  const j = ctx.index.get(b);
  return i === undefined || j === undefined ? 0 : ctx.similarity[i][j];
}

function themeLabel(cluster: SurveyCluster): string {
  return cluster.terms.length > 0 ? joinList(cluster.terms.slice(0, 2)) : cluster.label;
}

function orderedClusters(ctx: Context): SurveyCluster[] {
  return [...ctx.clusters].sort((a, b) => b.paperIds.length - a.paperIds.length || a.id - b.id);
}

function yearSpan(ctx: Context): string {
  const years = ctx.comparison.map((r) => r.year).filter((y): y is number => typeof y === "number");
  if (years.length === 0) return "";
  const min = Math.min(...years);
  const max = Math.max(...years);
  return min === max ? ` published in ${min}` : ` published between ${min} and ${max}`;
}

function fieldMap(ctx: Context): string {
  const clusters = orderedClusters(ctx);
  if (clusters.length === 1) {
    return `All ${countWord(ctx.papers.length)} papers form a single theme centred on ${themeLabel(clusters[0])}.`;
  }
  const parts = clusters.map((cluster, i) => {
    const members = cites(ctx, cluster.paperIds);
    const label = themeLabel(cluster);
    if (i === 0) {
      return cluster.paperIds.length > 1
        ? `${countWord(cluster.paperIds.length, true)} papers (${members}) approach the topic from the ${label} side`
        : `${members} approaches the topic from the ${label} side`;
    }
    const connective = i === 1 ? "whereas" : "while";
    return cluster.paperIds.length > 1 ? `${connective} ${members} centre on ${label}` : `${connective} ${members} focuses on ${label}`;
  });
  return `${parts.join(", ")}.`;
}

function closestAndMostDistinct(ctx: Context): string[] {
  const ids = ctx.papers.map((p) => p.meta.id);
  if (ids.length < 2) return [];
  let best = { a: ids[0], b: ids[1], s: Number.NEGATIVE_INFINITY };
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const s = ctx.similarity[i][j];
      if (s > best.s) best = { a: ids[i], b: ids[j], s };
    }
  }
  const lines = [`The most closely related pair is ${cites(ctx, [best.a, best.b])} (cosine similarity ${best.s.toFixed(2)})`];
  if (ids.length >= 3) {
    const meanSimilarity = ids.map((_, i) => ctx.similarity[i].reduce((s, v, j) => s + (i === j ? 0 : v), 0) / (ids.length - 1));
    const outlier = meanSimilarity.indexOf(Math.min(...meanSimilarity));
    lines[0] += `, while ${cite(ctx, ids[outlier])} is the most distinct from the rest (mean similarity ${meanSimilarity[outlier].toFixed(2)})`;
  }
  lines[0] += ".";
  return lines;
}

function couplingSentences(ctx: Context): string[] {
  const withReferences = ctx.papers.filter((p) => p.references.length > 0).length;
  const out: string[] = [];
  const strongest = ctx.coupling[0];
  if (strongest) {
    out.push(
      `${cites(ctx, [strongest.source, strongest.target])} cite ${countWord(strongest.shared)} ${
        strongest.shared === 1 ? "work" : "works"
      } in common (Jaccard ${strongest.jaccard.toFixed(2)}), the strongest bibliographic coupling in the set.`,
    );
  } else if (withReferences >= 2) {
    out.push("No two papers share a reference that could be matched, so their bibliographies do not overlap.");
  } else {
    out.push("Reference lists were not available for enough papers to measure bibliographic coupling.");
  }
  for (const { citing, cited } of ctx.citations) out.push(`${cite(ctx, citing)} cites ${cite(ctx, cited)} directly.`);
  return out;
}

function overview(ctx: Context): string {
  const n = ctx.papers.length;
  const intro = [`This review covers ${countWord(n)} ${n === 1 ? "paper" : "papers"}${yearSpan(ctx)}.`];
  if (ctx.sharedVocabulary.length > 0) {
    intro.push(`Their shared vocabulary centres on ${joinList(ctx.sharedVocabulary.slice(0, 4).map((t) => `*${t}*`))}.`);
  }
  intro.push(fieldMap(ctx));
  const relations = [...closestAndMostDistinct(ctx), ...couplingSentences(ctx)];
  const how =
    `_Themes come from average-linkage clustering of ${
      ctx.representation === "embedding" ? "sentence-embedding" : "TF-IDF"
    } representations of the full text` +
    (ctx.papers.length > 2 && ctx.clusters.length > 1 ? ` (mean silhouette ${ctx.silhouette.toFixed(2)})` : "") +
    "; theme names are their most distinctive terms (c-TF-IDF)._";
  return ["## Overview", intro.join(" "), relations.join(" "), how].join("\n\n");
}

function transition(ctx: Context, paper: SurveyPaperInput, previous: SurveyPaperInput[], themeMean: number): string {
  const id = paper.meta.id;
  const citedEarlier = previous.find((p) => ctx.citations.some((c) => c.citing === id && c.cited === p.meta.id));
  if (citedEarlier) return `Building directly on ${cite(ctx, citedEarlier.meta.id)}, `;
  const last = previous[previous.length - 1];
  const year = ctx.comparison[ctx.index.get(id) ?? 0]?.year;
  const lastYear = ctx.comparison[ctx.index.get(last.meta.id) ?? 0]?.year;
  if (year !== undefined && lastYear !== undefined && year > lastYear) return "More recently, ";
  return similarityOf(ctx, id, last.meta.id) >= themeMean ? "In a closely related direction, " : "From a different angle, ";
}

/** Lower-cases the generated subject when a transition precedes it ("..., the paper by ..."). */
function afterTransition(prefix: string, sentence: string): string {
  if (!prefix) return sentence;
  return /^(?:The|As) /.test(sentence) ? `${prefix}${sentence.charAt(0).toLowerCase()}${sentence.slice(1)}` : `${prefix}${sentence}`;
}

function themeMeanSimilarity(ctx: Context, ids: string[]): number {
  let total = 0;
  let pairs = 0;
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      total += similarityOf(ctx, ids[i], ids[j]);
      pairs++;
    }
  }
  return pairs > 0 ? total / pairs : 1;
}

function themeSection(ctx: Context, cluster: SurveyCluster, position: number): string {
  const members = cluster.paperIds
    .map((id) => ctx.byId.get(id))
    .filter((p): p is SurveyPaperInput => Boolean(p))
    .sort((a, b) => {
      const ya = ctx.comparison[ctx.index.get(a.meta.id) ?? 0]?.year ?? 9999;
      const yb = ctx.comparison[ctx.index.get(b.meta.id) ?? 0]?.year ?? 9999;
      return ya - yb || (ctx.number.get(a.meta.id) ?? 0) - (ctx.number.get(b.meta.id) ?? 0);
    });
  const heading = `### ${position}. ${capitalise(cluster.label)}`;
  const vocabulary = cluster.terms.slice(0, 4).map((t) => `*${t}*`);
  const intro =
    members.length > 1
      ? `${countWord(members.length, true)} papers (${cites(ctx, cluster.paperIds)}) fall under this theme${
          vocabulary.length ? `, distinguished by ${joinList(vocabulary)}` : ""
        }.`
      : `${cite(ctx, cluster.paperIds[0])} stands on its own here${vocabulary.length ? `, distinguished by ${joinList(vocabulary)}` : ""}.`;

  const themeStems = new Set(tokenize(cluster.terms.join(" ")));
  const themeMean = themeMeanSimilarity(ctx, cluster.paperIds);
  const sentences: string[] = [];
  members.forEach((paper, i) => {
    const c = cite(ctx, paper.meta.id);
    const summary = summarisePaper(paper, themeStems);
    const prefix = i === 0 ? "" : transition(ctx, paper, members.slice(0, i), themeMean);
    const who = attributionFor(paper.meta, c);
    const lead = summary.lead
      ? attribute(summary.lead.text, paper.meta, c, summary.lead.zone, ctx.vocabulary.get(paper.meta.id))
      : `${who.subject} ${who.plural ? "present" : "presents"} *${displayTitle(paper.meta.title)}*.`;
    sentences.push(afterTransition(prefix, lead));
    if (summary.method) sentences.push(continueAbout(summary.method, c));
  });
  return [heading, intro, sentences.join(" ")].join("\n\n");
}

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function themes(ctx: Context): string {
  return ["## Themes", ...orderedClusters(ctx).map((cluster, i) => themeSection(ctx, cluster, i + 1))].join("\n\n");
}

/** Groups papers by a value they report ("GSM8K" -> [1], [3]). */
function usage(ctx: Context, pick: (row: ComparisonRow) => string[]): Map<string, string[]> {
  const map = new Map<string, string[]>();
  for (const row of ctx.comparison) for (const value of pick(row)) map.set(value, [...(map.get(value) ?? []), row.paperId]);
  return map;
}

function datasetParagraph(ctx: Context): string {
  const datasets = usage(ctx, (r) => r.datasets);
  if (datasets.size === 0) {
    return "None of the papers names a benchmark from PaperTrail's dataset lexicon, so the evaluation setups cannot be compared dataset by dataset.";
  }
  const shared = [...datasets.entries()].filter(([, ids]) => ids.length > 1).sort((a, b) => b[1].length - a[1].length);
  const out: string[] = [];
  if (shared.length > 0) {
    const [name, ids] = shared[0];
    out.push(`${name} is the most common testbed, used by ${cites(ctx, ids)}.`);
    const others = shared.slice(1, 4).map(([n, i]) => `${n} (${cites(ctx, i)})`);
    if (others.length > 0) out.push(`Other shared benchmarks are ${joinList(others)}.`);
  } else {
    out.push("No benchmark is shared by two papers, so their reported numbers are not directly comparable.");
  }
  for (const row of ctx.comparison) {
    const unique = row.datasets.filter((d) => (datasets.get(d)?.length ?? 0) === 1);
    if (unique.length > 0 && shared.length > 0) out.push(`${cite(ctx, row.paperId)} alone evaluates on ${joinList(unique.slice(0, 3))}.`);
  }
  if (shared.length === 0) {
    const perPaper = ctx.comparison
      .filter((r) => r.datasets.length > 0)
      .map((r) => `${cite(ctx, r.paperId)} on ${joinList(r.datasets.slice(0, 3))}`);
    out.push(`Each evaluates on its own data: ${joinList(perPaper)}.`);
  }
  return out.join(" ");
}

function metricParagraph(ctx: Context): string {
  const metrics = [...usage(ctx, (r) => r.metrics).entries()].sort((a, b) => b[1].length - a[1].length).slice(0, 4);
  const out: string[] = [];
  if (metrics.length > 0) out.push(`Results are reported mainly as ${joinList(metrics.map(([m, ids]) => `${m} (${cites(ctx, ids)})`))}.`);
  const withCode = ctx.comparison.filter((r) => r.code).map((r) => r.paperId);
  const withoutCode = ctx.comparison.filter((r) => !r.code).map((r) => r.paperId);
  if (withCode.length === ctx.comparison.length) out.push("All papers link to released code.");
  else if (withCode.length === 0) out.push("No paper links to released code in the analysed text.");
  else out.push(`Code is linked for ${cites(ctx, withCode)}; no code link was found in the text of ${cites(ctx, withoutCode)}.`);
  if (ctx.comparison.length >= 2) {
    const sorted = [...ctx.comparison].sort((a, b) => b.rigor - a.rigor);
    const top = sorted[0];
    const bottom = sorted[sorted.length - 1];
    if (top.rigor !== bottom.rigor) {
      out.push(
        `On PaperTrail's rigor profile, ${cite(ctx, top.paperId)} scores highest (${top.rigor}/100) and ${cite(ctx, bottom.paperId)} lowest (${bottom.rigor}/100).`,
      );
    }
  }
  return out.join(" ");
}

function approachParagraph(ctx: Context): string {
  const parts = ctx.comparison
    .filter((r) => r.approach.length > 0)
    .map((r, i) => `${cite(ctx, r.paperId)} ${i === 0 ? "centres on" : "on"} ${joinList(r.approach.slice(0, 3).map((a) => `*${a}*`))}`);
  return parts.length > 0 ? `Judged by their key phrases, ${parts.join("; ")}.` : "";
}

function cell(text: string): string {
  return text.replace(/\|/g, "\\|").replace(/\s+/g, " ").trim() || "—";
}

function comparisonTable(ctx: Context): string {
  const header = "| Ref | Year | Approach | Datasets | Metrics | Code | Rigor |\n|---|---|---|---|---|---|---|";
  const rows = ctx.comparison.map((r) =>
    [
      cite(ctx, r.paperId),
      r.year ?? "—",
      cell(r.approach.slice(0, 3).join(", ")),
      cell(r.datasets.join(", ")),
      cell(r.metrics.join(", ")),
      r.code ? "yes" : "no",
      `${r.rigor}`,
    ].join(" | "),
  );
  return [header, ...rows.map((r) => `| ${r} |`)].join("\n");
}

function methodsCompared(ctx: Context): string {
  return ["## Methods compared", approachParagraph(ctx), datasetParagraph(ctx), metricParagraph(ctx), comparisonTable(ctx)]
    .filter(Boolean)
    .join("\n\n");
}

function resultsAndConsensus(ctx: Context): string {
  const headline = ctx.papers
    .map((paper) => {
      const result = pickKeyResult(paper);
      return result ? attribute(result, paper.meta, cite(ctx, paper.meta.id), "result", ctx.vocabulary.get(paper.meta.id)) : null;
    })
    .filter((s): s is string => Boolean(s));
  const parts = ["## Results and consensus"];
  if (headline.length > 0) parts.push(headline.join(" "));
  if (ctx.consensus.length > 0) {
    const statements = ctx.consensus.map((c) => {
      const [first, ...rest] = c.support;
      const echoed = rest.length > 0 ? `, echoed by ${cites(ctx, [...new Set(rest.map((e) => e.paperId))])}` : "";
      return `- ${cites(ctx, c.paperIds)} converge on a similar point: “${c.statement}” ${cite(ctx, first.paperId)}${echoed}.`;
    });
    parts.push(`Some findings recur across papers, stated in closely matching terms:\n\n${statements.join("\n")}`);
  } else if (ctx.papers.length > 1) {
    parts.push(
      `No result is stated in closely matching terms by more than one paper (cosine ≥ ${ctx.consensusThreshold}), so the findings are complementary rather than mutually confirming.`,
    );
  }
  return parts.join("\n\n");
}

function openProblems(ctx: Context): string {
  if (ctx.gaps.length === 0) {
    return "## Open problems and gaps\n\nNone of the papers states future work or limitations that the others leave open, and no weakness is shared by most of them.";
  }
  const lines = ctx.gaps.map((gap) =>
    gap.source === "critique"
      ? `- ${gap.statement.replace(/\.$/, "")} (${cites(ctx, gap.paperIds)}; from PaperTrail's critique).`
      : `- “${gap.statement}” — raised by ${cites(ctx, gap.paperIds)}${gap.paperIds.length > 1 ? ", and still open" : ", not addressed by the other papers"}.`,
  );
  return `## Open problems and gaps\n\n${lines.join("\n")}`;
}

const MINOR_TITLE_WORDS = new Set(["a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "of", "on", "or", "the", "to", "via", "with", "your"]);

/** PDF-extracted titles are sometimes set in capitals ("EFFICIENT STREAMING ..."); show them in title case. */
export function displayTitle(title: string): string {
  const clean = title.replace(/\s+/g, " ").trim();
  if (/\p{Ll}/u.test(clean)) return clean;
  return clean
    .toLowerCase()
    .split(" ")
    .map((word, i) => (i > 0 && MINOR_TITLE_WORDS.has(word) ? word : word.replace(/(^|-)(\p{L})/gu, (_, sep: string, ch: string) => sep + ch.toUpperCase())))
    .join(" ");
}

function formatAuthors(authors: string[]): string {
  if (authors.length === 0) return "Unknown authors";
  return authors.length > 3 ? `${authors.slice(0, 3).join(", ")}, et al` : joinList(authors);
}

function referenceEntry(ctx: Context, paper: SurveyPaperInput): string {
  const { meta } = paper;
  const year = ctx.comparison[ctx.index.get(meta.id) ?? 0]?.year;
  const parts = [`${cite(ctx, meta.id)} ${formatAuthors(meta.authors)}.`, `*${displayTitle(meta.title).replace(/\.$/, "")}*.`];
  if (meta.venue) parts.push(`${meta.venue}.`);
  if (year) parts.push(`${year}.`);
  if (meta.arxivId) parts.push(`arXiv:${meta.arxivId}.`);
  if (meta.doi && !/^10\.48550\/arxiv/i.test(meta.doi)) parts.push(`https://doi.org/${meta.doi}`);
  return parts.join(" ");
}

function referenceList(ctx: Context): string {
  return ["## References", ...ctx.papers.map((p) => referenceEntry(ctx, p))].join("\n\n");
}

/** The full review. Papers are numbered [1]..[n] in input order. */
export function writeNarrative(input: NarrativeInput): string {
  const ctx: Context = {
    ...input,
    number: new Map(input.papers.map((p, i) => [p.meta.id, i + 1])),
    byId: new Map(input.papers.map((p) => [p.meta.id, p])),
    index: new Map(input.papers.map((p, i) => [p.meta.id, i])),
    vocabulary: new Map(input.papers.map((p) => [p.meta.id, lowercaseVocabulary(p.sentences)])),
  };
  const title = `# Literature review of ${countWord(input.papers.length)} ${input.papers.length === 1 ? "paper" : "papers"}`;
  return [title, overview(ctx), themes(ctx), methodsCompared(ctx), resultsAndConsensus(ctx), openProblems(ctx), referenceList(ctx)].join(
    "\n\n",
  );
}

/**
 * Extractive briefing: every line is a sentence from the paper, chosen by
 * graph centrality and rhetorical role, never generated text.
 *
 * 1. Sentences become vectors (MiniLM embeddings when available, otherwise
 *    TF-IDF) and LexRank scores each one by how central it is in the
 *    similarity graph: a sentence that many other sentences echo states
 *    something the paper keeps coming back to.
 * 2. Argumentative zones decide which list a sentence can enter (problem,
 *    method, result), and section priors favour the abstract and
 *    introduction for the TL;DR.
 * 3. Maximal Marginal Relevance picks bullets that are relevant but not
 *    redundant with each other.
 * Results are additionally ranked by metric density (how many measured
 * quantities the sentence reports), and key numbers come from the metric
 * extractor, so every figure on screen is a quote.
 */
import type { Briefing, BriefingItem, Critique, CritiqueFinding, Evidence, KeyNumber, ZonedSentence } from "@/types";
import { extractKeyphrases } from "@/nlp/keyphrases";
import { lexRank, similarityMatrix } from "@/nlp/lexrank";
import { mmrSelect } from "@/nlp/mmr";
import { extractMetrics, type Metric } from "@/nlp/numbers";
import { tokenize, words } from "@/nlp/text";
import { cosineSparse, tfidf } from "@/nlp/tfidf";
import { generateFollowUps } from "./followups";
import { DATASET_MATCHERS, HYPERPARAMETER_RE, datasetKey } from "./lexicons";
import { type AnalyzeInput, CHECKLIST_RE, type PaperSignals, sentenceEvidence } from "./signals";

const KEYPHRASES = 18;
const TLDR_MAX_WORDS = 95;
/**
 * LexRank edge thresholds. MiniLM cosines between unrelated sentences of the
 * same paper sit around 0.1-0.25, so embeddings need a higher bar than
 * sparse TF-IDF vectors, where any shared content word already gives > 0.
 */
const EMBEDDING_THRESHOLD = 0.3;
const TFIDF_THRESHOLD = 0.1;

// ───────────────────────────────── Cleanup ──────────────────────────────────

/** Removes "[12]", "[3, 5-7]", "(Liu et al., 2024; Wei, 2022)" and "et al. (2023)" years. */
export function stripCitations(text: string): string {
  return text
    .replace(/\s*\[\d+(?:\s*[-–,]\s*\d+)*\]/gu, "")
    .replace(/\s*\((?:e\.g\.,?\s*|see\s+|cf\.\s*)?[A-Z][^()]{0,160}?\b(?:19|20)\d{2}[a-z]?\)/gu, "")
    .replace(/(et al\.)\s*\((?:19|20)\d{2}[a-z]?\)/gu, "$1")
    .replace(/\s+([.,;:!?])/gu, "$1")
    .replace(/\s{2,}/gu, " ")
    .trim();
}

/** Discourse links that only make sense after the previous sentence of the paper. */
const LEADING_CONNECTIVE_RE = /^(?:however|moreover|furthermore|in addition|additionally|in contrast|therefore|thus|hence|consequently|yet|also|specifically|in particular|to this end|to address this|as a result),\s+/iu;

/** Citation cleanup plus repairs for display: no dangling "(Rae", no leading "However,", a closing full stop. */
export function displaySentence(text: string): string {
  let out = stripCitations(text)
    .replace(/\s*\([^()]*$/u, "") // a parenthesis cut off by sentence splitting
    .replace(LEADING_CONNECTIVE_RE, "");
  out = out.charAt(0).toUpperCase() + out.slice(1);
  return /[.!?…:]$/u.test(out) ? out : `${out}.`;
}

function item(input: AnalyzeInput, s: ZonedSentence): BriefingItem {
  return { text: displaySentence(s.text), evidence: [sentenceEvidence(input, s)] };
}

const wordCount = (s: ZonedSentence) => words(s.text).length;

/** Long equations, table residue, checklist lines and list headers make poor bullets. */
function readable(s: ZonedSentence): boolean {
  const n = wordCount(s);
  if (n < 8 || n > 70) return false;
  if (CHECKLIST_RE.test(s.text) || /:\s*$|\bthe following\b/iu.test(s.text)) return false;
  // Formula-heavy sentences read badly out of context.
  if ((s.text.match(/[=∼≈|∑∏∫≤≥→←∈∀∃^_{}\\]|\p{Script=Greek}/gu) ?? []).length >= 3) return false;
  // Figure-panel fragments ("(d) ...", "Figure 4: ..."), prompt exemplars ("Q: ...") and quoted dialogue.
  if (/^\(?[a-h]\)\s|\b(?:Figure|Fig\.|Table) \d+:/u.test(s.text)) return false;
  if (/^(?:[A-Z][A-Z ]{1,15}|Question|Answer|Input|Output|Prompt|Response|Human|Assistant|User|Context)\s*:/u.test(s.text)) return false;
  if (/\bI (?:am|was|have|had|asked|think|want|will|would|did|feel|got|need|just)\b/u.test(s.text)) return false;
  const letters = (s.text.match(/\p{L}/gu) ?? []).length;
  return letters / Math.max(1, s.text.length) > 0.6;
}

// ──────────────────────────────── Centrality ────────────────────────────────

interface Graph {
  sentences: ZonedSentence[];
  similarity: number[][];
  /** LexRank scores min-max scaled to 0..1. */
  centrality: number[];
  threshold: number;
}

function pairwiseSparse(vectors: Map<string, number>[]): number[][] {
  const n = vectors.length;
  const m = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    m[i][i] = 1;
    for (let j = i + 1; j < n; j++) m[i][j] = m[j][i] = cosineSparse(vectors[i], vectors[j]);
  }
  return m;
}

function buildGraph(input: AnalyzeInput, sentences: ZonedSentence[]): Graph {
  const embeddings = input.sentenceEmbeddings;
  const dense = embeddings && sentences.every((s) => embeddings.has(s.id));
  const similarity = dense
    ? similarityMatrix(sentences.map((s) => embeddings.get(s.id) as Float32Array))
    : pairwiseSparse(tfidf(sentences.map((s) => tokenize(s.text))).vectors);
  const threshold = dense ? EMBEDDING_THRESHOLD : TFIDF_THRESHOLD;
  return { sentences, similarity, centrality: minMax(lexRank(similarity, { threshold })), threshold };
}

function minMax(xs: number[]): number[] {
  if (xs.length === 0) return [];
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  return xs.map((x) => (hi > lo ? (x - lo) / (hi - lo) : 1));
}

function subgraph(graph: Graph, keep: (s: ZonedSentence) => boolean): Graph {
  const idx = graph.sentences.map((s, i) => (keep(s) ? i : -1)).filter((i) => i >= 0);
  const similarity = idx.map((i) => idx.map((j) => graph.similarity[i][j]));
  const sentences = idx.map((i) => graph.sentences[i]);
  return { sentences, similarity, centrality: minMax(lexRank(similarity, { threshold: graph.threshold })), threshold: graph.threshold };
}

interface Ranked {
  s: ZonedSentence;
  i: number;
  relevance: number;
}

/** MMR over sentence candidates using the graph's similarity matrix. */
function diverse(graph: Graph, candidates: Ranked[], k: number, lambda = 0.7): Ranked[] {
  return mmrSelect(candidates, (c) => c.relevance, (a, b) => graph.similarity[a.i][b.i], k, lambda);
}

const byDocumentOrder = (a: Ranked, b: Ranked) => a.s.index - b.s.index;

// ─────────────────────────────────── TL;DR ──────────────────────────────────

type Role = "problem" | "contribution" | "result";
const ROLE_ORDER: Role[] = ["problem", "contribution", "result"];
/** Contributions matter most for "why this paper matters", then the problem, then a headline result. */
const ROLE_PRIORITY: Role[] = ["contribution", "problem", "result"];

function namesMethod(name: string | undefined): (s: ZonedSentence) => boolean {
  if (!name) return () => false;
  const re = new RegExp(`(?<![\\p{L}\\p{N}])${name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![\\p{L}\\p{N}])`, "u");
  return (s) => re.test(s.text);
}

function tldr(input: AnalyzeInput, graph: Graph, methodName: string | undefined): BriefingItem {
  const front = subgraph(graph, (s) => s.sectionKind === "abstract" || s.sectionKind === "introduction");
  const pool = front.sentences.length >= 2 ? front : graph;
  const named = namesMethod(methodName);
  // The sentence that names the proposed system is usually the paper's thesis.
  const score = (s: ZonedSentence, i: number, role: Role) =>
    0.55 * pool.centrality[i] +
    0.3 * s.scores[role] +
    (s.sectionKind === "abstract" ? 0.25 : 0) +
    (role === "contribution" && named(s) ? 0.15 : 0) -
    (/^(?:This|These|It|Such|They)\b/u.test(s.text) ? 0.1 : 0); // anaphora needs the previous sentence

  const picked = new Map<Role, ZonedSentence>();
  let budget = TLDR_MAX_WORDS;
  for (const role of ROLE_PRIORITY) {
    const best = pool.sentences
      .map((s, i) => ({ s, value: score(s, i, role) }))
      .filter(({ s }) => s.zone === role && readable(s) && ![...picked.values()].includes(s))
      .filter(({ s }) => role !== "result" || (s.sectionKind === "abstract" && extractMetrics(s.text).length > 0))
      .sort((a, b) => b.value - a.value)[0];
    if (!best || wordCount(best.s) > budget) continue;
    picked.set(role, best.s);
    budget -= wordCount(best.s);
  }
  if (picked.size === 0) {
    const central = pool.sentences
      .map((s, i) => ({ s, value: pool.centrality[i] }))
      .filter(({ s }) => readable(s))
      .sort((a, b) => b.value - a.value)
      .slice(0, 2)
      .map(({ s }) => s);
    const fallback = central.length ? central : pool.sentences.slice(0, 2);
    return {
      text: fallback.map((s) => displaySentence(s.text)).join(" "),
      evidence: fallback.map((s) => sentenceEvidence(input, s)),
    };
  }
  const ordered = ROLE_ORDER.map((r) => picked.get(r)).filter((s): s is ZonedSentence => s !== undefined);
  return {
    text: ordered.map((s) => displaySentence(s.text)).join(" "),
    evidence: ordered.map((s) => sentenceEvidence(input, s)),
  };
}

// ──────────────────────────────── Section lists ─────────────────────────────

function ranked(graph: Graph, keep: (s: ZonedSentence) => boolean, relevance: (s: ZonedSentence, centrality: number) => number): Ranked[] {
  return graph.sentences
    .map((s, i) => ({ s, i, relevance: relevance(s, graph.centrality[i]) }))
    .filter(({ s }) => keep(s) && readable(s));
}

function problemItems(input: AnalyzeInput, graph: Graph, exclude: Set<string>): BriefingItem[] {
  const front = (s: ZonedSentence) => s.sectionKind === "abstract" || s.sectionKind === "introduction";
  let candidates = ranked(graph, (s) => s.zone === "problem", (s, c) => 0.5 * c + 0.3 * s.scores.problem + (front(s) ? 0.2 : 0));
  const fresh = candidates.filter((c) => !exclude.has(c.s.id));
  if (fresh.length >= 2) candidates = fresh;
  return diverse(graph, candidates, 3).sort(byDocumentOrder).map((c) => item(input, c.s));
}

/** Position of a sentence inside its section ("s3:0" is the section's opening sentence). */
function positionInSection(s: ZonedSentence): number {
  const n = Number(s.id.split(":").pop());
  return Number.isFinite(n) ? n : 99;
}

function methodItems(input: AnalyzeInput, graph: Graph, exclude: Set<string>, methodName: string | undefined): BriefingItem[] {
  const inMethod = (s: ZonedSentence) => s.sectionKind === "method";
  const named = namesMethod(methodName);
  // Opening sentences of a section state its idea; hyperparameter values are details for the appendix.
  const sectionPrior = (s: ZonedSentence) =>
    inMethod(s) ? 0.15 : s.sectionKind === "experiments" || s.sectionKind === "appendix" || s.sectionKind === "results" ? -0.15 : 0;
  const relevance = (s: ZonedSentence, c: number) =>
    0.5 * c +
    0.3 * s.scores.method +
    sectionPrior(s) +
    (positionInSection(s) <= 1 ? 0.15 : 0) +
    (named(s) ? 0.1 : 0) -
    (HYPERPARAMETER_RE.test(s.text) ? 0.25 : 0) -
    (/\b(?:shown|see|illustrated|given|listed|presented|found) in (?:Table|Figure|Fig\.|Appendix)/iu.test(s.text) ? 0.15 : 0);
  let candidates = ranked(graph, (s) => s.zone === "method" || (inMethod(s) && s.zone === "contribution"), relevance);
  if (candidates.length === 0) {
    // Abstract-only: the contribution sentences are the closest thing to a method description.
    candidates = ranked(graph, (s) => s.zone === "contribution" && !exclude.has(s.id), (s, c) => 0.6 * c + 0.4 * s.scores.contribution);
  }
  return diverse(graph, candidates, 5, 0.65).sort(byDocumentOrder).map((c) => item(input, c.s));
}

const HEADLINE_KINDS = new Set(["abstract", "conclusion", "results", "experiments", "introduction"]);

function resultItems(input: AnalyzeInput, graph: Graph): BriefingItem[] {
  const metricCounts = new Map(graph.sentences.map((s) => [s.id, measurements(s.text).length]));
  const maxMetrics = Math.max(1, ...metricCounts.values());
  const candidates = ranked(
    graph,
    (s) => s.zone === "result" || ((metricCounts.get(s.id) ?? 0) > 0 && HEADLINE_KINDS.has(s.sectionKind) && s.zone !== "background"),
    (s, c) =>
      0.45 * ((metricCounts.get(s.id) ?? 0) / maxMetrics) +
      0.3 * c +
      0.15 * s.scores.result +
      (s.sectionKind === "abstract" || s.sectionKind === "conclusion" ? 0.1 : 0),
  );
  return diverse(graph, candidates, 5).map((c) => item(input, c.s));
}

function authorLimitations(input: AnalyzeInput, graph: Graph, signals: PaperSignals): BriefingItem[] {
  const ids = new Set(signals.authorLimitations.map((e) => e.sentenceId));
  // Admissions in a limitations, discussion or conclusion section are the authors' considered view.
  const placement = (s: ZonedSentence) =>
    s.sectionKind === "limitations" ? 0.3 : s.sectionKind === "discussion" || s.sectionKind === "conclusion" ? 0.15 : 0;
  const candidates = graph.sentences
    .map((s, i) => ({ s, i, relevance: 0.5 * s.scores.limitation + placement(s) }))
    .filter(({ s }) => ids.has(s.id));
  return diverse(graph, candidates, 5, 0.6).map((c) => item(input, c.s));
}

// ──────────────────────────── Detected limitations ──────────────────────────

function firstSentence(text: string): string {
  const match = /^.+?(?<!\be\.g|\bi\.e|\bvs|\bal)[.!?](?=\s+[A-Z]|$)/u.exec(text);
  return (match ? match[0] : text).trim();
}

function asLimitation(f: CritiqueFinding): BriefingItem {
  return { text: `${f.title}. ${firstSentence(f.rationale)}`, evidence: f.evidence };
}

/** Critique findings phrased as limitations; falls back to the most relevant residual caveat, never empty. */
function detectedLimitations(signals: PaperSignals, critique: Critique): BriefingItem[] {
  const weaknesses = critique.findings.filter((f) => f.severity !== "strength");
  const serious = weaknesses.filter((f) => f.severity === "major" || f.severity === "moderate");
  const chosen = (serious.length > 0 ? serious : weaknesses).slice(0, 5);
  if (chosen.length > 0) return chosen.map(asLimitation);

  const names = signals.datasets.map((d) => d.name);
  if (names.length > 0) {
    return [{
      text: `No review check flagged a weakness. The main residual caveat: conclusions are bounded by the ${names.length === 1 ? "benchmark" : "benchmarks"} evaluated (${names.slice(0, 4).join(", ")}${names.length > 4 ? ", …" : ""}) and may not transfer to other data.`,
      evidence: signals.datasets.slice(0, 2).map((d) => d.evidence),
    }];
  }
  return [{
    text: `No review check flagged a weakness. The main residual caveat: no standard benchmark was recognised (${signals.resultSentences} result sentences), so the generality of the findings is hard to assess.`,
    evidence: [],
  }];
}

// ──────────────────────────────── Key numbers ───────────────────────────────

/** Configuration sizes ("7B parameters", "32K context") are settings, not results. */
const SETTING_UNITS = new Set<Metric["unit"]>(["B", "M", "K"]);

function measurements(text: string): Metric[] {
  return extractMetrics(text).filter((m) => !SETTING_UNITS.has(m.unit) && (m.unit !== undefined || m.metric !== undefined));
}

const UNIT_LABELS: Partial<Record<NonNullable<Metric["unit"]>, string>> = {
  x: "factor",
  "%": "percentage",
  pp: "percentage-point change",
  points: "point change",
  ms: "latency",
  s: "time",
  GB: "memory",
  MB: "memory",
};

function displayValue(m: Metric): string {
  if (/^[+\-−]/u.test(m.raw)) return m.raw;
  const delta = m.unit === "pp" || m.unit === "points";
  if (delta && m.direction === "up") return `+${m.raw}`;
  if (delta && m.direction === "down") return `−${m.raw}`;
  return m.raw;
}

/** "Memory reduction", "Accuracy on GSM8K", "Coin Flip": what the number measures, best effort. */
function metricLabel(m: Metric, dataset: string | undefined): string {
  const change = m.unit === "x" || m.unit === "pp" || m.unit === "points" || m.unit === "%";
  const direction = !change || !m.metric ? "" : m.direction === "down" ? " reduction" : m.direction === "up" ? " gain" : "";
  const on = dataset && dataset.toLowerCase() !== m.metric?.toLowerCase() ? dataset : undefined;
  let label: string;
  if (m.metric) label = `${m.metric}${direction}${on ? ` on ${on}` : ""}`;
  else if (on) label = on;
  else label = UNIT_LABELS[m.unit ?? "%"] ?? "reported value";
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function matcherFor(name: string): RegExp | undefined {
  const key = datasetKey(name);
  return DATASET_MATCHERS.find((m) => datasetKey(m.entry.name) === key)?.re;
}

function keyNumbers(input: AnalyzeInput, graph: Graph, signals: PaperSignals): KeyNumber[] {
  const datasetNames = signals.datasets.map((d) => d.name);
  // Headline numbers live in the abstract and conclusion; then the most central result sentences.
  const sourceRank = (s: ZonedSentence) => (s.sectionKind === "abstract" ? 0 : s.sectionKind === "conclusion" ? 1 : 2);
  const sources = graph.sentences
    .map((s, i) => ({ s, c: graph.centrality[i] }))
    .filter(({ s }) => s.zone === "result" || s.sectionKind === "abstract" || s.sectionKind === "conclusion")
    .filter(({ s }) => s.zone !== "background" && readable(s))
    .sort((a, b) => sourceRank(a.s) - sourceRank(b.s) || b.c - a.c);

  // Numbers whose meaning is known (a named metric, a dataset, or a ×/pp change) beat bare percentages.
  const candidates: { value: string; label: string; tier: number; order: number; s: ZonedSentence }[] = [];
  const seen = new Set<string>();
  sources.forEach(({ s }, order) => {
    for (const m of measurements(s.text)) {
      const value = displayValue(m);
      if (seen.has(value)) continue;
      seen.add(value);
      // Attach a dataset only when the sentence names exactly one; otherwise the pairing is a guess.
      const named = datasetNames.filter((name) => matcherFor(name)?.test(s.text));
      const dataset = named.length === 1 ? named[0] : undefined;
      const tier = m.metric || m.unit === "x" || m.unit === "pp" || m.unit === "points" ? 0 : dataset ? 1 : 2;
      // A bare percentage is only meaningful where the authors headline it.
      if (tier === 2 && sourceRank(s) > 1) continue;
      candidates.push({ value, label: metricLabel(m, dataset), tier, order, s });
    }
  });
  return candidates
    .sort((a, b) => a.tier - b.tier || a.order - b.order)
    .slice(0, 6)
    .map(({ value, label, s }) => ({ value, label, evidence: sentenceEvidence(input, s) }));
}

// ────────────────────────────────── Entry ───────────────────────────────────

function keyphraseText(input: AnalyzeInput): string {
  const body = input.sentences.map((s) => s.text).join(" ");
  return `${input.meta.title.replace(/[.:]?\s*$/u, ".")} ${body}`;
}

export async function buildBriefing(input: AnalyzeInput, signals: PaperSignals, critique: Critique): Promise<Briefing> {
  const graph = buildGraph(input, input.sentences);
  const summary = tldr(input, graph, signals.methodName);
  const inSummary = new Set(summary.evidence.map((e: Evidence) => e.sentenceId ?? ""));
  const keyphrases = input.sentences.length > 0 ? await extractKeyphrases(keyphraseText(input), { embed: input.embed, top: KEYPHRASES }) : [];

  return {
    tldr: summary,
    problem: problemItems(input, graph, inSummary),
    method: methodItems(input, graph, inSummary, signals.methodName),
    results: resultItems(input, graph),
    limitations: {
      authorStated: authorLimitations(input, graph, signals),
      detected: detectedLimitations(signals, critique),
    },
    followUps: generateFollowUps(input, signals, keyphrases),
    keyphrases,
    keyNumbers: keyNumbers(input, graph, signals),
  };
}

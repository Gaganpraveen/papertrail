/**
 * Grounded question answering over one paper, run as an explicit state graph:
 *
 *   classify ─┬─ summary / limitations ──> compose_from_analysis ──> END
 *             └─ otherwise ──> retrieve ──> rerank ──> read ──> verify ─┬─> respond ──> END
 *                                  └─ nothing found ──> abstain         └─> abstain ──> END
 *
 * Extractive by design: every answer is copied from the paper, so the system
 * can be wrong about *where* the answer is but cannot invent a fact.
 *
 *  - retrieve  hybrid BM25 + dense search over chunks (retrieval/hybrid.ts).
 *  - rerank    a cross-encoder reads (question, passage) pairs jointly; its
 *              logit becomes a calibrated passage relevance.
 *  - read      an extractive reader (DistilBERT-SQuAD) proposes a span in
 *              each of the top passages, with a span probability.
 *  - verify    the span must occur verbatim in its passage, and
 *              confidence = relevance × reader probability × sentence support
 *              must clear ABSTAIN_THRESHOLD; otherwise we abstain rather than
 *              guess. "Support" checks that the sentence holding the span is
 *              itself about the question. A question that names something the
 *              paper never mentions ("MMLU" in a paper without it) gets 0.
 *
 * Some questions have no span-shaped answer (yes/no, "What is X?", overviews):
 * those quote the best-matching sentence instead, under a stricter threshold.
 *
 * Every model is optional. Without the reranker, relevance comes from term
 * coverage and embedding similarity; without the reader, sentences are quoted;
 * with no models at all this degrades to purely lexical retrieval, and still
 * abstains when nothing matches.
 */
import { CompiledGraph, END, START, StateGraph, type NodeContext } from "@/agent/graph";
import type { MLApi } from "@/ml/client";
import { alignSpan } from "@/ml/models";
import { splitSentences, tokenize, words } from "@/nlp/text";
import { cosine } from "@/nlp/vector";
import { embeddingText } from "@/retrieval/chunker";
import type { HybridIndex } from "@/retrieval/hybrid";
import type {
  Answer,
  BriefingItem,
  Chunk,
  EmbedFn,
  Evidence,
  PaperAnalysis,
  QuestionIntent,
  RetrievalHit,
  SectionKind,
  TraceEvent,
  Zone,
  ZonedSentence,
} from "@/types";

export interface QAContext {
  paperId: string;
  index: HybridIndex;
  sentences: ZonedSentence[];
  analysis?: PaperAnalysis;
  ml: { embed?: EmbedFn; rerank?: MLApi["rerank"]; answer?: MLApi["answer"] };
}

// ───────────────────────────── Tuned constants ─────────────────────────────
//
// Calibrated with the deployed models on 33 questions over three real papers
// (KIVI 2402.02750, Chain-of-Thought 2201.11903, ReAct 2210.03629), mixing
// answerable questions with off-topic ones ("What is the capital of
// France?") and plausible-but-absent ones ("What learning rate was used for
// fine-tuning?" on a paper that never fine-tunes). ABSTAIN_THRESHOLD was
// later re-chosen on a separate labelled set, as its comment says.
// tests/ml.node.test.ts re-checks the behaviour on real passages.

/**
 * Minimum relevance × reader probability × support for a span answer.
 *
 * Chosen by `npm run eval` on eval/questions.json (51 questions over the same
 * three papers: 39 answerable, 12 not). The rule: over thresholds 0.05, 0.10,
 * ..., 0.80, take the one that maximises accuracy on answerable questions
 * minus the false-answer rate on unanswerable ones; on a tie, the lower one.
 * On all 51 questions that is 0.55: 23/39 correct, 0/12 false answers.
 *
 * That number is optimistic, because it is measured on the questions the
 * threshold was chosen on. The honest estimate comes from leaving one paper
 * out: choose on two papers, score the third. The three folds chose 0.45,
 * 0.55 and 0.55, and the held-out papers together got 23/39 right with 1/12
 * false answers. With everything else unchanged, the old value, 0.2, gets
 * 26/39 right but 3/12 false answers; it was not chosen on this set, so that
 * score is already a held-out one.
 *
 * Correct and unanswerable scores overlap (0.13–0.91 against up to 0.54), so
 * no value separates them, and 0.55 also withholds some right answers.
 * Confident reader mistakes remain, which is why every answer ships with its
 * quoted sentence and page.
 */
export const ABSTAIN_THRESHOLD = 0.55;

/**
 * Quoting a whole sentence is a weaker claim than extracting a span, and the
 * lexical/cosine match behind it is less selective than a reader, so the bar
 * is higher. Lexical-only, unanswerable questions still matched a sentence at
 * up to 0.39 through generic shared terms; real answers matched at 0.46+.
 */
export const SENTENCE_ABSTAIN_THRESHOLD = 0.45;

/**
 * Cross-encoder logits are calibrated for web search. On paper passages we
 * saw relevant passages mostly at -3.5 .. +7.5 and off-topic ones at -5 .. -11, so
 * sigmoid((logit - CENTER) / TEMPERATURE) puts the coin flip at -3 and
 * softens the slope: a mildly negative but correct passage keeps most of its
 * credit while off-topic passages still fall towards zero.
 */
const RERANK_CENTER = -3;
const RERANK_TEMPERATURE = 2;

/**
 * Cosine between MiniLM embeddings of a question and an unrelated passage
 * sits around 0.0–0.3; strong matches reach 0.55–0.75. Map [0.2, 0.7] -> [0, 1].
 */
const COSINE_FLOOR = 0.2;
const COSINE_SPAN = 0.5;

const RETRIEVE_K = 20;
/** Cross-encoding is the costliest step per passage; 8 covers the useful head. */
const RERANK_DEPTH = 8;
/**
 * A reader span whose sentence shares no terms or meaning with the question
 * keeps only half its confidence. Answers are often paraphrased, so this is a
 * discount, not a veto; on real papers it pushed confidently wrong spans
 * lifted from off-topic sentences ("six" examples for "How many GPUs ...?")
 * below the threshold while correct spans kept most of their score.
 */
const SUPPORT_FLOOR = 0.5;

/** Passages the reader examines (one forward pass each). */
const READ_DEPTH = 3;
/** Hits kept on the Answer for the UI's "sources" list. */
const HITS_KEPT = 8;

/**
 * Soft section prior: a 5% boost on the fused score moves a passage from a
 * section that fits the question's intent up by about three ranks.
 */
const SECTION_PRIOR_BOOST = 0.05;
const PREFERRED_SECTIONS: Partial<Record<QuestionIntent, SectionKind[]>> = {
  method: ["method", "background"],
  results: ["results", "experiments"],
  comparison: ["results", "experiments", "related"],
  limitations: ["limitations", "discussion", "conclusion"],
  summary: ["abstract", "introduction", "conclusion"],
};

/** Zones whose sentences best answer a composite question in sentence mode. */
const PREFERRED_ZONES: Partial<Record<QuestionIntent, Zone>> = {
  summary: "contribution",
  limitations: "limitation",
};
const ZONE_BONUS = 0.15;
/** Bonus for a sentence that has the shape of a definition of the asked-about term. */
const DEFINITION_BONUS = 0.2;

const NOT_ANSWERABLE_SECTIONS = new Set<SectionKind>(["references", "acknowledgments"]);

/** Content words that every question about a paper shares; they carry no signal. */
const QUESTION_TERMS = new Set(
  tokenize("paper papers author authors work study propose proposed use used using"),
);

// ──────────────────────────── Intent classification ───────────────────────

const INTENT_RULES: [QuestionIntent, RegExp][] = [
  [
    "limitations",
    /\b(limitations?|weakness(es)?|shortcomings?|drawbacks?|downsides?|caveats?|failure (cases?|modes?)|threats? to validity|fall short|does ?n[o']t work)\b/i,
  ],
  [
    "summary",
    /\b(summar(y|ise|ize)|tl;?dr|overview|gist|in a nutshell|(main|key|core) (idea|point|takeaway|contribution|finding)s?|contributions?|what is (this|the) paper about|what does (this|the) paper (do|propose|present))\b/i,
  ],
  ["comparison", /\b(compare[sd]?|comparison|versus|vs\.?|differ(s|ence|ences)? (from|between)|relative to|better than|worse than)\b/i],
  [
    "results",
    /\b(results?|performance|accuracy|scores?|improve(s|d|ment|ments)?|speed-?ups?|faster|outperform(s|ed)?|achieve[sd]?|gains?|bleu|rouge|f1|perplexity|throughput|latency|how (well|much|many))\b/i,
  ],
  [
    "method",
    /\b(how (does|do|is|are|was|were) .+ (work|built|trained|computed|implemented|designed)|method(ology)?|approach|architecture|algorithm|mechanism|procedure|pipeline|objective|loss function|training)\b/i,
  ],
];

export function classifyQuestion(question: string): QuestionIntent {
  for (const [intent, pattern] of INTENT_RULES) if (pattern.test(question)) return intent;
  return "factual";
}

/** Yes/no questions start with an auxiliary verb; no extracted span can answer them. */
function isPolarQuestion(question: string): boolean {
  return /^\s*(is|are|was|were|does|do|did|can|could|will|would|should|has|have|had)\b/i.test(question);
}

/**
 * The subject of a definition question ("What is chain-of-thought
 * prompting?" -> "chain-of-thought prompting"). Questions about a value of
 * something ("What is the batch size?") are not definitions and return null.
 */
export function definitionSubject(question: string): string | null {
  const match = /^\s*what\s+(?:is|are)\s+(?:an?\s+)?([\p{L}\p{N}][\p{L}\p{N}\s-]*?)\s*\??\s*$/iu.exec(question);
  if (!match) return null;
  const subject = match[1].trim();
  if (/^(the|this|that|these|those|its|their|our)\b/i.test(subject)) return null;
  return subject.split(/\s+/).length <= 4 ? subject : null;
}

/**
 * Why a question is answered with a whole sentence instead of a span: a short
 * span cannot say yes or no, define a term, or give an overview. Null when a
 * span is the right shape.
 */
function sentenceModeReason(question: string, intent: QuestionIntent): string | null {
  if (isPolarQuestion(question)) return "a yes/no question is answered with the best supporting sentence";
  if (definitionSubject(question)) return "a definition question is answered with the defining sentence";
  if (intent === "summary" || intent === "limitations") return "an overview question is answered with the best-matching sentence";
  return null;
}

/** A term right after one of these is the preposition's object, not the subject of the sentence. */
const PREPOSITIONS =
  "about|across|after|against|among|at|before|between|by|during|for|from|in|into|of|on|onto|over|than|through|to|under|unlike|with|within|without";

/**
 * The shapes definitions take in papers: "X is a ...", "X refers to / denotes /
 * means ...", "X, a ...", "We define X as ...". Bracketed asides may sit
 * between the term and the definer, nested one level deep ("ALFWorld
 * (Shridhar et al., 2020b) (Figure 1(2)) is a ..."). The term must be the
 * subject: in "one challenge built into ALFWorld is the need to ..." and "For
 * last letter concatenation, the model ..." it is a preposition's object, and
 * the sentence is about something else.
 */
export function definesSubject(sentence: string, subject: string): boolean {
  const term = subject.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\s+/g, "\\s+");
  const notObject = String.raw`(?<!\b(?:${PREPOSITIONS})\s+)`;
  const asides = String.raw`(?:\s*(?:\((?:[^()]|\([^()]*\))*\)|\[[^\]]*\]))*\s*`; // "(RAG)", "(Smith et al., 2020)", "[12]"
  const definer = String.raw`(?:(?:is|are)\s+(?:an?|the|one)\b|(?:refers? to|denotes?|means)\b|,\s*(?:an?|the)\b)`;
  const shaped = new RegExp(String.raw`${notObject}\b${term}${asides}${definer}`, "iu");
  const declared = new RegExp(String.raw`\bdefines?\s+(?:an?\s+|the\s+)?${term}\s+as\b`, "iu");
  return shaped.test(sentence) || declared.test(sentence);
}

/**
 * Answer-type check, as in classic factoid QA: a question that asks for a
 * quantity ("How many layers…?", "What is the group size?") can only be
 * answered by a span that contains a number. The reader has no notion of
 * answer types and will happily return "GSM8K" for "What is the group size?".
 */
export function expectsQuantity(question: string): boolean {
  return /\b(how (many|much|long|large|big|fast|often)|what (is|are|was|were) the [\p{L}\s-]*?\b(sizes?|numbers?|counts?|rates?|ratios?|percentages?|dimensions?|lengths?|values?|amounts?|budgets?|depths?|widths?))\b/iu.test(question);
}

/** A standalone number ("2.6×", "128", "4-bit") or a number word; "GSM8K" does not count. */
export function containsQuantity(span: string): boolean {
  return /(?:^|[^\p{L}\p{N}])\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|million|billion|half|double|twice|triple)\b/iu.test(span);
}

/**
 * Names in a question: words written with a capital after the first letter,
 * or mixing letters and digits ("MMLU", "WikiText-2", "MiniWoB++", "GSM8K").
 * Ordinary words are left out on purpose. A question may say "hardware" where
 * the paper says "TPU v4", but nobody paraphrases a benchmark's name.
 */
export function namedTerms(question: string): string[] {
  return words(question).filter((w) => /\p{Lu}/u.test(w.slice(1)) || (/\p{N}/u.test(w) && /\p{L}/u.test(w)));
}

const squash = (token: string) => token.replace(/[-'’]/gu, "");

/** "gpt-3" also as "gpt3", "gpt" and "3", so "GPT3" and "GPT-3", or "PaLM 540B" and "PaLM-540B", match. */
function wordForms(token: string): string[] {
  return [token, squash(token), ...token.split("-")];
}

/** Every word form in the paper's passages (lowercased and stemmed, as `tokenize` gives them), cached per index. */
const vocabularyCache = new WeakMap<HybridIndex, Set<string>>();

function paperVocabulary(index: HybridIndex): Set<string> {
  let vocabulary = vocabularyCache.get(index);
  if (!vocabulary) {
    vocabulary = new Set(index.chunks.flatMap((chunk) => tokenize(chunk.text).flatMap(wordForms)));
    vocabularyCache.set(index, vocabulary);
  }
  return vocabulary;
}

/**
 * Names the question uses that appear nowhere in the paper's passages, the
 * references included. No passage can answer a question about something it
 * never mentions, however well the reader scores a span in it.
 */
export function unmentionedNames(question: string, index: HybridIndex): string[] {
  const vocabulary = paperVocabulary(index);
  const mentioned = (token: string) =>
    vocabulary.has(token) ||
    vocabulary.has(squash(token)) ||
    token.split("-").filter((part) => part.length >= 2).every((part) => vocabulary.has(part));
  return namedTerms(question).filter((name) => !tokenize(name).every(mentioned));
}

// ─────────────────────────────── Graph state ──────────────────────────────

type AnswerMode = "reader" | "sentence";

/** A candidate answer before verification. Offsets are into `hit.chunk.text`. */
interface Proposal {
  hit: RetrievalHit;
  span: string;
  start: number;
  end: number;
  mode: AnswerMode;
  /** Reader span probability, or sentence match score; 0..1. */
  evidenceScore: number;
  /** How relevant the passage is to the question; 0..1. */
  relevance: number;
  /** For spans: how well the span's sentence matches the question, in [SUPPORT_FLOOR, 1]. */
  support: number;
  /** relevance × evidenceScore × support, filled in by verify. */
  confidence: number;
}

export interface QAState {
  question: string;
  ctx: QAContext;
  intent: QuestionIntent;
  queryEmbedding: Float32Array | null;
  hits: RetrievalHit[];
  proposals: Proposal[];
  best: Proposal | null;
  /** Names in the question that the paper never mentions (see unmentionedNames). */
  unmentioned: string[];
  answer: Answer | null;
  /** How the answer was produced, including any model that was unavailable. */
  notes: string[];
}

// ───────────────────────────── Scoring helpers ────────────────────────────

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

export function rerankRelevance(logit: number): number {
  return sigmoid((logit - RERANK_CENTER) / RERANK_TEMPERATURE);
}

function cosineRelevance(cos: number): number {
  return clamp01((cos - COSINE_FLOOR) / COSINE_SPAN);
}

function questionTerms(question: string): string[] {
  return [...new Set(tokenize(question))].filter((t) => !QUESTION_TERMS.has(t));
}

/** Inverse document frequency over the paper's chunks, cached per index. */
const idfCache = new WeakMap<HybridIndex, Map<string, number>>();

function idfFor(index: HybridIndex): (term: string) => number {
  let table = idfCache.get(index);
  if (!table) {
    const df = new Map<string, number>();
    for (const chunk of index.chunks) {
      for (const term of new Set(tokenize(embeddingText(chunk)))) df.set(term, (df.get(term) ?? 0) + 1);
    }
    const n = index.chunks.length;
    table = new Map([...df].map(([term, count]) => [term, Math.log(1 + (n - count + 0.5) / (count + 0.5))]));
    idfCache.set(index, table);
  }
  // A term absent from the paper gets the highest possible weight: missing it
  // is the strongest sign the paper does not cover the question.
  const unseen = Math.log(1 + (index.chunks.length + 0.5) / 0.5);
  return (term) => table.get(term) ?? unseen;
}

/**
 * Share of the question's IDF mass present in `text`: 1 when every content
 * term appears, 0 when none does. Rare terms (method names, datasets) weigh
 * most, so "Which GPU ..." hinges on "gpu", not on "experiment".
 */
function termCoverage(terms: string[], text: string, idf: (t: string) => number): number {
  if (terms.length === 0) return 0;
  const present = new Set(tokenize(text));
  let total = 0;
  let matched = 0;
  for (const term of terms) {
    const weight = idf(term);
    total += weight;
    if (present.has(term)) matched += weight;
  }
  return total > 0 ? matched / total : 0;
}

/** Lexical coverage, blended 50/50 with embedding similarity when available. */
function matchScore(coverage: number, cos: number | null): number {
  return cos === null ? coverage : 0.5 * coverage + 0.5 * cosineRelevance(cos);
}

// ───────────────────────────── Sentence lookup ────────────────────────────

interface Located {
  text: string;
  start: number;
  end: number;
  sentence?: ZonedSentence;
}

/**
 * The sentences of a chunk with their offsets in its text. Chunks are joined
 * sentences, so we find each section sentence in order; if the sentence list
 * does not line up (e.g. it was not provided), we re-split the chunk text.
 */
function sentencesOf(chunk: Chunk, bySection: Map<string, ZonedSentence[]>): Located[] {
  const located: Located[] = [];
  let cursor = 0;
  for (const sentence of bySection.get(chunk.sectionId) ?? []) {
    const text = sentence.text.trim();
    const at = text ? chunk.text.indexOf(text, cursor) : -1;
    if (at < 0) continue;
    located.push({ text, start: at, end: at + text.length, sentence });
    cursor = at + text.length;
  }
  if (located.length > 0) return located;

  cursor = 0;
  for (const text of splitSentences(chunk.text)) {
    const at = chunk.text.indexOf(text, cursor);
    if (at < 0) continue;
    located.push({ text, start: at, end: at + text.length });
    cursor = at + text.length;
  }
  return located.length > 0 ? located : [{ text: chunk.text, start: 0, end: chunk.text.length }];
}

function groupBySection(sentences: ZonedSentence[]): Map<string, ZonedSentence[]> {
  const groups = new Map<string, ZonedSentence[]>();
  for (const s of sentences) {
    const group = groups.get(s.sectionId);
    if (group) group.push(s);
    else groups.set(s.sectionId, [s]);
  }
  return groups;
}

/** Sentences overlapping [start, end): the span's context and its evidence. */
function sentencesAround(proposal: Proposal, ctx: QAContext): Located[] {
  const all = sentencesOf(proposal.hit.chunk, groupBySection(ctx.sentences));
  const overlapping = all.filter((s) => s.start < proposal.end && s.end > proposal.start);
  return overlapping.length > 0 ? overlapping : [{ text: proposal.span, start: proposal.start, end: proposal.end }];
}

// ─────────────────────────────────── Nodes ─────────────────────────────────

function classifyNode(state: QAState, node: NodeContext): Partial<QAState> {
  const intent = classifyQuestion(state.question);
  node.log(`Intent: ${intent}`);
  return { intent };
}

function routeAfterClassify(state: QAState): string {
  const composite = state.intent === "summary" || state.intent === "limitations";
  return composite && state.ctx.analysis ? "compose_from_analysis" : "retrieve";
}

/** One block of a composed answer: an optional heading over briefing items. */
interface ComposedPart {
  heading?: string;
  items: BriefingItem[];
  /** Line shown when the part has no items; parts without one are skipped. */
  whenEmpty?: string;
}

function composedParts(analysis: PaperAnalysis, intent: QuestionIntent): ComposedPart[] {
  const { briefing } = analysis;
  if (intent === "limitations") {
    return [
      {
        heading: "Stated by the authors",
        items: briefing.limitations.authorStated,
        whenEmpty: "The authors do not discuss limitations explicitly.",
      },
      { heading: "Flagged by PaperTrail's review rules", items: briefing.limitations.detected },
    ];
  }
  return [
    { items: [briefing.tldr] },
    { heading: "Method", items: briefing.method.slice(0, 2) },
    { heading: "Results", items: briefing.results.slice(0, 2) },
  ];
}

/** Plain text with line breaks: headed parts become "Heading:" plus "• item" lines. */
function renderParts(parts: ComposedPart[]): string {
  const lines: string[] = [];
  for (const { heading, items, whenEmpty } of parts) {
    if (items.length === 0 && !whenEmpty) continue;
    if (heading) lines.push(`${heading}:`);
    if (items.length === 0 && whenEmpty) lines.push(whenEmpty);
    for (const item of items) lines.push(heading ? `• ${item.text}` : item.text);
  }
  return lines.join("\n");
}

function composeNode(state: QAState, node: NodeContext): Partial<QAState> {
  const parts = composedParts(state.ctx.analysis as PaperAnalysis, state.intent);
  const items = parts.flatMap((p) => p.items);
  // No model probability exists for a composed answer; report the share of it
  // that is backed by quoted evidence instead.
  const backed = items.filter((i) => i.evidence.length > 0).length;
  node.log(`Composed from ${items.length} briefing items`);

  return {
    answer: {
      ...baseAnswer(state),
      status: "answered",
      answer: renderParts(parts),
      confidence: items.length > 0 ? round(backed / items.length) : 0,
      evidence: uniqueEvidence(items),
      hits: [],
      reason:
        state.intent === "limitations"
          ? "Composed from the paper's briefing: author-stated limitations are quoted from the paper; flagged ones come from PaperTrail's review rules, each with its evidence."
          : "Composed from the paper's briefing, with the quoted sentences each line is drawn from.",
    },
  };
}

async function retrieveNode(state: QAState, node: NodeContext): Promise<Partial<QAState>> {
  const { ctx, question } = state;
  if (!/[\p{L}\p{N}]/u.test(question)) {
    node.log("Empty question; nothing to retrieve");
    return { hits: [] };
  }
  let queryEmbedding: Float32Array | null = null;
  const notes = [...state.notes];

  if (ctx.ml.embed) {
    try {
      const [vector] = await ctx.ml.embed([question]);
      queryEmbedding = vector ?? null;
    } catch (err) {
      notes.push("embedding model unavailable, so retrieval was lexical only");
      node.log(`Embedder unavailable (${errorText(err)}); continuing with BM25 only`);
    }
  }

  const hits = ctx.index.search(question, ctx.index.hasDense ? queryEmbedding : null, {
    k: RETRIEVE_K,
    filter: (c) => !NOT_ANSWERABLE_SECTIONS.has(c.sectionKind),
  });
  const ordered = applySectionPrior(hits, state.intent);
  node.log(`${ordered.length} passages via ${ctx.index.hasDense && queryEmbedding ? "BM25 + dense (RRF)" : "BM25"}`);
  return { queryEmbedding, hits: ordered, notes };
}

function applySectionPrior(hits: RetrievalHit[], intent: QuestionIntent): RetrievalHit[] {
  const preferred = PREFERRED_SECTIONS[intent];
  if (!preferred) return hits;
  const boosted = (h: RetrievalHit) => h.rrf * (preferred.includes(h.chunk.sectionKind) ? 1 + SECTION_PRIOR_BOOST : 1);
  return withFinalRanks([...hits].sort((a, b) => boosted(b) - boosted(a) || a.rank.final - b.rank.final));
}

function withFinalRanks(hits: RetrievalHit[]): RetrievalHit[] {
  return hits.map((h, i) => ({ ...h, rank: { ...h.rank, final: i + 1 } }));
}

async function rerankNode(state: QAState, node: NodeContext): Promise<Partial<QAState>> {
  const { ctx, question, hits } = state;
  if (!ctx.ml.rerank) {
    node.log("Reranker not available; keeping the fused order");
    return { notes: [...state.notes, "reranker unavailable, so passage relevance comes from term and embedding overlap"] };
  }
  const head = hits.slice(0, RERANK_DEPTH);
  try {
    const logits = await ctx.ml.rerank(question, head.map((h) => embeddingText(h.chunk)));
    if (logits.length !== head.length) throw new Error(`expected ${head.length} scores, got ${logits.length}`);
    const reranked = head.map((h, i) => ({ ...h, rerank: logits[i] })).sort((a, b) => b.rerank - a.rerank);
    node.log(`Top passage logit ${reranked[0]?.rerank.toFixed(2) ?? "n/a"}`);
    return { hits: withFinalRanks([...reranked, ...hits.slice(RERANK_DEPTH)]) };
  } catch (err) {
    node.log(`Reranker failed (${errorText(err)}); keeping the fused order`);
    return { notes: [...state.notes, "reranker unavailable, so passage relevance comes from term and embedding overlap"] };
  }
}

/** Passage relevance: calibrated cross-encoder score, else lexical + dense match. */
function passageRelevance(hit: RetrievalHit, terms: string[], idf: (t: string) => number, dense: boolean): number {
  if (hit.rerank !== undefined) return rerankRelevance(hit.rerank);
  return matchScore(termCoverage(terms, embeddingText(hit.chunk), idf), dense ? hit.dense : null);
}

async function readNode(state: QAState, node: NodeContext): Promise<Partial<QAState>> {
  const { ctx, question } = state;
  const top = state.hits.slice(0, READ_DEPTH);
  const terms = questionTerms(question);
  const idf = idfFor(ctx.index);
  const dense = state.queryEmbedding !== null && ctx.index.hasDense;
  const notes = [...state.notes];

  const sentenceReason = sentenceModeReason(question, state.intent);
  if (ctx.ml.answer && !sentenceReason) {
    try {
      const proposals: Proposal[] = [];
      for (const hit of top) {
        const result = await ctx.ml.answer(question, hit.chunk.text);
        proposals.push({
          hit,
          span: result.answer,
          start: result.start,
          end: result.end,
          mode: "reader",
          evidenceScore: result.score,
          relevance: passageRelevance(hit, terms, idf, dense),
          support: 1,
          confidence: 0,
        });
      }
      node.log(`Reader proposed ${proposals.length} spans`);
      return { proposals, notes };
    } catch (err) {
      notes.push("reader model unavailable, so the best-matching sentence is quoted");
      node.log(`Reader failed (${errorText(err)}); falling back to sentence matching`);
    }
  } else if (!ctx.ml.answer) {
    notes.push("reader model unavailable, so the best-matching sentence is quoted");
  } else if (sentenceReason) {
    notes.push(sentenceReason);
  }

  const proposals = await bestSentences(state, top, node);
  return { proposals, notes };
}

/**
 * Scores sentences against the question: IDF-weighted term coverage, blended
 * with embedding cosine when the embedder works. 0..1.
 */
async function scoreSentences(state: QAState, texts: string[], node: NodeContext): Promise<number[]> {
  const { ctx, queryEmbedding } = state;
  const terms = questionTerms(state.question);
  const idf = idfFor(ctx.index);

  let vectors: Float32Array[] | null = null;
  if (ctx.ml.embed && queryEmbedding && texts.length > 0) {
    try {
      vectors = await ctx.ml.embed(texts);
    } catch (err) {
      node.log(`Sentence embeddings unavailable (${errorText(err)}); scoring by term coverage`);
    }
  }
  return texts.map((text, i) => {
    const cos = vectors && queryEmbedding ? cosine(queryEmbedding, vectors[i]) : null;
    return matchScore(termCoverage(terms, text, idf), cos);
  });
}

/** Sentence-mode fallback: the best sentence per top passage. */
async function bestSentences(state: QAState, top: RetrievalHit[], node: NodeContext): Promise<Proposal[]> {
  const bySection = groupBySection(state.ctx.sentences);
  const perHit = top.map((hit) => ({ hit, sentences: sentencesOf(hit.chunk, bySection) }));
  const texts = [...new Set(perHit.flatMap((p) => p.sentences.map((s) => s.text)))];
  const scores = await scoreSentences(state, texts, node);
  const scoreOf = new Map(texts.map((t, i) => [t, scores[i]]));

  const preferredZone = PREFERRED_ZONES[state.intent];
  const subject = definitionSubject(state.question);
  const candidates: { proposal: Proposal; defines: boolean }[] = [];
  for (const { hit, sentences } of perHit) {
    let best: { located: Located; score: number; defines: boolean } | null = null;
    for (const located of sentences) {
      // Bonuses rank sentences unclamped (so they still separate perfect lexical
      // matches); the reported score is clamped back to 0..1 below.
      let score = scoreOf.get(located.text) ?? 0;
      if (preferredZone && located.sentence?.zone === preferredZone) score += ZONE_BONUS;
      const defines = subject !== null && definesSubject(located.text, subject);
      if (defines) score += DEFINITION_BONUS;
      // A definition question wants a definition: one beats any sentence that is not one.
      if (!best || (defines && !best.defines) || (defines === best.defines && score > best.score)) best = { located, score, defines };
    }
    if (!best) continue;
    candidates.push({
      defines: best.defines,
      proposal: {
        hit,
        span: best.located.text,
        start: best.located.start,
        end: best.located.end,
        mode: "sentence",
        evidenceScore: clamp01(best.score),
        // The sentence score already measures relevance; only a cross-encoder adds independent evidence.
        relevance: hit.rerank !== undefined ? rerankRelevance(hit.rerank) : 1,
        support: 1,
        confidence: 0,
      },
    });
  }
  node.log(`Sentence matching proposed ${candidates.length} candidates`);

  // A definition beats a non-definition across passages too. The bonus alone
  // cannot do that: clamping caps every strong match at 1, so a short sentence
  // that merely names the term ("Here fake quantization is exactly the same as
  // in Table 1.") ties with the definition and wins on passage relevance.
  const definitions = candidates.filter((c) => c.defines);
  if (definitions.length > 0 && definitions.length < candidates.length) {
    node.log(`Kept the ${definitions.length} shaped like a definition of "${subject}"`);
  }
  return (definitions.length > 0 ? definitions : candidates).map((c) => c.proposal);
}

/**
 * Reader spans sometimes run a token past a sentence end ("32. C" from "...
 * set as 32. C stands for ..."). Cut the span at its first sentence's end
 * when the overflow is the smaller part, then drop trailing punctuation.
 */
function tidySpan(text: string, span: { start: number; end: number }, sentences: Located[]): { start: number; end: number } {
  const { start } = span;
  let { end } = span;
  const first = sentences.find((s) => s.start <= start && start < s.end);
  if (first && end > first.end && end - first.end < first.end - start) end = first.end;
  while (end > start && /[\s.,;:]/u.test(text[end - 1])) end--;
  return { start, end };
}

async function verifyNode(state: QAState, node: NodeContext): Promise<Partial<QAState>> {
  const bySection = groupBySection(state.ctx.sentences);
  const wantsQuantity = expectsQuantity(state.question);
  const verbatim: Proposal[] = [];
  for (const proposal of state.proposals) {
    const { chunk } = proposal.hit;
    const text = chunk.text;
    const exact = proposal.start >= 0 && text.slice(proposal.start, proposal.end) === proposal.span;
    const found = exact ? { start: proposal.start, end: proposal.end } : alignSpan(proposal.span, text);
    if (!found) {
      node.log(`Rejected a span not found verbatim in ${chunk.id}`);
      continue;
    }
    const located = proposal.mode === "reader" ? tidySpan(text, found, sentencesOf(chunk, bySection)) : found;
    const span = text.slice(located.start, located.end);
    if (!/[\p{L}\p{N}]/u.test(span)) continue; // punctuation-only spans answer nothing
    if (proposal.mode === "reader" && wantsQuantity && !containsQuantity(span)) {
      node.log(`Rejected "${span}": the question asks for a quantity`);
      continue;
    }
    verbatim.push({ ...proposal, ...located, span });
  }

  // A reader span must also sit in a sentence that is about the question.
  const spans = verbatim.filter((p) => p.mode === "reader");
  const contexts = spans.map((p) => sentencesAround(p, state.ctx).map((s) => s.text).join(" "));
  const support = await scoreSentences(state, contexts, node);
  spans.forEach((p, i) => {
    p.support = SUPPORT_FLOOR + (1 - SUPPORT_FLOOR) * support[i];
  });

  const unmentioned = unmentionedNames(state.question, state.ctx.index);
  if (unmentioned.length > 0) node.log(`The paper never mentions ${unmentioned.join(", ")}; no candidate can answer`);

  const verified = verbatim
    .map((p) => ({ ...p, confidence: p.relevance * p.evidenceScore * p.support }))
    .sort((a, b) => b.confidence - a.confidence);
  // Ranked as usual, so the closest passage can still be shown, but none may answer.
  if (unmentioned.length > 0) for (const p of verified) p.confidence = 0;
  const best = verified[0] ?? null;
  node.log(
    best
      ? `Best candidate ${best.confidence.toFixed(3)} (threshold ${thresholdFor(best.mode)})`
      : "No candidate survived verification",
  );
  return { proposals: verified, best, unmentioned };
}

function thresholdFor(mode: AnswerMode): number {
  return mode === "reader" ? ABSTAIN_THRESHOLD : SENTENCE_ABSTAIN_THRESHOLD;
}

function routeAfterVerify(state: QAState): string {
  const { best } = state;
  return best && best.confidence >= thresholdFor(best.mode) ? "respond" : "abstain";
}

function respondNode(state: QAState): Partial<QAState> {
  const best = state.best as Proposal;
  const { chunk } = best.hit;
  const context = sentencesAround(best, state.ctx);
  const quote = context.map((s) => s.text).join(" ");
  const anchor = context.find((s) => s.sentence)?.sentence;
  const page = anchor?.page ?? chunk.page;

  const how =
    best.mode === "reader"
      ? `Span extracted verbatim from "${chunk.sectionTitle}" (p. ${page}) by the reader model`
      : `Sentence quoted verbatim from "${chunk.sectionTitle}" (p. ${page})`;
  const relevance = best.hit.rerank !== undefined ? "; passage relevance from the cross-encoder" : "";

  return {
    answer: {
      ...baseAnswer(state),
      status: "answered",
      answer: best.span,
      context: quote,
      confidence: round(best.confidence),
      evidence: [
        {
          paperId: chunk.paperId,
          page,
          section: chunk.sectionTitle,
          quote,
          chunkId: chunk.id,
          sentenceId: anchor?.id,
        },
      ],
      hits: state.hits.slice(0, HITS_KEPT),
      reason: withNotes(`${how}${relevance}.`, state.notes),
    },
  };
}

function abstainNode(state: QAState): Partial<QAState> {
  const { best } = state;
  const evidence: Evidence[] = [];
  let detail =
    state.hits.length === 0
      ? "No passage in the paper matched the question."
      : "No candidate answer could be verified against the paper's text.";
  if (best) {
    const closest = sentencesAround(best, state.ctx);
    const anchor = closest.find((s) => s.sentence)?.sentence;
    const chunk = best.hit.chunk;
    evidence.push({
      paperId: chunk.paperId,
      page: anchor?.page ?? chunk.page,
      section: chunk.sectionTitle,
      quote: closest.map((s) => s.text).join(" "),
      chunkId: chunk.id,
      sentenceId: anchor?.id,
    });
    detail =
      state.unmentioned.length > 0
        ? `It never mentions ${state.unmentioned.join(" or ")}, which the question asks about; the closest passage is shown as the nearest evidence.`
        : `The closest passage scored ${best.confidence.toFixed(2)}, below the ${thresholdFor(best.mode).toFixed(2)} needed to answer; it is shown as the nearest evidence.`;
  }
  return {
    answer: {
      ...baseAnswer(state),
      status: "abstained",
      confidence: round(best?.confidence ?? 0),
      evidence,
      hits: state.hits.slice(0, HITS_KEPT),
      reason: withNotes(`The paper does not appear to answer this question. ${detail}`, state.notes),
    },
  };
}

// ───────────────────────────── Answer assembly ────────────────────────────

function baseAnswer(state: QAState): Omit<Answer, "status" | "confidence" | "evidence" | "hits"> {
  return {
    id: answerId(),
    question: state.question,
    intent: state.intent,
    timingsMs: {},
    createdAt: Date.now(),
  };
}

function answerId(): string {
  const random =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().slice(0, 8)
      : Math.random().toString(36).slice(2, 10);
  return `qa-${Date.now().toString(36)}-${random}`;
}

function uniqueEvidence(items: BriefingItem[]): Evidence[] {
  const seen = new Set<string>();
  const out: Evidence[] = [];
  for (const e of items.flatMap((i) => i.evidence)) {
    const key = `${e.page}|${e.quote}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(e);
  }
  return out;
}

function withNotes(reason: string, notes: string[]): string {
  const unique = [...new Set(notes)];
  return unique.length > 0 ? `${reason} Note: ${unique.join("; ")}.` : reason;
}

const round = (x: number) => Math.round(x * 1000) / 1000;

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

// ─────────────────────────────── Graph wiring ─────────────────────────────

export const QA_NODES: { name: string; label: string; description: string }[] = [
  { name: "classify", label: "Classify", description: "Detect the question's intent (factual, method, results, summary, limitations...)." },
  { name: "compose_from_analysis", label: "Compose", description: "Answer overview questions from the briefing, with its quoted evidence." },
  { name: "retrieve", label: "Retrieve", description: "Hybrid BM25 + embedding search over the paper's passages, fused with RRF." },
  { name: "rerank", label: "Rerank", description: "Cross-encoder rescoring of the top passages against the question." },
  { name: "read", label: "Read", description: "Extractive reader proposes an answer span in each top passage." },
  { name: "verify", label: "Verify", description: "Check the span is verbatim in the paper, the paper mentions what the question names, and the confidence clears the bar." },
  { name: "respond", label: "Respond", description: "Return the span with its sentence, page and section as evidence." },
  { name: "abstain", label: "Abstain", description: "Decline to answer and show the closest passage instead." },
];

export function buildQAGraph(): CompiledGraph<QAState> {
  const spec = (name: string) => {
    const node = QA_NODES.find((n) => n.name === name);
    return { label: node?.label, description: node?.description };
  };
  return new StateGraph<QAState>()
    .addNode("classify", classifyNode, spec("classify"))
    .addNode("compose_from_analysis", composeNode, spec("compose_from_analysis"))
    .addNode("retrieve", retrieveNode, spec("retrieve"))
    .addNode("rerank", rerankNode, spec("rerank"))
    .addNode("read", readNode, spec("read"))
    .addNode("verify", verifyNode, spec("verify"))
    .addNode("respond", respondNode, spec("respond"))
    .addNode("abstain", abstainNode, spec("abstain"))
    .addEdge(START, "classify")
    .addConditionalEdges("classify", routeAfterClassify, ["compose_from_analysis", "retrieve"])
    .addEdge("compose_from_analysis", END)
    .addConditionalEdges("retrieve", (s) => (s.hits.length > 0 ? "rerank" : "abstain"), ["rerank", "abstain"])
    .addEdge("rerank", "read")
    .addEdge("read", "verify")
    .addConditionalEdges("verify", routeAfterVerify, ["respond", "abstain"])
    .addEdge("respond", END)
    .addEdge("abstain", END)
    .compile();
}

let compiled: CompiledGraph<QAState> | null = null;

export async function answerQuestion(
  question: string,
  ctx: QAContext,
  onEvent?: (e: TraceEvent) => void,
): Promise<Answer> {
  compiled ??= buildQAGraph();
  const timings: Record<string, number> = {};
  const started = performance.now();

  const final = await compiled.run(
    {
      question: question.trim(),
      ctx,
      intent: "factual",
      queryEmbedding: null,
      hits: [],
      proposals: [],
      best: null,
      unmentioned: [],
      answer: null,
      notes: [],
    },
    {
      onEvent: (event) => {
        if (event.type === "node:end" && event.durationMs !== undefined) {
          timings[event.node] = Math.round(event.durationMs);
        }
        onEvent?.(event);
      },
    },
  );

  if (!final.answer) throw new Error("Question answering finished without an answer");
  return { ...final.answer, timingsMs: { ...timings, total: Math.round(performance.now() - started) } };
}

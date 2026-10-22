/**
 * Candidate ranking for topic search: a transparent, weighted blend of four
 * signals, then Maximal Marginal Relevance (MMR) to diversify the order.
 *
 *   final = 0.35 lexical + 0.40 semantic + 0.15 recency + 0.10 impact
 *
 *  - lexical  BM25 over title + abstract (title counted twice, a light
 *             BM25F), min-max normalised over the result set. Exact
 *             terminology matters in science: method names, datasets, acronyms
 *             a small embedding model may not know.
 *  - semantic cosine between query and paper embeddings (MiniLM). The largest
 *             weight, because it catches paraphrases a keyword match misses
 *             ("KV cache quantization" for "kv cache compression").
 *  - recency  exponential decay with a 3-year half-life: the field moves fast,
 *             but a 3-year-old paper still keeps half of its recency credit.
 *  - impact   log1p(citations) / max over the set. Citations are heavy-tailed,
 *             lag behind publication and are zero for new preprints, so they
 *             are only a weak prior.
 *
 * Without an embedding model the semantic weight is redistributed:
 * 0.65 lexical + 0.20 recency + 0.15 impact.
 *
 * MMR then picks results greedily, trading relevance against similarity to
 * what is already picked (lambda = 0.75 leans towards relevance), so five
 * near-duplicate papers do not crowd out a distinct line of work.
 * Everything is deterministic: same inputs, same order.
 */
import type { Candidate, CandidateScores, EmbedFn, PaperMeta } from "@/types";
import { BM25 } from "@/nlp/bm25";
import { mmrSelect } from "@/nlp/mmr";
import { tokenize } from "@/nlp/text";
import { cosine } from "@/nlp/vector";

interface Weights {
  lexical: number;
  semantic: number;
  recency: number;
  impact: number;
}

export const RANK_WEIGHTS: { withEmbeddings: Weights; lexicalOnly: Weights } = {
  withEmbeddings: { lexical: 0.35, semantic: 0.4, recency: 0.15, impact: 0.1 },
  lexicalOnly: { lexical: 0.65, semantic: 0, recency: 0.2, impact: 0.15 },
};

const RECENCY_HALF_LIFE_YEARS = 3;
const MMR_LAMBDA = 0.75;
const MS_PER_YEAR = 365.25 * 24 * 3600 * 1000;
/** Abstract prefix sent to the embedder; MiniLM truncates at 256 word pieces anyway. */
const EMBED_ABSTRACT_CHARS = 1000;
/**
 * Observed cosine range of all-MiniLM-L6-v2 for a short query against an
 * abstract: unrelated pairs mostly fall below ~0.15, close matches reach
 * ~0.75. Rescaling that band to 0..1 keeps the semantic bar comparable to
 * the lexical one while preserving absolute meaning (unlike min-max, an
 * all-irrelevant result set stays low).
 */
const COSINE_FLOOR = 0.15;
const COSINE_CEIL = 0.75;

interface Prepared {
  meta: PaperMeta;
  titleTokens: string[];
  tokenSet: Set<string>;
  scores: CandidateScores;
  vector?: Float32Array;
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

export function minMaxNormalize(values: number[]): number[] {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  // All equal: every document matched equally well (1) or not at all (0).
  if (max - min < 1e-12) return values.map(() => (max > 0 ? 1 : 0));
  return values.map((v) => (v - min) / (max - min));
}

/** ISO date or year -> epoch ms; partial dates use the middle of their period to halve the worst-case error. */
function publicationTime(meta: PaperMeta): number | null {
  const match = /^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?/.exec(meta.published ?? "");
  const year = match ? Number(match[1]) : meta.year;
  if (!year) return null;
  const month = match?.[2] ? Number(match[2]) : undefined;
  const day = match?.[3] ? Number(match[3]) : undefined;
  if (month === undefined) return Date.UTC(year, 6, 1);
  return Date.UTC(year, month - 1, day ?? 15);
}

/** 0.5 ^ (age / half-life); unknown dates get no recency credit rather than a guess. */
export function recencyScore(meta: PaperMeta, now: Date): number {
  const time = publicationTime(meta);
  if (time === null) return 0;
  const ageYears = Math.max(0, (now.getTime() - time) / MS_PER_YEAR);
  return Math.pow(0.5, ageYears / RECENCY_HALF_LIFE_YEARS);
}

function impactScores(metas: PaperMeta[]): number[] {
  const logs = metas.map((m) => Math.log1p(Math.max(0, m.citationCount ?? 0)));
  const max = Math.max(0, ...logs);
  return logs.map((v) => (max > 0 ? v / max : 0));
}

function calibrateCosine(similarity: number): number {
  return clamp01((similarity - COSINE_FLOOR) / (COSINE_CEIL - COSINE_FLOOR));
}

function embeddingText(meta: PaperMeta): string {
  const abstract = (meta.abstract ?? "").slice(0, EMBED_ABSTRACT_CHARS);
  return abstract ? `${meta.title}. ${abstract}` : meta.title;
}

/** Query + paper vectors, or null when the model is unavailable or misbehaves. */
async function embedAll(
  embed: EmbedFn | null,
  query: string,
  metas: PaperMeta[],
): Promise<Float32Array[] | null> {
  if (!embed) return null;
  try {
    const vectors = await embed([query, ...metas.map(embeddingText)]);
    return vectors.length === metas.length + 1 ? vectors : null;
  } catch {
    return null;
  }
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 0;
  let shared = 0;
  for (const token of a) if (b.has(token)) shared++;
  return shared / (a.size + b.size - shared);
}

/** Code-unit order, unlike localeCompare, is identical in every browser and locale. */
function compareIds(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function uniqueById(metas: PaperMeta[]): PaperMeta[] {
  const seen = new Set<string>();
  return metas.filter((meta) => (seen.has(meta.id) ? false : (seen.add(meta.id), true)));
}

function formatCount(n: number): string {
  if (n >= 10_000) return `${Math.round(n / 1000)}k`;
  if (n >= 1_000) return `${(n / 1000).toFixed(1).replace(/\.0$/, "")}k`;
  return String(n);
}

/** One short, honest sentence fragment built from the strongest signals. */
function explain(item: Prepared, queryTokens: Set<string>): string {
  const { scores, meta } = item;
  const reasons: string[] = [];
  const titleHits = [...queryTokens].filter((t) => item.titleTokens.includes(t)).length;

  if (queryTokens.size > 0 && titleHits === queryTokens.size) reasons.push("title matches every query term");
  else if (scores.lexical >= 0.7) reasons.push("strong keyword match");
  else if (scores.lexical >= 0.35) reasons.push("partial keyword match");

  if (scores.semantic >= 0.7) reasons.push("very close in meaning");
  else if (scores.semantic >= 0.45) reasons.push("related in meaning");

  const citations = meta.citationCount ?? 0;
  if (citations >= 100) reasons.push(`${formatCount(citations)} citations`);

  const parts = reasons.slice(0, 2);
  if (meta.year) parts.push(scores.recency >= 0.8 ? `recent (${meta.year})` : String(meta.year));
  const text = parts.join(", ") || "returned by the search index";
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** Scores, diversifies and explains search candidates. */
export async function rankCandidates(
  query: string,
  metas: PaperMeta[],
  embed: EmbedFn | null,
  opts: { now?: Date; limit?: number } = {},
): Promise<Candidate[]> {
  const papers = uniqueById(metas);
  if (papers.length === 0) return [];
  const now = opts.now ?? new Date();
  const limit = Math.max(0, Math.min(papers.length, opts.limit ?? papers.length));

  const queryTokens: string[] = tokenize(query);
  const titleTokens: string[][] = papers.map((m) => tokenize(m.title));
  const abstractTokens: string[][] = papers.map((m) => tokenize(m.abstract ?? ""));
  // Counting the title twice is a light-weight BM25F: title words are stronger evidence.
  const documents = papers.map((_, i) => [...titleTokens[i], ...titleTokens[i], ...abstractTokens[i]]);
  const lexical =
    queryTokens.length > 0 ? minMaxNormalize(new BM25(documents).score(queryTokens)) : papers.map(() => 0);

  const vectors = await embedAll(embed, query, papers);
  const semantic = vectors ? papers.map((_, i) => calibrateCosine(cosine(vectors[0], vectors[i + 1]))) : null;
  const weights = semantic ? RANK_WEIGHTS.withEmbeddings : RANK_WEIGHTS.lexicalOnly;
  const impact = impactScores(papers);

  const prepared: Prepared[] = papers.map((meta, i) => {
    const parts = {
      lexical: clamp01(lexical[i] ?? 0),
      semantic: semantic ? semantic[i] : 0,
      recency: recencyScore(meta, now),
      impact: impact[i],
    };
    const final =
      weights.lexical * parts.lexical +
      weights.semantic * parts.semantic +
      weights.recency * parts.recency +
      weights.impact * parts.impact;
    return {
      meta,
      titleTokens: titleTokens[i],
      tokenSet: new Set([...titleTokens[i], ...abstractTokens[i]]),
      scores: { ...parts, final },
      vector: vectors ? vectors[i + 1] : undefined,
    };
  });

  // Stable pre-order (score, then id) so MMR tie-breaks never depend on input order.
  prepared.sort((a, b) => b.scores.final - a.scores.final || compareIds(a.meta.id, b.meta.id));
  const similarity = (a: Prepared, b: Prepared): number =>
    a.vector && b.vector ? cosine(a.vector, b.vector) : jaccard(a.tokenSet, b.tokenSet);
  const ordered: Prepared[] = mmrSelect<Prepared>(
    prepared,
    (p: Prepared) => p.scores.final,
    similarity,
    limit,
    MMR_LAMBDA,
  );

  const querySet = new Set<string>(queryTokens);
  return ordered.map((item: Prepared, position: number) => ({
    meta: item.meta,
    scores: item.scores,
    rank: position + 1,
    why: explain(item, querySet),
  }));
}

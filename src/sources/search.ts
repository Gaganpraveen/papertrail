/**
 * Topic search over arXiv through two independent indexes, queried in parallel.
 *
 *  - DataCite (client arxiv.content): every arXiv paper with clean metadata,
 *    matched on title terms and sorted by relevance.
 *  - OpenAlex (works hosted at the arXiv source): full-text matching, which
 *    finds papers whose titles do not contain the query words, plus
 *    citation counts for ranking.
 *
 * Results are merged on the arXiv ID (DataCite facts first, OpenAlex fills
 * gaps). Either index may fail; the search only fails when both do.
 *
 * `broadenQuery` is the recovery step of the search agent: when a query finds
 * nothing it first drops conversational filler, then keeps only the few most
 * specific terms.
 */
import type { PaperMeta } from "@/types";
import { STOPWORDS } from "@/nlp/text";
import { DATACITE_API, mapDataCite, withArxivLinks } from "./arxiv";
import { asRecords, fetchJson, isRecord, mostInformativeError, SourceError, type FetchOptions } from "./http";
import { mapOpenAlexWork, mergeMeta, OPENALEX_API, OPENALEX_ARXIV_SOURCE, OPENALEX_SELECT } from "./openalex";

const DEFAULT_LIMIT = 25;
const MAX_LIMIT = 100;

// ───────────────────────────── Query sanitising ─────────────────────────────

function queryWords(text: string): string[] {
  return text.toLowerCase().match(/[\p{L}\p{N}]+(?:-[\p{L}\p{N}]+)*/gu) ?? [];
}

/**
 * DataCite passes `query` to Elasticsearch query_string syntax, so operators
 * and special characters are removed and stopwords dropped (titles must match
 * every remaining term, so a stray "the" would only cost recall).
 */
export function dataCiteTerms(query: string): string {
  const words = queryWords(query.replace(/-/g, " ")).filter((w) => !STOPWORDS.has(w));
  return words.join(" ");
}

/** OpenAlex filter values cannot contain its own separators (",", "|", ":", "!", "+"). */
export function openAlexTerms(query: string): string {
  return query
    .replace(/[,|:!+"]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

// ─────────────────────────────── Index queries ───────────────────────────────

export function dataCiteSearchUrl(terms: string, limit: number): string {
  const params = new URLSearchParams({
    query: `titles.title:(${terms})`,
    "client-id": "arxiv.content",
    "page[size]": String(limit),
    sort: "relevance",
  });
  return `${DATACITE_API}/dois?${params.toString()}`;
}

export function openAlexSearchUrl(terms: string, limit: number): string {
  const params = new URLSearchParams({
    filter: `fulltext.search:${terms},locations.source.id:${OPENALEX_ARXIV_SOURCE}`,
    per_page: String(limit),
    select: OPENALEX_SELECT,
  });
  return `${OPENALEX_API}/works?${params.toString()}`;
}

async function searchDataCite(query: string, limit: number, opts: FetchOptions): Promise<PaperMeta[]> {
  const terms = dataCiteTerms(query);
  if (!terms) return [];
  const json = await fetchJson(dataCiteSearchUrl(terms, limit), opts);
  const items = isRecord(json) ? asRecords(json.data) : [];
  return items.map(mapDataCite).filter((meta): meta is PaperMeta => meta !== null);
}

async function searchOpenAlex(query: string, limit: number, opts: FetchOptions): Promise<PaperMeta[]> {
  const terms = openAlexTerms(query);
  if (!terms) return [];
  const json = await fetchJson(openAlexSearchUrl(terms, limit), opts);
  const items = isRecord(json) ? asRecords(json.results) : [];
  return items
    .map(mapOpenAlexWork)
    .filter((meta): meta is PaperMeta => meta !== null)
    .map((meta) => (meta.arxivId ? withArxivLinks(meta, meta.arxivId) : meta));
}

function mergeKey(meta: PaperMeta): string {
  return meta.arxivId ? `arxiv:${meta.arxivId}` : meta.id;
}

/** De-duplicates by arXiv ID; DataCite records stay primary and OpenAlex enriches them. */
export function mergeSearchResults(primary: PaperMeta[], secondary: PaperMeta[]): PaperMeta[] {
  const merged = new Map<string, PaperMeta>();
  for (const meta of primary) if (!merged.has(mergeKey(meta))) merged.set(mergeKey(meta), meta);
  for (const meta of secondary) {
    const key = mergeKey(meta);
    const existing = merged.get(key);
    merged.set(key, existing ? mergeMeta(existing, meta) : meta);
  }
  return [...merged.values()];
}

/**
 * Searches arXiv papers by topic. `limit` is the page size requested from
 * each index, so up to 2 x limit unique papers can come back before ranking.
 */
export async function searchArxiv(
  query: string,
  opts: { limit?: number; signal?: AbortSignal } = {},
): Promise<PaperMeta[]> {
  const limit = Math.min(MAX_LIMIT, Math.max(1, Math.round(opts.limit ?? DEFAULT_LIMIT)));
  const fetchOpts: FetchOptions = { signal: opts.signal };
  const [dataCite, openAlex] = await Promise.allSettled([
    searchDataCite(query, limit, fetchOpts),
    searchOpenAlex(query, limit, fetchOpts),
  ]);
  if (opts.signal?.aborted) {
    throw opts.signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
  }
  if (dataCite.status === "rejected" && openAlex.status === "rejected") {
    throw mostInformativeError(
      [dataCite.reason, openAlex.reason],
      new SourceError("Both search indexes failed", "network"),
    );
  }
  return mergeSearchResults(
    dataCite.status === "fulfilled" ? dataCite.value : [],
    openAlex.status === "fulfilled" ? openAlex.value : [],
  );
}

// ───────────────────────────── Query broadening ─────────────────────────────

/** Conversational wrappers around the actual topic. */
const FILLER_PATTERNS: RegExp[] = [
  // "recent work on", "latest advances in"
  /\b(?:recent|latest|new|current|existing)\s+(?:works?|research|papers?|advances|developments|progress|literature|studies)\s+(?:on|in|about|for|regarding|into)\b/g,
  // "papers about", "literature related to"
  /\b(?:papers?|articles?|works?|research|studies|literature|publications?)\s+(?:on|about|regarding|related\s+to|in|into)\b/g,
  /\b(?:find|show|give|get|recommend|suggest)\s+(?:me\s+)?/g,
  /\bi\s+(?:want|need|would\s+like)\s+(?:to\s+(?:read|find|know|learn)(?:\s+about)?)?/g,
  /\b(?:looking|searching)\s+for\b/g,
  /\b(?:what\s+(?:is|are)|how\s+(?:to|do|does|can))\b/g,
  /\bstate[\s-]of[\s-]the[\s-]art\b/g,
];

// prettier-ignore
const FILLER_WORDS = new Set([
  "paper", "papers", "work", "works", "research", "study", "studies", "article", "articles",
  "literature", "publication", "publications", "recent", "latest", "please", "anything", "something",
]);

/**
 * Words that appear in a large share of ML paper titles and therefore carry
 * little signal on their own (a hand-made stand-in for low IDF).
 */
// prettier-ignore
const GENERIC_TERMS = new Set([
  "approach", "approaches", "method", "methods", "methodology", "technique", "techniques",
  "framework", "frameworks", "system", "systems", "model", "models", "modeling", "analysis",
  "novel", "new", "efficient", "effective", "improved", "improving", "better", "using", "based",
  "via", "towards", "toward", "task", "tasks", "problem", "problems", "application",
  "applications", "performance", "learning", "deep", "neural", "network", "networks", "data", "large",
  "language",
]);

function stripFiller(query: string): string[] {
  let text = query.toLowerCase();
  for (const pattern of FILLER_PATTERNS) text = text.replace(pattern, " ");
  return queryWords(text).filter((w) => !STOPWORDS.has(w) && !FILLER_WORDS.has(w));
}

/**
 * Specificity without corpus statistics: longer words tend to be rarer
 * (Zipf's law of abbreviation), tokens with digits are usually names
 * ("gpt-4", "llama2"), and generic research vocabulary is pushed to the back.
 */
function specificity(term: string): number {
  const lengthScore = Math.min(term.length, 12);
  const digitBonus = /\d/.test(term) ? 3 : 0;
  const genericPenalty = GENERIC_TERMS.has(term) ? 6 : 0;
  return lengthScore + digitBonus - genericPenalty;
}

/** Keeps the 2-3 most specific terms, in their original order so phrases stay readable. */
function keepSpecificTerms(terms: string[]): string[] {
  const keep = terms.length >= 5 ? 3 : 2;
  if (terms.length <= keep) return terms;
  const chosen = new Set(
    terms
      .map((term, position) => ({ term, position, score: specificity(term) }))
      .sort((a, b) => b.score - a.score || a.position - b.position)
      .slice(0, keep)
      .map((t) => t.position),
  );
  return terms.filter((_, position) => chosen.has(position));
}

function firstNovel(candidates: string[], seen: (string | null)[]): string | null {
  return candidates.find((c) => c.length > 0 && !seen.includes(c)) ?? null;
}

/**
 * Attempt 1 strips filler ("recent work on", "papers about", stopwords);
 * attempt 2 keeps the 2-3 most specific terms. Returns null when an attempt
 * would not change the query (another search would be wasted) or after attempt 2.
 */
export function broadenQuery(query: string, attempt: number): string | null {
  const original = queryWords(query).join(" ");
  const stripped = stripFiller(query);
  const withoutFiller = stripped.join(" ");
  const specific = keepSpecificTerms(stripped).join(" ");

  if (attempt === 1) return firstNovel([withoutFiller, specific], [original]);
  if (attempt === 2) return firstNovel([specific], [original, broadenQuery(query, 1)]);
  return null;
}

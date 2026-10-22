/**
 * The search agent: finds arXiv papers for a free-text topic.
 *
 *   understand -> search ─┬─ results ───────────────> rank -> END
 *                  ^      └─ nothing & can broaden ─> broaden ─┐
 *                  └───────────────────────────────────────────┘
 *
 * A self-correcting loop: when the indexes return nothing, the query is
 * broadened (first strip conversational filler, then keep only the most
 * specific terms) and searched again, at most three attempts in total.
 * Results are ranked by a blend of BM25, embedding similarity, recency and
 * citation impact, then diversified with MMR (sources/rank.ts).
 */
import { END, START, StateGraph, type CompiledGraph, type NodeContext } from "@/agent/graph";
import type { MLApi } from "@/ml/client";
import { rankCandidates } from "@/sources/rank";
import { broadenQuery, searchArxiv } from "@/sources/search";
import type { Candidate, PaperMeta, TraceEvent } from "@/types";

export const SEARCH_NODES: { name: string; label: string; description: string }[] = [
  { name: "understand", label: "Understand", description: "Normalise the topic into a search query" },
  { name: "search", label: "Search", description: "Query DataCite and OpenAlex arXiv indexes in parallel, merge duplicates" },
  { name: "broaden", label: "Broaden", description: "No hits: drop filler words, then keep only the most specific terms" },
  { name: "rank", label: "Rank", description: "Blend BM25, semantic, recency and impact scores; diversify with MMR" },
];

/** Total searches allowed, including the original query. */
export const MAX_SEARCH_ATTEMPTS = 3;
/** Page size per index; up to twice this many unique papers reach the ranker. */
const SEARCH_LIMIT = 25;
/** Candidates shown to the user. */
const RESULT_LIMIT = 20;

interface SearchState {
  query: string;
  /** The query sent on the latest search. */
  current: string;
  attempts: string[];
  results: PaperMeta[];
  candidates: Candidate[];
  exhausted: boolean;
}

function spec(name: string): { label: string; description: string } {
  const node = SEARCH_NODES.find((n) => n.name === name)!;
  return { label: node.label, description: node.description };
}

function understand(state: SearchState, ctx: NodeContext): Partial<SearchState> {
  const query = state.query.replace(/\s+/g, " ").trim();
  if (!query) throw new Error("Type a topic to search for");
  ctx.log(`Searching arXiv for "${query}"`);
  return { query, current: query };
}

async function search(state: SearchState, ctx: NodeContext): Promise<Partial<SearchState>> {
  const results = await searchArxiv(state.current, { limit: SEARCH_LIMIT, signal: ctx.signal });
  ctx.log(`${results.length} papers for "${state.current}"`);
  return { results, attempts: [...state.attempts, state.current] };
}

function routeAfterSearch(state: SearchState): string {
  if (state.results.length > 0) return "rank";
  return state.attempts.length < MAX_SEARCH_ATTEMPTS ? "broaden" : END;
}

/** Tries successive broadening strategies until one yields a query not searched yet. */
function broaden(state: SearchState, ctx: NodeContext): Partial<SearchState> {
  for (let attempt = state.attempts.length; attempt < MAX_SEARCH_ATTEMPTS; attempt++) {
    const next = broadenQuery(state.query, attempt);
    if (next && !state.attempts.includes(next)) {
      ctx.log(`Retrying with "${next}"`);
      return { current: next };
    }
  }
  ctx.log("The query cannot be broadened further");
  return { exhausted: true };
}

function routeAfterBroaden(state: SearchState): string {
  return state.exhausted ? END : "search";
}

function makeRank(deps: { ml: MLApi }) {
  return async (state: SearchState, ctx: NodeContext): Promise<Partial<SearchState>> => {
    const candidates = await rankCandidates(state.query, state.results, deps.ml.embed, { limit: RESULT_LIMIT });
    const semantic = candidates.some((c) => c.scores.semantic > 0);
    ctx.log(`Ranked ${candidates.length} candidates (${semantic ? "hybrid" : "keyword-only"} scoring)`);
    return { candidates };
  };
}

export function buildSearchGraph(deps: { ml: MLApi }): CompiledGraph<SearchState> {
  return new StateGraph<SearchState>()
    .addNode("understand", understand, spec("understand"))
    .addNode("search", search, spec("search"))
    .addNode("broaden", broaden, spec("broaden"))
    .addNode("rank", makeRank(deps), spec("rank"))
    .addEdge(START, "understand")
    .addEdge("understand", "search")
    .addConditionalEdges("search", routeAfterSearch, ["rank", "broaden", END])
    .addConditionalEdges("broaden", routeAfterBroaden, ["search", END])
    .addEdge("rank", END)
    .compile();
}

export async function runTopicSearch(
  query: string,
  deps: { ml: MLApi },
  onEvent?: (e: TraceEvent) => void,
  signal?: AbortSignal,
): Promise<{ candidates: Candidate[]; attempts: string[] }> {
  const final = await buildSearchGraph(deps).run(
    { query, current: query, attempts: [], results: [], candidates: [], exhausted: false },
    { onEvent, signal },
  );
  return { candidates: final.candidates, attempts: final.attempts };
}

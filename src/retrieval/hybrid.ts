/**
 * Hybrid retrieval: BM25 (exact terms) + dense cosine (meaning), fused with
 * Reciprocal Rank Fusion.
 *
 * The two retrievers fail differently. BM25 nails rare exact tokens that a
 * small embedding model blurs ("GSM8K", "A100", "group size 32"); embeddings
 * catch paraphrases that share no words ("memory savings" vs "reduces peak
 * memory usage"). Their scores live on incomparable scales, so instead of
 * blending scores we blend ranks: RRF gives each document sum(1 / (k + rank))
 * over the lists it appears in. k = 60 (Cormack et al., 2009) flattens the
 * head of each list so neither retriever can dominate on its own.
 *
 * Without embeddings the index is simply BM25; with an empty lexical match it
 * is simply dense search.
 */
import type { Chunk, RetrievalHit } from "@/types";
import { BM25 } from "@/nlp/bm25";
import { tokenize } from "@/nlp/text";
import { cosine } from "@/nlp/vector";
import { embeddingText } from "./chunker";

export const RRF_K = 60;

/**
 * Only the head of each ranking takes part in fusion: deep ranks carry noise
 * (every chunk has *some* cosine with the query), not evidence.
 */
const FUSION_DEPTH = 100;

/** Fuses ranked id lists (best first) into id -> sum over lists of 1 / (k + rank). */
export function reciprocalRankFusion(rankings: string[][], k: number = RRF_K): Map<string, number> {
  const fused = new Map<string, number>();
  for (const ranking of rankings) {
    ranking.forEach((id, i) => {
      fused.set(id, (fused.get(id) ?? 0) + 1 / (k + i + 1));
    });
  }
  return fused;
}

export interface SearchOptions {
  k?: number;
  filter?: (chunk: Chunk) => boolean;
}

/** Positions sorted by descending score, keeping only positive scores. */
function rankPositive(positions: number[], scores: ArrayLike<number>): number[] {
  return positions
    .filter((p) => scores[p] > 0)
    .sort((a, b) => scores[b] - scores[a] || a - b)
    .slice(0, FUSION_DEPTH);
}

export class HybridIndex {
  private readonly bm25: BM25;
  private readonly embeddings: Float32Array[] | null;

  constructor(
    readonly chunks: Chunk[],
    embeddings: Float32Array[] | null,
  ) {
    this.bm25 = new BM25(chunks.map((c) => tokenize(embeddingText(c))));
    // A stale cache could hand us vectors for a different chunking; searching
    // lexically is better than pairing passages with the wrong vectors.
    this.embeddings = embeddings && embeddings.length === chunks.length ? embeddings : null;
  }

  /** True when dense vectors are available for every chunk. */
  get hasDense(): boolean {
    return this.embeddings !== null;
  }

  search(query: string, queryEmbedding: Float32Array | null, opts: SearchOptions = {}): RetrievalHit[] {
    const { k = 20, filter } = opts;
    const positions = this.chunks.map((_, i) => i).filter((i) => !filter || filter(this.chunks[i]));
    if (positions.length === 0) return [];

    const queryTokens = tokenize(query);
    const bm25 = queryTokens.length > 0 ? this.bm25.score(queryTokens) : new Array<number>(this.chunks.length).fill(0);
    const dense = this.denseScores(queryEmbedding);

    const bm25Ranking = rankPositive(positions, bm25);
    const denseRanking = dense ? rankPositive(positions, dense) : [];

    const idOf = (p: number) => this.chunks[p].id;
    const fused = reciprocalRankFusion([bm25Ranking.map(idOf), denseRanking.map(idOf)]);

    const bm25Rank = new Map(bm25Ranking.map((p, i) => [p, i + 1]));
    const denseRank = new Map(denseRanking.map((p, i) => [p, i + 1]));

    return positions
      .filter((p) => fused.has(idOf(p)))
      .sort(
        (a, b) =>
          (fused.get(idOf(b)) ?? 0) - (fused.get(idOf(a)) ?? 0) ||
          bm25[b] - bm25[a] ||
          (dense ? dense[b] - dense[a] : 0) ||
          a - b,
      )
      .slice(0, k)
      .map((p, i) => ({
        chunk: this.chunks[p],
        bm25: bm25[p],
        dense: dense ? dense[p] : 0,
        rrf: fused.get(idOf(p)) ?? 0,
        rank: { bm25: bm25Rank.get(p), dense: denseRank.get(p), final: i + 1 },
      }));
  }

  private denseScores(queryEmbedding: Float32Array | null): number[] | null {
    if (!this.embeddings || !queryEmbedding) return null;
    return this.embeddings.map((v) => cosine(queryEmbedding, v));
  }
}

/**
 * Grouping papers (and sentences) by meaning.
 *
 * Representation: every text becomes a vector, either a sentence-encoder
 * embedding or, without a model, a TF-IDF vector. Similarity is cosine and
 * distance is 1 - cosine.
 *
 * Agglomerative clustering (average linkage, also called UPGMA) starts with
 * every item in its own cluster and repeatedly merges the two clusters whose
 * members are, on average, closest. Recording the merges gives a dendrogram
 * that can be cut into any number of clusters without re-running.
 *
 * The number of clusters is chosen with the silhouette score: for each item,
 * a = mean distance to its own cluster, b = mean distance to the nearest
 * other cluster, s = (b - a) / max(a, b). s near 1 means tight, well
 * separated clusters; near 0 means overlapping ones. We keep the k with the
 * highest mean silhouette.
 */
import type { EmbedFn } from "@/types";
import { similarityMatrix } from "@/nlp/lexrank";
import { tokenize } from "@/nlp/text";
import { cosineSparse, tfidf } from "@/nlp/tfidf";

// ─────────────────────────────── Vector spaces ──────────────────────────────

export type Representation = "embedding" | "tfidf";

export interface TextSpace {
  kind: Representation;
  /** Symmetric cosine similarities, ones on the diagonal. */
  similarity: number[][];
}

/** Cosine similarity matrix of sparse vectors. */
export function sparseSimilarity(vectors: Map<string, number>[]): number[][] {
  const n = vectors.length;
  const matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    matrix[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      const s = cosineSparse(vectors[i], vectors[j]);
      matrix[i][j] = s;
      matrix[j][i] = s;
    }
  }
  return matrix;
}

/** TF-IDF cosine similarity between texts, with idf estimated over the texts themselves. */
export function lexicalSimilarity(texts: string[]): number[][] {
  return sparseSimilarity(tfidf(texts.map(tokenize)).vectors);
}

/**
 * Embeds the texts when a model is available and falls back to TF-IDF when it
 * is not (or fails), so callers always get a usable similarity matrix and know
 * which kind it is (thresholds differ between the two).
 */
export async function textSpace(texts: string[], embed: EmbedFn | null): Promise<TextSpace> {
  if (embed && texts.length > 0) {
    try {
      const vectors = await embed(texts);
      if (vectors.length === texts.length && vectors.every((v) => v.length > 0)) {
        return { kind: "embedding", similarity: similarityMatrix(vectors) };
      }
    } catch {
      // A missing or failing model is not fatal: the lexical space still works.
    }
  }
  return { kind: "tfidf", similarity: lexicalSimilarity(texts) };
}

export function toDistance(similarity: number[][]): number[][] {
  return similarity.map((row) => row.map((s) => Math.max(0, 1 - s)));
}

// ──────────────────────────── Average linkage ───────────────────────────────

export interface Merge {
  a: number;
  b: number;
  distance: number;
}

/**
 * Builds the dendrogram. Clusters are identified by the index of their first
 * member; after merging b into a, a's distances are updated with the
 * Lance-Williams formula for average linkage:
 *   d(a ∪ b, x) = (|a| d(a, x) + |b| d(b, x)) / (|a| + |b|).
 */
export function averageLinkage(distance: number[][]): Merge[] {
  const n = distance.length;
  const d = distance.map((row) => [...row]);
  const size = new Array<number>(n).fill(1);
  const active = new Set(Array.from({ length: n }, (_, i) => i));
  const merges: Merge[] = [];

  while (active.size > 1) {
    let best: Merge | null = null;
    for (const i of active) {
      for (const j of active) {
        if (j <= i) continue;
        if (!best || d[i][j] < best.distance) best = { a: i, b: j, distance: d[i][j] };
      }
    }
    if (!best) break;
    const { a, b } = best;
    for (const x of active) {
      if (x === a || x === b) continue;
      const merged = (size[a] * d[a][x] + size[b] * d[b][x]) / (size[a] + size[b]);
      d[a][x] = merged;
      d[x][a] = merged;
    }
    size[a] += size[b];
    active.delete(b);
    merges.push(best);
  }
  return merges;
}

/** Replays merges until `k` clusters remain; labels are 0..k-1 in order of first appearance. */
export function cutTree(merges: Merge[], n: number, k: number): number[] {
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  const steps = Math.max(0, Math.min(merges.length, n - Math.max(1, k)));
  for (let s = 0; s < steps; s++) parent[find(merges[s].b)] = find(merges[s].a);
  return relabel(Array.from({ length: n }, (_, i) => find(i)));
}

/** Merges only while the clusters are closer than `maxDistance` (a similarity floor for the group). */
export function cutAtDistance(merges: Merge[], n: number, maxDistance: number): number[] {
  const steps = merges.findIndex((m) => m.distance > maxDistance);
  return cutTree(merges, n, n - (steps === -1 ? merges.length : steps));
}

function relabel(roots: number[]): number[] {
  const ids = new Map<number, number>();
  return roots.map((r) => {
    if (!ids.has(r)) ids.set(r, ids.size);
    return ids.get(r) as number;
  });
}

// ──────────────────────────────── Silhouette ────────────────────────────────

/** Mean silhouette in [-1, 1]; 0 when there is only one cluster (it is undefined there). */
export function silhouette(distance: number[][], labels: number[]): number {
  const n = labels.length;
  const clusters = new Set(labels);
  if (clusters.size < 2 || clusters.size >= n) return 0;

  let total = 0;
  for (let i = 0; i < n; i++) {
    const own = labels[i];
    const sums = new Map<number, { sum: number; count: number }>();
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const entry = sums.get(labels[j]) ?? { sum: 0, count: 0 };
      entry.sum += distance[i][j];
      entry.count += 1;
      sums.set(labels[j], entry);
    }
    const ownEntry = sums.get(own);
    // By convention a point alone in its cluster scores 0.
    if (!ownEntry) continue;
    const a = ownEntry.sum / ownEntry.count;
    let b = Number.POSITIVE_INFINITY;
    for (const [label, { sum, count }] of sums) if (label !== own) b = Math.min(b, sum / count);
    const denominator = Math.max(a, b);
    total += denominator > 0 ? (b - a) / denominator : 0;
  }
  return total / n;
}

// ─────────────────────────────── Choosing k ─────────────────────────────────

export interface Clustering {
  labels: number[];
  k: number;
  silhouette: number;
  /** Silhouette of every k that was tried, for the UI and for debugging. */
  candidates: { k: number; silhouette: number }[];
}

/** Upper bound on themes: more than six stops reading like a survey and starts reading like a list. */
const MAX_THEMES = 6;
/** A larger k must beat a smaller one by this much: prefer fewer, broader themes on near-ties. */
const SILHOUETTE_TIE = 0.01;

/**
 * Clusters papers from their similarity matrix. One paper is one cluster;
 * two papers are two clusters (with a single pair there is nothing to compare
 * a split against); otherwise every k in [2, min(6, n - 1)] is scored.
 */
export function clusterPapers(similarity: number[][]): Clustering {
  const n = similarity.length;
  if (n <= 2) {
    const labels = Array.from({ length: n }, (_, i) => i);
    return { labels, k: n, silhouette: 0, candidates: [] };
  }
  const distance = toDistance(similarity);
  const merges = averageLinkage(distance);
  const candidates: Clustering["candidates"] = [];
  let best: Clustering | null = null;
  for (let k = 2; k <= Math.min(MAX_THEMES, n - 1); k++) {
    const labels = cutTree(merges, n, k);
    const score = silhouette(distance, labels);
    candidates.push({ k, silhouette: score });
    if (!best || score > best.silhouette + SILHOUETTE_TIE) best = { labels, k, silhouette: score, candidates };
  }
  return best ?? { labels: new Array<number>(n).fill(0), k: 1, silhouette: 0, candidates };
}

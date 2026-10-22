/**
 * LexRank (Erkan & Radev, 2004): graph-based sentence salience.
 *
 * Sentences are nodes; an edge joins two sentences whose cosine similarity
 * reaches a threshold, weighted by that similarity. Each row is normalised
 * into transition probabilities and PageRank is computed by power iteration:
 * a sentence is central when it is similar to many other central sentences,
 * which is exactly what a good summary sentence looks like. The damping
 * factor mixes in a uniform jump so the random walk always converges, and a
 * sentence with no neighbours jumps uniformly (no probability is lost).
 */
import { cosine } from "./vector";

export interface LexRankOptions {
  /** Probability of following an edge rather than jumping anywhere. */
  damping?: number;
  /** Minimum similarity for an edge; filters out incidental word overlap. */
  threshold?: number;
  /** Upper bound on power-iteration steps (it usually converges in < 30). */
  iterations?: number;
}

const CONVERGENCE_TOLERANCE = 1e-9;

/** Symmetric cosine-similarity matrix with ones on the diagonal. */
export function similarityMatrix(vectors: ArrayLike<number>[]): number[][] {
  const n = vectors.length;
  const matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    matrix[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      const s = cosine(vectors[i], vectors[j]);
      matrix[i][j] = s;
      matrix[j][i] = s;
    }
  }
  return matrix;
}

/** Row-stochastic transition matrix of the thresholded graph; null rows mark dangling nodes. */
function transitionRows(sim: number[][], threshold: number): (number[] | null)[] {
  return sim.map((row, i) => {
    const weights = row.map((s, j) => (j !== i && s >= threshold ? s : 0));
    const total = weights.reduce((sum, w) => sum + w, 0);
    return total > 0 ? weights.map((w) => w / total) : null;
  });
}

/** Stationary distribution of the damped random walk; scores sum to 1. */
export function lexRank(sim: number[][], opts: LexRankOptions = {}): number[] {
  const n = sim.length;
  if (n === 0) return [];
  const damping = opts.damping ?? 0.85;
  const threshold = opts.threshold ?? 0.1;
  const iterations = opts.iterations ?? 100;
  const rows = transitionRows(sim, threshold);

  let scores = new Array<number>(n).fill(1 / n);
  for (let step = 0; step < iterations; step++) {
    // Mass held by dangling nodes is spread uniformly, like the teleport term.
    const dangling = rows.reduce((sum, row, i) => (row ? sum : sum + scores[i]), 0);
    const base = (1 - damping) / n + (damping * dangling) / n;
    const next = new Array<number>(n).fill(base);
    rows.forEach((row, i) => {
      if (!row) return;
      for (let j = 0; j < n; j++) if (row[j] > 0) next[j] += damping * scores[i] * row[j];
    });
    const change = next.reduce((sum, v, i) => sum + Math.abs(v - scores[i]), 0);
    scores = next;
    if (change < CONVERGENCE_TOLERANCE) break;
  }

  const total = scores.reduce((sum, v) => sum + v, 0);
  return scores.map((v) => v / total);
}

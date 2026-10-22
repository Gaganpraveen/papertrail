/**
 * Maximal Marginal Relevance (Carbonell & Goldstein, 1998).
 *
 * Greedy selection that balances relevance against redundancy: at each step
 * pick the item maximising
 *   lambda * relevance(item) - (1 - lambda) * max similarity(item, picked).
 * lambda = 1 is plain ranking; lower values favour diversity. Ties keep the
 * input order, so the result is deterministic.
 */
export function mmrSelect<T>(
  items: T[],
  relevance: (t: T) => number,
  similarity: (a: T, b: T) => number,
  k: number,
  lambda = 0.7,
): T[] {
  const limit = Math.min(items.length, Math.max(0, Math.floor(k)));
  const scores = items.map(relevance);
  // Highest similarity of each remaining item to anything picked so far.
  const redundancy = new Array<number>(items.length).fill(Number.NEGATIVE_INFINITY);
  const remaining = new Set(items.map((_, i) => i));
  const picked: T[] = [];

  while (picked.length < limit) {
    let best = -1;
    let bestValue = Number.NEGATIVE_INFINITY;
    for (const i of remaining) {
      const penalty = picked.length === 0 ? 0 : redundancy[i];
      const value = lambda * scores[i] - (1 - lambda) * penalty;
      if (value > bestValue) {
        bestValue = value;
        best = i;
      }
    }
    if (best < 0) break;
    remaining.delete(best);
    const chosen = items[best];
    picked.push(chosen);
    for (const i of remaining) redundancy[i] = Math.max(redundancy[i], similarity(items[i], chosen));
  }
  return picked;
}

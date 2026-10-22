/**
 * TF-IDF document vectors as sparse maps.
 *
 * Weight = (1 + ln tf) * idf, with smoothed idf = ln((1 + N) / (1 + df)) + 1
 * (the scikit-learn convention, so a term present everywhere still counts a
 * little). Sub-linear tf stops one repeated word from dominating a sentence.
 * Each vector is L2-normalised, so cosine similarity is a sparse dot product.
 */

export interface TfidfResult {
  vectors: Map<string, number>[];
  idf: Map<string, number>;
}

export function tfidf(docs: string[][]): TfidfResult {
  const documentFrequency = new Map<string, number>();
  const counts = docs.map((doc) => {
    const tf = new Map<string, number>();
    for (const term of doc) tf.set(term, (tf.get(term) ?? 0) + 1);
    for (const term of tf.keys()) documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
    return tf;
  });

  const n = docs.length;
  const idf = new Map<string, number>();
  for (const [term, df] of documentFrequency) idf.set(term, Math.log((1 + n) / (1 + df)) + 1);

  const vectors = counts.map((tf) => {
    const vector = new Map<string, number>();
    let squaredNorm = 0;
    for (const [term, f] of tf) {
      const weight = (1 + Math.log(f)) * (idf.get(term) ?? 0);
      vector.set(term, weight);
      squaredNorm += weight * weight;
    }
    const length = Math.sqrt(squaredNorm);
    if (length > 0) for (const [term, weight] of vector) vector.set(term, weight / length);
    return vector;
  });

  return { vectors, idf };
}

/** Cosine similarity of two sparse vectors (iterates over the smaller one). */
export function cosineSparse(a: Map<string, number>, b: Map<string, number>): number {
  const [small, large] = a.size <= b.size ? [a, b] : [b, a];
  let product = 0;
  for (const [term, weight] of small) {
    const other = large.get(term);
    if (other !== undefined) product += weight * other;
  }
  if (product === 0) return 0;
  const denominator = sparseNorm(a) * sparseNorm(b);
  return denominator > 0 ? product / denominator : 0;
}

function sparseNorm(v: Map<string, number>): number {
  let sum = 0;
  for (const weight of v.values()) sum += weight * weight;
  return Math.sqrt(sum);
}

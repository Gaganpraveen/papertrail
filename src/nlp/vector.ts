/**
 * Dense vector helpers for embeddings.
 *
 * Embeddings from the sentence encoder are L2-normalised, so cosine
 * similarity equals the dot product; `cosine` still divides by the norms so
 * it is safe for un-normalised inputs (TF-IDF rows, averaged vectors).
 */

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < n; i++) sum += a[i] * b[i];
  return sum;
}

function norm(v: ArrayLike<number>): number {
  return Math.sqrt(dot(v, v));
}

/** Cosine similarity in [-1, 1]; 0 when either vector is all zeros. */
export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const denominator = norm(a) * norm(b);
  return denominator > 0 ? dot(a, b) / denominator : 0;
}

/** Returns a new unit-length copy (a zero vector stays zero). */
export function normalize(v: Float32Array): Float32Array {
  const length = norm(v);
  const out = new Float32Array(v.length);
  if (length === 0) return out;
  for (let i = 0; i < v.length; i++) out[i] = v[i] / length;
  return out;
}

/** Centroid of the vectors, L2-normalised; empty input gives an empty vector. */
export function meanVector(vs: Float32Array[]): Float32Array {
  if (vs.length === 0) return new Float32Array(0);
  const dimension = vs[0].length;
  const sum = new Float32Array(dimension);
  for (const v of vs) {
    for (let i = 0; i < dimension; i++) sum[i] += v[i] ?? 0;
  }
  return normalize(sum);
}

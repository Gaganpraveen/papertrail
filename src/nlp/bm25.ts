/**
 * Okapi BM25: the classic lexical ranking function behind most search engines.
 *
 * For each query term, a document earns idf(term) * tf * (k1 + 1) /
 * (tf + k1 * (1 - b + b * docLength / avgdl)).
 *  - idf rewards rare terms; we use the Lucene form
 *    ln(1 + (N - df + 0.5) / (df + 0.5)), which is never negative.
 *  - k1 saturates term frequency: the 10th "cache" adds far less than the 1st.
 *  - b normalises by length so long documents do not win just by being long.
 */

export interface BM25Options {
  /** Term-frequency saturation; 1.2–2.0 is the usual range. */
  k1?: number;
  /** Length normalisation strength, 0 (none) to 1 (full). */
  b?: number;
}

export class BM25 {
  readonly avgdl: number;
  private readonly k1: number;
  private readonly b: number;
  private readonly termFrequencies: Map<string, number>[];
  private readonly lengths: number[];
  private readonly idf = new Map<string, number>();

  constructor(docs: string[][], opts: BM25Options = {}) {
    this.k1 = opts.k1 ?? 1.5;
    this.b = opts.b ?? 0.75;
    this.lengths = docs.map((doc) => doc.length);
    this.avgdl = docs.length > 0 ? this.lengths.reduce((sum, n) => sum + n, 0) / docs.length : 0;
    this.termFrequencies = docs.map(countTerms);

    const documentFrequency = new Map<string, number>();
    for (const tf of this.termFrequencies) {
      for (const term of tf.keys()) documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
    }
    const n = docs.length;
    for (const [term, df] of documentFrequency) {
      this.idf.set(term, Math.log(1 + (n - df + 0.5) / (df + 0.5)));
    }
  }

  /** One score per document, in input order; 0 for documents sharing no query term. */
  score(query: string[]): number[] {
    const terms = [...new Set(query)].filter((t) => this.idf.has(t));
    return this.termFrequencies.map((tf, i) => {
      // Guard avgdl = 0 (every document empty): length normalisation is then moot.
      const relativeLength = this.avgdl > 0 ? this.lengths[i] / this.avgdl : 1;
      const lengthNorm = this.k1 * (1 - this.b + this.b * relativeLength);
      let total = 0;
      for (const term of terms) {
        const f = tf.get(term);
        if (!f) continue;
        total += (this.idf.get(term) ?? 0) * ((f * (this.k1 + 1)) / (f + lengthNorm));
      }
      return total;
    });
  }
}

function countTerms(doc: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const term of doc) counts.set(term, (counts.get(term) ?? 0) + 1);
  return counts;
}

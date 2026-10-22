/**
 * Class-based TF-IDF (c-TF-IDF, from BERTopic, Grootendorst 2022) for theme labels.
 *
 * All papers of a cluster are concatenated into one "class document". A term's
 * weight in class c is
 *   W(t, c) = tf(t, c) * log(1 + A / f(t))
 * where tf(t, c) is the term's share of the class's words, f(t) its total
 * frequency over all classes and A the average number of words per class.
 * Terms that are frequent inside one cluster but rare elsewhere rise to the
 * top, which is exactly what a theme label should contain.
 *
 * Terms are unigrams and bigrams of content words (stopwords and generic
 * paper vocabulary removed), keyed by their stems so "embedding"/"embeddings"
 * count together, and displayed in their most frequent surface form.
 */
import { STOPWORDS, stem, words } from "@/nlp/text";

export interface TermStats {
  count: number;
  /** Surface spellings seen for this term, with counts. */
  surfaces: Map<string, number>;
}

export type TermCounts = Map<string, TermStats>;

/** Words every ML paper uses; they would win every label without telling themes apart. */
const GENERIC_TERMS = new Set(
  `paper work method approach model result propose show use task performance study experiment baseline
  evaluation evaluate dataset data set number problem time new different large small existing prior previous
  table figure section appendix et al arxiv preprint proceeding conference journal http https www github com
  org doi vol pp ieee acm neurips iclr icml acl emnlp naacl cvpr significant significantly state art
  approaches technique framework system various specific achieve improve improvement outperform
  first second third main key important good better best high higher low lower`
    .split(/\s+/)
    .filter(Boolean)
    .flatMap((word) => [word, stem(word)]),
);

/**
 * Keep acronyms and identifiers as written ("KV", "GPT-4", "LLMs", "Self-RAG");
 * lower-case ordinary and Title-Case words ("Knowledge-Intensive").
 */
function surfaceForm(word: string): string {
  const identifier = word.split("-").some((part) => /\p{Lu}.*\p{Lu}|\p{N}|^\p{Ll}+\p{Lu}/u.test(part));
  return identifier ? word : word.toLowerCase();
}

function isContentWord(word: string, key: string): boolean {
  const lower = word.toLowerCase();
  if (STOPWORDS.has(lower) || GENERIC_TERMS.has(key) || GENERIC_TERMS.has(lower)) return false;
  if (/^[\p{N}.,-]+$/u.test(word)) return false; // bare numbers and years
  // Two-letter words only survive as acronyms ("KV", "RL").
  return word.length >= 3 || /^\p{Lu}{2}$/u.test(word);
}

function addTerm(counts: TermCounts, key: string, surface: string, weight: number): void {
  const stats = counts.get(key) ?? { count: 0, surfaces: new Map<string, number>() };
  stats.count += weight;
  stats.surfaces.set(surface, (stats.surfaces.get(surface) ?? 0) + weight);
  counts.set(key, stats);
}

/**
 * Unigram and bigram counts of a text. Bigrams are two adjacent content words
 * inside one sentence, so they never bridge a stopword or a sentence break.
 */
export function countTerms(text: string, weight = 1, into: TermCounts = new Map()): TermCounts {
  // A plain punctuation split is enough here: bigrams only need to stop at sentence ends.
  for (const sentence of text.split(/(?<=[.!?;:])\s+/)) {
    let previous: { key: string; surface: string } | null = null;
    for (const word of words(sentence)) {
      const key = stem(word.toLowerCase());
      if (!isContentWord(word, key)) {
        previous = null;
        continue;
      }
      const surface = surfaceForm(word);
      addTerm(into, key, surface, weight);
      if (previous) addTerm(into, `${previous.key} ${key}`, `${previous.surface} ${surface}`, weight);
      previous = { key, surface };
    }
  }
  return into;
}

export function mergeCounts(parts: TermCounts[]): TermCounts {
  const merged: TermCounts = new Map();
  for (const part of parts) {
    for (const [key, stats] of part) {
      for (const [surface, count] of stats.surfaces) addTerm(merged, key, surface, count);
    }
  }
  return merged;
}

/** c-TF-IDF weights for each class document. */
export function classTfidf(classes: TermCounts[]): Map<string, number>[] {
  const totalPerClass = classes.map((c) => [...c.values()].reduce((s, t) => s + t.count, 0));
  const average = totalPerClass.reduce((s, v) => s + v, 0) / Math.max(1, classes.length);
  const frequency = new Map<string, number>();
  for (const c of classes) for (const [key, { count }] of c) frequency.set(key, (frequency.get(key) ?? 0) + count);

  return classes.map((c, i) => {
    const weights = new Map<string, number>();
    for (const [key, { count }] of c) {
      const tf = totalPerClass[i] > 0 ? count / totalPerClass[i] : 0;
      weights.set(key, tf * Math.log(1 + average / (frequency.get(key) ?? 1)));
    }
    return weights;
  });
}

function displayForm(stats: TermStats | undefined, key: string): string {
  if (!stats) return key;
  return [...stats.surfaces.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0][0];
}

/** Six shared leading letters mean one word family: "retrieval" / "retrieved", "quantization" / "quantize". */
const ROOT_LENGTH = 6;

function sameRoot(a: string, b: string): boolean {
  return !a.includes(" ") && a.length >= ROOT_LENGTH && b.length >= ROOT_LENGTH && a.slice(0, ROOT_LENGTH) === b.slice(0, ROOT_LENGTH);
}

/** A term seen fewer times than this in a whole cluster is noise, not a theme. */
const MIN_TERM_COUNT = 2;

/**
 * The best `k` terms by weight, without redundancy: a bigram absorbs the
 * unigrams it contains ("kv cache" replaces "cache").
 */
export function topTerms(weights: Map<string, number>, counts: TermCounts, k: number): string[] {
  const sorted = [...weights.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const frequent = (key: string) => (counts.get(key)?.count ?? 0) >= MIN_TERM_COUNT;
  // Frequent terms first; rare ones only fill the list when a cluster has very little text.
  const ranked = [...sorted.filter(([key]) => frequent(key)), ...sorted.filter(([key]) => !frequent(key))];
  const chosen: string[] = [];
  for (const [key] of ranked) {
    if (chosen.length >= k) break;
    const parts = key.split(" ");
    if (parts.length === 1 && chosen.some((c) => c.split(" ").includes(key) || sameRoot(c, key))) continue;
    if (chosen.includes(key)) continue;
    if (parts.length === 2 && chosen.some((c) => parts.includes(c))) {
      // A frequent bigram is the more specific label; a rare one must not displace a frequent word.
      if (!frequent(key)) continue;
      for (let i = chosen.length - 1; i >= 0; i--) if (parts.includes(chosen[i])) chosen.splice(i, 1);
    }
    chosen.push(key);
  }
  return chosen.map((key) => displayForm(counts.get(key), key));
}

export interface ClusterTerms {
  terms: string[];
  label: string;
}

const LABEL_TERMS = 3;
const KEPT_TERMS = 8;

/** Labels every cluster from the term counts of its member papers. */
export function labelClusters(paperCounts: TermCounts[], labels: number[], k: number): ClusterTerms[] {
  const classes = Array.from({ length: k }, (_, c) => mergeCounts(paperCounts.filter((_, i) => labels[i] === c)));
  return classTfidf(classes).map((weights, c) => {
    const terms = topTerms(weights, classes[c], KEPT_TERMS);
    return { terms, label: terms.slice(0, LABEL_TERMS).join(" · ") || `Theme ${c + 1}` };
  });
}

/**
 * Vocabulary the whole collection shares: terms used by the most papers,
 * ties broken by total frequency. Describes what the survey is about.
 */
export function sharedTerms(paperCounts: TermCounts[], k: number): string[] {
  const documentFrequency = new Map<string, number>();
  for (const counts of paperCounts) for (const key of counts.keys()) documentFrequency.set(key, (documentFrequency.get(key) ?? 0) + 1);
  const merged = mergeCounts(paperCounts);
  const minimumPapers = Math.max(2, Math.ceil(paperCounts.length / 2));
  const weights = new Map<string, number>();
  for (const [key, df] of documentFrequency) {
    if (df < Math.min(minimumPapers, paperCounts.length)) continue;
    weights.set(key, df * 1000 + (merged.get(key)?.count ?? 0));
  }
  return topTerms(weights, merged, k);
}

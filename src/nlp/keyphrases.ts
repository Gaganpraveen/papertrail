/**
 * Keyphrase extraction.
 *
 * Candidates are noun-phrase-like spans: maximal runs of content words
 * between punctuation, stopwords and a curated list of verbs, adverbs and
 * vague adjectives ("propose", "significantly", "novel"), from which all 1-3
 * word n-grams are counted. Variants that differ only by inflection or
 * hyphenation share a key ("attention head" / "attention heads").
 *
 * Statistical score (YAKE, Campos et al. 2020, simplified): each word gets
 *   H(w) = Rel * Pos / (Case + Freq / Rel + Spread / Rel)
 * where Freq is its normalised frequency, Spread the share of sentences it
 * appears in (an IDF-like penalty for words used everywhere), Pos favours
 * words that appear early, Case rewards acronyms and proper nouns, and Rel
 * penalises words that combine with many different neighbours (generic
 * words such as "method"). A phrase scores prod H(w) / (count * (1 + sum H(w))),
 * lower being better. No corpus statistics are needed, so it works on one
 * paper in the browser.
 *
 * With an embedding model the ranking is KeyBERT-style: the cosine between
 * the document embedding and each candidate is blended with the statistical
 * score. Either way, Maximal Marginal Relevance picks a diverse final list
 * and near-duplicates (same stems, mostly overlapping stems, or an acronym
 * and its expansion such as "KV cache" / "key-value cache") are merged.
 */
import type { EmbedFn } from "@/types";
import { mmrSelect } from "./mmr";
import { STOPWORDS, splitSentences, stem } from "./text";
import { cosine, meanVector } from "./vector";

export interface CandidatePhrase {
  phrase: string;
  count: number;
  /** Word offset of the first occurrence in the text (0-based). */
  firstPos: number;
}

export interface Keyphrase {
  phrase: string;
  score: number;
}

// ──────────────────────────────── Lexicons ─────────────────────────────────

/** Verbs, adverbs and vague adjectives that end a noun phrase. */
const PHRASE_BREAKERS = new Set(
  `propose proposes proposed proposing present presents presented introduce introduces introduced introducing
  show shows showed shown showing demonstrate demonstrates demonstrated achieve achieves achieved achieving
  outperform outperforms outperformed outperforming improve improves improved use uses used using utilize
  utilizes utilized leverage leverages leveraged leveraging apply applies applied applying perform performs
  performed obtain obtains obtained provide provides provided providing require requires required requiring
  enable enables enabled enabling allow allows allowed allowing make makes made making take takes taken give
  gives given find finds found observe observes observed report reports reported consider considers considered
  evaluate evaluates evaluated compare compares compared based include includes included including lead leads
  led leading yield yields yielded reach reaches reached exceed exceeds surpass surpasses reduce reduces reduced
  reducing increase increases increased increasing decrease decreases decreased remain remains remained become
  becomes became contain contains consist consists focus focuses focused aim aims aimed seek seeks investigate
  investigates investigated explore explores explored analyze analyzes analyzed describe describes described
  discuss discusses discussed note noted suggest suggests suggested indicate indicates indicated highlight
  highlights highlighted address addresses addressed tackle tackles solve solves solved designed develop develops
  developed build builds built help helps helped need needs needed ensure ensures ensured maintain maintains
  maintained preserve preserves preserved deliver delivers boost boosts boosted enhance enhances enhanced
  represent represents represented denote denotes denoted define defines defined refer refers call called known
  termed named see seen according due respectively thereby thus get gets got let lets try tries tried want
  wants allow allowing serve serves served work works worked hold holds held keep keeps kept run runs ran
  generate generates generated generating compress compresses predict predicts retrieve retrieves produce
  produces encode encodes decode decodes select selects compute computes calculate calculates handle handles
  handled support supports supported
  novel existing various different several numerous significant extensive comprehensive effective efficient
  simple respective additional further overall main certain specific particular typical common current recent
  prior previous original same similar own whole entire total great good better best strong stronger superior
  impressive remarkable promising important crucial critical challenging difficult easy possible potential able
  unable necessary first second third last final initial next above below following new many much most other
  also even still yet already able often usually typically generally specifically particularly simply directly
  higher lower larger smaller faster slower greater fewer more less better worse longer shorter cheaper
  two three four five six seven eight nine ten hundred thousand million billion dozen single double multiple
  well-known widely-used so-called state-of-the-art off-the-shelf real-world up-to-date
  et al etc fig figs eq eqs tab sec ref refs vs cf ie eg resp approx`
    .split(/\s+/)
    .filter(Boolean),
);

/** Words too vague to be a keyphrase on their own or with only each other. */
const GENERIC_WORDS = new Set(
  `model method approach result performance task paper work data dataset problem framework system technique
  algorithm experiment evaluation analysis study set number way case example time value table figure section
  appendix equation baseline setting input output step process term type form part level order use effect
  impact improvement state art idea insight observation finding question answer issue challenge limitation
  contribution field area domain application practice research literature range variety amount aspect manner
  factor point line side end fact lot kind sort size length rate score accuracy quality cost efficiency scale
  version variant component strategy scheme procedure pipeline architecture parameter feature sample instance
  token layer network training learning large small high low long short better best wide large-scale
  state-of-the-art benchmark metric experiment implementation detail details overview summary goal purpose
  property structure function operation information knowledge representation space solution design trade-off
  objective response capability ability behavior behaviour advantage benefit gain standard`
    .split(/\s+/)
    .filter(Boolean)
    .map(stem),
);

/** Multi-word phrases that are boilerplate in almost every paper. */
const GENERIC_PHRASES = new Set(
  [
    "proposed method", "experimental results", "related work", "future work", "previous work", "prior work",
    "extensive experiments", "main results", "large margin", "wide range", "recent years", "recent work",
    "best knowledge", "this paper", "main contributions", "empirical results", "experimental setup",
    "ablation study", "case study", "open source", "source code", "supplementary material", "github repository",
  ].map((p) => p.split(" ").map(stem).join(" ")),
);

// ──────────────────────────────── Scanning ─────────────────────────────────

interface Token {
  surface: string;
  lower: string;
  stem: string;
  breaker: boolean;
  /** Offset among word tokens (punctuation breakers excluded). */
  position: number;
  sentence: number;
  sentenceInitial: boolean;
}

const WORD_RE = /[\p{L}\p{N}]+(?:[-'’][\p{L}\p{N}]+)*/gu;
/** "4-bit", "2bit", "7B", "16x": sizes and settings, not concepts. */
const NUMERIC_WORD_RE = /^\d+(?:[.,]\d+)?(?:-?(?:bit|bits|b|m|k|x|shot|way|layer|fold|th|st|nd|rd|d))?$/iu;
/** Greek letters and math operators: notation ("πSFT", "θ"), not vocabulary. */
const NOTATION_RE = /[\u0370-\u03ff\u2200-\u22ff]/u;
/** Third-person verb endings ("optimizes", "facilitates"), minus the nouns that share them. */
const VERB_SUFFIX_RE = /(?:izes|ises|yzes|ifies|ates)$/u;
const NOUNS_ENDING_ATES = new Set(
  "states rates gates updates candidates templates estimates aggregates coordinates dates plates predicates substrates climates delegates duplicates certificates surrogates affiliates".split(" "),
);
/** Short mixed-case tokens are math notation ("XVr", "XKg", "Wq"), not words. */
const SYMBOL_RE = /^\p{Lu}{1,3}\p{Ll}{1,2}$/u;
/** A word ending in -s/-ed/-ing directly before one of these is acting as a verb ("outperforms the"). */
const DETERMINERS = new Set(["the", "a", "an", "this", "these", "that", "those", "its", "their", "our", "each", "every", "any"]);
const BREAK = "|";

/**
 * Removes material that would create fake phrases: URLs, citations ("Brown et
 * al., 2020", "[12]"), cross-references ("Table 2") and inline acronym
 * definitions, so "key-value (KV) cache" reads as "key-value cache".
 */
function preprocess(text: string): string {
  return text
    .replace(/https?:\/\/\S+|www\.\S+|\S+@\S+\.\w+/g, ` ${BREAK} `)
    .replace(/\([^()]*\b(?:19|20)\d{2}[a-z]?\b[^()]*\)/g, ` ${BREAK} `)
    .replace(/\[[\d\s,;–-]+\]/g, ` ${BREAK} `)
    .replace(/\b[\p{Lu}][\p{L}'’-]+(?:\s+(?:and|&)\s+[\p{Lu}][\p{L}'’-]+)?\s+et\s+al\.?/gu, ` ${BREAK} `)
    .replace(
      /\b(?:Table|Tab\.|Figure|Fig\.|Section|Sec\.|Eq\.|Equation|Appendix|Algorithm|Theorem|Lemma)\s*~?\(?[A-Z]?\d*(?:\.\d+)*[a-z]?\)?/g,
      ` ${BREAK} `,
    )
    .replace(/\s*\(\s*[\p{Lu}][\p{L}\d-]{0,10}\s*\)/gu, "");
}

function isAcronym(word: string): boolean {
  return /^[\p{Lu}\d][\p{Lu}\d-]*[\p{Lu}]s?$/u.test(word) && /\p{Lu}.*\p{Lu}/u.test(word);
}

function isBreakerWord(lower: string, surface: string): boolean {
  if (lower.length < 2 || !/\p{L}/u.test(lower) || NOTATION_RE.test(lower)) return true;
  if (STOPWORDS.has(lower) || PHRASE_BREAKERS.has(lower)) return true;
  if (NUMERIC_WORD_RE.test(lower)) return true;
  if (VERB_SUFFIX_RE.test(lower) && !NOUNS_ENDING_ATES.has(lower)) return true;
  if (/(?:ize|yze)$/u.test(lower) && lower.length > 6) return true; // "quantize", "optimize" ("size" is short)
  if (SYMBOL_RE.test(surface) && surface.length <= 4) return true;
  // Adverbs ("significantly"), except nouns that happen to end in -ly.
  return /ly$/u.test(lower) && !/(?:ply|family|anomaly|assembly|monopoly|italy)$/u.test(lower);
}

function actsAsVerb(token: Token, next: Token | undefined): boolean {
  if (!next || !DETERMINERS.has(next.lower)) return false;
  return /(?:[^su]s|ed|ing)$/u.test(token.lower);
}

function scanSentences(sentences: string[]): Token[] {
  const tokens: Token[] = [];
  let wordIndex = 0;
  sentences.forEach((sentence, s) => {
    const sentenceStart = tokens.length;
    let previousEnd = 0;
    let first = true;
    for (const match of sentence.matchAll(WORD_RE)) {
      const start = match.index ?? 0;
      // Any punctuation between two words ends the phrase: insert a breaker.
      if (!first && /\S/u.test(sentence.slice(previousEnd, start))) tokens.push(breakerToken(wordIndex, s));
      const surface = match[0].replace(/’/gu, "'").replace(/'s$/u, "");
      const lower = surface.toLowerCase();
      tokens.push({
        surface,
        lower,
        stem: stem(lower),
        breaker: isBreakerWord(lower, surface),
        position: wordIndex++,
        sentence: s,
        sentenceInitial: first,
      });
      first = false;
      previousEnd = start + match[0].length;
    }
    for (let i = sentenceStart; i < tokens.length; i++) {
      if (!tokens[i].breaker && actsAsVerb(tokens[i], tokens[i + 1])) tokens[i].breaker = true;
    }
    tokens.push(breakerToken(wordIndex, s));
  });
  return tokens;
}

function breakerToken(position: number, sentence: number): Token {
  return { surface: BREAK, lower: BREAK, stem: BREAK, breaker: true, position, sentence, sentenceInitial: false };
}

// ──────────────────────────────── Candidates ───────────────────────────────

interface PhraseStats {
  key: string;
  /** Content words of the phrase (the "of" in "mixture of experts" is not stored). */
  words: Token[];
  /** True for "X of Y" phrases. */
  linked: boolean;
  count: number;
  firstPos: number;
  /** Display form -> occurrences. */
  variants: Map<string, number>;
  /** Title-case surface forms seen mid-sentence ("Direct Preference Optimization"). */
  titleCased: number;
  sentences: Set<number>;
}

/** Lowercase unless the word is an acronym, mixed-case name, or a proper noun capitalised mid-sentence. */
function displayWord(token: Token, properNouns: Set<string>): string {
  if (/\p{Lu}/u.test(token.surface.slice(1))) return token.surface; // "KV", "GPT-4", "LLaMA"
  return properNouns.has(token.lower) ? token.surface : token.lower;
}

/** Words capitalised in most of their non-sentence-initial occurrences ("Mamba", "Transformer"). */
function findProperNouns(tokens: Token[]): Set<string> {
  const capitalised = new Map<string, number>();
  const total = new Map<string, number>();
  for (const t of tokens) {
    if (t.breaker || t.sentenceInitial) continue;
    total.set(t.lower, (total.get(t.lower) ?? 0) + 1);
    if (/^\p{Lu}/u.test(t.surface)) capitalised.set(t.lower, (capitalised.get(t.lower) ?? 0) + 1);
  }
  const out = new Set<string>();
  for (const [word, n] of total) if ((capitalised.get(word) ?? 0) / n >= 0.6) out.add(word);
  return out;
}

function letterCount(word: string): number {
  return word.match(/\p{L}/gu)?.length ?? 0;
}

function isAcceptablePhrase(words: Token[]): boolean {
  const last = words[words.length - 1].lower;
  // Participles end verb phrases, not noun phrases ("model trained", "attention proposed").
  if (/[^e]ed$/u.test(last) && last.length > 4 && !/(?:embed|speed|seed|feed|bed)$/u.test(last)) return false;
  if (/^per-/u.test(last)) return false; // "key cache per-channel"
  const length = words.reduce((sum, w) => sum + w.lower.length + 1, -1);
  if (length < 3 || length > 60) return false;
  if (words.length === 1 && last.length < 4 && !isAcronym(words[0].surface)) return false;
  if (!words.some((w) => letterCount(w.lower) >= 3)) return false;
  const stems = words.map((w) => w.stem);
  if (new Set(stems).size < stems.length) return false; // "Mamba Mamba Mamba" from a table header
  if (GENERIC_PHRASES.has(stems.join(" "))) return false;
  return !stems.every((s) => GENERIC_WORDS.has(s));
}

/** "chain of thought", "mixture of experts": two specific words joined by "of". */
function isLinkedPhrase(head: Token, tail: Token): boolean {
  return !GENERIC_WORDS.has(head.stem) && !GENERIC_WORDS.has(tail.stem) && !/(?:ing|ed)$/u.test(head.lower);
}

function addOccurrence(phrases: Map<string, PhraseStats>, words: Token[], linked: boolean, properNouns: Set<string>): void {
  const key = words.map((w) => w.stem).join(linked ? " of " : " ");
  const display = words.map((w) => displayWord(w, properNouns)).join(linked ? " of " : " ");
  const stats = phrases.get(key) ?? {
    key,
    words,
    linked,
    count: 0,
    firstPos: words[0].position,
    variants: new Map<string, number>(),
    titleCased: 0,
    sentences: new Set<number>(),
  };
  stats.count++;
  stats.variants.set(display, (stats.variants.get(display) ?? 0) + 1);
  if (words.length > 1 && !words[0].sentenceInitial && words.every((w) => /^\p{Lu}/u.test(w.surface))) stats.titleCased++;
  stats.sentences.add(words[0].sentence);
  phrases.set(key, stats);
}

function collectPhrases(tokens: Token[]): Map<string, PhraseStats> {
  const phrases = new Map<string, PhraseStats>();
  const properNouns = findProperNouns(tokens);
  let run: Token[] = [];
  /** Tails of "X of Y" phrases: "thought prompting" in "chain of thought prompting" is a fragment. */
  const linkedTails = new Set<number>();
  const flush = (): void => {
    for (let n = 1; n <= 3; n++) {
      for (let i = 0; i + n <= run.length; i++) {
        if (i === 0 && n > 1 && linkedTails.has(run[0].position)) continue;
        const words = run.slice(i, i + n);
        if (isAcceptablePhrase(words)) addOccurrence(phrases, words, false, properNouns);
      }
    }
    run = [];
  };
  tokens.forEach((token, i) => {
    if (!token.breaker) {
      run.push(token);
      return;
    }
    if (token.lower === "of" && run.length === 1) {
      const tail = tokens[i + 1];
      if (tail && !tail.breaker && isLinkedPhrase(run[0], tail)) {
        addOccurrence(phrases, [run[0], tail], true, properNouns);
        linkedTails.add(tail.position);
      }
    }
    flush();
  });
  flush();
  return dropFragments(phrases);
}

/** A prefix that is almost always continued is a fragment ("large language" before "models"). */
const PREFIX_FRAGMENT_SHARE = 0.8;
/** Suffixes are usually heads that stand alone ("language models"), so only drop those that never do. */
const SUFFIX_FRAGMENT_SHARE = 0.95;

function dropFragments(phrases: Map<string, PhraseStats>): Map<string, PhraseStats> {
  const continuedAfter = new Map<string, number>();
  const continuedBefore = new Map<string, number>();
  const add = (map: Map<string, number>, key: string, n: number): void => {
    map.set(key, (map.get(key) ?? 0) + n);
  };
  for (const p of phrases.values()) {
    if (p.words.length < 2) continue;
    const stems = p.words.map((w) => w.stem);
    add(continuedAfter, stems.slice(0, -1).join(" "), p.count);
    if (!p.linked) add(continuedBefore, stems.slice(1).join(" "), p.count);
  }
  for (const [key, p] of phrases) {
    if (p.linked) continue;
    const after = continuedAfter.get(key) ?? 0;
    const before = continuedBefore.get(key) ?? 0;
    if (after >= PREFIX_FRAGMENT_SHARE * p.count || before >= SUFFIX_FRAGMENT_SHARE * p.count) phrases.delete(key);
  }
  return phrases;
}

function preferredVariant(stats: PhraseStats): string {
  let best = "";
  let bestCount = -1;
  for (const [variant, n] of stats.variants) {
    if (n > bestCount) {
      best = variant;
      bestCount = n;
    }
  }
  // A name written in title case mid-sentence keeps its capitals ("Direct Preference Optimization").
  if (stats.titleCased >= 2 && stats.titleCased > stats.count / 2) {
    return stats.words.map((w) => w.surface.charAt(0).toUpperCase() + w.surface.slice(1)).join(stats.linked ? " of " : " ");
  }
  return best;
}

/** Size and quality adjectives that rarely belong at the start of a concept ("small Mamba model"). */
const WEAK_LEADING = new Set(["small", "large", "big", "high", "low", "long", "short", "simple", "good", "wide", "full", "naive", "vanilla", "standard", "typical", "baseline"]);
const WEAK_HEADS = new Set(["performance", "result", "accuracy", "score", "approach", "method", "baseline"].map(stem));
/** Preference by word count (1, 2, 3 words). */
const LENGTH_PRIOR = [0.9, 1, 0.85];
/** Established terms that start with an otherwise weak adjective. */
const LEXICALISED = new Set(["large language model", "long context", "low rank", "small language model"].map((p) => p.split(" ").map(stem).join(" ")));

/**
 * Multiplier below 1 for phrases whose form suggests a weaker concept:
 * participle modifiers ("learned reward function"), leading size adjectives
 * ("small Mamba model"), vague heads ("StreamingLLM performance"). Among
 * near-duplicates this lets the cleaner phrase win.
 */
function formPenalty(stats: PhraseStats): number {
  const words = stats.words;
  let factor = 1;
  for (const w of words) if (/[^e]ed$/u.test(w.lower) && w.lower.length > 4) factor *= 0.85;
  // A leading derived-verb gerund is usually an action, not a concept ("quantizing key cache").
  if (words.length > 1 && /(?:iz|is|yz|at|ify)ing$/u.test(words[0].lower)) factor *= 0.7;
  if (words.length > 1 && WEAK_LEADING.has(words[0].lower) && !LEXICALISED.has(stats.key)) factor *= 0.8;
  if (words.length > 1 && WEAK_HEADS.has(words[words.length - 1].stem)) factor *= 0.85;
  if (words.length > 1 && GENERIC_WORDS.has(words[0].stem)) factor *= 0.8; // "benchmark StreamingLLM"
  if (words.length === 1 && /(?:ive|ous|able|ible|ful)$/u.test(words[0].lower)) factor *= 0.7; // lone adjectives
  return factor;
}

function sentencesOf(text: string): string[] {
  const cleaned = preprocess(text);
  const sentences = splitSentences(cleaned);
  return sentences.length > 0 ? sentences : [cleaned];
}

/** 1–3-word noun-phrase-like candidates with their frequency and first position. */
export function candidatePhrases(text: string): CandidatePhrase[] {
  const phrases = collectPhrases(scanSentences(sentencesOf(text)));
  return [...phrases.values()]
    .sort((a, b) => a.firstPos - b.firstPos)
    .map((p) => ({ phrase: preferredVariant(p), count: p.count, firstPos: p.firstPos }));
}

// ───────────────────────────── YAKE-style scoring ──────────────────────────

interface WordStats {
  tf: number;
  tfCased: number;
  sentences: number[];
  left: Set<string>;
  right: Set<string>;
}

function wordStatistics(tokens: Token[]): Map<string, WordStats> {
  const stats = new Map<string, WordStats>();
  tokens.forEach((t, i) => {
    if (t.breaker) return;
    const s = stats.get(t.stem) ?? { tf: 0, tfCased: 0, sentences: [], left: new Set(), right: new Set() };
    s.tf++;
    if (isAcronym(t.surface) || (!t.sentenceInitial && /^\p{Lu}/u.test(t.surface))) s.tfCased++;
    s.sentences.push(t.sentence);
    const before = tokens[i - 1];
    const after = tokens[i + 1];
    if (before && before.lower !== BREAK && before.sentence === t.sentence) s.left.add(before.lower);
    if (after && after.lower !== BREAK && after.sentence === t.sentence) s.right.add(after.lower);
    stats.set(t.stem, s);
  });
  return stats;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** YAKE word weight H(w); lower means more keyphrase-like. */
function wordWeights(stats: Map<string, WordStats>, sentenceCount: number): Map<string, number> {
  const tfs = [...stats.values()].map((s) => s.tf);
  const mean = tfs.reduce((a, b) => a + b, 0) / Math.max(1, tfs.length);
  const std = Math.sqrt(tfs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, tfs.length));
  const maxTf = Math.max(1, ...tfs);
  const weights = new Map<string, number>();
  for (const [word, s] of stats) {
    const casing = s.tfCased / (1 + Math.log(s.tf));
    const position = Math.log(Math.log(3 + median(s.sentences)));
    const frequency = s.tf / (mean + std);
    // Distinct neighbours per occurrence: near 1 for words that pair with anything.
    const relatedness = 1 + ((s.left.size + s.right.size) / (2 * s.tf)) * (s.tf / maxTf);
    const spread = new Set(s.sentences).size / Math.max(1, sentenceCount);
    weights.set(word, (relatedness * position) / (casing + frequency / relatedness + spread / relatedness));
  }
  return weights;
}

/** YAKE phrase score: lower is better. */
function phraseScore(phrase: PhraseStats, weights: Map<string, number>): number {
  let product = 1;
  let sum = 0;
  for (const w of phrase.words) {
    const h = weights.get(w.stem) ?? 1;
    product *= h;
    sum += h;
  }
  return product / (phrase.count * (1 + sum));
}

// ─────────────────────────────── Near-duplicates ───────────────────────────

interface Scored {
  stats: PhraseStats;
  phrase: string;
  stemSet: Set<string>;
  relevance: number;
  vector?: Float32Array;
}

function jaccard(a: Set<string>, b: Set<string>): number {
  let shared = 0;
  for (const x of a) if (b.has(x)) shared++;
  const union = a.size + b.size - shared;
  return union === 0 ? 0 : shared / union;
}

/** Tokens for acronym matching: acronyms become letters, other words their hyphen parts. */
function acronymForm(words: Token[], expandAcronyms: boolean): string[] {
  return words.flatMap((w) => {
    if (expandAcronyms && isAcronym(w.surface)) return [...w.surface.replace(/s$/u, "").toLowerCase()];
    return w.lower.split("-").map(stem);
  });
}

/** "KV cache" ~ "key-value cache", "LLMs" ~ "large language models". */
function acronymMatches(short: Token[], long: Token[]): boolean {
  if (!short.some((w) => isAcronym(w.surface))) return false;
  const a = acronymForm(short, true);
  const b = acronymForm(long, false);
  if (a.length !== b.length) return false;
  return a.every((x, i) => x === b[i] || (x.length === 1 && b[i].startsWith(x)));
}

/** Stems of every hyphen part, plus "of" for linked phrases: "chain-of-thought" ~ "chain of thought". */
function partStems(stats: PhraseStats): string[] {
  const parts = stats.words.flatMap((w) => w.lower.split("-").map(stem));
  return stats.linked ? [...parts, "of"] : parts;
}

function nearDuplicate(a: Scored, b: Scored): boolean {
  const partsA = partStems(a.stats);
  const partsB = partStems(b.stats);
  if ([...partsA].sort().join(" ") === [...partsB].sort().join(" ")) return true; // "key-value cache" / "key value cache"
  if (jaccard(a.stemSet, b.stemSet) >= 0.5) return true; // "cache" / "KV cache", "attention" / "attention heads"
  // Hyphen parts overlap strongly: "chain-of-thought prompting" / "chain of thought".
  if (jaccard(new Set(partsA), new Set(partsB)) > 0.6) return true;
  return acronymMatches(a.stats.words, b.stats.words) || acronymMatches(b.stats.words, a.stats.words);
}

/**
 * Greedy de-duplication in relevance order: a candidate that is a
 * near-duplicate of a kept phrase is dropped, except that a phrase may
 * replace a bare word it contains when it is nearly as relevant, so
 * "KV cache" wins over a slightly higher-scoring "cache".
 */
function dedupe(sorted: Scored[]): Scored[] {
  const kept: Scored[] = [];
  for (const candidate of sorted) {
    const clashes = kept.flatMap((k, i) => (nearDuplicate(k, candidate) ? [i] : []));
    if (clashes.length === 0) {
      kept.push(candidate);
      continue;
    }
    // Replacing one of several clashing phrases would let near-duplicates back in.
    if (clashes.length > 1) continue;
    const clash = clashes[0];
    const existing = kept[clash];
    // Only a bare word gives way to a phrase containing it ("cache" -> "KV cache"); longer
    // phrases keep their place, otherwise every term would grow into its longest variant.
    const moreSpecific =
      existing.stats.words.length === 1 &&
      candidate.stemSet.has(existing.stats.words[0].stem) &&
      formPenalty(candidate.stats) >= formPenalty(existing.stats);
    if (moreSpecific && candidate.relevance >= SPECIFICITY_RATIO * existing.relevance) {
      kept[clash] = { ...candidate, relevance: existing.relevance };
    }
  }
  return kept;
}

// ──────────────────────────────── Extraction ───────────────────────────────

const DEFAULT_TOP = 12;
/** Candidates kept after statistical ranking; bounds embedding work on long papers. */
const POOL_SIZE = 48;
/** Only phrases seen at least this often compete, unless that leaves fewer than twice the requested count. */
const MIN_COUNT = 2;
/** A containing phrase may replace a near-duplicate if it scores at least this fraction of it. */
const SPECIFICITY_RATIO = 0.6;
/** Blend of embedding relevance and statistical score in KeyBERT mode. */
const SEMANTIC_WEIGHT = 0.5;
const MMR_LAMBDA = { semantic: 0.6, lexical: 0.7 };
/** Document embedding = centroid of up to this many evenly spaced windows. */
const DOC_WINDOWS = 8;
const WINDOW_WORDS = 120;

function minMax(values: number[]): number[] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  return values.map((v) => (max - min > 1e-12 ? (v - min) / (max - min) : 1));
}

/** Evenly spaced word windows covering the document, so long papers are not judged by their opening only. */
function documentWindows(text: string): string[] {
  const all = text.split(/\s+/).filter(Boolean);
  const windows: string[] = [];
  const count = Math.min(DOC_WINDOWS, Math.max(1, Math.ceil(all.length / WINDOW_WORDS)));
  const stride = Math.max(WINDOW_WORDS, Math.floor(all.length / count));
  for (let k = 0; k < count; k++) {
    const slice = all.slice(k * stride, k * stride + WINDOW_WORDS);
    if (slice.length > 0) windows.push(slice.join(" "));
  }
  return windows;
}

async function embedCandidates(pool: Scored[], text: string, embed: EmbedFn): Promise<number[] | null> {
  try {
    const windows = documentWindows(text);
    const vectors = await embed([...windows, ...pool.map((p) => p.phrase)]);
    if (vectors.length !== windows.length + pool.length) return null;
    const docVector = meanVector(vectors.slice(0, windows.length));
    pool.forEach((p, i) => (p.vector = vectors[windows.length + i]));
    return pool.map((p) => cosine(docVector, p.vector ?? new Float32Array(0)));
  } catch {
    return null;
  }
}

function lexicalSimilarity(a: Scored, b: Scored): number {
  return nearDuplicate(a, b) ? 1 : jaccard(a.stemSet, b.stemSet);
}

/** Ranked, de-duplicated keyphrases with scores in [0, 1]. */
export async function extractKeyphrases(
  text: string,
  opts: { embed?: EmbedFn | null; top?: number } = {},
): Promise<Keyphrase[]> {
  const top = opts.top ?? DEFAULT_TOP;
  const sentences = sentencesOf(text);
  const tokens = scanSentences(sentences);
  const phrases = [...collectPhrases(tokens).values()];
  if (phrases.length === 0 || top <= 0) return [];

  const weights = wordWeights(wordStatistics(tokens), sentences.length);
  const frequent = phrases.filter((p) => p.count >= MIN_COUNT);
  // Min-max scaling needs a pool wider than the answer, or the weakest kept phrase is pinned to 0.
  const eligible = frequent.length >= 2 * top ? frequent : phrases;
  const raw = eligible.map((p) => -Math.log(phraseScore(p, weights)));
  // YAKE favours long n-grams; the length prior nudges towards readable two-word terms.
  const statistical = minMax(raw).map((v, i) => v * formPenalty(eligible[i]) * (LENGTH_PRIOR[eligible[i].words.length - 1] ?? 1));
  let pool: Scored[] = eligible
    .map((stats, i) => ({
      stats,
      phrase: preferredVariant(stats),
      stemSet: new Set(stats.words.map((w) => w.stem)),
      relevance: statistical[i],
    }))
    .sort((a, b) => b.relevance - a.relevance || a.stats.firstPos - b.stats.firstPos)
    .slice(0, POOL_SIZE);

  const semantic = opts.embed ? await embedCandidates(pool, text, opts.embed) : null;
  if (semantic) {
    const scaled = minMax(semantic);
    pool = pool.map((p, i) => ({ ...p, relevance: SEMANTIC_WEIGHT * scaled[i] + (1 - SEMANTIC_WEIGHT) * p.relevance }));
    pool.sort((a, b) => b.relevance - a.relevance);
  }

  const unique = dedupe(pool);
  const similarity = (a: Scored, b: Scored): number =>
    a.vector && b.vector ? Math.max(cosine(a.vector, b.vector), lexicalSimilarity(a, b)) : lexicalSimilarity(a, b);
  const chosen = mmrSelect(unique, (p) => p.relevance, similarity, top, semantic ? MMR_LAMBDA.semantic : MMR_LAMBDA.lexical);
  return chosen.map((p) => ({ phrase: p.phrase, score: Math.round(p.relevance * 1000) / 1000 }));
}

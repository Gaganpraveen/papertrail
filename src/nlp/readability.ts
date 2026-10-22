/**
 * Flesch Reading Ease: 206.835 - 1.015 * (words / sentences)
 *                              - 84.6 * (syllables / words).
 *
 * Higher is easier: 60–70 is plain English, research papers usually land
 * between 0 and 30. Syllables are estimated with the usual heuristic of
 * counting vowel groups, corrected for a silent final "e" and the "-ed" /
 * "-es" endings that do not add a syllable.
 */
import { splitSentences, words as wordTokens } from "./text";

export interface ReadabilityStats {
  fleschReadingEase: number;
  avgSentenceLength: number;
  avgSyllablesPerWord: number;
  words: number;
}

/** Heuristic syllable count for an English word (at least 1 for any word with letters). */
export function countSyllables(word: string): number {
  let w = word.toLowerCase().replace(/[^a-z]/g, "");
  if (w.length === 0) return 0;
  if (w.length <= 3) return 1;
  // Silent endings: "used", "makes", "large" — but not "-ted"/"-ded" ("trained" vs "added").
  w = w.replace(/(?:[^laeiouy]es|[^laeiouytd]ed|[^laeiouy]e)$/, (m) => m.charAt(0));
  w = w.replace(/^y/, "");
  const groups = w.match(/[aeiouy]+/g)?.length ?? 0;
  // A trailing consonant + "le" is its own syllable ("sample", "table") and survives the rule above.
  return Math.max(1, groups);
}

/** Flesch Reading Ease plus its inputs; zeros for text without words. */
export function readability(text: string): ReadabilityStats {
  const tokens = wordTokens(text).filter((w) => /\p{L}/u.test(w));
  if (tokens.length === 0) {
    return { fleschReadingEase: 0, avgSentenceLength: 0, avgSyllablesPerWord: 0, words: 0 };
  }
  const syllables = tokens.reduce((sum, w) => sum + countSyllables(w), 0);
  // Sentence length is measured over real sentences only, so stray headings and
  // labels (which splitSentences drops) neither shorten nor lengthen the average.
  const sentences = splitSentences(text);
  const sentenceWords = sentences.reduce((sum, s) => sum + wordTokens(s).filter((w) => /\p{L}/u.test(w)).length, 0);
  const avgSentenceLength = sentences.length > 0 ? sentenceWords / sentences.length : tokens.length;
  const avgSyllablesPerWord = syllables / tokens.length;
  const fleschReadingEase = 206.835 - 1.015 * avgSentenceLength - 84.6 * avgSyllablesPerWord;
  return {
    fleschReadingEase: round(fleschReadingEase, 1),
    avgSentenceLength: round(avgSentenceLength, 1),
    avgSyllablesPerWord: round(avgSyllablesPerWord, 2),
    words: tokens.length,
  };
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

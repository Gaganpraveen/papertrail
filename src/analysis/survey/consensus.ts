/**
 * Cross-paper consensus: findings that more than one paper states.
 *
 * Candidates are each paper's result and contribution sentences (by zone).
 * Two sentences from different papers "agree" when their cosine similarity
 * clears a high bar (0.72 with sentence embeddings, roughly the point where
 * MiniLM pairs read as paraphrases). Agreeing sentences are grouped with star
 * clustering: the sentence with neighbours in the most other papers becomes a
 * centre, takes its best match from each of those papers, and the process
 * repeats on what is left. Unlike connected components this cannot chain
 * A ~ B ~ C into a group where A and C say different things.
 */
import type { Evidence, SurveyResult, ZonedSentence } from "@/types";
import type { Representation } from "./clustering";
import { cleanQuote, isOwnClaim, isQuotable, toEvidence } from "./comparison";

export type ConsensusStatement = SurveyResult["consensus"][number];

export interface PooledSentence {
  paperId: string;
  sentence: ZonedSentence;
}

/**
 * Agreement thresholds per representation. TF-IDF cosines of true
 * paraphrases are much lower than embedding cosines (synonyms share no
 * terms), so the lexical fallback needs a lower bar to find anything.
 */
export const CONSENSUS_THRESHOLD: Record<Representation, number> = { embedding: 0.72, tfidf: 0.5 };

const FINDING_ZONES = new Set<ZonedSentence["zone"]>(["result", "contribution"]);

/** Enough to cover a paper's main claims without letting a long paper dominate the pool. */
const FINDINGS_PER_PAPER = 15;
const MAX_STATEMENTS = 6;

/** The result and contribution sentences of one paper, most confident first. */
export function findingCandidates(paperId: string, sentences: ZonedSentence[]): PooledSentence[] {
  return sentences
    .filter((s) => FINDING_ZONES.has(s.zone) && isOwnClaim(s) && isQuotable(s.text))
    .sort((a, b) => b.confidence - a.confidence || a.index - b.index)
    .slice(0, FINDINGS_PER_PAPER)
    .map((sentence) => ({ paperId, sentence }));
}

interface Star {
  centre: number;
  members: number[];
  strength: number;
}

/** The best unused match for `centre` in every other paper. */
function bestMatchPerPaper(
  centre: number,
  members: number[],
  pool: PooledSentence[],
  similarity: number[][],
  threshold: number,
  used: Set<number>,
): Map<string, number> {
  const best = new Map<string, number>();
  for (const j of members) {
    if (used.has(j) || pool[j].paperId === pool[centre].paperId) continue;
    if (similarity[centre][j] < threshold) continue;
    const current = best.get(pool[j].paperId);
    if (current === undefined || similarity[centre][j] > similarity[centre][current]) best.set(pool[j].paperId, j);
  }
  return best;
}

function nextStar(
  members: number[],
  pool: PooledSentence[],
  similarity: number[][],
  threshold: number,
  used: Set<number>,
): Star | null {
  let best: Star | null = null;
  for (const centre of members) {
    if (used.has(centre)) continue;
    const matches = [...bestMatchPerPaper(centre, members, pool, similarity, threshold, used).values()];
    if (matches.length === 0) continue;
    const strength = matches.reduce((s, j) => s + similarity[centre][j], 0) / matches.length;
    const better =
      !best ||
      matches.length > best.members.length - 1 ||
      (matches.length === best.members.length - 1 && strength > best.strength);
    if (better) best = { centre, members: [centre, ...matches], strength };
  }
  return best;
}

/**
 * Groups cross-paper agreements. `members` are indices into `pool` (and into
 * the rows of `similarity`) that are eligible as findings.
 */
export function findConsensus(
  pool: PooledSentence[],
  members: number[],
  similarity: number[][],
  kind: Representation,
): ConsensusStatement[] {
  const threshold = CONSENSUS_THRESHOLD[kind];
  const used = new Set<number>();
  const stars: Star[] = [];
  for (;;) {
    const star = nextStar(members, pool, similarity, threshold, used);
    if (!star) break;
    stars.push(star);
    // Each paper contributes one sentence per statement; its other near-copies would only repeat it.
    for (const m of star.members) used.add(m);
  }
  return stars
    .sort((a, b) => b.members.length - a.members.length || b.strength - a.strength)
    .slice(0, MAX_STATEMENTS)
    .map((star) => {
      const support: Evidence[] = star.members.map((m) => toEvidence(pool[m].paperId, pool[m].sentence));
      return {
        statement: cleanQuote(pool[star.centre].sentence.text),
        support,
        paperIds: [...new Set(star.members.map((m) => pool[m].paperId))],
      };
    });
}

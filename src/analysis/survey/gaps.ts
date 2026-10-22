/**
 * Open problems and gaps across a collection.
 *
 * 1. Author-stated gaps: every paper's future-work and limitation sentences
 *    (by zone) are pooled and grouped with average-linkage clustering, so the
 *    same open problem raised by two papers becomes one item. A group is
 *    dropped when another paper's contribution sentences already address it
 *    (high similarity to a contribution of a paper outside the group): what
 *    one paper leaves for future work, another may have done.
 * 2. Shared weaknesses: a critique finding (e.g. no variance reported) that
 *    recurs in at least half of the papers is a gap of the field, not of one
 *    paper. These are labelled as coming from PaperTrail's critique.
 */
import type { Evidence, SectionKind, SurveyPaperInput, ZonedSentence } from "@/types";
import { tokenize } from "@/nlp/text";
import { averageLinkage, cutAtDistance, toDistance, type Representation } from "./clustering";
import { cleanQuote, isOwnClaim, isQuotable, startsWithAnaphor, toEvidence } from "./comparison";
import type { PooledSentence } from "./consensus";

export interface GapItem {
  statement: string;
  evidence: Evidence[];
  paperIds: string[];
  source: "authors" | "critique";
}

/** Two gap sentences belong together above this similarity. */
export const GAP_GROUP_THRESHOLD: Record<Representation, number> = { embedding: 0.6, tfidf: 0.3 };
/** A gap counts as addressed when another paper's contribution is at least this similar. */
export const ADDRESSED_THRESHOLD: Record<Representation, number> = { embedding: 0.65, tfidf: 0.35 };

const GAP_ZONES = new Set<ZonedSentence["zone"]>(["future", "limitation"]);
const GAPS_PER_PAPER = 8;
const MAX_AUTHOR_GAPS = 8;
/** A zone guess below this confidence is too uncertain to present as the authors' own admission. */
const MIN_GAP_CONFIDENCE = 0.1;
/** "Our results raise several questions for future work." names no problem; a gap needs some content. */
const MIN_GAP_CONTENT_WORDS = 5;
/** Words that announce a gap without saying what it is. */
const GAP_FRAMING_WORDS = new Set(
  tokenize(
    `future work works direction directions limitation limitations question questions open important interesting
    several various further research study studies result results paper discuss related remain remains leave left`,
  ),
);
/**
 * Limitations stated in experiments sections are usually setup caveats ("we do
 * not consider instances where ...") and in the introduction they are mostly
 * prior work's flaws; the authors' own limits live in these sections.
 */
const OWN_LIMITATION_SECTIONS = new Set<SectionKind>(["limitations", "discussion", "conclusion"]);

function namesAProblem(text: string): boolean {
  return tokenize(text).filter((t) => !GAP_FRAMING_WORDS.has(t)).length >= MIN_GAP_CONTENT_WORDS;
}

export function gapCandidates(paperId: string, sentences: ZonedSentence[]): PooledSentence[] {
  return sentences
    .filter(
      (s) =>
        GAP_ZONES.has(s.zone) &&
        s.confidence >= MIN_GAP_CONFIDENCE &&
        isOwnClaim(s) &&
        (s.zone === "future" || OWN_LIMITATION_SECTIONS.has(s.sectionKind)) &&
        isQuotable(s.text) &&
        !startsWithAnaphor(s.text) &&
        namesAProblem(s.text),
    )
    .sort((a, b) => b.confidence - a.confidence || a.index - b.index)
    .slice(0, GAPS_PER_PAPER)
    .map((sentence) => ({ paperId, sentence }));
}

function subMatrix(similarity: number[][], indices: number[]): number[][] {
  return indices.map((i) => indices.map((j) => similarity[i][j]));
}

/** The member most similar on average to the rest of its group. */
function medoid(group: number[], similarity: number[][]): number {
  let best = group[0];
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const i of group) {
    const score = group.reduce((s, j) => s + (i === j ? 0 : similarity[i][j]), 0);
    if (score > bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return best;
}

/**
 * Author-stated open problems. `gapMembers` and `contributionMembers` index
 * into `pool` and `similarity`.
 */
export function authorGaps(
  pool: PooledSentence[],
  gapMembers: number[],
  contributionMembers: number[],
  similarity: number[][],
  kind: Representation,
): GapItem[] {
  if (gapMembers.length === 0) return [];
  const merges = averageLinkage(toDistance(subMatrix(similarity, gapMembers)));
  const labels = cutAtDistance(merges, gapMembers.length, 1 - GAP_GROUP_THRESHOLD[kind]);
  const groups = new Map<number, number[]>();
  labels.forEach((label, i) => groups.set(label, [...(groups.get(label) ?? []), gapMembers[i]]));

  const items: (GapItem & { confidence: number })[] = [];
  for (const group of groups.values()) {
    const papers = new Set(group.map((m) => pool[m].paperId));
    const addressed = contributionMembers.some(
      (c) => !papers.has(pool[c].paperId) && group.some((m) => similarity[m][c] >= ADDRESSED_THRESHOLD[kind]),
    );
    if (addressed) continue;
    const representative = medoid(group, similarity);
    items.push({
      statement: cleanQuote(pool[representative].sentence.text),
      evidence: group.map((m) => toEvidence(pool[m].paperId, pool[m].sentence)),
      paperIds: [...papers],
      source: "authors",
      confidence: Math.max(...group.map((m) => pool[m].sentence.confidence)),
    });
  }
  return items
    .sort((a, b) => b.paperIds.length - a.paperIds.length || b.confidence - a.confidence)
    .slice(0, MAX_AUTHOR_GAPS)
    .map(({ confidence: _confidence, ...item }) => item);
}

/** Findings that only matter when they recur: weaknesses, not strengths or nitpicks. */
const SHARED_SEVERITIES = new Set(["major", "moderate"]);

/** Groups the same critique rule across papers; digits are ignored so "1 dataset" and "2 datasets" match. */
function findingKey(category: string, title: string): string {
  return `${category}|${title.toLowerCase().replace(/\d+/g, "#").replace(/\s+/g, " ").trim()}`;
}

/** Critique findings shared by at least half of the papers (and at least two). */
export function sharedWeaknesses(papers: SurveyPaperInput[]): GapItem[] {
  const groups = new Map<string, { title: string; paperIds: string[]; evidence: Evidence[] }>();
  for (const paper of papers) {
    const seen = new Set<string>();
    for (const finding of paper.analysis.critique.findings) {
      if (!SHARED_SEVERITIES.has(finding.severity)) continue;
      const key = findingKey(finding.category, finding.title);
      if (seen.has(key)) continue;
      seen.add(key);
      const group = groups.get(key) ?? { title: finding.title, paperIds: [], evidence: [] };
      group.paperIds.push(paper.meta.id);
      group.evidence.push(...finding.evidence.slice(0, 1));
      groups.set(key, group);
    }
  }
  const quorum = Math.max(2, Math.ceil(papers.length / 2));
  return [...groups.values()]
    .filter((g) => g.paperIds.length >= quorum)
    .sort((a, b) => b.paperIds.length - a.paperIds.length || a.title.localeCompare(b.title))
    .map((g) => ({
      statement: `${g.paperIds.length} of ${papers.length} papers share this weakness: ${g.title.replace(/\.$/, "")}.`,
      evidence: g.evidence,
      paperIds: g.paperIds,
      source: "critique" as const,
    }));
}

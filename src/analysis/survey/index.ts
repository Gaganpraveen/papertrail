/**
 * Literature survey over a set of analysed papers.
 *
 * Pipeline: represent every paper as a vector (L2-normalised mean of its
 * chunk embeddings, or of its chunks' TF-IDF vectors when any paper lacks
 * embeddings, so all papers share one space) -> cluster into themes
 * (average linkage + silhouette; embedding centroids are first centred on the
 * collection mean, because papers from one field share a large common
 * component that compresses every cosine into a narrow band and hides the
 * structure) -> name themes (c-TF-IDF) -> project chunks
 * and papers onto a 2-D map (UMAP / classical MDS) -> bibliographic coupling
 * -> cross-paper consensus and open problems from zoned sentences ->
 * comparison matrix -> a cited, Markdown literature review.
 */
import type { EmbedFn, SurveyCluster, SurveyPaperInput, SurveyResult } from "@/types";
import { similarityMatrix } from "@/nlp/lexrank";
import { tokenize } from "@/nlp/text";
import { tfidf } from "@/nlp/tfidf";
import { meanVector, normalize } from "@/nlp/vector";
import { clusterPapers, sparseSimilarity, textSpace, type Representation } from "./clustering";
import { buildComparison } from "./comparison";
import { CONSENSUS_THRESHOLD, findConsensus, findingCandidates, type PooledSentence } from "./consensus";
import { bibliographicCoupling, directCitations } from "./coupling";
import { countTerms, labelClusters, sharedTerms, type TermCounts } from "./ctfidf";
import { authorGaps, gapCandidates, sharedWeaknesses } from "./gaps";
import { writeNarrative } from "./narrative";
import { projectDense, projectSimilarity, type MapItem, type SurveyMap } from "./projection";

/** Caps the semantic map at a readable, quickly computed number of points. */
const MAP_CHUNKS_PER_PAPER = 40;
const SHARED_VOCABULARY = 6;

// ───────────────────────────── Representation ───────────────────────────────

interface PaperSpace {
  kind: Representation;
  /** Plain cosine between paper vectors: what the UI and the review report. */
  similarity: number[][];
  /** What clustering sees (centred for embeddings). */
  clusterSimilarity: number[][];
  map: SurveyMap;
}

function hasUsableEmbeddings(papers: SurveyPaperInput[]): boolean {
  const dimension = papers[0]?.chunkEmbeddings?.[0]?.length ?? 0;
  return (
    dimension > 0 &&
    papers.every(
      (p) =>
        p.chunkEmbeddings !== null &&
        p.chunkEmbeddings.length > 0 &&
        p.chunkEmbeddings.length === p.chunks.length &&
        p.chunkEmbeddings.every((v) => v.length === dimension),
    )
  );
}

/** Evenly spaced indices so a long paper is sampled from start to end, deterministically. */
function sampleIndices(length: number, limit: number): number[] {
  if (length <= limit) return Array.from({ length }, (_, i) => i);
  return Array.from({ length: limit }, (_, i) => Math.floor((i * length) / limit));
}

/**
 * Subtracts the mean paper vector before comparing papers. Needs three or more
 * papers: with two, the centred vectors are exact opposites by construction.
 */
function centredSimilarity(centroids: Float32Array[]): number[][] {
  if (centroids.length < 3) return similarityMatrix(centroids);
  const dimension = centroids[0].length;
  const mean = new Float32Array(dimension);
  for (const c of centroids) for (let d = 0; d < dimension; d++) mean[d] += c[d] / centroids.length;
  return similarityMatrix(centroids.map((c) => normalize(c.map((v, d) => v - mean[d]))));
}

function embeddingSpace(papers: SurveyPaperInput[]): PaperSpace {
  const centroids = papers.map((p) => meanVector(p.chunkEmbeddings ?? []));
  const items: MapItem[] = [];
  const vectors: Float32Array[] = [];
  papers.forEach((paper, p) => {
    for (const i of sampleIndices(paper.chunks.length, MAP_CHUNKS_PER_PAPER)) {
      items.push({ paperId: paper.meta.id, chunkId: paper.chunks[i].id, isCentroid: false });
      vectors.push((paper.chunkEmbeddings ?? [])[i]);
    }
    items.push({ paperId: paper.meta.id, isCentroid: true });
    vectors.push(centroids[p]);
  });
  return {
    kind: "embedding",
    similarity: similarityMatrix(centroids),
    clusterSimilarity: centredSimilarity(centroids),
    map: projectDense(items, vectors),
  };
}

/** The text that stands in for a paper with no chunks (abstract-only, or nothing parsed). */
function fallbackText(paper: SurveyPaperInput): string {
  return paper.sentences.map((s) => s.text).join(" ") || paper.meta.abstract || paper.meta.title;
}

function meanSparse(vectors: Map<string, number>[]): Map<string, number> {
  const sum = new Map<string, number>();
  for (const v of vectors) for (const [term, w] of v) sum.set(term, (sum.get(term) ?? 0) + w);
  const length = Math.sqrt([...sum.values()].reduce((s, w) => s + w * w, 0));
  if (length > 0) for (const [term, w] of sum) sum.set(term, w / length);
  return sum;
}

/** Same recipe as the embedding path, with chunk TF-IDF vectors (idf estimated over all chunks). */
function lexicalSpace(papers: SurveyPaperInput[]): PaperSpace {
  const docs = papers.map((p) => (p.chunks.length > 0 ? p.chunks.map((c) => c.text) : [fallbackText(p)]));
  const offsets: number[] = [];
  const flat: string[][] = [];
  for (const texts of docs) {
    offsets.push(flat.length);
    for (const text of texts) flat.push(tokenize(text));
  }
  const { vectors } = tfidf(flat);
  const centroids = papers.map((_, p) => meanSparse(vectors.slice(offsets[p], offsets[p] + docs[p].length)));

  const items: MapItem[] = [];
  const mapVectors: Map<string, number>[] = [];
  papers.forEach((paper, p) => {
    for (const i of sampleIndices(paper.chunks.length, MAP_CHUNKS_PER_PAPER)) {
      items.push({ paperId: paper.meta.id, chunkId: paper.chunks[i].id, isCentroid: false });
      mapVectors.push(vectors[offsets[p] + i]);
    }
    items.push({ paperId: paper.meta.id, isCentroid: true });
    mapVectors.push(centroids[p]);
  });
  // IDF already discounts vocabulary every paper shares, so no centring is needed here.
  const similarity = sparseSimilarity(centroids);
  return { kind: "tfidf", similarity, clusterSimilarity: similarity, map: projectSimilarity(items, sparseSimilarity(mapVectors)) };
}

// ─────────────────────────────── Theme labels ───────────────────────────────

/**
 * Term counts for c-TF-IDF. Titles and the paper's keyphrases are counted
 * extra (x3, x2): they are the authors' own summary of what the paper is about.
 */
function paperTerms(paper: SurveyPaperInput): TermCounts {
  const counts = countTerms(paper.meta.title, 3);
  for (const { phrase } of paper.analysis.briefing.keyphrases) countTerms(phrase, 2, counts);
  return countTerms(fallbackText(paper), 1, counts);
}

function buildClusters(papers: SurveyPaperInput[], labels: number[], k: number, terms: TermCounts[]): SurveyCluster[] {
  const named = labelClusters(terms, labels, k);
  return named.map(({ terms: clusterTerms, label }, id) => ({
    id,
    label,
    terms: clusterTerms,
    paperIds: papers.filter((_, i) => labels[i] === id).map((p) => p.meta.id),
    color: id,
  }));
}

// ─────────────────────────────── Identifiers ────────────────────────────────

/** FNV-1a: a short, stable fingerprint of the paper set. */
function fingerprint(text: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(36);
}

function timelineOf(papers: SurveyPaperInput[]): SurveyResult["timeline"] {
  return papers
    .map((p) => ({ paperId: p.meta.id, date: p.meta.published ?? (p.meta.year ? String(p.meta.year) : ""), title: p.meta.title }))
    .filter((t) => t.date !== "")
    .sort((a, b) => a.date.localeCompare(b.date));
}

function uniqueById(papers: SurveyPaperInput[]): SurveyPaperInput[] {
  const seen = new Set<string>();
  return papers.filter((p) => !seen.has(p.meta.id) && Boolean(seen.add(p.meta.id)));
}

// ──────────────────────────────── Survey ────────────────────────────────────

export async function buildSurvey(
  inputs: SurveyPaperInput[],
  embed: EmbedFn | null,
  onProgress?: (step: string) => void,
): Promise<SurveyResult> {
  const papers = uniqueById(inputs);
  if (papers.length === 0) throw new Error("A literature survey needs at least one analysed paper.");
  const progress = (step: string) => onProgress?.(step);

  progress("Representing papers");
  const space = hasUsableEmbeddings(papers) ? embeddingSpace(papers) : lexicalSpace(papers);

  progress("Clustering themes");
  const clustering = clusterPapers(space.clusterSimilarity);
  const terms = papers.map(paperTerms);
  const clusters = buildClusters(papers, clustering.labels, clustering.k, terms);

  progress("Measuring bibliographic coupling");
  const linked = papers.map((p) => ({ id: p.meta.id, meta: p.meta, references: p.references }));
  const coupling = bibliographicCoupling(linked);
  const citations = directCitations(linked);

  progress("Comparing findings across papers");
  const findings = papers.flatMap((p) => findingCandidates(p.meta.id, p.sentences));
  const gapsPool = papers.flatMap((p) => gapCandidates(p.meta.id, p.sentences));
  const pool: PooledSentence[] = [...findings, ...gapsPool];
  const sentenceSpace = await textSpace(pool.map((s) => s.sentence.text), embed);
  const findingIndices = findings.map((_, i) => i);
  const gapIndices = gapsPool.map((_, i) => findings.length + i);
  const contributionIndices = findingIndices.filter((i) => pool[i].sentence.zone === "contribution");
  const consensus = findConsensus(pool, findingIndices, sentenceSpace.similarity, sentenceSpace.kind);

  progress("Mining open problems");
  const gaps = [
    ...authorGaps(pool, gapIndices, contributionIndices, sentenceSpace.similarity, sentenceSpace.kind),
    ...sharedWeaknesses(papers),
  ];

  progress("Building the comparison matrix");
  const comparison = buildComparison(papers);

  progress("Writing the review");
  const narrative = writeNarrative({
    papers,
    clusters,
    similarity: space.similarity,
    silhouette: clustering.silhouette,
    representation: space.kind,
    comparison,
    coupling,
    citations,
    consensus,
    consensusThreshold: CONSENSUS_THRESHOLD[sentenceSpace.kind],
    gaps,
    sharedVocabulary: papers.length > 1 ? sharedTerms(terms, SHARED_VOCABULARY) : [],
  });

  const createdAt = Date.now();
  const paperIds = papers.map((p) => p.meta.id);
  return {
    id: `survey-${fingerprint(paperIds.join("|"))}-${createdAt.toString(36)}`,
    createdAt,
    paperIds,
    similarity: space.similarity,
    clusters,
    silhouette: clustering.silhouette,
    map: space.map,
    comparison,
    timeline: timelineOf(papers),
    coupling,
    consensus,
    gaps: gaps.map(({ statement, evidence }) => ({ statement, evidence })),
    narrative,
  };
}

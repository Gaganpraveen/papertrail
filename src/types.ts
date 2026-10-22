/**
 * Domain model shared by every layer of PaperTrail.
 *
 * Data flows one way: sources -> parsing -> nlp -> retrieval -> analysis -> ui.
 * Every claim the app shows carries `Evidence` pointing back to a page and a
 * verbatim quote, so the UI can always answer "where did this come from?".
 */

// ───────────────────────────── Input & sources ─────────────────────────────

export type InputKind = "arxiv" | "doi" | "pdf-url" | "url" | "topic" | "upload";

export interface ParsedInput {
  kind: InputKind;
  raw: string;
  arxivId?: string; // e.g. "2402.02750" or "hep-th/9901001" (no version)
  version?: string; // e.g. "v2"
  doi?: string; // lower-cased, without the https://doi.org/ prefix
  url?: string;
  query?: string; // topic text
  fileName?: string; // uploads
}

export type MetaSource =
  | "datacite"
  | "openalex"
  | "semanticscholar"
  | "crossref"
  | "upload"
  | "pdf";

export interface PaperMeta {
  /** Stable key: "arxiv:2402.02750", "doi:10.1145/...", "upload:<sha-12>" */
  id: string;
  title: string;
  authors: string[];
  abstract?: string;
  /** ISO date (YYYY-MM-DD) or year (YYYY). */
  published?: string;
  year?: number;
  arxivId?: string;
  version?: string;
  doi?: string;
  categories?: string[];
  venue?: string;
  citationCount?: number;
  pdfUrl?: string;
  landingUrl?: string;
  source: MetaSource;
}

export interface CandidateScores {
  lexical: number; // BM25 over title + abstract, normalised 0..1
  semantic: number; // cosine(query, title+abstract), 0..1
  recency: number; // exponential decay on age, 0..1
  impact: number; // log-scaled citations, 0..1
  final: number; // weighted blend after MMR re-ordering
}

export interface Candidate {
  meta: PaperMeta;
  scores: CandidateScores;
  rank: number;
  /** One-line reason shown in the UI ("strong title match, 2025"). */
  why: string;
}

// ───────────────────────────────── Parsing ─────────────────────────────────

/** One visual line of text reconstructed from pdf.js text items. */
export interface TextLine {
  text: string;
  page: number; // 1-based
  x: number;
  y: number; // distance from top of page, in PDF units
  width: number;
  fontSize: number;
  bold: boolean;
  column: 0 | 1 | 2; // 0 = full width, 1 = left, 2 = right
}

export type SectionKind =
  | "abstract"
  | "introduction"
  | "related"
  | "background"
  | "method"
  | "experiments"
  | "results"
  | "discussion"
  | "limitations"
  | "conclusion"
  | "references"
  | "appendix"
  | "acknowledgments"
  | "other";

export interface Section {
  id: string; // "s0", "s1", ...
  title: string;
  number?: string; // "3.2"
  level: number; // 1 = top level
  kind: SectionKind;
  pageStart: number;
  pageEnd: number;
  text: string;
}

export interface Reference {
  raw: string;
  title?: string;
  authors?: string;
  year?: number;
  arxivId?: string;
  doi?: string;
}

export interface Caption {
  kind: "figure" | "table";
  label: string; // "Figure 3"
  text: string;
  page: number;
}

export interface ParsedDocument {
  pageCount: number;
  pagesParsed: number;
  pages: { page: number; text: string; charCount: number }[];
  title?: string;
  abstract?: string;
  sections: Section[];
  references: Reference[];
  captions: Caption[];
  links: string[]; // URLs found in the text (code, data, project pages)
  warnings: string[];
  quality: "good" | "degraded" | "image-only";
}

// ─────────────────────────────────── NLP ───────────────────────────────────

export interface Sentence {
  id: string; // "<sectionId>:<n>"
  text: string;
  sectionId: string;
  sectionTitle: string;
  sectionKind: SectionKind;
  page: number;
  index: number; // global order in the document
}

/** Argumentative zones (after Teufel's Argumentative Zoning). */
export type Zone =
  | "background"
  | "problem"
  | "contribution"
  | "method"
  | "result"
  | "limitation"
  | "future"
  | "other";

export interface ZonedSentence extends Sentence {
  zone: Zone;
  confidence: number; // 0..1, margin-based
  scores: Record<Zone, number>;
}

/** Texts in, L2-normalised vectors out. */
export type EmbedFn = (texts: string[]) => Promise<Float32Array[]>;

// ──────────────────────────────── Retrieval ────────────────────────────────

export interface Chunk {
  id: string; // "<paperId>#<n>"
  paperId: string;
  index: number;
  sectionId: string;
  sectionTitle: string;
  sectionKind: SectionKind;
  page: number;
  text: string;
  tokenCount: number;
}

export interface Evidence {
  paperId: string;
  page: number;
  section: string;
  quote: string;
  chunkId?: string;
  sentenceId?: string;
}

export interface RetrievalHit {
  chunk: Chunk;
  bm25: number;
  dense: number;
  rrf: number;
  rerank?: number;
  rank: { bm25?: number; dense?: number; final: number };
}

export type QuestionIntent =
  | "factual"
  | "summary"
  | "limitations"
  | "method"
  | "results"
  | "comparison";

export interface Answer {
  id: string;
  question: string;
  intent: QuestionIntent;
  status: "answered" | "abstained";
  /** Extracted span (verbatim from the paper) or a routed summary. */
  answer?: string;
  /** The full sentence that contains the span, for context. */
  context?: string;
  confidence: number; // 0..1
  evidence: Evidence[];
  hits: RetrievalHit[];
  reason?: string; // why we abstained, or how the answer was produced
  timingsMs: Record<string, number>;
  createdAt: number;
}

// ───────────────────────────────── Analysis ────────────────────────────────

export interface BriefingItem {
  text: string;
  evidence: Evidence[];
}

export interface KeyNumber {
  value: string; // "2.6×", "+4.1 BLEU", "93.2%"
  label: string; // what it measures, best effort
  evidence: Evidence;
}

export interface Briefing {
  /** "Why this paper matters" in one paragraph. */
  tldr: BriefingItem;
  problem: BriefingItem[];
  method: BriefingItem[];
  results: BriefingItem[];
  limitations: {
    authorStated: BriefingItem[];
    /** Inferred by PaperTrail's critique rules; never empty. */
    detected: BriefingItem[];
  };
  followUps: string[];
  keyphrases: { phrase: string; score: number }[];
  keyNumbers: KeyNumber[];
}

export type Severity = "major" | "moderate" | "minor" | "strength";

export interface CritiqueFinding {
  id: string;
  category:
    | "evaluation"
    | "statistics"
    | "reproducibility"
    | "claims"
    | "scope"
    | "presentation"
    | "related-work";
  severity: Severity;
  title: string;
  rationale: string;
  evidence: Evidence[];
}

export interface Critique {
  findings: CritiqueFinding[];
  reviewerQuestions: string[];
  /** Two to three sentence overall assessment. */
  verdict: string;
}

export interface RigorSignal {
  label: string;
  value: string | number | boolean;
  evidence?: Evidence;
}

export interface RigorAxis {
  key:
    | "empirical"
    | "baselines"
    | "ablation"
    | "statistics"
    | "reproducibility"
    | "transparency"
    | "scholarship";
  label: string;
  score: number; // 0..100
  weight: number; // sums to 1 across axes
  signals: RigorSignal[];
  explanation: string;
}

export interface RigorProfile {
  overall: number; // 0..100
  grade: "A" | "B" | "C" | "D";
  axes: RigorAxis[];
}

export interface VisualData {
  /** Every sentence placed on 0..1 through the paper, coloured by zone. */
  rhetoricalMap: { position: number; zone: Zone; page: number; sentenceId: string }[];
  sectionComposition: { kind: SectionKind; title: string; words: number }[];
  zoneDistribution: { zone: Zone; count: number }[];
  keyphraseGraph: {
    nodes: { id: string; weight: number }[];
    links: { source: string; target: string; weight: number }[];
  };
  citationYears: { year: number; count: number }[];
  readability: {
    fleschReadingEase: number;
    avgSentenceLength: number;
    readingMinutes: number;
    jargonDensity: number; // share of tokens that are keyphrase terms
  };
}

export interface PaperStats {
  pages: number;
  words: number;
  sentences: number;
  chunks: number;
  references: number;
  figures: number;
  tables: number;
}

export interface PaperAnalysis {
  briefing: Briefing;
  critique: Critique;
  rigor: RigorProfile;
  visuals: VisualData;
  stats: PaperStats;
  mode: "full" | "abstract-only";
  computedMs: number;
}

// ────────────────────────────────── Survey ─────────────────────────────────

export interface SurveyPaperInput {
  meta: PaperMeta;
  analysis: PaperAnalysis;
  sentences: ZonedSentence[];
  chunks: Chunk[];
  chunkEmbeddings: Float32Array[] | null;
  references: Reference[];
}

export interface SurveyCluster {
  id: number;
  label: string; // c-TF-IDF top terms joined
  terms: string[];
  paperIds: string[];
  color: number; // categorical index
}

export interface SurveyResult {
  id: string;
  createdAt: number;
  paperIds: string[];
  similarity: number[][]; // paper x paper cosine
  clusters: SurveyCluster[];
  silhouette: number;
  /** 2-D semantic map: one point per chunk plus one per paper centroid. */
  map: {
    points: { x: number; y: number; paperId: string; chunkId?: string; isCentroid: boolean }[];
    method: "umap" | "mds";
  };
  comparison: {
    paperId: string;
    year?: number;
    problem: string;
    approach: string[];
    datasets: string[];
    metrics: string[];
    keyResult: string;
    limitations: number;
    rigor: number;
    code: boolean;
  }[];
  timeline: { paperId: string; date: string; title: string }[];
  coupling: { source: string; target: string; shared: number; jaccard: number }[];
  consensus: { statement: string; support: Evidence[]; paperIds: string[] }[];
  gaps: { statement: string; evidence: Evidence[] }[];
  /** Full literature review, Markdown, with [n] citations. */
  narrative: string;
}

// ────────────────────────────── Agent runtime ──────────────────────────────

export type TraceEventType =
  | "node:start"
  | "node:end"
  | "node:error"
  | "node:retry"
  | "route"
  | "log";

export interface TraceEvent {
  type: TraceEventType;
  node: string;
  at: number; // epoch ms
  durationMs?: number;
  message?: string;
  to?: string; // for "route"
}

export type ModelKey = "embedder" | "reranker" | "reader";

export interface ModelStatus {
  state: "idle" | "loading" | "ready" | "error";
  progress: number; // 0..1 download progress
  error?: string;
}

export type MLStatus = Record<ModelKey, ModelStatus>;

/**
 * Shape of the application store (implemented in ./library.ts with zustand).
 * UI components read state and call actions; they never call the pipeline
 * directly.
 */
import type {
  Answer,
  Candidate,
  Chunk,
  Evidence,
  MLStatus,
  PaperAnalysis,
  PaperMeta,
  ParsedDocument,
  SurveyResult,
  TraceEvent,
  ZonedSentence,
} from "@/types";

export type PaperTab = "briefing" | "visuals" | "critique" | "ask";

export type PaperStatus = "queued" | "running" | "ready" | "failed";

export interface PaperRecord {
  id: string;
  meta: PaperMeta;
  status: PaperStatus;
  /** Name of the graph node currently executing (for the live trace). */
  currentNode?: string;
  /** 0..1 coarse progress through the pipeline. */
  progress: number;
  trace: TraceEvent[];
  mode?: "full" | "abstract-only";
  analysis?: PaperAnalysis;
  doc?: ParsedDocument;
  sentences?: ZonedSentence[];
  chunks?: Chunk[];
  /** True when a PDF is cached locally (enables page previews). */
  hasPdf: boolean;
  qa: Answer[];
  qaPending: boolean;
  warnings: string[];
  error?: string;
  addedAt: number;
}

export interface SearchSession {
  query: string;
  status: "searching" | "ready" | "empty" | "failed";
  /** Queries actually tried, including automatic broadening. */
  attempts: string[];
  candidates: Candidate[];
  trace: TraceEvent[];
  error?: string;
}

export interface SurveyRecord {
  id: string;
  paperIds: string[];
  status: "running" | "ready" | "failed";
  result?: SurveyResult;
  trace: TraceEvent[];
  error?: string;
  createdAt: number;
}

export type View =
  | { kind: "home" }
  | { kind: "search" }
  | { kind: "paper"; id: string; tab: PaperTab }
  | { kind: "survey"; id: string };

export interface Toast {
  id: number;
  tone: "info" | "good" | "warn" | "bad";
  message: string;
}

export interface LibraryState {
  papers: Record<string, PaperRecord>;
  /** Paper ids, most recent first. */
  order: string[];
  view: View;
  search: SearchSession | null;
  /** Paper ids ticked for a literature survey (max 10). */
  surveySelection: string[];
  surveys: Record<string, SurveyRecord>;
  surveyOrder: string[];
  ml: MLStatus;
  /** Evidence currently shown in the side drawer. */
  evidence: Evidence | null;
  theme: "light" | "dark";
  toasts: Toast[];
  hydrated: boolean;

  // ── actions ──
  hydrate(): Promise<void>;
  /** Free-text entry: arXiv id/url, DOI, PDF url, or a topic. */
  submit(raw: string): Promise<void>;
  submitFiles(files: File[]): Promise<void>;
  /** Analyse candidates picked from a topic search. */
  analyzeCandidates(candidates: Candidate[]): Promise<void>;
  retry(paperId: string): Promise<void>;
  remove(paperId: string): Promise<void>;
  ask(paperId: string, question: string): Promise<void>;
  toggleSurveySelection(paperId: string): void;
  clearSurveySelection(): void;
  runSurvey(paperIds?: string[]): Promise<void>;
  removeSurvey(id: string): Promise<void>;
  setView(view: View): void;
  openEvidence(evidence: Evidence | null): void;
  toggleTheme(): void;
  toast(message: string, tone?: Toast["tone"]): void;
  dismissToast(id: number): void;
  /** Raw PDF bytes for page previews (from IndexedDB). */
  getPdf(paperId: string): Promise<ArrayBuffer | null>;
}

export const MAX_SURVEY_PAPERS = 10;

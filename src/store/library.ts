/**
 * Application store: the one place where UI intent turns into pipeline runs.
 *
 * - Papers go through a FIFO job queue with concurrency 1, so the browser's
 *   single inference worker and the network are never oversubscribed. Each
 *   run's trace events stream into its PaperRecord (current node, progress,
 *   full trace) so the UI can show the agent working live.
 * - Finished records, PDF bytes and passage embeddings are persisted to
 *   IndexedDB and restored on load; retrieval indexes are rebuilt lazily the
 *   first time a question is asked.
 * - Topic searches and surveys run their own graphs and report into
 *   `search` / `surveys` the same way.
 */
import { create } from "zustand";
import type { CompiledGraph } from "@/agent/graph";
import { ml } from "@/ml/client";
import {
  buildPaperGraph,
  PAPER_PATH_LENGTH,
  paperIdFor,
  titleFromFileName,
  uploadId,
  type PaperPipelineState,
} from "@/pipeline/paper";
import { runTopicSearch } from "@/pipeline/search";
import { MIN_SURVEY_PAPERS, runSurveyPipeline } from "@/pipeline/survey";
import { answerQuestion } from "@/qa/answer";
import { HybridIndex } from "@/retrieval/hybrid";
import { parseInput } from "@/sources/input";
import type { Candidate, PaperMeta, ParsedInput, SurveyPaperInput, TraceEvent } from "@/types";
import * as db from "./db";
import {
  MAX_SURVEY_PAPERS,
  type LibraryState,
  type PaperRecord,
  type SurveyRecord,
  type Toast,
  type View,
} from "./types";

const THEME_KEY = "papertrail:theme";
const TOAST_MS = 5000;
/** Progress never shows 100% until the record is actually ready. */
const MAX_RUNNING_PROGRESS = 0.97;

interface PaperJob {
  id: string;
  input: ParsedInput;
  file?: File;
  pdf?: ArrayBuffer;
  /** Known metadata (from a search result) lets the graph skip the lookup. */
  meta?: PaperMeta;
}

// ───────────────────────── Runtime (non-UI) state ──────────────────────────

const queue: PaperJob[] = [];
let active: { id: string; controller: AbortController } | null = null;
/** Jobs seen this session, so retry can replay uploads without asking again. */
const jobsById = new Map<string, PaperJob>();
const embeddingCache = new Map<string, Float32Array[] | null>();
const indexCache = new Map<string, HybridIndex>();
let paperGraph: CompiledGraph<PaperPipelineState> | null = null;
let searchController: AbortController | null = null;
let hydrating: Promise<void> | null = null;
let toastSeq = 0;

// ───────────────────────────────── Helpers ─────────────────────────────────

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function readTheme(): LibraryState["theme"] {
  try {
    return globalThis.localStorage?.getItem(THEME_KEY) === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

function applyTheme(theme: LibraryState["theme"]): void {
  if (typeof document !== "undefined") document.documentElement.dataset.theme = theme;
  try {
    globalThis.localStorage?.setItem(THEME_KEY, theme);
  } catch {
    // Storage can be disabled; the theme still applies for this session.
  }
}

function newRecord(id: string, meta: PaperMeta): PaperRecord {
  return {
    id,
    meta,
    status: "queued",
    progress: 0,
    trace: [],
    hasPdf: false,
    qa: [],
    qaPending: false,
    warnings: [],
    addedAt: Date.now(),
  };
}

/** What the sidebar shows while the real metadata is being looked up. */
function placeholderMeta(input: ParsedInput, id: string): PaperMeta {
  const base = { id, authors: [], source: "pdf" as const };
  switch (input.kind) {
    case "arxiv":
      return { ...base, title: `arXiv:${input.arxivId}`, arxivId: input.arxivId, version: input.version };
    case "doi":
      return { ...base, title: `doi:${input.doi}`, doi: input.doi };
    default:
      return { ...base, title: titleFromFileName(input.url ?? input.raw), pdfUrl: input.url };
  }
}

/** Rebuilds a typed input from stored metadata (retries after a reload). */
function inputFromMeta(meta: PaperMeta): ParsedInput {
  if (meta.arxivId) return { kind: "arxiv", raw: meta.arxivId, arxivId: meta.arxivId, version: meta.version };
  if (meta.doi) return { kind: "doi", raw: meta.doi, doi: meta.doi };
  const url = meta.pdfUrl ?? meta.landingUrl ?? meta.id;
  return { kind: meta.pdfUrl ? "pdf-url" : "url", raw: url, url };
}

function isPdfFile(file: File): boolean {
  return file.type === "application/pdf" || /\.pdf$/i.test(file.name);
}

// ────────────────────────────────── Store ──────────────────────────────────

export const useLibrary = create<LibraryState>()((set, get) => {
  const patchPaper = (id: string, patch: Partial<PaperRecord> | ((r: PaperRecord) => Partial<PaperRecord>)) =>
    set((s) => {
      const record = s.papers[id];
      if (!record) return {};
      const update = typeof patch === "function" ? patch(record) : patch;
      return { papers: { ...s.papers, [id]: { ...record, ...update } } };
    });

  const patchSurvey = (id: string, patch: Partial<SurveyRecord> | ((r: SurveyRecord) => Partial<SurveyRecord>)) =>
    set((s) => {
      const record = s.surveys[id];
      if (!record) return {};
      const update = typeof patch === "function" ? patch(record) : patch;
      return { surveys: { ...s.surveys, [id]: { ...record, ...update } } };
    });

  const openPaper = (id: string) => set({ view: { kind: "paper", id, tab: "briefing" } });

  const addPaper = (id: string, meta: PaperMeta) =>
    set((s) => ({
      papers: { ...s.papers, [id]: newRecord(id, meta) },
      order: [id, ...s.order.filter((x) => x !== id)],
    }));

  const persistPaper = (id: string) => {
    const record = get().papers[id];
    return record ? db.savePaper(record) : Promise.resolve();
  };

  // ── Paper queue ──

  const enqueue = (job: PaperJob) => {
    jobsById.set(job.id, job);
    queue.push(job);
    void pump();
  };

  const pump = async (): Promise<void> => {
    if (active) return;
    const job = queue.shift();
    if (!job) return;
    active = { id: job.id, controller: new AbortController() };
    try {
      await runJob(job, active.controller.signal);
    } finally {
      active = null;
      void pump();
    }
  };

  const onTrace = (id: string) => {
    let completed = 0;
    return (event: TraceEvent) => {
      if (event.type === "node:end") completed++;
      patchPaper(id, (r) => ({
        trace: [...r.trace, event],
        currentNode: event.type === "node:start" ? event.node : r.currentNode,
        progress: Math.min(MAX_RUNNING_PROGRESS, Math.max(r.progress, completed / PAPER_PATH_LENGTH)),
      }));
    };
  };

  const runJob = async (job: PaperJob, signal: AbortSignal) => {
    const { id } = job;
    patchPaper(id, { status: "running", progress: 0.02, trace: [], error: undefined, warnings: [] });
    paperGraph ??= buildPaperGraph({ ml });

    try {
      const final = await paperGraph.run(
        { input: job.input, file: job.file, pdf: job.pdf, meta: job.meta, paperId: id, mode: "full", warnings: [] },
        {
          signal,
          onEvent: onTrace(id),
          onState: (state) => {
            if (state.meta) patchPaper(id, { meta: state.meta, mode: state.mode, warnings: state.warnings });
          },
        },
      );
      if (!get().papers[id] || !final.meta || !final.analysis) return;

      embeddingCache.set(id, final.embeddings ?? null);
      indexCache.delete(id);
      patchPaper(id, {
        status: "ready",
        progress: 1,
        currentNode: undefined,
        meta: final.meta,
        mode: final.mode,
        analysis: final.analysis,
        doc: final.doc ?? undefined,
        sentences: final.sentences,
        chunks: final.chunks,
        hasPdf: Boolean(final.pdf),
        warnings: final.warnings,
      });
      await Promise.all([
        persistPaper(id),
        final.pdf ? db.savePdf(id, final.pdf) : undefined,
        final.embeddings ? db.saveEmbeddings(id, final.embeddings) : undefined,
      ]);
      get().toast(`Analysis ready: ${final.meta.title}`, "good");
    } catch (error) {
      if (signal.aborted || !get().papers[id]) return;
      const message = errorMessage(error);
      patchPaper(id, { status: "failed", currentNode: undefined, error: message });
      await persistPaper(id);
      get().toast(message, "bad");
    }
  };

  // ── Embeddings & retrieval ──

  const getEmbeddings = async (id: string): Promise<Float32Array[] | null> => {
    if (!embeddingCache.has(id)) embeddingCache.set(id, await db.loadEmbeddings(id));
    return embeddingCache.get(id) ?? null;
  };

  const getIndex = async (record: PaperRecord): Promise<HybridIndex> => {
    let index = indexCache.get(record.id);
    if (!index) {
      index = new HybridIndex(record.chunks ?? [], await getEmbeddings(record.id));
      indexCache.set(record.id, index);
    }
    return index;
  };

  // ── Topic search ──

  const startSearch = async (query: string) => {
    searchController?.abort();
    const controller = new AbortController();
    searchController = controller;
    const isCurrent = () => searchController === controller;
    set({
      view: { kind: "search" },
      search: { query, status: "searching", attempts: [], candidates: [], trace: [] },
    });

    try {
      const { candidates, attempts } = await runTopicSearch(
        query,
        { ml },
        (event) => {
          if (isCurrent()) set((s) => (s.search ? { search: { ...s.search, trace: [...s.search.trace, event] } } : {}));
        },
        controller.signal,
      );
      if (!isCurrent()) return;
      set((s) => ({
        search: {
          query,
          trace: s.search?.trace ?? [],
          attempts,
          candidates,
          status: candidates.length > 0 ? "ready" : "empty",
        },
      }));
    } catch (error) {
      if (!isCurrent() || controller.signal.aborted) return;
      set((s) => (s.search ? { search: { ...s.search, status: "failed", error: errorMessage(error) } } : {}));
    }
  };

  return {
    papers: {},
    order: [],
    view: { kind: "home" },
    search: null,
    surveySelection: [],
    surveys: {},
    surveyOrder: [],
    ml: ml.getStatus(),
    evidence: null,
    theme: readTheme(),
    toasts: [],
    hydrated: false,

    hydrate() {
      hydrating ??= (async () => {
        applyTheme(get().theme);
        set({ ml: ml.getStatus() });
        ml.subscribe((status) => set({ ml: { ...status } }));

        const [papers, surveys] = await Promise.all([db.loadPapers(), db.loadSurveys()]);
        set((s) => {
          const loaded: Record<string, PaperRecord> = {};
          for (const p of papers) {
            // A run cannot survive a reload; anything unfinished is shown as interrupted.
            const unfinished = p.status === "running" || p.status === "queued";
            loaded[p.id] = unfinished
              ? { ...p, status: "failed", error: "Interrupted by a page reload", trace: [], qaPending: false }
              : { ...p, trace: [], qaPending: false };
          }
          const merged = { ...loaded, ...s.papers }; // records created this session win
          const loadedSurveys = Object.fromEntries(surveys.map((sv) => [sv.id, sv]));
          const mergedSurveys = { ...loadedSurveys, ...s.surveys };
          return {
            papers: merged,
            order: Object.values(merged)
              .sort((a, b) => b.addedAt - a.addedAt)
              .map((p) => p.id),
            surveys: mergedSurveys,
            surveyOrder: Object.values(mergedSurveys)
              .sort((a, b) => b.createdAt - a.createdAt)
              .map((sv) => sv.id),
            hydrated: true,
          };
        });
      })();
      return hydrating;
    },

    async submit(raw) {
      const text = raw.trim();
      if (!text) return;
      const input = parseInput(text);
      if (input.kind === "topic") return startSearch(input.query ?? text);

      const id = await paperIdFor(input);
      if (!id) {
        get().toast("That input could not be recognised as a paper", "warn");
        return;
      }
      if (!get().papers[id]) {
        addPaper(id, placeholderMeta(input, id));
        enqueue({ id, input });
      }
      openPaper(id);
    },

    async submitFiles(files) {
      const pdfs = files.filter(isPdfFile);
      if (pdfs.length < files.length) get().toast("Only PDF files can be analysed", "warn");
      let first: string | null = null;
      for (const file of pdfs) {
        try {
          const pdf = await file.arrayBuffer();
          const id = await uploadId(pdf);
          first ??= id;
          if (get().papers[id]) continue;
          addPaper(id, { id, title: titleFromFileName(file.name), authors: [], source: "upload" });
          enqueue({ id, input: { kind: "upload", raw: file.name, fileName: file.name }, file, pdf });
        } catch (error) {
          get().toast(`Could not read ${file.name}: ${errorMessage(error)}`, "bad");
        }
      }
      if (first) openPaper(first);
    },

    async analyzeCandidates(candidates: Candidate[]) {
      const ids: string[] = [];
      for (const { meta } of candidates) {
        ids.push(meta.id);
        if (get().papers[meta.id]) continue;
        addPaper(meta.id, meta);
        enqueue({ id: meta.id, input: inputFromMeta(meta), meta });
      }
      if (ids.length === 0) return;
      // Papers picked together are usually meant to be compared: pre-select them for a survey.
      set((s) => ({
        surveySelection: [...new Set([...s.surveySelection, ...ids])].slice(0, MAX_SURVEY_PAPERS),
      }));
      get().toast(`Analysing ${ids.length} paper${ids.length === 1 ? "" : "s"}`, "info");
      openPaper(ids[0]);
    },

    async retry(paperId) {
      const record = get().papers[paperId];
      if (!record || record.status === "running" || record.status === "queued") return;

      let job = jobsById.get(paperId);
      if (!job && record.meta.source === "upload") {
        const pdf = await db.loadPdf(paperId);
        if (pdf) job = { id: paperId, input: { kind: "upload", raw: record.meta.title, fileName: record.meta.title }, pdf };
      }
      job ??= record.meta.source === "upload" ? undefined : { id: paperId, input: inputFromMeta(record.meta) };
      if (!job) {
        get().toast("The original file is no longer available; upload it again", "warn");
        return;
      }

      embeddingCache.delete(paperId);
      indexCache.delete(paperId);
      patchPaper(paperId, {
        status: "queued",
        progress: 0,
        trace: [],
        error: undefined,
        analysis: undefined,
        doc: undefined,
        sentences: undefined,
        chunks: undefined,
      });
      enqueue(job);
    },

    async remove(paperId) {
      const queued = queue.findIndex((j) => j.id === paperId);
      if (queued >= 0) queue.splice(queued, 1);
      if (active?.id === paperId) active.controller.abort();
      jobsById.delete(paperId);
      embeddingCache.delete(paperId);
      indexCache.delete(paperId);

      set((s) => {
        const papers = { ...s.papers };
        delete papers[paperId];
        const viewing = s.view.kind === "paper" && s.view.id === paperId;
        return {
          papers,
          order: s.order.filter((id) => id !== paperId),
          surveySelection: s.surveySelection.filter((id) => id !== paperId),
          view: viewing ? { kind: "home" } : s.view,
          evidence: s.evidence?.paperId === paperId ? null : s.evidence,
        };
      });
      await db.deletePaper(paperId);
    },

    async ask(paperId, question) {
      const q = question.trim();
      const record = get().papers[paperId];
      if (!q || !record || record.qaPending) return;
      if (record.status !== "ready" || !record.chunks || !record.sentences) {
        get().toast("Questions can be asked once the analysis is ready", "info");
        return;
      }

      patchPaper(paperId, { qaPending: true });
      try {
        const answer = await answerQuestion(q, {
          paperId,
          index: await getIndex(record),
          sentences: record.sentences,
          analysis: record.analysis,
          ml: { embed: ml.embed, rerank: ml.rerank, answer: ml.answer },
        });
        patchPaper(paperId, (r) => ({ qa: [...r.qa, answer], qaPending: false }));
        await persistPaper(paperId);
      } catch (error) {
        patchPaper(paperId, { qaPending: false });
        get().toast(`Could not answer: ${errorMessage(error)}`, "bad");
      }
    },

    toggleSurveySelection(paperId) {
      const selected = get().surveySelection;
      if (selected.includes(paperId)) {
        set({ surveySelection: selected.filter((id) => id !== paperId) });
      } else if (selected.length >= MAX_SURVEY_PAPERS) {
        get().toast(`A survey can include up to ${MAX_SURVEY_PAPERS} papers`, "warn");
      } else {
        set({ surveySelection: [...selected, paperId] });
      }
    },

    clearSurveySelection() {
      set({ surveySelection: [] });
    },

    async runSurvey(paperIds) {
      const requested = [...new Set(paperIds ?? get().surveySelection)];
      const ready = requested
        .map((id) => get().papers[id])
        .filter((r): r is PaperRecord => r?.status === "ready" && !!r.analysis && !!r.chunks && !!r.sentences);
      if (ready.length < MIN_SURVEY_PAPERS) {
        get().toast(`A survey needs at least ${MIN_SURVEY_PAPERS} analysed papers`, "warn");
        return;
      }
      const skipped = requested.length - ready.length;
      if (skipped > 0) get().toast(`Skipping ${skipped} paper${skipped === 1 ? "" : "s"} still being analysed`, "info");

      const inputs: SurveyPaperInput[] = await Promise.all(
        ready.map(async (r) => ({
          meta: r.meta,
          analysis: r.analysis!,
          sentences: r.sentences!,
          chunks: r.chunks!,
          chunkEmbeddings: await getEmbeddings(r.id),
          references: r.doc?.references ?? [],
        })),
      );

      const id = `survey-${Date.now().toString(36)}`;
      const record: SurveyRecord = {
        id,
        paperIds: ready.map((r) => r.id),
        status: "running",
        trace: [],
        createdAt: Date.now(),
      };
      set((s) => ({
        surveys: { ...s.surveys, [id]: record },
        surveyOrder: [id, ...s.surveyOrder],
        view: { kind: "survey", id },
      }));

      try {
        const result = await runSurveyPipeline(inputs, { ml }, (event) =>
          patchSurvey(id, (r) => ({ trace: [...r.trace, event] })),
        );
        if (!get().surveys[id]) return;
        patchSurvey(id, { status: "ready", result });
        await db.saveSurvey(get().surveys[id]);
        get().toast(`Literature survey of ${ready.length} papers is ready`, "good");
      } catch (error) {
        if (!get().surveys[id]) return;
        patchSurvey(id, { status: "failed", error: errorMessage(error) });
        get().toast(`Survey failed: ${errorMessage(error)}`, "bad");
      }
    },

    async removeSurvey(id) {
      set((s) => {
        const surveys = { ...s.surveys };
        delete surveys[id];
        const viewing = s.view.kind === "survey" && s.view.id === id;
        return {
          surveys,
          surveyOrder: s.surveyOrder.filter((x) => x !== id),
          view: viewing ? { kind: "home" } : s.view,
        };
      });
      await db.deleteSurvey(id);
    },

    setView(view: View) {
      set({ view });
    },

    openEvidence(evidence) {
      set({ evidence });
    },

    toggleTheme() {
      const theme = get().theme === "dark" ? "light" : "dark";
      applyTheme(theme);
      set({ theme });
    },

    toast(message, tone: Toast["tone"] = "info") {
      const id = ++toastSeq;
      set((s) => ({ toasts: [...s.toasts, { id, tone, message }] }));
      setTimeout(() => get().dismissToast(id), TOAST_MS);
    },

    dismissToast(id) {
      set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
    },

    getPdf(paperId) {
      return db.loadPdf(paperId);
    },
  };
});

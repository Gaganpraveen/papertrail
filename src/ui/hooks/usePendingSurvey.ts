/**
 * "Survey these papers once they are analysed."
 *
 * The sample survey submits several arXiv ids at once, but a survey can only
 * run on finished analyses. This small store remembers the intent and
 * `useSurveyIntentRunner` starts the survey as soon as every paper has
 * settled; papers that failed are left out (a survey needs at least MIN_SURVEY_PAPERS).
 */
import { useEffect } from "react";
import { create } from "zustand";
import { MIN_SURVEY_PAPERS } from "@/pipeline/survey";
import type { PaperRecord } from "@/store/types";
import { store, useStore } from "./store";

interface PendingSurveyState {
  arxivIds: string[] | null;
  start(arxivIds: string[]): void;
  cancel(): void;
}

export const usePendingSurvey = create<PendingSurveyState>()((set) => ({
  arxivIds: null,
  start: (arxivIds) => set({ arxivIds }),
  cancel: () => set({ arxivIds: null }),
}));

/** Records are keyed "arxiv:<id>" once resolved; match on metadata too. */
export function findArxivRecord(
  papers: Record<string, PaperRecord>,
  arxivId: string,
): PaperRecord | undefined {
  return (
    papers[`arxiv:${arxivId}`] ??
    Object.values(papers).find((record) => record.meta.arxivId === arxivId)
  );
}

export interface SurveyIntentProgress {
  total: number;
  ready: number;
  failed: number;
  settled: boolean;
}

export function intentProgress(
  papers: Record<string, PaperRecord>,
  arxivIds: string[],
): SurveyIntentProgress {
  const records = arxivIds.map((id) => findArxivRecord(papers, id));
  const ready = records.filter((r) => r?.status === "ready").length;
  const failed = records.filter((r) => r?.status === "failed").length;
  return { total: arxivIds.length, ready, failed, settled: ready + failed === arxivIds.length };
}

/** Submits the papers (sequentially, so the queue keeps their order) and arms the intent. */
export async function submitForSurvey(arxivIds: string[]): Promise<void> {
  usePendingSurvey.getState().start(arxivIds);
  const { submit } = store();
  for (const id of arxivIds) await submit(id);
  // Each submit opens its paper; land on the first one, which the queue processes first.
  const first = findArxivRecord(store().papers, arxivIds[0]);
  if (first) store().setView({ kind: "paper", id: first.id, tab: "briefing" });
}

/** Mount once: watches the library and launches the survey when the papers are ready. */
export function useSurveyIntentRunner(): void {
  const arxivIds = usePendingSurvey((s) => s.arxivIds);
  const papers = useStore((s) => s.papers);

  useEffect(() => {
    if (!arxivIds) return;
    const progress = intentProgress(papers, arxivIds);
    if (!progress.settled) return;

    usePendingSurvey.getState().cancel();
    const readyIds = arxivIds
      .map((id) => findArxivRecord(papers, id))
      .filter((r): r is PaperRecord => r?.status === "ready")
      .map((r) => r.id);
    const { runSurvey, toast } = store();
    if (readyIds.length < MIN_SURVEY_PAPERS) {
      toast(`Survey cancelled: fewer than ${MIN_SURVEY_PAPERS} papers could be analysed.`, "bad");
      return;
    }
    if (progress.failed > 0) {
      toast(`${progress.failed} paper(s) failed; surveying the other ${readyIds.length}.`, "warn");
    }
    void runSurvey(readyIds);
  }, [arxivIds, papers]);
}

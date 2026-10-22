/**
 * Picks which pipeline run the trace dock shows: the job behind the current
 * view, or on the home screen the paper being analysed right now.
 */
import { PAPER_NODES } from "@/pipeline/paper";
import { SEARCH_NODES } from "@/pipeline/search";
import { SURVEY_NODES } from "@/pipeline/survey";
import type { PaperRecord, SearchSession, SurveyRecord } from "@/store/types";
import type { TraceEvent } from "@/types";
import { useStore } from "@/ui/hooks/store";
import type { JobStatus, NodeInfo } from "./traceModel";

export interface TraceJob {
  kind: "paper" | "search" | "survey";
  key: string;
  title: string;
  nodes: NodeInfo[];
  events: TraceEvent[];
  status: JobStatus;
  /** Node the job is in right now, when the store reports it. */
  currentNode?: string;
  error?: string;
}

function paperJob(record: PaperRecord): TraceJob {
  return {
    kind: "paper",
    key: `paper:${record.id}`,
    title: record.meta.title || record.id,
    nodes: PAPER_NODES,
    events: record.trace,
    status: record.status === "ready" ? "done" : record.status === "failed" ? "failed" : "running",
    currentNode: record.currentNode,
    error: record.error,
  };
}

function searchJob(search: SearchSession): TraceJob {
  return {
    kind: "search",
    key: `search:${search.query}`,
    title: `Topic search · “${search.query}”`,
    nodes: SEARCH_NODES,
    events: search.trace,
    status: search.status === "searching" ? "running" : search.status === "failed" ? "failed" : "done",
    error: search.error,
  };
}

function surveyJob(survey: SurveyRecord): TraceJob {
  return {
    kind: "survey",
    key: `survey:${survey.id}`,
    title: `Literature survey · ${survey.paperIds.length} papers`,
    nodes: SURVEY_NODES,
    events: survey.trace,
    status: survey.status === "running" ? "running" : survey.status === "failed" ? "failed" : "done",
    error: survey.error,
  };
}

export function useActiveJob(): TraceJob | null {
  const view = useStore((s) => s.view);
  const paper = useStore((s) => (s.view.kind === "paper" ? s.papers[s.view.id] : undefined));
  const survey = useStore((s) => (s.view.kind === "survey" ? s.surveys[s.view.id] : undefined));
  const search = useStore((s) => s.search);
  // On other screens, follow whatever paper is being analysed (else the latest one).
  const background = useStore((s) => {
    const running = s.order.find((id) => s.papers[id]?.status === "running");
    return s.papers[running ?? s.order[0] ?? ""];
  });

  if (view.kind === "paper") return paper ? paperJob(paper) : null;
  if (view.kind === "survey") return survey ? surveyJob(survey) : null;
  if (view.kind === "search") return search ? searchJob(search) : null;
  if (background?.status === "running") return paperJob(background);
  if (search?.status === "searching") return searchJob(search);
  return background ? paperJob(background) : null;
}

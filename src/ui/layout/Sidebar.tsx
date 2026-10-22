/**
 * Left rail: library of analysed papers (with live status and survey
 * selection), saved surveys, model status and theme toggle.
 */
import { BookOpen, Check, Layers, Moon, Plus, Sun, Trash2, X } from "lucide-react";
import { PAPER_NODES } from "@/pipeline/paper";
import { MIN_SURVEY_PAPERS } from "@/pipeline/survey";
import { MAX_SURVEY_PAPERS, type PaperRecord, type SurveyRecord } from "@/store/types";
import { Button, IconButton, SectionLabel, Spinner, cn } from "@/ui/kit";
import { store, useStore } from "@/ui/hooks/store";
import { focusCommandBar } from "@/ui/hooks/useHotkeys";
import { Brand } from "./Brand";
import { ModelStatus } from "./ModelStatus";

const statusDot: Record<PaperRecord["status"], string> = {
  queued: "bg-line-strong",
  running: "bg-accent animate-soft-pulse",
  ready: "bg-good",
  failed: "bg-bad",
};

function shortAuthors(authors: string[]): string {
  if (authors.length === 0) return "";
  const first = authors[0].split(/\s+/).pop() ?? authors[0];
  return authors.length > 1 ? `${first} et al.` : first;
}

function paperSubtitle(record: PaperRecord, nodeLabel?: string): string {
  if (record.status === "queued") return "Queued";
  if (record.status === "running") return nodeLabel ? `${nodeLabel}…` : "Analysing…";
  if (record.status === "failed") return "Failed · click for details";
  const parts = [shortAuthors(record.meta.authors), record.meta.year?.toString()].filter(Boolean);
  if (parts.length === 0 && record.meta.source === "upload") parts.push("Uploaded PDF");
  if (record.mode === "abstract-only") parts.push("abstract only");
  return parts.join(" · ") || record.id;
}

function LibraryRow({
  record,
  active,
  selected,
  selectionMode,
  onNavigate,
}: {
  record: PaperRecord;
  active: boolean;
  selected: boolean;
  selectionMode: boolean;
  onNavigate: () => void;
}) {
  const selectable = record.status === "ready";
  const open = () => {
    store().setView({ kind: "paper", id: record.id, tab: "briefing" });
    onNavigate();
  };
  const nodeLabel = PAPER_NODES.find((n) => n.name === record.currentNode)?.label;

  return (
    <li
      className={cn(
        "group relative flex items-start gap-2 rounded-lg px-2 py-2 transition-colors",
        active ? "bg-panel-2" : "hover:bg-panel-2/60",
      )}
    >
      <button
        role="checkbox"
        aria-checked={selected}
        aria-label={selected ? "Remove from survey" : "Add to survey"}
        title={selectable ? "Select for literature survey" : "Available once analysed"}
        disabled={!selectable}
        onClick={() => store().toggleSurveySelection(record.id)}
        className={cn(
          "mt-0.5 flex size-4 shrink-0 items-center justify-center rounded border transition-all",
          selected ? "border-accent bg-accent text-accent-ink" : "border-line-strong bg-panel",
          !selected && !selectionMode && "opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
          !selectable && "hidden",
        )}
      >
        {selected && <Check className="size-3" strokeWidth={3} />}
      </button>
      {!selectable && (
        <span className="mt-1.5 flex size-4 shrink-0 items-center justify-center">
          <span className={cn("size-2 rounded-full", statusDot[record.status])} />
        </span>
      )}
      <button onClick={open} className="min-w-0 flex-1 text-left">
        <span className="line-clamp-2 text-[13px] leading-snug font-medium text-ink">
          {record.meta.title || record.id}
        </span>
        <span
          className={cn(
            "mt-0.5 block truncate text-[11px]",
            record.status === "failed" ? "text-bad" : "text-ink-3",
          )}
        >
          {paperSubtitle(record, nodeLabel)}
        </span>
        {record.status === "running" && (
          <span className="mt-1.5 block h-0.5 overflow-hidden rounded-full bg-line">
            <span
              className="block h-full rounded-full bg-accent transition-[width] duration-500"
              style={{ width: `${Math.max(4, record.progress * 100)}%` }}
            />
          </span>
        )}
      </button>
      {selectable && (
        <span className={cn("mt-1.5 size-1.5 shrink-0 rounded-full", statusDot[record.status])} />
      )}
      <button
        onClick={() => void store().remove(record.id)}
        aria-label="Remove paper"
        title="Remove from library"
        className="absolute top-1.5 right-1.5 hidden rounded-md bg-panel-2 p-1 text-ink-3 group-hover:block hover:text-bad"
      >
        <Trash2 className="size-3.5" />
      </button>
    </li>
  );
}

function SurveyRow({ survey, active, onNavigate }: { survey: SurveyRecord; active: boolean; onNavigate: () => void }) {
  const date = new Date(survey.createdAt).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return (
    <li>
      <button
        onClick={() => {
          store().setView({ kind: "survey", id: survey.id });
          onNavigate();
        }}
        className={cn(
          "flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left transition-colors",
          active ? "bg-panel-2" : "hover:bg-panel-2/60",
        )}
      >
        <Layers className="size-4 shrink-0 text-ink-3" />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] font-medium text-ink">
            Survey · {survey.paperIds.length} papers
          </span>
          <span className="block text-[11px] text-ink-3">
            {survey.status === "running" ? "Synthesising…" : survey.status === "failed" ? "Failed" : date}
          </span>
        </span>
        {survey.status === "running" && <Spinner className="size-3.5" />}
      </button>
    </li>
  );
}

export function Sidebar({ onNavigate = () => {}, onClose }: { onNavigate?: () => void; onClose?: () => void }) {
  const papers = useStore((s) => s.papers);
  const order = useStore((s) => s.order);
  const view = useStore((s) => s.view);
  const selection = useStore((s) => s.surveySelection);
  const surveys = useStore((s) => s.surveys);
  const surveyOrder = useStore((s) => s.surveyOrder);
  const theme = useStore((s) => s.theme);

  const records = order.map((id) => papers[id]).filter((r): r is PaperRecord => Boolean(r));
  const readyCount = records.filter((r) => r.status === "ready").length;
  const selectionMode = selection.length > 0;

  const newAnalysis = () => {
    store().setView({ kind: "home" });
    onNavigate();
    window.setTimeout(focusCommandBar, 50);
  };

  const selectAllReady = () => {
    const { toggleSurveySelection } = store();
    records
      .filter((r) => r.status === "ready" && !selection.includes(r.id))
      .slice(0, MAX_SURVEY_PAPERS - selection.length)
      .forEach((r) => toggleSurveySelection(r.id));
  };

  return (
    <aside className="flex h-full w-[264px] shrink-0 flex-col border-r border-line bg-panel">
      <div className="flex h-14 items-center justify-between px-4">
        <Brand onClick={newAnalysis} />
        {onClose ? (
          <IconButton label="Close menu" onClick={onClose}>
            <X className="size-4" />
          </IconButton>
        ) : (
          <IconButton label="New analysis" onClick={newAnalysis}>
            <Plus className="size-4" />
          </IconButton>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        <div className="flex items-center justify-between px-2 pt-2 pb-1.5">
          <SectionLabel>Library {records.length > 0 && <span className="tabular">· {records.length}</span>}</SectionLabel>
          {readyCount >= MIN_SURVEY_PAPERS && !selectionMode && (
            <button onClick={selectAllReady} className="text-[11px] text-ink-3 hover:text-accent">
              Select all
            </button>
          )}
          {selectionMode && (
            <button onClick={() => store().clearSurveySelection()} className="text-[11px] text-ink-3 hover:text-ink">
              Clear
            </button>
          )}
        </div>

        {records.length === 0 ? (
          <div className="mx-2 mt-1 rounded-xl border border-dashed border-line px-3 py-5 text-center">
            <BookOpen className="mx-auto size-5 text-ink-3" />
            <p className="mt-2 text-xs font-medium text-ink-2">No papers yet</p>
            <p className="mt-1 text-[11px] leading-relaxed text-ink-3">
              Paste an arXiv ID or drop a PDF anywhere to start.
            </p>
          </div>
        ) : (
          <ul className="space-y-0.5">
            {records.map((record) => (
              <LibraryRow
                key={record.id}
                record={record}
                active={view.kind === "paper" && view.id === record.id}
                selected={selection.includes(record.id)}
                selectionMode={selectionMode}
                onNavigate={onNavigate}
              />
            ))}
          </ul>
        )}

        {records.length > 0 && (
          <div className="mx-2 mt-3">
            <Button
              variant={selection.length >= MIN_SURVEY_PAPERS ? "primary" : "secondary"}
              size="sm"
              className="w-full"
              disabled={selection.length < MIN_SURVEY_PAPERS}
              onClick={() => {
                void store().runSurvey(selection);
                onNavigate();
              }}
            >
              <Layers className="size-3.5" />
              Literature survey ({selection.length})
            </Button>
            <p className="mt-1.5 text-center text-[11px] text-ink-3">
              {selection.length < MIN_SURVEY_PAPERS
                ? `Tick ${MIN_SURVEY_PAPERS}–${MAX_SURVEY_PAPERS} analysed papers to compare them`
                : `${selection.length} of ${MAX_SURVEY_PAPERS} selected`}
            </p>
          </div>
        )}

        {surveyOrder.length > 0 && (
          <>
            <SectionLabel className="px-2 pt-5 pb-1.5">Surveys</SectionLabel>
            <ul className="space-y-0.5">
              {surveyOrder
                .map((id) => surveys[id])
                .filter((s): s is SurveyRecord => Boolean(s))
                .map((survey) => (
                  <SurveyRow
                    key={survey.id}
                    survey={survey}
                    active={view.kind === "survey" && view.id === survey.id}
                    onNavigate={onNavigate}
                  />
                ))}
            </ul>
          </>
        )}
      </div>

      <div className="flex items-center gap-1 border-t border-line p-2">
        <ModelStatus />
        <IconButton
          label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          onClick={() => store().toggleTheme()}
        >
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </IconButton>
      </div>
    </aside>
  );
}

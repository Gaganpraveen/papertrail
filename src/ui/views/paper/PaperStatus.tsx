/**
 * Non-ready states of a paper: a live pipeline view while it is queued or
 * running, and a recovery card when it failed.
 */
import { useRef } from "react";
import { Check, RotateCw, TriangleAlert, Upload, X } from "lucide-react";
import { PAPER_NODES } from "@/pipeline/paper";
import { useLibrary } from "@/store/library";
import type { PaperRecord } from "@/store/types";
import type { TraceEvent } from "@/types";
import { Button, Card, Progress, Skeleton, Spinner, cn } from "@/ui/kit";
import { ModelDownload } from "@/ui/layout/ModelDownload";

/** What each step is doing, in plain words, while the user waits. */
const STEP_MESSAGES: Record<string, string> = {
  understand: "Working out what you pasted.",
  resolve: "Finding the paper's title, authors and PDF.",
  fetch: "Downloading the PDF.",
  parse: "Reading the PDF: sections, references and captions.",
  fallback: "No readable full text, so continuing from the abstract.",
  segment: "Splitting the text into sentences, keeping page numbers.",
  zone: "Tagging each sentence as problem, method, result or limitation.",
  chunk: "Grouping sentences into short passages.",
  embed: "Indexing the passages so you can search them by meaning.",
  analyze: "Writing the briefing, critique and charts.",
};

type StepState = "pending" | "running" | "done" | "error";

function stepStates(trace: TraceEvent[], currentNode?: string): Map<string, { state: StepState; ms?: number }> {
  const states = new Map<string, { state: StepState; ms?: number }>();
  for (const e of trace) {
    if (e.type === "node:start") states.set(e.node, { state: "running" });
    else if (e.type === "node:end") states.set(e.node, { state: "done", ms: e.durationMs });
    else if (e.type === "node:error") states.set(e.node, { state: "error", ms: e.durationMs });
  }
  if (currentNode && states.get(currentNode)?.state !== "done") states.set(currentNode, { state: "running" });
  return states;
}

function StepPill({ label, state, ms }: { label: string; state: StepState; ms?: number }) {
  return (
    <li
      className={cn(
        "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors",
        state === "done" && "border-line text-ink-2",
        state === "running" && "border-accent bg-accent-soft text-accent",
        state === "error" && "border-bad bg-bad-soft text-bad",
        state === "pending" && "border-dashed border-line text-ink-3",
      )}
    >
      {state === "done" && <Check className="size-3 text-good" />}
      {state === "running" && <Spinner className="size-3" />}
      {state === "error" && <X className="size-3" />}
      {label}
      {ms !== undefined && state === "done" && <span className="tabular text-[10px] text-ink-3">{ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`}</span>}
    </li>
  );
}

function ContentSkeleton() {
  return (
    <div className="space-y-4" aria-hidden>
      <div className="flex gap-4 border-b border-line pb-2.5">
        {["w-16", "w-24", "w-16", "w-11"].map((w, i) => (
          <Skeleton key={i} className={cn("h-4", w)} />
        ))}
      </div>
      <Skeleton className="h-28 w-full rounded-2xl" />
      <div className="grid gap-4 lg:grid-cols-3">
        <Skeleton className="h-56 rounded-2xl lg:col-span-2" />
        <Skeleton className="h-56 rounded-2xl" />
      </div>
    </div>
  );
}

export function RunningState({ record }: { record: PaperRecord }) {
  const states = stepStates(record.trace, record.currentNode);
  const current = PAPER_NODES.find((n) => n.name === record.currentNode);
  // The abstract fallback only appears in the stepper once the graph routes to it.
  const steps = PAPER_NODES.filter((n) => n.name !== "fallback" || states.has("fallback"));
  const queued = record.status === "queued";

  return (
    <div className="space-y-6 animate-in">
      <Card className="p-5">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent-soft">
            <Spinner />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline justify-between gap-3">
              <p className="text-sm font-semibold text-ink">
                {queued ? "Waiting in queue" : current ? current.label : "Starting"}
              </p>
              <span className="tabular text-xs text-ink-3">{Math.round(record.progress * 100)}%</span>
            </div>
            <p className="mt-0.5 text-xs text-ink-3">
              {queued
                ? "Papers are analysed one at a time so the in-browser models stay responsive."
                : current
                  ? (STEP_MESSAGES[current.name] ?? current.description)
                  : "Getting ready."}
            </p>
            <Progress value={record.progress} className="mt-3" />
          </div>
        </div>
        <div className="mt-4 empty:hidden">
          <ModelDownload keys={["embedder"]} />
        </div>
        <ol className="mt-4 flex flex-wrap gap-1.5" aria-label="Pipeline steps">
          {steps.map((n) => {
            const s = states.get(n.name);
            return <StepPill key={n.name} label={n.label} state={s?.state ?? "pending"} ms={s?.ms} />;
          })}
        </ol>
      </Card>
      <ContentSkeleton />
    </div>
  );
}

/** Button that opens a file picker and hands the chosen PDF to the library. */
export function UploadPdfButton({ variant = "primary" }: { variant?: "primary" | "secondary" }) {
  const submitFiles = useLibrary((s) => s.submitFiles);
  const fileInput = useRef<HTMLInputElement>(null);
  return (
    <>
      <Button size="sm" variant={variant} onClick={() => fileInput.current?.click()}>
        <Upload className="size-3.5" />
        Upload PDF
      </Button>
      <input
        ref={fileInput}
        type="file"
        accept="application/pdf,.pdf"
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) void submitFiles(files);
          e.target.value = "";
        }}
      />
    </>
  );
}

export function FailedState({ record }: { record: PaperRecord }) {
  const retry = useLibrary((s) => s.retry);

  return (
    <Card className="overflow-hidden animate-in">
      <div className="flex items-start gap-3 border-b border-line bg-bad-soft/40 px-5 py-4">
        <TriangleAlert className="mt-0.5 size-4 shrink-0 text-bad" />
        <div className="min-w-0">
          <p className="text-sm font-semibold text-ink">This paper could not be analysed</p>
          <p className="mt-1 text-xs leading-relaxed break-words text-ink-2">
            {record.error ?? "The pipeline stopped before producing an analysis."}
          </p>
        </div>
      </div>
      <div className="flex flex-col gap-4 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
        <p className="max-w-xl text-xs leading-relaxed text-ink-3">
          Some publishers block downloads from other websites. Upload the PDF instead: it is parsed on your
          device and never leaves it.
        </p>
        <div className="flex shrink-0 gap-2">
          <Button size="sm" variant="secondary" onClick={() => void retry(record.id)}>
            <RotateCw className="size-3.5" />
            Retry
          </Button>
          <UploadPdfButton />
        </div>
      </div>
    </Card>
  );
}

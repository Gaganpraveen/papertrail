/**
 * The trace dock: a live view of the state graph executing the active job.
 * Each node is a pill whose status, duration and retries update as trace
 * events stream in; conditional routes and log lines appear alongside, so
 * a viewer can watch the pipeline decide what to do next.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ChevronDown,
  ChevronUp,
  CircleCheck,
  CircleDashed,
  CircleSlash,
  CircleX,
  GitBranch,
  LoaderCircle,
} from "lucide-react";
import { cn } from "@/ui/kit";
import { usePersistentState } from "@/ui/hooks/usePersistentState";
import type { TraceEvent } from "@/types";
import { useActiveJob, type TraceJob } from "./useActiveJob";
import {
  deriveNodes,
  deriveRoutes,
  describeEvent,
  elapsedMs,
  formatDuration,
  type NodeStatus,
  type NodeView,
} from "./traceModel";

const END_NODE = "__end__";

const statusStyle: Record<NodeStatus, string> = {
  pending: "border-line text-ink-3",
  running: "border-accent text-ink bg-accent-soft shadow-[0_0_0_3px_var(--accent-soft)]",
  done: "border-line-strong text-ink-2 bg-panel",
  failed: "border-bad text-bad bg-bad-soft",
  skipped: "border-dashed border-line text-ink-3 opacity-55",
};

const dotStyle: Record<NodeStatus, string> = {
  pending: "bg-line-strong",
  running: "bg-accent animate-soft-pulse",
  done: "bg-good",
  failed: "bg-bad",
  skipped: "bg-transparent border border-dashed border-line-strong",
};

function StatusIcon({ status }: { status: NodeStatus }) {
  const cls = "size-3.5 shrink-0";
  switch (status) {
    case "running":
      return <LoaderCircle className={cn(cls, "animate-spin text-accent")} />;
    case "done":
      return <CircleCheck className={cn(cls, "text-good")} />;
    case "failed":
      return <CircleX className={cn(cls, "text-bad")} />;
    case "skipped":
      return <CircleSlash className={cls} />;
    default:
      return <CircleDashed className={cls} />;
  }
}

/** Re-renders on an interval while `active`, for the live elapsed timer. */
function useNow(active: boolean, intervalMs = 250): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(timer);
  }, [active, intervalMs]);
  return now;
}

function NodePill({ node }: { node: NodeView }) {
  const tooltip = [node.description, node.message].filter(Boolean).join("\n\n");
  return (
    <div
      title={tooltip}
      className={cn(
        "flex min-w-[104px] shrink-0 flex-col gap-1 rounded-xl border px-3 py-2 transition-all duration-300",
        statusStyle[node.status],
      )}
    >
      <div className="flex items-center gap-1.5">
        <StatusIcon status={node.status} />
        <span className="text-xs font-medium whitespace-nowrap">{node.label}</span>
        {node.visits > 1 && (
          <span className="rounded bg-panel-2 px-1 text-[10px] tabular text-ink-3">×{node.visits}</span>
        )}
      </div>
      <div className="flex items-center gap-1.5 text-[10px] tabular text-ink-3">
        {node.status === "done" || node.status === "failed"
          ? formatDuration(node.durationMs)
          : node.status === "running"
            ? "running…"
            : node.status}
        {node.retries > 0 && <span className="text-warn">· {node.retries} retry</span>}
      </div>
    </div>
  );
}

function Connector({ active }: { active: boolean }) {
  return (
    <div className="relative flex w-6 shrink-0 items-center" aria-hidden>
      <div className={cn("h-px w-full", active ? "bg-accent" : "bg-line-strong")} />
      <div
        className={cn(
          "absolute right-0 size-1.5 rotate-45 border-t border-r",
          active ? "border-accent" : "border-line-strong",
        )}
      />
    </div>
  );
}

function EventLog({ events, labelOf }: { events: TraceEvent[]; labelOf: (n: string) => string }) {
  const ref = useRef<HTMLOListElement>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [events.length]);
  const start = events[0]?.at ?? 0;
  return (
    <ol ref={ref} className="max-h-28 min-h-0 overflow-y-auto font-mono text-[11px] leading-5">
      {events.map((event, i) => (
        <li key={i} className="flex gap-3 px-1">
          <span className="w-14 shrink-0 text-right tabular text-ink-3">
            +{((event.at - start) / 1000).toFixed(2)}s
          </span>
          <span
            className={cn(
              "min-w-0 break-words",
              event.type === "node:error" && "text-bad",
              event.type === "node:retry" && "text-warn",
              event.type === "route" && "text-accent",
              event.type === "log" && "text-ink-2",
              (event.type === "node:start" || event.type === "node:end") && "text-ink-3",
            )}
          >
            {describeEvent(event, labelOf)}
          </span>
        </li>
      ))}
    </ol>
  );
}

function summary(job: TraceJob, nodes: NodeView[], elapsed: number): string {
  if (job.status === "running") {
    const current = nodes.find((n) => n.status === "running") ?? nodes.find((n) => n.name === job.currentNode);
    return current ? `${current.label} · ${formatDuration(elapsed)}` : `Queued`;
  }
  if (job.status === "failed") {
    const failed = nodes.find((n) => n.status === "failed");
    return failed ? `Failed at ${failed.label}` : "Failed";
  }
  return `Completed in ${formatDuration(elapsed)}`;
}

export function TraceDock() {
  const job = useActiveJob();
  // null = automatic: expanded while a job runs, tucked away once it finishes.
  const [preference, setPreference] = usePersistentState<boolean | null>("papertrail:trace-open", null);
  const running = job?.status === "running";
  const open = preference ?? running;
  const setOpen = (next: boolean) => setPreference(next);
  const now = useNow(running);

  const nodeList = job?.nodes;
  const events = job?.events;
  const status = job?.status;
  const nodes = useMemo(
    () => (nodeList && events && status ? deriveNodes(nodeList, events, status) : []),
    [nodeList, events, status],
  );
  const routes = useMemo(() => (events ? deriveRoutes(events) : []), [events]);
  const labelOf = (name: string) =>
    name === END_NODE ? "End" : (nodes.find((n) => n.name === name)?.label ?? name);

  if (!job) return null;
  const elapsed = elapsedMs(job.events, job.status, now);

  return (
    <section
      aria-label="Agent trace"
      className="shrink-0 border-t border-line bg-panel/80 backdrop-blur supports-[backdrop-filter]:bg-panel/70"
    >
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex h-10 w-full items-center gap-3 px-4 text-left hover:bg-panel-2/60"
      >
        <span className="flex items-center gap-2 text-xs font-semibold text-ink">
          <Activity className={cn("size-4", running ? "text-accent" : "text-ink-3")} />
          Agent trace
        </span>
        <span className="hidden h-4 w-px bg-line sm:block" />
        <span className="hidden min-w-0 flex-1 truncate text-xs text-ink-3 sm:block">{job.title}</span>
        <span className="flex items-center gap-1" aria-hidden>
          {nodes.map((n) => (
            <span key={n.name} className={cn("size-1.5 rounded-full", dotStyle[n.status])} />
          ))}
        </span>
        <span
          className={cn(
            "ml-auto shrink-0 text-xs tabular sm:ml-0",
            job.status === "failed" ? "text-bad" : running ? "text-accent" : "text-ink-3",
          )}
        >
          {summary(job, nodes, elapsed)}
        </span>
        {open ? <ChevronDown className="size-4 text-ink-3" /> : <ChevronUp className="size-4 text-ink-3" />}
      </button>

      {open && (
        <div className="animate-in border-t border-line px-4 pt-3 pb-4">
          <div className="flex items-center overflow-x-auto pb-2">
            {nodes.map((node, i) => (
              <div key={node.name} className="flex shrink-0 items-center">
                {i > 0 && <Connector active={node.status === "done" || node.status === "running"} />}
                <NodePill node={node} />
              </div>
            ))}
          </div>
          <div className="mt-2 grid gap-4 md:grid-cols-[minmax(0,220px)_minmax(0,1fr)]">
            <div className="min-w-0">
              <div className="mb-1.5 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-ink-3">
                <GitBranch className="size-3.5" /> Routing decisions
              </div>
              {routes.length === 0 ? (
                <p className="text-xs text-ink-3">No conditional branch taken yet.</p>
              ) : (
                <ul className="flex flex-wrap gap-1.5">
                  {routes.map((r, i) => (
                    <li
                      key={i}
                      className="rounded-md border border-line bg-panel-2 px-1.5 py-0.5 text-[11px] text-ink-2"
                    >
                      {labelOf(r.from)} <span className="text-accent">→</span> {labelOf(r.to)}
                    </li>
                  ))}
                </ul>
              )}
              {job.error && <p className="mt-2 text-xs leading-relaxed text-bad">{job.error}</p>}
            </div>
            <div className="min-w-0 rounded-lg border border-line bg-bg/60 py-1.5">
              {job.events.length === 0 ? (
                <p className="px-3 py-2 text-xs text-ink-3">Waiting in the queue…</p>
              ) : (
                <EventLog events={job.events} labelOf={labelOf} />
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

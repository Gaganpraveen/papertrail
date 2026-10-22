/**
 * Turns the raw event stream of a graph run into what the trace dock draws:
 * one status per node, total time per node, and the conditional routes taken.
 *
 * The graph runtime only emits events (start / end / error / retry / route /
 * log), so status is reconstructed by replaying them in order. A node that
 * never started is "pending" while the run is live and "skipped" once the
 * run has finished, which is exactly how a fallback branch shows up.
 */
import type { TraceEvent } from "@/types";

export type NodeStatus = "pending" | "running" | "done" | "failed" | "skipped";

export interface NodeInfo {
  name: string;
  label: string;
  description: string;
}

export interface NodeView extends NodeInfo {
  status: NodeStatus;
  /** Sum over visits, so loops (search -> broaden -> search) add up. */
  durationMs: number;
  visits: number;
  retries: number;
  /** Latest log or error line for this node. */
  message?: string;
}

export interface RouteView {
  from: string;
  to: string;
  at: number;
}

export type JobStatus = "running" | "done" | "failed";

export function deriveNodes(nodes: NodeInfo[], events: TraceEvent[], status: JobStatus): NodeView[] {
  const views = new Map<string, NodeView>();
  const ensure = (name: string): NodeView => {
    let view = views.get(name);
    if (!view) {
      const info = nodes.find((n) => n.name === name) ?? { name, label: name, description: "" };
      view = { ...info, status: "pending", durationMs: 0, visits: 0, retries: 0 };
      views.set(name, view);
    }
    return view;
  };
  nodes.forEach((n) => ensure(n.name));

  for (const event of events) {
    const view = ensure(event.node);
    switch (event.type) {
      case "node:start":
        view.status = "running";
        view.visits += 1;
        break;
      case "node:end":
        view.status = "done";
        view.durationMs += event.durationMs ?? 0;
        break;
      case "node:error":
        view.status = "failed";
        view.message = event.message;
        break;
      case "node:retry":
        view.retries += 1;
        view.message = event.message;
        break;
      case "log":
        view.message = event.message;
        break;
      case "route":
        break;
    }
  }

  const result = [...views.values()];
  if (status !== "running") {
    for (const view of result) {
      if (view.status === "pending") view.status = "skipped";
      // A run that stopped (error or cancel) leaves no node running.
      if (view.status === "running") view.status = status === "failed" ? "failed" : "done";
    }
  }
  return result;
}

export function deriveRoutes(events: TraceEvent[]): RouteView[] {
  return events
    .filter((e) => e.type === "route" && e.to)
    .map((e) => ({ from: e.node, to: e.to as string, at: e.at }));
}

/** Wall-clock span of the run; still-running jobs count up to `now`. */
export function elapsedMs(events: TraceEvent[], status: JobStatus, now: number): number {
  if (events.length === 0) return 0;
  const first = events[0].at;
  const last = status === "running" ? now : events[events.length - 1].at;
  return Math.max(0, last - first);
}

export function formatDuration(ms: number): string {
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
  const minutes = Math.floor(ms / 60_000);
  return `${minutes} m ${Math.round((ms % 60_000) / 1000)} s`;
}

/** One human-readable line per event, for the dock's event log. */
export function describeEvent(event: TraceEvent, labelOf: (node: string) => string): string {
  const label = labelOf(event.node);
  switch (event.type) {
    case "node:start":
      return `${label} started`;
    case "node:end":
      return `${label} finished in ${formatDuration(event.durationMs ?? 0)}`;
    case "node:error":
      return `${label} failed: ${event.message ?? "unknown error"}`;
    case "node:retry":
      return `${label} retrying (${event.message ?? "transient error"})`;
    case "route":
      return `Route ${label} → ${labelOf(event.to ?? "")}`;
    case "log":
      return `${label}: ${event.message ?? ""}`;
  }
}

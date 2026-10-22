import { describe, expect, it } from "vitest";
import type { TraceEvent } from "@/types";
import { deriveNodes, deriveRoutes, elapsedMs, formatDuration } from "@/ui/trace/traceModel";
import { parseCitation, parseInline, parseMarkdown } from "@/ui/views/survey/markdown";
import { previewInput, splitBatch } from "@/ui/layout/inputKind";
import { hashToRoute, viewToHash } from "@/ui/hooks/routes";

const NODES = ["understand", "resolve", "fetch", "parse", "fallback", "segment"].map((name) => ({
  name,
  label: name,
  description: "",
}));

const ev = (type: TraceEvent["type"], node: string, at: number, extra: Partial<TraceEvent> = {}): TraceEvent => ({
  type,
  node,
  at,
  ...extra,
});

describe("trace model", () => {
  const events: TraceEvent[] = [
    ev("node:start", "understand", 0),
    ev("node:end", "understand", 5, { durationMs: 5 }),
    ev("node:start", "resolve", 5),
    ev("node:end", "resolve", 50, { durationMs: 45 }),
    ev("route", "resolve", 50, { to: "fetch" }),
    ev("node:start", "fetch", 50),
    ev("node:retry", "fetch", 60, { message: "timeout" }),
    ev("node:start", "fetch", 400),
    ev("node:end", "fetch", 500, { durationMs: 100 }),
    ev("route", "fetch", 500, { to: "parse" }),
    ev("node:start", "parse", 500),
  ];

  it("replays events into per-node status while running", () => {
    const nodes = deriveNodes(NODES, events, "running");
    const by = Object.fromEntries(nodes.map((n) => [n.name, n]));
    expect(by.understand.status).toBe("done");
    expect(by.fetch.retries).toBe(1);
    expect(by.fetch.visits).toBe(2);
    expect(by.parse.status).toBe("running");
    expect(by.fallback.status).toBe("pending");
  });

  it("marks unvisited nodes as skipped once the run has finished", () => {
    const finished = [...events, ev("node:end", "parse", 900, { durationMs: 400 })];
    const by = Object.fromEntries(deriveNodes(NODES, finished, "done").map((n) => [n.name, n]));
    expect(by.fallback.status).toBe("skipped");
    expect(by.parse.durationMs).toBe(400);
  });

  it("closes a dangling running node when the job failed", () => {
    const by = Object.fromEntries(deriveNodes(NODES, events, "failed").map((n) => [n.name, n]));
    expect(by.parse.status).toBe("failed");
  });

  it("lists routing decisions and elapsed time", () => {
    expect(deriveRoutes(events).map((r) => `${r.from}>${r.to}`)).toEqual(["resolve>fetch", "fetch>parse"]);
    expect(elapsedMs(events, "done", 10_000)).toBe(500);
    expect(elapsedMs(events, "running", 1_000)).toBe(1_000);
    expect(formatDuration(1530)).toBe("1.53 s");
  });
});

describe("survey markdown", () => {
  it("parses headings, lists, paragraphs and reference lines", () => {
    const blocks = parseMarkdown(
      "## Overview\nFirst line\ncontinues here [1].\n\n- one\n- two\n\n## References\n[1] Wei et al. Chain of thought. 2022.\n[2] Yao et al. ReAct. 2022.",
    );
    expect(blocks.map((b) => b.type)).toEqual(["heading", "paragraph", "list", "heading", "reference", "reference"]);
    expect(blocks[1]).toEqual({ type: "paragraph", text: "First line continues here [1]." });
  });

  it("turns [n] groups into citations but leaves other brackets alone", () => {
    expect(parseCitation("1, 3")).toEqual([1, 3]);
    expect(parseCitation("2–4")).toEqual([2, 3, 4]);
    expect(parseCitation("see 3")).toBeNull();
    const inline = parseInline("**Bold** claim [1, 2] and [link](https://arxiv.org) [x]");
    expect(inline.map((n) => n.type)).toEqual(["strong", "text", "cite", "text", "link", "text"]);
  });

  it("never produces raw HTML nodes", () => {
    const inline = parseInline("<script>alert(1)</script>");
    expect(inline).toEqual([{ type: "text", text: "<script>alert(1)</script>" }]);
  });
});

describe("command bar input preview", () => {
  it("detects a batch of identifiers but not a topic", () => {
    expect(splitBatch("2201.11903, 2210.03629\n10.18653/v1/N19-1423")).toHaveLength(3);
    expect(splitBatch("kv cache compression")).toEqual([]);
    expect(previewInput("2402.02750")?.label).toBe("arXiv ID");
    expect(previewInput("kv cache compression")?.label).toBe("Topic search");
    expect(previewInput("2201.11903 2210.03629")?.label).toBe("2 papers");
    expect(previewInput("   ")).toBeNull();
  });
});

describe("hash routes", () => {
  it("round-trips views, including ids with slashes", () => {
    const view = { kind: "paper", id: "doi:10.18653/v1/N19-1423", tab: "critique" } as const;
    expect(hashToRoute(viewToHash(view))).toEqual(view);
    expect(hashToRoute("#/survey/survey-abc")).toEqual({ kind: "survey", id: "survey-abc" });
    expect(hashToRoute("#/analyze/2402.02750")).toEqual({ kind: "analyze", input: "2402.02750" });
    expect(hashToRoute("")).toEqual({ kind: "home" });
    expect(hashToRoute("#/nonsense")).toBeNull();
  });
});

describe("survey markdown tables", () => {
  it("parses pipe tables and skips the divider row", () => {
    const blocks = parseMarkdown("| Ref | Year |\n|---|---|\n| [1] | 2022 |\n| [2] | — |\n\nAfter.");
    expect(blocks[0]).toEqual({ type: "table", header: ["Ref", "Year"], rows: [["[1]", "2022"], ["[2]", "—"]] });
    expect(blocks[1]).toEqual({ type: "paragraph", text: "After." });
  });
});

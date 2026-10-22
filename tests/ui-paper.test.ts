import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { PaperRecord } from "@/store/types";
import type { Answer, PaperAnalysis, ZonedSentence } from "@/types";

const evidence = { paperId: "arxiv:0000.00000", page: 3, section: "4 Results", quote: "Our method improves accuracy by 4.1 points." };

const sentences: ZonedSentence[] = (["problem", "method", "result", "limitation"] as const).map((zone, i) => ({
  id: `s${i}:0`,
  text: `Sentence number ${i} describing the ${zone} of the paper in some detail.`,
  sectionId: `s${i}`,
  sectionTitle: `Section ${i}`,
  sectionKind: "method",
  page: i + 1,
  index: i,
  zone,
  confidence: 0.6,
  scores: { background: 0, problem: 0, contribution: 0, method: 0, result: 0, limitation: 0, future: 0, other: 0 },
}));

const analysis: PaperAnalysis = {
  mode: "full",
  computedMs: 1234,
  stats: { pages: 9, words: 6000, sentences: 4, chunks: 12, references: 30, figures: 4, tables: 2 },
  briefing: {
    tldr: { text: "The paper shows a cheaper way to do the thing.", evidence: [evidence] },
    problem: [{ text: "Existing methods are slow.", evidence: [evidence] }],
    method: [{ text: "We quantise the cache.", evidence: [evidence] }],
    results: [{ text: "Accuracy improves.", evidence: [evidence] }],
    limitations: { authorStated: [], detected: [{ text: "No variance across seeds is reported.", evidence: [] }] },
    followUps: ["How does it scale to longer contexts?"],
    keyphrases: [{ phrase: "kv cache", score: 0.9 }, { phrase: "quantization", score: 0.5 }],
    keyNumbers: [{ value: "2.6×", label: "less memory", evidence }, { value: "93.2%", label: "accuracy", evidence }],
  },
  critique: {
    verdict: "Solid engineering with thin statistics.",
    reviewerQuestions: ["How many seeds were used?"],
    findings: [
      { id: "f1", category: "statistics", severity: "major", title: "No variance reported", rationale: "Single runs only.", evidence: [] },
      { id: "f2", category: "reproducibility", severity: "strength", title: "Code released", rationale: "A repository is linked.", evidence: [evidence] },
    ],
  },
  rigor: {
    overall: 62,
    grade: "C",
    axes: (["empirical", "baselines", "ablation", "statistics", "reproducibility", "transparency", "scholarship"] as const).map((key) => ({
      key,
      label: key,
      score: 60,
      weight: 1 / 7,
      signals: [{ label: "datasets", value: 2 }],
      explanation: "Because.",
    })),
  },
  visuals: {
    rhetoricalMap: sentences.map((s, i) => ({ position: i / 3, zone: s.zone, page: s.page, sentenceId: s.id })),
    sectionComposition: [{ kind: "method", title: "Method", words: 1200 }, { kind: "results", title: "Results", words: 800 }],
    zoneDistribution: [{ zone: "method", count: 2 }, { zone: "result", count: 1 }],
    keyphraseGraph: { nodes: [{ id: "kv cache", weight: 1 }, { id: "quantization", weight: 0.5 }], links: [{ source: "kv cache", target: "quantization", weight: 1 }] },
    citationYears: [{ year: 2019, count: 3 }, { year: 2022, count: 5 }, { year: 2023, count: 2 }],
    readability: { fleschReadingEase: 32, avgSentenceLength: 24, readingMinutes: 30, jargonDensity: 0.08 },
  },
};

const answer: Answer = {
  id: "qa-1",
  question: "What memory saving does it get?",
  intent: "results",
  status: "answered",
  answer: "2.6× less peak memory",
  context: "KIVI enables 2.6× less peak memory usage at the same accuracy.",
  confidence: 0.71,
  evidence: [{ ...evidence, chunkId: "c1" }],
  hits: [],
  reason: "Span extracted verbatim.",
  timingsMs: { retrieve: 12, total: 40 },
  createdAt: 0,
};

const record: PaperRecord = {
  id: "arxiv:0000.00000",
  meta: { id: "arxiv:0000.00000", title: "A Test Paper", authors: ["A. One", "B. Two", "C. Three", "D. Four", "E. Five"], arxivId: "0000.00000", year: 2024, source: "datacite" },
  status: "ready",
  progress: 1,
  trace: [],
  mode: "full",
  analysis,
  sentences,
  hasPdf: true,
  qa: [answer],
  qaPending: false,
  warnings: ["Parsed 9 of 9 pages."],
  addedAt: 0,
};

let currentTab: "briefing" | "visuals" | "critique" | "ask" = "briefing";
const state = () => ({
  papers: { [record.id]: record },
  view: { kind: "paper", id: record.id, tab: currentTab },
  surveySelection: [],
  setView: vi.fn(),
  ask: vi.fn(async () => {}),
  retry: vi.fn(async () => {}),
  submitFiles: vi.fn(async () => {}),
  openEvidence: vi.fn(),
  toggleSurveySelection: vi.fn(),
});

vi.mock("@/store/library", () => ({
  useLibrary: <T>(selector: (s: ReturnType<typeof state>) => T) => selector(state()),
}));
vi.mock("@/store/export", () => ({ briefingMarkdown: () => "", download: () => {} }));
vi.mock("@/pipeline/paper", () => ({ PAPER_NODES: [{ name: "parse", label: "Parse", description: "Parse the PDF" }] }));
vi.mock("@/qa/answer", () => ({
  ABSTAIN_THRESHOLD: 0.2,
  SENTENCE_ABSTAIN_THRESHOLD: 0.45,
  QA_NODES: [{ name: "retrieve", label: "Retrieve", description: "Hybrid search" }],
}));

describe("chart helpers", () => {
  it("niceTicks covers the maximum with round steps", async () => {
    const { niceTicks } = await import("@/ui/charts/common");
    expect(niceTicks(7, 3)).toEqual([0, 5, 10]);
    expect(niceTicks(40, 4)).toEqual([0, 10, 20, 30, 40]);
    expect(niceTicks(0)).toEqual([0]);
  });

  it("medianYear walks the cumulative counts", async () => {
    const { medianYear } = await import("@/ui/charts/CitationAges");
    expect(medianYear([{ year: 2010, count: 1 }, { year: 2020, count: 5 }, { year: 2015, count: 1 }])).toBe(2020);
    expect(medianYear([])).toBeNull();
  });

  it("parseValue separates magnitude and unit", async () => {
    const { parseValue } = await import("@/ui/charts/MetricBars");
    expect(parseValue("93.2%")).toEqual({ magnitude: 93.2, unit: "%" });
    expect(parseValue("2.6×")).toEqual({ magnitude: 2.6, unit: "x" });
    expect(parseValue("+4.1 BLEU")).toEqual({ magnitude: 4.1, unit: "bleu" });
    expect(parseValue("n/a")).toBeNull();
  });

  it("key-phrase layout is deterministic and fits the viewport", async () => {
    const { computeLayout } = await import("@/ui/charts/KeyphraseNetwork");
    const nodes = Array.from({ length: 18 }, (_, i) => ({ id: `phrase ${i}`, weight: 1 - i / 20 }));
    const links = nodes.slice(1).map((n, i) => ({ source: nodes[i % 5].id, target: n.id, weight: (i % 4) + 1 }));
    const a = computeLayout({ nodes, links }, 600, 340);
    const b = computeLayout({ nodes, links }, 600, 340);
    expect(a).toEqual(b);
    for (const n of a.nodes) {
      expect(n.x).toBeGreaterThanOrEqual(0);
      expect(n.x).toBeLessThanOrEqual(600);
      expect(n.y).toBeGreaterThanOrEqual(0);
      expect(n.y).toBeLessThanOrEqual(340);
    }
    expect(a.links).toHaveLength(links.length);
  });

  it("splitAroundSpan highlights the extracted span inside its sentence", async () => {
    const { splitAroundSpan } = await import("@/ui/views/paper/AskTab");
    expect(splitAroundSpan("KIVI enables 2.6× less memory.", "2.6× less memory")).toEqual(["KIVI enables ", "2.6× less memory", "."]);
    expect(splitAroundSpan("Abc", "zzz")).toBeNull();
  });
});

describe("PaperView renders every tab", () => {
  for (const tab of ["briefing", "visuals", "critique", "ask"] as const) {
    it(`renders the ${tab} tab`, async () => {
      currentTab = tab;
      const { default: PaperView } = await import("@/ui/views/paper/PaperView");
      const html = renderToStaticMarkup(createElement(PaperView, { id: record.id }));
      expect(html).toContain("A Test Paper");
      expect(html).toContain("+1");
      if (tab === "briefing") expect(html).toContain("Detected by PaperTrail");
      if (tab === "visuals") expect(html).toContain("Rhetorical map");
      if (tab === "critique") expect(html).toContain("No variance reported");
      if (tab === "ask") expect(html).toContain("<mark");
    });
  }

  it("shows the running state with the pipeline steps", async () => {
    const { RunningState } = await import("@/ui/views/paper/PaperStatus");
    const html = renderToStaticMarkup(
      createElement(RunningState, { record: { ...record, status: "running", currentNode: "parse", progress: 0.4 } }),
    );
    expect(html).toContain("Parse");
    expect(html).toContain("40%");
  });
});

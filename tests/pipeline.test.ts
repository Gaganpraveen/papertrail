import { beforeEach, describe, expect, it, vi } from "vitest";
import { GraphError } from "@/agent/graph";
import { MLUnavailableError, type MLApi } from "@/ml/client";
import { analyzePaper } from "@/analysis";
import { buildSurvey } from "@/analysis/survey";
import { parsePdf } from "@/parsing";
import { fetchArxivMeta } from "@/sources/arxiv";
import { resolveDoi } from "@/sources/doi";
import { SourceError } from "@/sources/http";
import { downloadPdf } from "@/sources/pdf";
import { searchArxiv } from "@/sources/search";
import {
  buildPaperGraph,
  PAPER_NODES,
  paperIdFor,
  SEMANTIC_SEARCH_WARNING,
  type PaperPipelineState,
} from "@/pipeline/paper";
import { buildSearchGraph, MAX_SEARCH_ATTEMPTS, runTopicSearch, SEARCH_NODES } from "@/pipeline/search";
import { runSurveyPipeline, SURVEY_NODES } from "@/pipeline/survey";
import { packEmbeddings, unpackEmbeddings } from "@/store/db";
import { useLibrary } from "@/store/library";
import { briefingMarkdown } from "@/store/export";
import type { PaperRecord } from "@/store/types";
import type {
  EmbedFn,
  MLStatus,
  PaperAnalysis,
  PaperMeta,
  ParsedDocument,
  ParsedInput,
  SurveyPaperInput,
  SurveyResult,
  TraceEvent,
} from "@/types";

// ───────────────────────────── Module fakes ─────────────────────────────

vi.mock("@/sources/arxiv", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/sources/arxiv")>()),
  fetchArxivMeta: vi.fn(),
}));
vi.mock("@/sources/doi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/sources/doi")>()),
  resolveDoi: vi.fn(),
}));
vi.mock("@/sources/pdf", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/sources/pdf")>()),
  downloadPdf: vi.fn(),
}));
vi.mock("@/sources/search", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/sources/search")>()),
  searchArxiv: vi.fn(),
}));
vi.mock("@/parsing", () => ({ parsePdf: vi.fn() }));
// The store uses the shared client; replace only the worker-backed instance.
vi.mock("@/ml/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/ml/client")>();
  const embed = async (texts: string[]) =>
    texts.map((text) => {
      const v = new Float32Array(16);
      for (let i = 0; i < text.length; i++) v[text.charCodeAt(i) % 16] += 1;
      const norm = Math.hypot(...v) || 1;
      return v.map((x) => x / norm);
    });
  const status = {
    embedder: { state: "idle", progress: 0 },
    reranker: { state: "idle", progress: 0 },
    reader: { state: "idle", progress: 0 },
  };
  return {
    ...actual,
    ml: {
      embed,
      rerank: async (_q: string, passages: string[]) => passages.map(() => 2),
      answer: async (_q: string, context: string) => ({ answer: context.slice(0, 40), score: 0.9, start: 0, end: 40 }),
      warmup: async () => undefined,
      getStatus: () => status,
      subscribe: () => () => undefined,
    },
  };
});
vi.mock("@/analysis", () => ({ analyzePaper: vi.fn() }));
vi.mock("@/analysis/survey", () => ({ buildSurvey: vi.fn() }));

const mockedArxiv = vi.mocked(fetchArxivMeta);
const mockedDoi = vi.mocked(resolveDoi);
const mockedDownload = vi.mocked(downloadPdf);
const mockedParse = vi.mocked(parsePdf);
const mockedAnalyze = vi.mocked(analyzePaper);
const mockedSearch = vi.mocked(searchArxiv);
const mockedSurvey = vi.mocked(buildSurvey);

// ─────────────────────────────── Fixtures ───────────────────────────────

/** Deterministic bag-of-characters vectors: similar texts get similar vectors. */
const fakeEmbed: EmbedFn = async (texts) =>
  texts.map((text) => {
    const v = new Float32Array(16);
    for (let i = 0; i < text.length; i++) v[text.charCodeAt(i) % 16] += 1;
    const norm = Math.hypot(...v) || 1;
    return v.map((x) => x / norm);
  });

const failingEmbed: EmbedFn = async () => {
  throw new MLUnavailableError("embedder failed to load", "embedder");
};

function fakeMl(embed: EmbedFn = fakeEmbed): MLApi {
  const status = {} as MLStatus;
  return {
    embed,
    rerank: async (_q, passages) => passages.map(() => 0),
    answer: async () => ({ answer: "", score: 0, start: -1, end: -1 }),
    warmup: async () => undefined,
    getStatus: () => status,
    subscribe: () => () => undefined,
  };
}

const ABSTRACT =
  "Large language models need a key-value cache that grows with context length. " +
  "We propose a tuning-free two-bit quantization scheme for the cache. " +
  "Our method reduces peak memory by a factor of two point six while preserving accuracy.";

const META: PaperMeta = {
  id: "arxiv:2402.02750",
  title: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
  authors: ["Zirui Liu", "Jiayi Yuan"],
  abstract: ABSTRACT,
  year: 2024,
  arxivId: "2402.02750",
  pdfUrl: "https://arxiv.org/pdf/2402.02750",
  source: "datacite",
};

function section(id: string, title: string, kind: ParsedDocument["sections"][number]["kind"], text: string, page: number) {
  return { id, title, level: 1, kind, pageStart: page, pageEnd: page, text };
}

const INTRO =
  "Serving large language models is limited by the memory consumed by the key-value cache during decoding. " +
  "However, existing quantization methods suffer from large accuracy drops at very low bit widths. " +
  "In this paper we propose a simple asymmetric quantization scheme that needs no fine-tuning. " +
  "Our contributions are a per-channel key quantizer and a per-token value quantizer.";
const METHOD =
  "We quantize the key cache per channel because outliers concentrate in a few fixed channels. " +
  "The value cache is quantized per token since its errors affect the attention output locally. " +
  "A small window of recent tokens is kept in full precision to preserve accuracy on generation tasks.";
const RESULTS =
  "Our method achieves a two point six times reduction in peak memory usage on Llama models. " +
  "It improves throughput by up to three times compared with the full precision baseline. " +
  "A limitation is that we do not evaluate the approach on encoder-decoder architectures.";

function goodDoc(overrides: Partial<ParsedDocument> = {}): ParsedDocument {
  const sections = [
    section("s0", "Abstract", "abstract", ABSTRACT, 1),
    section("s1", "1 Introduction", "introduction", INTRO, 1),
    section("s2", "3 Method", "method", METHOD, 3),
    section("s3", "4 Experiments", "experiments", RESULTS, 5),
  ];
  const text = sections.map((s) => s.text).join(" ");
  return {
    pageCount: 6,
    pagesParsed: 6,
    pages: [1, 2, 3, 4, 5, 6].map((page) => ({ page, text, charCount: text.length })),
    title: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
    abstract: ABSTRACT,
    sections,
    references: [],
    captions: [],
    links: [],
    warnings: [],
    quality: "good",
    ...overrides,
  };
}

function fakeAnalysis(mode: "full" | "abstract-only"): PaperAnalysis {
  const evidence = { paperId: META.id, page: 3, section: "Method", quote: "We quantize the key cache per channel" };
  return {
    mode,
    computedMs: 1234,
    stats: { pages: 6, words: 5400, sentences: 200, chunks: 30, references: 42, figures: 5, tables: 3 },
    rigor: {
      overall: 71,
      grade: "B",
      axes: [
        {
          key: "empirical",
          label: "Empirical breadth",
          score: 80,
          weight: 0.2,
          signals: [],
          explanation: "Four benchmarks | two model families",
        },
      ],
    },
    critique: {
      verdict: "A focused, well-motivated method with convincing memory savings.",
      reviewerQuestions: ["How does the method behave on 1-bit caches?"],
      findings: [
        {
          id: "no-variance",
          category: "statistics",
          severity: "major",
          title: "No variance reported",
          rationale: "Results come from single runs.",
          evidence: [evidence],
        },
        {
          id: "code",
          category: "reproducibility",
          severity: "strength",
          title: "Code released",
          rationale: "A public repository is linked.",
          evidence: [],
        },
      ],
    },
    briefing: {
      tldr: { text: "KIVI quantizes the KV cache to 2 bits without tuning.", evidence: [evidence] },
      problem: [{ text: "The KV cache dominates serving memory.", evidence: [{ ...evidence, page: 1 }] }],
      method: [{ text: "Keys per channel, values per token.", evidence: [evidence] }],
      results: [],
      limitations: {
        authorStated: [],
        detected: [{ text: "Single runs without variance.", evidence: [] }],
      },
      followUps: ["Does the residual window size matter?"],
      keyphrases: [{ phrase: "kv cache", score: 0.9 }],
      keyNumbers: [{ value: "2.6×", label: "peak memory reduction", evidence }],
    },
    visuals: {
      rhetoricalMap: [],
      sectionComposition: [],
      zoneDistribution: [],
      keyphraseGraph: { nodes: [], links: [] },
      citationYears: [],
      readability: { fleschReadingEase: 30, avgSentenceLength: 22, readingMinutes: 25, jargonDensity: 0.1 },
    },
  };
}

const ARXIV_INPUT: ParsedInput = { kind: "arxiv", raw: "2402.02750", arxivId: "2402.02750" };

function initial(input: ParsedInput, extra: Partial<PaperPipelineState> = {}): PaperPipelineState {
  return { input, mode: "full", warnings: [], ...extra };
}

async function runPaper(state: PaperPipelineState, ml: MLApi = fakeMl()) {
  const events: TraceEvent[] = [];
  const final = await buildPaperGraph({ ml }).run(state, { onEvent: (e) => events.push(e) });
  return { final, events, visited: visitedNodes(events), routes: routesOf(events) };
}

const visitedNodes = (events: TraceEvent[]) => events.filter((e) => e.type === "node:start").map((e) => e.node);
const routesOf = (events: TraceEvent[]) =>
  events.filter((e) => e.type === "route").map((e) => `${e.node}->${e.to}`);

beforeEach(() => {
  vi.clearAllMocks();
  mockedArxiv.mockResolvedValue(META);
  mockedDownload.mockResolvedValue(new ArrayBuffer(2048));
  mockedParse.mockResolvedValue(goodDoc());
  mockedAnalyze.mockImplementation(async (input) => fakeAnalysis(input.doc ? "full" : "abstract-only"));
});

// ───────────────────────────── Paper graph ──────────────────────────────

describe("paper graph topology", () => {
  it("declares every UI node and the conditional routes", () => {
    const description = buildPaperGraph({ ml: fakeMl() }).describe();
    expect(description.nodes.map((n) => n.name)).toEqual(PAPER_NODES.map((n) => n.name));
    for (const node of description.nodes) expect(node.description.length).toBeGreaterThan(10);

    const conditional = description.edges.filter((e) => e.conditional).map((e) => `${e.from}->${e.to}`);
    expect(conditional).toEqual(
      expect.arrayContaining([
        "resolve->fetch",
        "resolve->fallback",
        "fetch->parse",
        "fetch->fallback",
        "parse->segment",
        "parse->fallback",
      ]),
    );
    expect(description.edges).toContainEqual({ from: "fallback", to: "segment", conditional: false });
  });

  it("derives stable ids before any network call", async () => {
    expect(await paperIdFor(ARXIV_INPUT)).toBe("arxiv:2402.02750");
    expect(await paperIdFor({ kind: "doi", raw: "x", doi: "10.18653/v1/n19-1423" })).toBe("doi:10.18653/v1/n19-1423");
    const url = await paperIdFor({ kind: "pdf-url", raw: "x", url: "https://example.org/a.pdf" });
    expect(url).toMatch(/^url:[0-9a-f]{12}$/);
    expect(await paperIdFor({ kind: "topic", raw: "x", query: "x" })).toBeNull();
  });
});

describe("paper graph routing", () => {
  it("runs the full-text path end to end", async () => {
    const { final, visited, routes } = await runPaper(initial(ARXIV_INPUT, { paperId: META.id }));

    expect(visited).toEqual(["understand", "resolve", "fetch", "parse", "segment", "zone", "chunk", "embed", "analyze"]);
    expect(routes).toEqual(["resolve->fetch", "fetch->parse", "parse->segment"]);
    expect(final.mode).toBe("full");
    expect(final.sentences!.length).toBeGreaterThan(5);
    expect(final.sentences!.every((s) => s.zone !== undefined)).toBe(true);
    expect(final.chunks!.length).toBeGreaterThan(0);
    expect(final.chunks!.every((c) => c.paperId === META.id)).toBe(true);
    expect(final.embeddings).toHaveLength(final.chunks!.length);
    expect(mockedAnalyze.mock.calls[0][0].embed).not.toBeNull();
    expect(final.analysis?.mode).toBe("full");
  });

  it("falls back to the abstract when the download is blocked", async () => {
    mockedDownload.mockRejectedValue(new SourceError("blocked by CORS", "cors", META.pdfUrl));
    const { final, visited, routes } = await runPaper(initial(ARXIV_INPUT));

    expect(visited).toContain("fallback");
    expect(visited).not.toContain("parse");
    expect(routes).toContain("fetch->fallback");
    expect(final.mode).toBe("abstract-only");
    expect(final.doc).toBeNull();
    expect(final.sentences!.every((s) => s.sectionKind === "abstract")).toBe(true);
    expect(final.warnings.some((w) => w.startsWith("Abstract-only analysis"))).toBe(true);
    expect(final.analysis?.mode).toBe("abstract-only");
  });

  it("falls back when the PDF is image-only", async () => {
    mockedParse.mockResolvedValue(
      goodDoc({ quality: "image-only", sections: [], pages: [{ page: 1, text: "", charCount: 0 }] }),
    );
    const { final, routes } = await runPaper(initial(ARXIV_INPUT));

    expect(routes).toContain("parse->fallback");
    expect(final.mode).toBe("abstract-only");
    expect(final.warnings.join(" ")).toMatch(/no text layer/);
  });

  it("goes straight to the fallback when no PDF link exists", async () => {
    mockedDoi.mockResolvedValue({ ...META, id: "doi:10.1/x", arxivId: undefined, pdfUrl: undefined, doi: "10.1/x" });
    const { visited, final } = await runPaper(initial({ kind: "doi", raw: "10.1/x", doi: "10.1/x" }, { paperId: "doi:10.1/x" }));

    expect(visited.slice(0, 3)).toEqual(["understand", "resolve", "fallback"]);
    expect(final.meta?.id).toBe("doi:10.1/x");
  });

  it("fails with an upload hint when there is neither full text nor abstract", async () => {
    mockedArxiv.mockResolvedValue({ ...META, abstract: undefined });
    mockedDownload.mockRejectedValue(new SourceError("blocked by CORS", "cors"));
    const events: TraceEvent[] = [];
    const run = buildPaperGraph({ ml: fakeMl() }).run(initial(ARXIV_INPUT), { onEvent: (e) => events.push(e) });

    await expect(run).rejects.toBeInstanceOf(GraphError);
    await expect(run).rejects.toThrow(/upload it/);
    expect(events.at(-1)).toMatchObject({ type: "node:error", node: "fallback" });
  });

  it("continues lexical-only when the embedding model is unavailable", async () => {
    const { final, visited } = await runPaper(initial(ARXIV_INPUT), fakeMl(failingEmbed));

    expect(visited.at(-1)).toBe("analyze");
    expect(final.embeddings).toBeNull();
    expect(final.warnings).toContain(SEMANTIC_SEARCH_WARNING);
    expect(final.sentences!.length).toBeGreaterThan(0);
    expect(mockedAnalyze.mock.calls[0][0].embed).toBeNull();
  });

  it("degrades to PDF metadata when the metadata services are unreachable", async () => {
    mockedArxiv.mockRejectedValue(new SourceError("offline", "network"));
    const { final } = await runPaper(initial(ARXIV_INPUT));

    expect(final.mode).toBe("full");
    expect(final.meta?.title).toBe(goodDoc().title);
    expect(final.meta?.pdfUrl).toBe("https://arxiv.org/pdf/2402.02750");
  });

  it("names uploads by content hash and takes the title from the document", async () => {
    const bytes = new TextEncoder().encode("%PDF-1.7 fake").buffer as ArrayBuffer;
    const file = new File([bytes], "kivi_paper.pdf", { type: "application/pdf" });
    const { final, visited } = await runPaper(initial({ kind: "upload", raw: file.name, fileName: file.name }, { file }));

    expect(final.paperId).toMatch(/^upload:[0-9a-f]{12}$/);
    expect(final.meta?.id).toBe(final.paperId);
    expect(final.meta?.title).toBe(goodDoc().title);
    expect(final.meta?.authors).toEqual([]);
    expect(mockedDownload).not.toHaveBeenCalled();
    expect(visited).toContain("parse");
  });

  it("skips the lookup when metadata is already known", async () => {
    const { visited } = await runPaper(initial(ARXIV_INPUT, { meta: META, paperId: META.id }));
    expect(mockedArxiv).not.toHaveBeenCalled();
    expect(visited).toContain("fetch");
  });

  it("rejects topic input", async () => {
    await expect(runPaper(initial({ kind: "topic", raw: "kv cache", query: "kv cache" }))).rejects.toThrow(/search/);
  });
});

// ───────────────────────────── Search graph ─────────────────────────────

describe("topic search", () => {
  const QUERY = "recent papers about efficient kv cache compression for long context llms";
  const results: PaperMeta[] = [
    META,
    { ...META, id: "arxiv:2309.17453", arxivId: "2309.17453", title: "Efficient Streaming Language Models with Attention Sinks", year: 2023 },
  ];

  it("declares its nodes", () => {
    expect(buildSearchGraph({ ml: fakeMl() }).describe().nodes.map((n) => n.name)).toEqual(SEARCH_NODES.map((n) => n.name));
  });

  it("broadens the query until results appear", async () => {
    mockedSearch.mockResolvedValueOnce([]).mockResolvedValueOnce([]).mockResolvedValueOnce(results);
    const events: TraceEvent[] = [];
    const { candidates, attempts } = await runTopicSearch(QUERY, { ml: fakeMl() }, (e) => events.push(e));

    expect(attempts).toHaveLength(MAX_SEARCH_ATTEMPTS);
    expect(attempts[0]).toBe(QUERY);
    expect(new Set(attempts).size).toBe(attempts.length);
    expect(attempts[2].split(" ").length).toBeLessThanOrEqual(3);
    expect(visitedNodes(events)).toEqual(["understand", "search", "broaden", "search", "broaden", "search", "rank"]);
    expect(candidates.map((c) => c.meta.id).sort()).toEqual(results.map((m) => m.id).sort());
  });

  it("stops after three attempts when nothing is found", async () => {
    mockedSearch.mockResolvedValue([]);
    const { candidates, attempts } = await runTopicSearch(QUERY, { ml: fakeMl() });
    expect(candidates).toEqual([]);
    expect(attempts).toHaveLength(MAX_SEARCH_ATTEMPTS);
    expect(mockedSearch).toHaveBeenCalledTimes(MAX_SEARCH_ATTEMPTS);
  });

  it("does not repeat a search when the query cannot be broadened", async () => {
    mockedSearch.mockResolvedValue([]);
    const { attempts } = await runTopicSearch("mamba", { ml: fakeMl(failingEmbed) });
    expect(attempts).toEqual(["mamba"]);
  });

  it("ranks without embeddings when the model is unavailable", async () => {
    mockedSearch.mockResolvedValue(results);
    const { candidates } = await runTopicSearch("streaming attention sinks", { ml: fakeMl(failingEmbed) });
    expect(candidates[0].meta.arxivId).toBe("2309.17453");
    expect(candidates.every((c) => c.scores.semantic === 0)).toBe(true);
  });
});

// ───────────────────────────── Survey graph ─────────────────────────────

describe("survey pipeline", () => {
  function surveyInput(id: string, withEmbeddings: boolean): SurveyPaperInput {
    const chunks = [0, 1].map((index) => ({
      id: `${id}#${index}`,
      paperId: id,
      index,
      sectionId: "s1",
      sectionTitle: "Introduction",
      sectionKind: "introduction" as const,
      page: 1,
      text: `${id} passage ${index} about cache compression`,
      tokenCount: 6,
    }));
    return {
      meta: { ...META, id },
      analysis: fakeAnalysis("full"),
      sentences: [],
      chunks,
      chunkEmbeddings: withEmbeddings ? chunks.map(() => new Float32Array(16).fill(0.25)) : null,
      references: [],
    };
  }

  beforeEach(() => {
    mockedSurvey.mockImplementation(async (papers, _embed, onProgress) => {
      onProgress?.("Clustering papers");
      return { id: "s", paperIds: papers.map((p) => p.meta.id), clusters: [], consensus: [], gaps: [] } as unknown as SurveyResult;
    });
  });

  it("fills missing passage embeddings before synthesis", async () => {
    const events: TraceEvent[] = [];
    await runSurveyPipeline([surveyInput("a", true), surveyInput("b", false)], { ml: fakeMl() }, (e) => events.push(e));

    const [papers, embed] = mockedSurvey.mock.calls[0];
    expect(papers.every((p) => p.chunkEmbeddings?.length === p.chunks.length)).toBe(true);
    expect(embed).not.toBeNull();
    expect(visitedNodes(events)).toEqual(SURVEY_NODES.map((n) => n.name));
    expect(events.some((e) => e.type === "log" && e.message === "Clustering papers")).toBe(true);
  });

  it("falls back to TF-IDF for every paper when the model is unavailable", async () => {
    await runSurveyPipeline([surveyInput("a", true), surveyInput("b", false)], { ml: fakeMl(failingEmbed) });
    const [papers, embed] = mockedSurvey.mock.calls[0];
    expect(embed).toBeNull();
    expect(papers.every((p) => p.chunkEmbeddings === null)).toBe(true);
  });

  it("refuses a survey of fewer than two papers", async () => {
    await expect(runSurveyPipeline([surveyInput("a", true)], { ml: fakeMl() })).rejects.toThrow(/at least 2/);
  });
});

// ──────────────────────────── Store helpers ─────────────────────────────

describe("persistence and export", () => {
  it("packs embeddings into one buffer and back", () => {
    const vectors = [new Float32Array([1, 2, 3]), new Float32Array([4, 5, 6])];
    const packed = packEmbeddings(vectors);
    expect(packed.data.byteLength).toBe(24);
    expect(unpackEmbeddings(packed).map((v) => [...v])).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
  });

  it("writes a briefing with page references and every section", () => {
    const record: PaperRecord = {
      id: META.id,
      meta: META,
      status: "ready",
      progress: 1,
      trace: [],
      mode: "abstract-only",
      analysis: fakeAnalysis("abstract-only"),
      hasPdf: false,
      qa: [],
      qaPending: false,
      warnings: [],
      addedAt: 0,
    };
    const md = briefingMarkdown(record);
    for (const heading of ["## TL;DR", "## Problem", "## Method", "## Results", "## Key numbers", "## Limitations", "## Critique", "## Rigor profile"]) {
      expect(md).toContain(heading);
    }
    expect(md).toContain("# KIVI");
    expect(md).toContain("_(p. 3)_");
    expect(md).toContain("Abstract-only analysis");
    expect(md).toContain("### Major concerns");
    expect(md).toContain("> We quantize the key cache per channel — Method, p. 3");
    expect(md).toContain("Four benchmarks \\| two model families");
    expect(md).toContain("[arXiv:2402.02750](https://arxiv.org/abs/2402.02750)");
  });
});

// ──────────────────────────────── Store ─────────────────────────────────

describe("library store", () => {
  beforeEach(() => {
    mockedArxiv.mockImplementation(async (id) => ({ ...META, id: `arxiv:${id}`, arxivId: id, pdfUrl: `https://arxiv.org/pdf/${id}` }));
    mockedSurvey.mockImplementation(
      async (papers) => ({ id: "x", paperIds: papers.map((p) => p.meta.id), clusters: [], consensus: [], gaps: [] }) as unknown as SurveyResult,
    );
  });

  const waitReady = (id: string) =>
    vi.waitFor(() => expect(useLibrary.getState().papers[id]?.status).toBe("ready"), { timeout: 4000 });

  it("queues, runs, persists and opens papers", async () => {
    const store = useLibrary.getState();
    await store.hydrate();
    await store.submit("2402.02750");
    await store.submit("https://arxiv.org/abs/2309.17453v2");
    expect(useLibrary.getState().order).toEqual(["arxiv:2309.17453", "arxiv:2402.02750"]);
    expect(useLibrary.getState().view).toEqual({ kind: "paper", id: "arxiv:2309.17453", tab: "briefing" });

    await waitReady("arxiv:2402.02750");
    await waitReady("arxiv:2309.17453");
    const record = useLibrary.getState().papers["arxiv:2402.02750"];
    expect(record.progress).toBe(1);
    expect(record.hasPdf).toBe(true);
    expect(record.trace.some((e) => e.type === "route")).toBe(true);
    expect(await store.getPdf("arxiv:2402.02750")).toBeInstanceOf(ArrayBuffer);

    // A duplicate submit only opens the existing record.
    await store.submit("arXiv:2402.02750");
    expect(Object.keys(useLibrary.getState().papers)).toHaveLength(2);
    expect(useLibrary.getState().view).toMatchObject({ id: "arxiv:2402.02750" });
  });

  it("answers questions and keeps the history", async () => {
    await useLibrary.getState().ask("arxiv:2402.02750", "How is the key cache quantized?");
    const qa = useLibrary.getState().papers["arxiv:2402.02750"].qa;
    expect(qa).toHaveLength(1);
    expect(qa[0].question).toBe("How is the key cache quantized?");
    expect(useLibrary.getState().papers["arxiv:2402.02750"].qaPending).toBe(false);
  });

  it("runs a survey over selected ready papers", async () => {
    const store = useLibrary.getState();
    store.toggleSurveySelection("arxiv:2402.02750");
    store.toggleSurveySelection("arxiv:2309.17453");
    await store.runSurvey();
    const [id] = useLibrary.getState().surveyOrder;
    const survey = useLibrary.getState().surveys[id];
    expect(survey.status).toBe("ready");
    expect(survey.trace.length).toBeGreaterThan(0);
    expect(mockedSurvey.mock.calls[0][0].every((p) => p.chunkEmbeddings !== null)).toBe(true);
    expect(useLibrary.getState().view).toEqual({ kind: "survey", id });
  });

  it("routes topics to the search agent", async () => {
    mockedSearch.mockResolvedValue([META]);
    await useLibrary.getState().submit("kv cache compression");
    const search = useLibrary.getState().search!;
    expect(useLibrary.getState().view).toEqual({ kind: "search" });
    expect(search.status).toBe("ready");
    expect(search.candidates).toHaveLength(1);
    expect(search.trace.length).toBeGreaterThan(0);
  });

  it("removes papers and marks failures with a message", async () => {
    mockedArxiv.mockRejectedValue(new SourceError("No arXiv paper found for 2501.00001", "not-found"));
    mockedDownload.mockRejectedValue(new SourceError("PDF returned 404", "not-found"));
    await useLibrary.getState().submit("2501.00001");
    await vi.waitFor(() => expect(useLibrary.getState().papers["arxiv:2501.00001"]?.status).toBe("failed"));
    expect(useLibrary.getState().papers["arxiv:2501.00001"].error).toMatch(/No arXiv paper/);

    await useLibrary.getState().remove("arxiv:2501.00001");
    expect(useLibrary.getState().papers["arxiv:2501.00001"]).toBeUndefined();
    expect(useLibrary.getState().view).toEqual({ kind: "home" });
  });
});

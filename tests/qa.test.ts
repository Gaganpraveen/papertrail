import { describe, expect, it } from "vitest";
import type { Evidence, PaperAnalysis, SectionKind, TraceEvent, Zone, ZonedSentence } from "@/types";
import { tokenize } from "@/nlp/text";
import { normalize } from "@/nlp/vector";
import { MLUnavailableError } from "@/ml/client";
import { chunkSentences, embeddingText } from "@/retrieval/chunker";
import { HybridIndex } from "@/retrieval/hybrid";
import {
  ABSTAIN_THRESHOLD,
  answerQuestion,
  buildQAGraph,
  classifyQuestion,
  containsQuantity,
  definitionSubject,
  expectsQuantity,
  rerankRelevance,
  type QAContext,
} from "@/qa/answer";

// ─────────────────────────────── A small paper ─────────────────────────────

const PAPER = "arxiv:0000.00001";
const ZONES: Zone[] = ["background", "problem", "contribution", "method", "result", "limitation", "future", "other"];

const SECTIONS: { id: string; title: string; kind: SectionKind; page: number; sentences: [string, Zone][] }[] = [
  {
    id: "s0",
    title: "Abstract",
    kind: "abstract",
    page: 1,
    sentences: [
      ["We present KVQ, a tuning-free method that compresses the key-value cache of large language models to two bits.", "contribution"],
      ["KVQ reduces peak memory usage by 2.6 times with almost no loss in accuracy.", "result"],
    ],
  },
  {
    id: "s1",
    title: "3 Method",
    kind: "method",
    page: 3,
    sentences: [
      ["Keys are quantized per channel because outlier channels persist across tokens.", "method"],
      ["Values are quantized per token so that new tokens can be appended in a streaming fashion.", "method"],
      ["A small residual window of recent tokens is kept in full precision.", "method"],
    ],
  },
  {
    id: "s2",
    title: "4 Experiments",
    kind: "experiments",
    page: 6,
    sentences: [
      ["All experiments run on a single NVIDIA A100 GPU with 80GB of memory.", "method"],
      ["We evaluate on LongBench and GSM8K using Llama-2 and Mistral models.", "method"],
      ["The batch size is set to 64 for all throughput measurements.", "method"],
    ],
  },
  {
    id: "s3",
    title: "6 Limitations",
    kind: "limitations",
    page: 9,
    sentences: [
      ["A limitation of KVQ is that the custom kernels currently support only NVIDIA hardware.", "limitation"],
      ["We do not evaluate models larger than 70B parameters.", "limitation"],
    ],
  },
  {
    id: "s4",
    title: "References",
    kind: "references",
    page: 11,
    sentences: [["Vaswani et al. Attention is all you need. In NeurIPS, 2017, which used eight P100 GPUs.", "other"]],
  },
];

const sentences: ZonedSentence[] = SECTIONS.flatMap((section) =>
  section.sentences.map(([text, zone], n) => ({
    id: `${section.id}:${n}`,
    text,
    sectionId: section.id,
    sectionTitle: section.title,
    sectionKind: section.kind,
    page: section.page,
    index: 0,
    zone,
    confidence: 0.8,
    scores: Object.fromEntries(ZONES.map((z) => [z, z === zone ? 1 : 0])) as Record<Zone, number>,
  })),
).map((s, index) => ({ ...s, index }));

const chunks = chunkSentences(PAPER, sentences, { targetWords: 25, maxWords: 40, overlap: 0 });

// ───────────────────────── Deterministic stand-in models ───────────────────

/** Hashed bag of words: texts sharing terms get positive cosine. */
const DIM = 1024; // wide enough that distinct terms rarely collide
async function fakeEmbed(texts: string[]): Promise<Float32Array[]> {
  return texts.map((text) => {
    const v = new Float32Array(DIM);
    for (const token of tokenize(text)) {
      let h = 0;
      for (const ch of token) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
      v[h % DIM] += 1;
    }
    return normalize(v);
  });
}

/** Logit from term overlap: full overlap +5, none -5. */
async function fakeRerank(query: string, passages: string[]): Promise<number[]> {
  const terms = [...new Set(tokenize(query))];
  return passages.map((p) => {
    const present = new Set(tokenize(p));
    const share = terms.length ? terms.filter((t) => present.has(t)).length / terms.length : 0;
    return 10 * share - 5;
  });
}

/** A rule-based "reader": knows how to find GPUs and batch sizes, unsure otherwise. */
async function fakeAnswer(question: string, context: string) {
  const patterns: [RegExp, RegExp, number][] = [
    [/\bGPU\b/i, /NVIDIA [A-Z]\d+/, 0.92],
    [/batch size/i, /\b\d+\b(?= for all throughput)/, 0.88],
  ];
  for (const [ask, find, score] of patterns) {
    const match = ask.test(question) ? find.exec(context) : null;
    if (match) return { answer: match[0], score, start: match.index, end: match.index + match[0].length };
  }
  const first = context.split(" ")[0];
  return { answer: first, score: 0.02, start: 0, end: first.length };
}

const fullML: QAContext["ml"] = { embed: fakeEmbed, rerank: fakeRerank, answer: fakeAnswer };

function context(ml: QAContext["ml"], analysis?: PaperAnalysis): QAContext {
  return { paperId: PAPER, index: new HybridIndex(chunks, null), sentences, analysis, ml };
}

async function denseContext(ml: QAContext["ml"]): Promise<QAContext> {
  const embeddings = await fakeEmbed(chunks.map(embeddingText));
  return { paperId: PAPER, index: new HybridIndex(chunks, embeddings), sentences, ml };
}

async function ask(question: string, ctx: QAContext) {
  const events: TraceEvent[] = [];
  const answer = await answerQuestion(question, ctx, (e) => events.push(e));
  const visited = events.filter((e) => e.type === "node:start").map((e) => e.node);
  const routes = events.filter((e) => e.type === "route").map((e) => `${e.node}->${e.to}`);
  return { answer, events, visited, routes };
}

// ─────────────────────────────── Analysis fixture ──────────────────────────

function evidenceFor(sentenceId: string): Evidence {
  const s = sentences.find((x) => x.id === sentenceId);
  if (!s) throw new Error(sentenceId);
  return { paperId: PAPER, page: s.page, section: s.sectionTitle, quote: s.text, sentenceId };
}

const analysis: PaperAnalysis = {
  briefing: {
    tldr: { text: sentences[0].text, evidence: [evidenceFor("s0:0")] },
    problem: [],
    method: [{ text: sentences[2].text, evidence: [evidenceFor("s1:0")] }],
    results: [{ text: sentences[1].text, evidence: [evidenceFor("s0:1")] }],
    limitations: {
      authorStated: [{ text: sentences[8].text, evidence: [evidenceFor("s3:0")] }],
      detected: [{ text: "Results are reported from single runs without variance.", evidence: [] }],
    },
    followUps: [],
    keyphrases: [],
    keyNumbers: [],
  },
  critique: { findings: [], reviewerQuestions: [], verdict: "" },
  rigor: { overall: 50, grade: "C", axes: [] },
  visuals: {
    rhetoricalMap: [],
    sectionComposition: [],
    zoneDistribution: [],
    keyphraseGraph: { nodes: [], links: [] },
    citationYears: [],
    readability: { fleschReadingEase: 40, avgSentenceLength: 20, readingMinutes: 1, jargonDensity: 0.1 },
  },
  stats: { pages: 11, words: 150, sentences: sentences.length, chunks: chunks.length, references: 1, figures: 0, tables: 0 },
  mode: "full",
  computedMs: 1,
};

// ─────────────────────────────────── Tests ─────────────────────────────────

describe("classifyQuestion", () => {
  it.each([
    ["Which GPU was used for the experiments?", "factual"],
    ["What are the limitations of this approach?", "limitations"],
    ["Where does the method fall short?", "limitations"],
    ["Summarize the paper.", "summary"],
    ["What are the main contributions?", "summary"],
    ["How does it compare to SmoothQuant?", "comparison"],
    ["How much memory does KVQ save?", "results"],
    ["What accuracy is reached on GSM8K?", "results"],
    ["How are keys quantized in the method?", "method"],
    ["What architecture is used?", "method"],
  ])("%s -> %s", (question, intent) => {
    expect(classifyQuestion(question)).toBe(intent);
  });
});

describe("definitionSubject", () => {
  it.each([
    ["What is chain-of-thought prompting?", "chain-of-thought prompting"],
    ["What is a KV cache?", "KV cache"],
    ["what are state space models", "state space models"],
    ["What is the batch size?", null],
    ["What is the capital of France?", null],
    ["What is this paper's main limitation in practice today?", null],
  ])("%s -> %s", (question, subject) => {
    expect(definitionSubject(question)).toBe(subject);
  });
});

describe("rerankRelevance", () => {
  it("maps the observed decision boundary (-3) to 0.5 and saturates at the extremes", () => {
    expect(rerankRelevance(-3)).toBeCloseTo(0.5);
    expect(rerankRelevance(6)).toBeGreaterThan(0.98);
    expect(rerankRelevance(-11)).toBeLessThan(0.02);
  });
});

describe("QA graph", () => {
  it("describes every node with a label and description", () => {
    const { nodes, edges } = buildQAGraph().describe();
    expect(nodes.map((n) => n.name)).toEqual([
      "classify",
      "compose_from_analysis",
      "retrieve",
      "rerank",
      "read",
      "verify",
      "respond",
      "abstain",
    ]);
    for (const n of nodes) {
      expect(n.label).not.toBe(n.name);
      expect(n.description.length).toBeGreaterThan(10);
    }
    expect(edges).toContainEqual({ from: "verify", to: "abstain", conditional: true });
  });

  it("answers a factual question with a verbatim span and its evidence", async () => {
    const { answer, visited, routes } = await ask("Which GPU was used for the experiments?", await denseContext(fullML));

    expect(answer.status).toBe("answered");
    expect(answer.intent).toBe("factual");
    expect(answer.answer).toBe("NVIDIA A100");
    expect(answer.context).toBe("All experiments run on a single NVIDIA A100 GPU with 80GB of memory.");
    expect(answer.confidence).toBeGreaterThanOrEqual(ABSTAIN_THRESHOLD);
    expect(answer.evidence[0]).toMatchObject({
      paperId: PAPER,
      page: 6,
      section: "4 Experiments",
      sentenceId: "s2:0",
    });
    expect(answer.evidence[0].chunkId).toMatch(/^arxiv:0000\.00001#\d+$/);
    expect(answer.hits[0].rerank).toBeDefined();
    expect(answer.hits.some((h) => h.chunk.sectionKind === "references")).toBe(false);
    expect(answer.reason).toMatch(/reader model/);

    expect(visited).toEqual(["classify", "retrieve", "rerank", "read", "verify", "respond"]);
    expect(routes).toEqual(["classify->retrieve", "retrieve->rerank", "verify->respond"]);
    expect(answer.timingsMs.total).toBeGreaterThanOrEqual(0);
    expect(Object.keys(answer.timingsMs)).toEqual(expect.arrayContaining(["retrieve", "read", "verify"]));
  });

  it("abstains when the paper does not contain the answer", async () => {
    const { answer, visited } = await ask("What is the capital of France?", await denseContext(fullML));

    expect(answer.status).toBe("abstained");
    expect(answer.answer).toBeUndefined();
    expect(answer.confidence).toBeLessThan(ABSTAIN_THRESHOLD);
    expect(answer.reason).toMatch(/^The paper does not appear to answer this question/);
    expect(visited.at(-1)).toBe("abstain");
  });

  it("abstains when the reader proposes a span that is not in the paper", async () => {
    const hallucinating: QAContext["ml"] = {
      ...fullML,
      answer: async () => ({ answer: "an H100 cluster", score: 0.99, start: -1, end: -1 }),
    };
    const { answer, events } = await ask("Which GPU was used for the experiments?", context(hallucinating));

    expect(answer.status).toBe("abstained");
    expect(events.some((e) => e.type === "log" && /not found verbatim/.test(e.message ?? ""))).toBe(true);
  });

  it("re-aligns a reader span whose spacing differs from the paper", async () => {
    const respaced: QAContext["ml"] = {
      ...fullML,
      answer: async (_q, ctx) =>
        ctx.includes("2.6 times") ? { answer: "2. 6 times", score: 0.9, start: -1, end: -1 } : { answer: "", score: 0, start: -1, end: -1 },
    };
    const { answer } = await ask("By how much is peak memory usage reduced?", context(respaced));

    expect(answer.status).toBe("answered");
    expect(answer.answer).toBe("2.6 times");
  });

  it("routes summary questions to the briefing", async () => {
    const { answer, visited, routes } = await ask("Summarize the paper.", context(fullML, analysis));

    expect(answer.status).toBe("answered");
    expect(answer.intent).toBe("summary");
    expect(answer.answer?.split("\n")[0]).toBe(analysis.briefing.tldr.text);
    expect(answer.answer).toContain(`• ${analysis.briefing.method[0].text}`);
    expect(answer.evidence.map((e) => e.sentenceId)).toEqual(["s0:0", "s1:0", "s0:1"]);
    expect(answer.confidence).toBe(1);
    expect(answer.hits).toEqual([]);
    expect(visited).toEqual(["classify", "compose_from_analysis"]);
    expect(routes).toEqual(["classify->compose_from_analysis"]);
  });

  it("routes limitation questions to the briefing, separating stated from flagged", async () => {
    const { answer } = await ask("What are the limitations?", context(fullML, analysis));

    expect(answer.intent).toBe("limitations");
    expect(answer.answer).toContain("Stated by the authors:");
    expect(answer.answer).toContain(analysis.briefing.limitations.authorStated[0].text);
    expect(answer.answer).toContain(analysis.briefing.limitations.detected[0].text);
    // One of the two items carries quoted evidence.
    expect(answer.confidence).toBeCloseTo(0.5);
  });

  it("falls back to extractive search for a summary question when no analysis exists", async () => {
    const { routes } = await ask("Summarize the paper.", context(fullML));
    expect(routes[0]).toBe("classify->retrieve");
  });

  it("answers lexical-only when no model is available", async () => {
    const { answer, visited } = await ask("Which GPU was used for the experiments?", context({}));

    expect(answer.status).toBe("answered");
    expect(answer.answer).toBe("All experiments run on a single NVIDIA A100 GPU with 80GB of memory.");
    expect(answer.evidence[0].page).toBe(6);
    expect(answer.hits.every((h) => h.rerank === undefined && h.dense === 0)).toBe(true);
    expect(answer.reason).toMatch(/reader model unavailable/);
    expect(answer.reason).toMatch(/reranker unavailable/);
    expect(visited).toEqual(["classify", "retrieve", "rerank", "read", "verify", "respond"]);
  });

  it("abstains on an empty question without calling any model", async () => {
    const calls: string[] = [];
    const spy: QAContext["ml"] = {
      embed: async (t) => (calls.push("embed"), fakeEmbed(t)),
      answer: async (q, c) => (calls.push("answer"), fakeAnswer(q, c)),
    };
    const { answer, routes } = await ask("  ?? ", await denseContext(spy));

    expect(answer.status).toBe("abstained");
    expect(answer.reason).toMatch(/No passage/);
    expect(routes).toContain("retrieve->abstain");
    expect(calls).toEqual([]);
  });

  it("still abstains lexical-only on an unanswerable question", async () => {
    const { answer } = await ask("What is the capital of France?", context({}));
    expect(answer.status).toBe("abstained");
  });

  it("degrades gracefully when every model call fails", async () => {
    const down = () => Promise.reject(new MLUnavailableError("offline"));
    const { answer, events } = await ask(
      "Which GPU was used for the experiments?",
      await denseContext({ embed: down, rerank: down, answer: down }),
    );

    expect(answer.status).toBe("answered");
    expect(answer.answer).toContain("NVIDIA A100");
    expect(answer.reason).toMatch(/embedding model unavailable/);
    expect(events.some((e) => e.type === "node:error")).toBe(false);
  });

  it("answers definition questions with the defining sentence", async () => {
    const { answer } = await ask("What is KVQ?", await denseContext(fullML));

    expect(answer.status).toBe("answered");
    expect(answer.answer).toBe(sentences[0].text);
    expect(answer.reason).toMatch(/definition question/);
  });

  it("trims a reader span that spills into the next sentence", async () => {
    const spilling: QAContext["ml"] = {
      ...fullML,
      answer: async (_q, ctx) => {
        const span = "Llama-2 and Mistral models. The";
        const at = ctx.indexOf(span);
        return at >= 0 ? { answer: span, score: 0.9, start: at, end: at + span.length } : { answer: "", score: 0, start: -1, end: -1 };
      },
    };
    const { answer } = await ask("Which models are evaluated?", context(spilling));

    expect(answer.answer).toBe("Llama-2 and Mistral models");
    expect(answer.context).toBe("We evaluate on LongBench and GSM8K using Llama-2 and Mistral models.");
  });

  it("answers yes/no questions with the best supporting sentence", async () => {
    const { answer } = await ask("Is a residual window of recent tokens kept in full precision?", await denseContext(fullML));

    expect(answer.status).toBe("answered");
    expect(answer.answer).toBe("A small residual window of recent tokens is kept in full precision.");
    expect(answer.reason).toMatch(/yes\/no/);
  });
});

describe("answer-type check", () => {
  it.each([
    ["How many GPUs were used?", true],
    ["What is the group size?", true],
    ["What was the learning rate?", true],
    ["How much memory does it save?", true],
    ["What datasets are used?", false],
    ["What is chain-of-thought prompting?", false],
  ])("%s -> expects a quantity: %s", (question, expected) => {
    expect(expectsQuantity(question)).toBe(expected);
  });

  it("accepts numbers and number words but not names that contain digits", () => {
    expect(containsQuantity("2.6× less peak memory")).toBe(true);
    expect(containsQuantity("group size of 32")).toBe(true);
    expect(containsQuantity("eight A100 GPUs")).toBe(true);
    expect(containsQuantity("GSM8K")).toBe(false);
    expect(containsQuantity("Llama-2-7B")).toBe(true);
  });
});

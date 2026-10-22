/**
 * ML layer tests.
 *
 *  1. Pure helpers (span alignment) and the worker client, against a fake
 *     worker: always run.
 *  2. The real quantised models, loaded with transformers.js in Node from
 *     public/models (no network): skipped when the model files are absent.
 *     These pin down the behaviour the QA thresholds were tuned on.
 */
import { existsSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import type { MLStatus, ZonedSentence, Zone } from "@/types";
import { MODELS, MODEL_DTYPE, alignSpan, type WorkerOutbound, type WorkerRequest } from "@/ml/models";
import { MLUnavailableError, createMLClient } from "@/ml/client";
import { chunkSentences, embeddingText } from "@/retrieval/chunker";
import { HybridIndex } from "@/retrieval/hybrid";
import { ABSTAIN_THRESHOLD, answerQuestion, rerankRelevance, type QAContext } from "@/qa/answer";
import { cosine } from "@/nlp/vector";

// ─────────────────────────────── alignSpan ─────────────────────────────────

describe("alignSpan", () => {
  const context = "KIVI uses 2.6× less peak memory; keys are quantized per-channel.";

  it("finds exact substrings", () => {
    const start = context.indexOf("peak memory");
    expect(alignSpan("peak memory", context)).toEqual({ start, end: start + "peak memory".length });
  });

  it("recovers spans the tokenizer re-spaced", () => {
    const hit = alignSpan("2. 6 ×", context);
    expect(hit && context.slice(hit.start, hit.end)).toBe("2.6×");
    const dash = alignSpan("per - channel", context);
    expect(dash && context.slice(dash.start, dash.end)).toBe("per-channel");
  });

  it("falls back to a case-insensitive match and rejects absent or empty spans", () => {
    const hit = alignSpan("kivi uses", context);
    expect(hit && context.slice(hit.start, hit.end)).toBe("KIVI uses");
    expect(alignSpan("H100 cluster", context)).toBeNull();
    expect(alignSpan("   ", context)).toBeNull();
  });
});

// ───────────────────────────── Worker client ───────────────────────────────

/** In-memory stand-in for the inference worker. */
class FakeWorker {
  onmessage: ((event: { data: WorkerOutbound }) => void) | null = null;
  onerror: ((event: { message: string; preventDefault(): void }) => void) | null = null;
  readonly received: WorkerRequest[] = [];
  terminated = false;

  constructor(private readonly behaviour: (req: WorkerRequest, worker: FakeWorker) => void) {}

  postMessage(request: WorkerRequest): void {
    this.received.push(request);
    queueMicrotask(() => this.behaviour(request, this));
  }

  reply(message: WorkerOutbound): void {
    this.onmessage?.({ data: message });
  }

  crash(message: string): void {
    this.onerror?.({ message, preventDefault: () => undefined });
  }

  terminate(): void {
    this.terminated = true;
  }
}

/** Embeds a text as [length, first char code]: enough to check vectors map back to their texts. */
const echoEmbed = (req: WorkerRequest, w: FakeWorker) => {
  if (req.type === "embed") {
    w.reply({ id: req.id, ok: true, result: req.payload.texts.map((t) => Float32Array.of(t.length, t.charCodeAt(0))) });
  }
};

function clientWith(behaviour: (req: WorkerRequest, worker: FakeWorker) => void) {
  const workers: FakeWorker[] = [];
  const client = createMLClient(() => {
    const worker = new FakeWorker(behaviour);
    workers.push(worker);
    return worker as unknown as Worker;
  });
  return { client, workers };
}

describe("createMLClient", () => {
  afterEach(() => vi.useRealTimers());

  it("spawns lazily and de-duplicates and caches embeddings by text", async () => {
    const { client, workers } = clientWith(echoEmbed);
    expect(workers).toHaveLength(0);

    const first = await client.embed(["alpha", "be", "alpha"]);
    expect(Array.from(first[0])).toEqual([5, "a".charCodeAt(0)]);
    expect(Array.from(first[1])).toEqual([2, "b".charCodeAt(0)]);
    expect(first[2]).toBe(first[0]);

    const second = await client.embed(["be", "gamma"]);
    expect(Array.from(second[1])).toEqual([5, "g".charCodeAt(0)]);
    expect(workers).toHaveLength(1);
    expect(workers[0].received.map((r) => (r.type === "embed" ? r.payload.texts : []))).toEqual([
      ["alpha", "be"],
      ["gamma"],
    ]);
  });

  it("mirrors worker status messages to subscribers", async () => {
    const { client } = clientWith((req, w) => {
      if (req.type !== "warmup") return;
      w.reply({ type: "status", key: "reader", status: { state: "loading", progress: 0.5 } });
      w.reply({ type: "status", key: "reader", status: { state: "ready", progress: 1 } });
      w.reply({ id: req.id, ok: true, result: null });
    });
    const seen: MLStatus[] = [];
    const unsubscribe = client.subscribe((s) => seen.push(s));
    const before = client.getStatus();

    await client.warmup("reader");
    unsubscribe();

    expect(before.reader.state).toBe("idle");
    expect(seen.map((s) => s.reader.state)).toEqual(["loading", "ready"]);
    expect(client.getStatus().reader).toEqual({ state: "ready", progress: 1 });
    expect(client.getStatus().embedder.state).toBe("idle");
  });

  it("turns a model failure into MLUnavailableError and fails fast during the cooldown", async () => {
    const { client, workers } = clientWith((req, w) => {
      w.reply({ type: "status", key: "reranker", status: { state: "error", progress: 0, error: "404 model" } });
      w.reply({ id: req.id, ok: false, error: "404 model", key: "reranker" });
    });

    const failure = await client.rerank("q", ["p"]).catch((e: unknown) => e);
    expect(failure).toBeInstanceOf(MLUnavailableError);
    expect((failure as MLUnavailableError).key).toBe("reranker");
    expect(client.getStatus().reranker.error).toBe("404 model");

    await expect(client.rerank("q", ["p"])).rejects.toBeInstanceOf(MLUnavailableError);
    expect(workers[0].received).toHaveLength(1); // the second call never reached the worker
  });

  it("rejects pending calls and marks loading models as failed when the worker crashes", async () => {
    const { client, workers } = clientWith((req, w) => {
      w.reply({ type: "status", key: "reader", status: { state: "loading", progress: 0.1 } });
      if (req.type === "answer") w.crash("out of memory");
    });

    await expect(client.answer("q", "c")).rejects.toThrow("out of memory");
    expect(workers[0].terminated).toBe(true);
    expect(client.getStatus().reader.state).toBe("error");
  });

  it("reports environments without workers as MLUnavailableError", async () => {
    const client = createMLClient(() => {
      throw new ReferenceError("Worker is not defined");
    });
    await expect(client.embed(["x"])).rejects.toBeInstanceOf(MLUnavailableError);
    await expect(client.rerank("q", [])).resolves.toEqual([]);
  });
});

// ──────────────────────────── Real models (Node) ───────────────────────────

const MODEL_ROOT = join(process.cwd(), "public", "models");
const modelsPresent = Object.values(MODELS).every((m) => existsSync(join(MODEL_ROOT, m.id, "onnx", "model_quantized.onnx")));

type Transformers = typeof import("@huggingface/transformers");

interface RealML {
  embed(texts: string[]): Promise<Float32Array[]>;
  rerank(query: string, passages: string[]): Promise<number[]>;
  answer(question: string, context: string): Promise<{ answer: string; score: number; start: number; end: number }>;
}

/** The worker's inference code paths, run directly in Node. */
async function loadRealML(): Promise<RealML> {
  const tf: Transformers = await import("@huggingface/transformers");
  tf.env.localModelPath = `${MODEL_ROOT}/`;
  tf.env.allowLocalModels = true;
  tf.env.allowRemoteModels = false;

  const load = tf.pipeline as (task: string, model: string, opts: object) => Promise<unknown>;
  const extractor = (await load("feature-extraction", MODELS.embedder.id, { dtype: MODEL_DTYPE })) as import("@huggingface/transformers").FeatureExtractionPipeline;
  const reader = (await load("question-answering", MODELS.reader.id, { dtype: MODEL_DTYPE })) as import("@huggingface/transformers").QuestionAnsweringPipeline;
  const tokenizer = await tf.AutoTokenizer.from_pretrained(MODELS.reranker.id);
  const crossEncoder = await tf.AutoModelForSequenceClassification.from_pretrained(MODELS.reranker.id, { dtype: MODEL_DTYPE });

  return {
    async embed(texts) {
      const tensor = await extractor(texts, { pooling: "mean", normalize: true });
      const [n, dim] = tensor.dims;
      const data = tensor.data as Float32Array;
      return Array.from({ length: n }, (_, i) => data.slice(i * dim, (i + 1) * dim));
    },
    async rerank(query, passages) {
      const inputs = tokenizer(new Array<string>(passages.length).fill(query), { text_pair: passages, padding: true, truncation: true });
      const { logits } = (await crossEncoder(inputs)) as { logits: import("@huggingface/transformers").Tensor };
      return Array.from(logits.data as Float32Array);
    },
    async answer(question, context) {
      const out = await reader(question, context);
      const best = Array.isArray(out) ? out[0] : out;
      const span = alignSpan(best.answer, context);
      return span
        ? { answer: context.slice(span.start, span.end), score: best.score, ...span }
        : { answer: best.answer, score: best.score, start: -1, end: -1 };
    },
  };
}

/**
 * Passages from KIVI (Liu et al., arXiv:2402.02750, CC BY 4.0), as they come
 * out of the PDF after de-hyphenation.
 */
const KIVI: { title: string; kind: "abstract" | "method" | "experiments"; page: number; text: string[] }[] = [
  {
    title: "Abstract",
    kind: "abstract",
    page: 1,
    text: [
      "Efficiently serving large language models (LLMs) requires batching of many requests to reduce the cost per request.",
      "Yet, with larger batch sizes and longer context lengths, the key-value (KV) cache, which stores attention keys and values to avoid re-computations, significantly increases memory demands and becomes the new bottleneck in speed and memory usage.",
      "Our findings indicate that the key cache should be quantized per-channel, i.e., group elements along the channel dimension and quantize them together.",
      "In contrast, the value cache should be quantized per-token.",
      "From this analysis, we developed a tuning-free 2bit KV cache quantization algorithm named KIVI.",
      "With hardware-friendly implementation, KIVI can enable Llama, Falcon, and Mistral models to maintain almost the same quality while using 2.6× less peak memory (including model weight).",
    ],
  },
  {
    title: "3.1 Setting",
    kind: "method",
    page: 3,
    text: [
      "In Table 1, we show the results of fake KV cache group-wise quantization with different configurations on the Llama-2-13B model for the CoQA and TruthfulQA tasks.",
      "We use a group size of 32 for all configurations.",
      "Here fake quantization means we simulate the quantization process by first quantizing KV cache into lower precision and then dequantizing it in the attention layer.",
    ],
  },
  {
    title: "4.1 Tasks",
    kind: "experiments",
    page: 6,
    text: [
      "As we analyzed in Section 2, the KV cache size grows larger with a longer context.",
      "Thus, we evaluate KIVI under the normal context length and long context setting, respectively.",
      "Specifically, we adopt generation tasks from LM-Eval for normal context length evaluation and LongBench for long context evaluation, respectively.",
    ],
  },
  {
    title: "4.3 Efficiency Comparison",
    kind: "experiments",
    page: 8,
    text: [
      "We report the peak memory usage and throughput between KIVI (with residual length 32 and 128) and FP16 baseline for the Llama-2-7B model.",
      "The hardware here is a single NVIDIA A100 GPU (80GB).",
      "As shown in Figure 5, with similar maximum memory usage, KIVI enables up to 4× larger batch size and gives 2.35× ∼ 3.47× larger throughput.",
    ],
  },
];

describe.skipIf(!modelsPresent)("real models (transformers.js in Node)", () => {
  let real: RealML;

  beforeAll(async () => {
    real = await loadRealML();
  }, 120_000);

  it("embeds paraphrases closer than unrelated sentences", async () => {
    const [a, b, c] = await real.embed([
      "The KV cache grows linearly with sequence length.",
      "Memory for cached keys and values increases with context size.",
      "A cat sat on the mat.",
    ]);
    expect(a.length).toBe(384);
    expect(Math.hypot(...a)).toBeCloseTo(1, 4);
    // Observed: paraphrase ~0.46, unrelated ~0.0.
    expect(cosine(a, b)).toBeGreaterThan(0.35);
    expect(cosine(a, c)).toBeLessThan(0.15);
    expect(cosine(a, b) - cosine(a, c)).toBeGreaterThan(0.25);
  });

  it("reranks a relevant passage far above an irrelevant one", async () => {
    const [relevant, irrelevant] = await real.rerank("How much memory does KIVI save?", [
      "KIVI can maintain almost the same quality while using 2.6× less peak memory.",
      "We thank the anonymous reviewers for their helpful feedback.",
    ]);
    // Observed: about +6 vs -11.
    expect(relevant).toBeGreaterThan(2);
    expect(irrelevant).toBeLessThan(-6);
    expect(rerankRelevance(relevant)).toBeGreaterThan(0.9);
    expect(rerankRelevance(irrelevant)).toBeLessThan(0.1);
  });

  it("extracts the answer span and aligns it to the exact source characters", async () => {
    const context = KIVI[0].text.join(" ");
    const memory = await real.answer("How much less peak memory does KIVI use?", context);
    expect(memory.answer).toBe("2.6×");
    expect(context.slice(memory.start, memory.end)).toBe("2.6×");
    expect(memory.score).toBeGreaterThan(0.5);

    const offTopic = await real.answer("Who won the 2018 World Cup?", context);
    expect(offTopic.score).toBeLessThan(0.1);
  });

  describe("grounded QA end to end", () => {
    let ctx: QAContext;

    beforeAll(async () => {
      const sentences: ZonedSentence[] = KIVI.flatMap((section, s) =>
        section.text.map((text, n) => ({
          id: `s${s}:${n}`,
          text,
          sectionId: `s${s}`,
          sectionTitle: section.title,
          sectionKind: section.kind,
          page: section.page,
          index: 0,
          zone: "other" as Zone,
          confidence: 0,
          scores: { background: 0, problem: 0, contribution: 0, method: 0, result: 0, limitation: 0, future: 0, other: 1 },
        })),
      ).map((s, index) => ({ ...s, index }));
      const chunks = chunkSentences("arxiv:2402.02750", sentences, { targetWords: 60, maxWords: 120 });
      const embeddings = await real.embed(chunks.map(embeddingText));
      ctx = { paperId: "arxiv:2402.02750", index: new HybridIndex(chunks, embeddings), sentences, ml: real };
    }, 60_000);

    it.each([
      ["How much less peak memory does KIVI use?", "2.6×"],
      ["Which GPU was used to measure efficiency?", "A100"],
      ["What group size is used?", "32"],
      ["Which benchmark is used for long context evaluation?", "LongBench"],
    ])("answers %s", async (question, expected) => {
      const answer = await answerQuestion(question, ctx);
      expect(answer.status).toBe("answered");
      expect(answer.answer).toContain(expected);
      expect(answer.confidence).toBeGreaterThanOrEqual(ABSTAIN_THRESHOLD);
      expect(answer.evidence[0].quote).toContain(answer.answer);
    }, 30_000);

    it.each([
      "What is the capital of France?",
      "What learning rate was used to fine-tune the model?",
      "How many layers does the vision encoder have?",
    ])("abstains on %s", async (question) => {
      const answer = await answerQuestion(question, ctx);
      expect(answer.status).toBe("abstained");
      expect(answer.confidence).toBeLessThan(ABSTAIN_THRESHOLD);
    }, 30_000);
  });
});

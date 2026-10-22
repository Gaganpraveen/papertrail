/**
 * Inference worker: runs the three transformers.js models off the main
 * thread so parsing and UI stay responsive while a paper is embedded.
 *
 * Models are served from the app's own origin (public/models, copied there at
 * build time) with the Hugging Face Hub as a fallback, and ONNX Runtime's
 * WebAssembly binaries come from public/ort. Each model loads lazily on first
 * use; download progress is streamed back as status messages.
 */
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  pipeline,
  type FeatureExtractionPipeline,
  type PreTrainedModel,
  type PreTrainedTokenizer,
  type PretrainedModelOptions,
  type QuestionAnsweringPipeline,
  type Tensor,
} from "@huggingface/transformers";
import type { ModelKey, ModelStatus } from "@/types";
import {
  INFERENCE_BATCH_SIZE,
  MODEL_DTYPE,
  MODELS,
  MODEL_FOR_REQUEST,
  alignSpan,
  type ReaderResult,
  type WorkerOutbound,
  type WorkerRequest,
  type WorkerRequestType,
  type WorkerResults,
} from "./models";

const scope = self as unknown as DedicatedWorkerGlobalScope;
const BASE_URL = import.meta.env.BASE_URL;

env.localModelPath = `${BASE_URL}models/`;
env.allowLocalModels = true;
env.allowRemoteModels = true; // falls back to the Hub when a file is not deployed
const wasm = env.backends.onnx.wasm;
if (wasm) {
  wasm.wasmPaths = `${BASE_URL}ort/`;
  // GitHub Pages cannot send COOP/COEP headers, so SharedArrayBuffer (and
  // with it multi-threaded WASM) is unavailable; one thread avoids a failed probe.
  wasm.numThreads = 1;
}

// ───────────────────────────── Status reporting ────────────────────────────

function post(message: WorkerOutbound, transfer: Transferable[] = []): void {
  scope.postMessage(message, { transfer });
}

function reportStatus(key: ModelKey, status: ModelStatus): void {
  post({ type: "status", key, status });
}

type ProgressCallback = NonNullable<PretrainedModelOptions["progress_callback"]>;

/** Aggregates per-file byte counts into one 0..1 progress value per model. */
function progressReporter(key: ModelKey): ProgressCallback {
  const files = new Map<string, { loaded: number; total: number }>();
  let lastSent = -1;
  return (info) => {
    if (info.status !== "progress") return;
    files.set(info.file, { loaded: info.loaded, total: info.total });
    let loaded = 0;
    let total = 0;
    for (const f of files.values()) {
      loaded += f.loaded;
      total += f.total;
    }
    const progress = total > 0 ? loaded / total : 0;
    // Throttle to whole percents: progress events fire per network chunk.
    if (Math.floor(progress * 100) === lastSent) return;
    lastSent = Math.floor(progress * 100);
    reportStatus(key, { state: "loading", progress });
  };
}

// ───────────────────────────── Lazy model loading ─────────────────────────

interface CrossEncoder {
  tokenizer: PreTrainedTokenizer;
  model: PreTrainedModel;
}

interface Loaded {
  embedder: FeatureExtractionPipeline;
  reranker: CrossEncoder;
  reader: QuestionAnsweringPipeline;
}

/**
 * `pipeline`'s generic signature unions every task's type, which is more than
 * tsc can represent (TS2590); the task is fixed per call site, so we narrow
 * the result there instead.
 */
const loadPipeline = pipeline as (task: string, model: string, options: PretrainedModelOptions) => Promise<unknown>;

const loaders: { [K in ModelKey]: () => Promise<Loaded[K]> } = {
  embedder: async () =>
    (await loadPipeline(MODELS.embedder.task, MODELS.embedder.id, {
      dtype: MODEL_DTYPE,
      progress_callback: progressReporter("embedder"),
    })) as FeatureExtractionPipeline,
  reranker: async () => {
    const id = MODELS.reranker.id;
    const progress_callback = progressReporter("reranker");
    const [tokenizer, model] = await Promise.all([
      AutoTokenizer.from_pretrained(id, { progress_callback }),
      AutoModelForSequenceClassification.from_pretrained(id, { dtype: MODEL_DTYPE, progress_callback }),
    ]);
    return { tokenizer, model };
  },
  reader: async () =>
    (await loadPipeline(MODELS.reader.task, MODELS.reader.id, {
      dtype: MODEL_DTYPE,
      progress_callback: progressReporter("reader"),
    })) as QuestionAnsweringPipeline,
};

const cache: { [K in ModelKey]?: Promise<Loaded[K]> } = {};

function load<K extends ModelKey>(key: K): Promise<Loaded[K]> {
  const cached = cache[key] as Promise<Loaded[K]> | undefined;
  if (cached) return cached;

  reportStatus(key, { state: "loading", progress: 0 });
  const promise = loaders[key]().then(
    (model) => {
      reportStatus(key, { state: "ready", progress: 1 });
      return model;
    },
    (err: unknown) => {
      delete cache[key]; // allow a later retry (e.g. after a network blip)
      reportStatus(key, { state: "error", progress: 0, error: errorMessage(err) });
      throw err;
    },
  );
  (cache as Record<ModelKey, Promise<unknown>>)[key] = promise;
  return promise;
}

// ─────────────────────────────── Inference ────────────────────────────────

function batches<T>(items: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

/** Splits a [n, dim] tensor into n standalone rows. */
function rows(tensor: Tensor): Float32Array[] {
  const [n, dim] = tensor.dims;
  const data = tensor.data as Float32Array;
  return Array.from({ length: n }, (_, i) => data.slice(i * dim, (i + 1) * dim));
}

async function embed(texts: string[]): Promise<Float32Array[]> {
  const extractor = await load("embedder");
  const out: Float32Array[] = [];
  for (const batch of batches(texts, INFERENCE_BATCH_SIZE)) {
    // Mean pooling over tokens + L2 normalisation = the sentence-transformers recipe.
    const tensor = await extractor(batch, { pooling: "mean", normalize: true });
    out.push(...rows(tensor));
  }
  return out;
}

async function rerank(query: string, passages: string[]): Promise<number[]> {
  const { tokenizer, model } = await load("reranker");
  const scores: number[] = [];
  for (const batch of batches(passages, INFERENCE_BATCH_SIZE)) {
    const inputs = tokenizer(new Array<string>(batch.length).fill(query), {
      text_pair: batch,
      padding: true,
      truncation: true,
    });
    const { logits } = (await model(inputs)) as { logits: Tensor };
    // One relevance logit per (query, passage) pair: shape [batch, 1].
    scores.push(...Array.from(logits.data as Float32Array));
  }
  return scores;
}

async function answer(question: string, context: string): Promise<ReaderResult> {
  const reader = await load("reader");
  const output = await reader(question, context);
  const best = Array.isArray(output) ? output[0] : output;
  if (!best) return { answer: "", score: 0, start: -1, end: -1 };

  // The pipeline reports no offsets and re-spaces decoded text, so locate the
  // span ourselves and return the paper's exact characters.
  const span = alignSpan(best.answer, context);
  return span
    ? { answer: context.slice(span.start, span.end), score: best.score, start: span.start, end: span.end }
    : { answer: best.answer, score: best.score, start: -1, end: -1 };
}

interface Handled {
  result: WorkerResults[WorkerRequestType];
  transfer: Transferable[];
}

async function handle(request: WorkerRequest): Promise<Handled> {
  switch (request.type) {
    case "embed": {
      const vectors = await embed(request.payload.texts);
      return { result: vectors, transfer: vectors.map((v) => v.buffer) };
    }
    case "rerank":
      return { result: await rerank(request.payload.query, request.payload.passages), transfer: [] };
    case "answer":
      return { result: await answer(request.payload.question, request.payload.context), transfer: [] };
    case "warmup":
      await load(request.payload.key);
      return { result: null, transfer: [] };
  }
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

scope.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  try {
    const { result, transfer } = await handle(request);
    post({ id: request.id, ok: true, result }, transfer);
  } catch (err) {
    const key = request.type === "warmup" ? request.payload.key : MODEL_FOR_REQUEST[request.type];
    post({ id: request.id, ok: false, error: errorMessage(err), key });
  }
};

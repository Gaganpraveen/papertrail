/**
 * The three small ONNX models PaperTrail runs in the browser, plus the message
 * protocol shared by the inference worker (worker.ts) and its main-thread
 * client (client.ts).
 *
 * All three are quantised to 8 bits ("q8"), which keeps the total download
 * near 110 MB while costing little accuracy on these encoder-only models.
 */
import type { ModelKey, ModelStatus } from "@/types";

export interface ModelSpec {
  /** Hugging Face repository id, also the folder name under public/models. */
  id: string;
  /** transformers.js task the model is loaded for. */
  task: "feature-extraction" | "text-classification" | "question-answering";
  /** Approximate download size of the quantised weights. */
  sizeMB: number;
  label: string;
  purpose: string;
}

export const MODELS: Record<ModelKey, ModelSpec> = {
  embedder: {
    id: "Xenova/all-MiniLM-L6-v2",
    task: "feature-extraction",
    sizeMB: 23,
    label: "MiniLM-L6 sentence embeddings",
    purpose: "Semantic search, zoning, keyphrases and paper clustering (384-d vectors).",
  },
  reranker: {
    id: "Xenova/ms-marco-MiniLM-L-6-v2",
    // A cross-encoder: a sequence classifier with one relevance logit per pair.
    task: "text-classification",
    sizeMB: 23,
    label: "MS MARCO cross-encoder",
    purpose: "Re-scores (question, passage) pairs jointly to sharpen retrieval.",
  },
  reader: {
    id: "Xenova/distilbert-base-cased-distilled-squad",
    task: "question-answering",
    sizeMB: 64,
    label: "DistilBERT SQuAD reader",
    purpose: "Extracts the answer span from a passage, so answers are copied, never invented.",
  },
};

export const MODEL_KEYS = Object.keys(MODELS) as ModelKey[];

/** Quantisation used for every model (maps to onnx/model_quantized.onnx). */
export const MODEL_DTYPE = "q8";

/** Texts per forward pass; bounds peak memory on long papers. */
export const INFERENCE_BATCH_SIZE = 16;

// ──────────────────────────── Worker protocol ─────────────────────────────

export interface ReaderResult {
  answer: string;
  /** Reader probability for the span (start prob × end prob), 0..1. */
  score: number;
  /** Character offsets of `answer` in the context, or -1 when not located. */
  start: number;
  end: number;
}

export interface WorkerPayloads {
  embed: { texts: string[] };
  rerank: { query: string; passages: string[] };
  answer: { question: string; context: string };
  warmup: { key: ModelKey };
}

export interface WorkerResults {
  embed: Float32Array[];
  rerank: number[];
  answer: ReaderResult;
  warmup: null;
}

export type WorkerRequestType = keyof WorkerPayloads;

export type WorkerRequest = {
  [T in WorkerRequestType]: { id: number; type: T; payload: WorkerPayloads[T] };
}[WorkerRequestType];

export type WorkerReply =
  | { id: number; ok: true; result: WorkerResults[WorkerRequestType] }
  | { id: number; ok: false; error: string; key?: ModelKey };

export interface WorkerStatusMessage {
  type: "status";
  key: ModelKey;
  status: ModelStatus;
}

export type WorkerOutbound = WorkerReply | WorkerStatusMessage;

/** Which model serves each request type. */
export const MODEL_FOR_REQUEST: Record<Exclude<WorkerRequestType, "warmup">, ModelKey> = {
  embed: "embedder",
  rerank: "reranker",
  answer: "reader",
};

// ─────────────────────── Reader span post-processing ───────────────────────

/**
 * Finds where a reader answer sits in its context.
 *
 * The QA pipeline returns text decoded from word pieces, which re-spaces
 * punctuation ("2.6×" comes back as "2. 6 ×", "per-channel" as
 * "per - channel"). We therefore match on a whitespace-free projection of both
 * strings and map the hit back to the original offsets, so the caller can
 * quote the exact characters from the paper. Case-insensitive matching is a
 * second chance for tokenizers that normalise case.
 */
export function alignSpan(answer: string, context: string): { start: number; end: number } | null {
  const exact = context.indexOf(answer);
  if (answer.length > 0 && exact >= 0) return { start: exact, end: exact + answer.length };

  const needle = answer.replace(/\s+/g, "");
  if (needle.length === 0) return null;

  const { compact, offsets } = compactWithOffsets(context);
  let at = compact.indexOf(needle);
  if (at < 0) at = compact.toLowerCase().indexOf(needle.toLowerCase());
  if (at < 0) return null;

  return { start: offsets[at], end: offsets[at + needle.length - 1] + 1 };
}

/** `text` without whitespace, plus the original index of every kept character. */
function compactWithOffsets(text: string): { compact: string; offsets: number[] } {
  const offsets: number[] = [];
  let compact = "";
  for (let i = 0; i < text.length; i++) {
    if (/\s/.test(text[i])) continue;
    compact += text[i];
    offsets.push(i);
  }
  return { compact, offsets };
}

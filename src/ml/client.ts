/**
 * Main-thread facade over the inference worker.
 *
 * The worker is spawned lazily on the first call, so pages that never need a
 * model never download one. Requests are matched to replies by id; model
 * status (idle -> loading with progress -> ready | error) is mirrored here so
 * the UI can subscribe to it. Every failure surfaces as MLUnavailableError,
 * which callers treat as "continue without this model".
 */
import type { EmbedFn, MLStatus, ModelKey, ModelStatus } from "@/types";
import {
  MODEL_KEYS,
  type ReaderResult,
  type WorkerOutbound,
  type WorkerPayloads,
  type WorkerRequest,
  type WorkerRequestType,
  type WorkerResults,
} from "./models";

export class MLUnavailableError extends Error {
  constructor(
    message: string,
    readonly key?: ModelKey,
  ) {
    super(message);
    this.name = "MLUnavailableError";
  }
}

export interface MLApi {
  /** L2-normalised embeddings; results are cached by text, treat them as read-only. */
  embed: EmbedFn;
  /** Raw cross-encoder logits, one per passage (higher = more relevant). */
  rerank(query: string, passages: string[]): Promise<number[]>;
  /** Best extractive span; start/end are offsets in `context` (-1 if not located). */
  answer(question: string, context: string): Promise<ReaderResult>;
  warmup(key: ModelKey): Promise<void>;
  getStatus(): MLStatus;
  subscribe(listener: (status: MLStatus) => void): () => void;
}

/** ~6 MB of 384-d vectors: enough for a few papers' chunks and sentences. */
const EMBED_CACHE_SIZE = 4096;

/**
 * After a model fails to load, calls fail fast for this long instead of
 * re-downloading on every request; an explicit warmup always retries.
 */
const RETRY_COOLDOWN_MS = 30_000;

/** Least-recently-used map: Map keeps insertion order, so the first key is the oldest. */
class LruCache<K, V> {
  private readonly map = new Map<K, V>();

  constructor(private readonly capacity: number) {}

  get(key: K): V | undefined {
    const value = this.map.get(key);
    if (value === undefined) return undefined;
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }

  set(key: K, value: V): void {
    this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.capacity) {
      const oldest = this.map.keys().next().value as K;
      this.map.delete(oldest);
    }
  }
}

interface Pending {
  key: ModelKey;
  resolve(value: unknown): void;
  reject(reason: MLUnavailableError): void;
}

function initialStatus(): MLStatus {
  const idle: ModelStatus = { state: "idle", progress: 0 };
  return { embedder: { ...idle }, reranker: { ...idle }, reader: { ...idle } };
}

/**
 * Builds a client around any worker factory (tests pass a fake). The factory
 * may throw, e.g. where Web Workers do not exist; that becomes an
 * MLUnavailableError on the call that needed the worker.
 */
export function createMLClient(spawn: () => Worker): MLApi {
  let worker: Worker | null = null;
  let nextId = 1;
  const pending = new Map<number, Pending>();
  const listeners = new Set<(status: MLStatus) => void>();
  const embedCache = new LruCache<string, Float32Array>(EMBED_CACHE_SIZE);
  const failedAt: Partial<Record<ModelKey, number>> = {};
  let status = initialStatus();

  function setStatus(key: ModelKey, next: ModelStatus): void {
    if (next.state === "error") failedAt[key] = Date.now();
    status = { ...status, [key]: next }; // new object => cheap change detection for subscribers
    for (const listener of listeners) listener(status);
  }

  function failAll(message: string): void {
    worker?.terminate();
    worker = null;
    for (const p of pending.values()) p.reject(new MLUnavailableError(message, p.key));
    pending.clear();
    for (const key of MODEL_KEYS) {
      if (status[key].state === "loading") setStatus(key, { state: "error", progress: 0, error: message });
    }
  }

  function onMessage(message: WorkerOutbound): void {
    if ("type" in message) {
      setStatus(message.key, message.status);
      return;
    }
    const entry = pending.get(message.id);
    if (!entry) return;
    pending.delete(message.id);
    if (message.ok) entry.resolve(message.result);
    else entry.reject(new MLUnavailableError(message.error, message.key ?? entry.key));
  }

  function ensureWorker(): Worker {
    if (worker) return worker;
    let spawned: Worker;
    try {
      spawned = spawn();
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      throw new MLUnavailableError(`Could not start the model worker: ${reason}`);
    }
    spawned.onmessage = (event: MessageEvent<WorkerOutbound>) => onMessage(event.data);
    spawned.onerror = (event: ErrorEvent) => {
      event.preventDefault();
      failAll(event.message || "The model worker stopped unexpectedly");
    };
    worker = spawned;
    return spawned;
  }

  function coolingDown(key: ModelKey): boolean {
    const at = failedAt[key];
    return status[key].state === "error" && at !== undefined && Date.now() - at < RETRY_COOLDOWN_MS;
  }

  function request<T extends WorkerRequestType>(
    type: T,
    payload: WorkerPayloads[T],
    key: ModelKey,
    opts: { force?: boolean } = {},
  ): Promise<WorkerResults[T]> {
    return new Promise<WorkerResults[T]>((resolve, reject) => {
      if (!opts.force && coolingDown(key)) {
        reject(new MLUnavailableError(status[key].error ?? `${key} is unavailable`, key));
        return;
      }
      let target: Worker;
      try {
        target = ensureWorker();
      } catch (err) {
        reject(err);
        return;
      }
      const id = nextId++;
      pending.set(id, { key, resolve: resolve as (value: unknown) => void, reject });
      target.postMessage({ id, type, payload } as WorkerRequest);
    });
  }

  async function embed(texts: string[]): Promise<Float32Array[]> {
    const known = new Map<string, Float32Array>();
    for (const text of texts) {
      const hit = embedCache.get(text);
      if (hit) known.set(text, hit);
    }
    const missing = [...new Set(texts.filter((t) => !known.has(t)))];
    if (missing.length > 0) {
      const vectors = await request("embed", { texts: missing }, "embedder");
      missing.forEach((text, i) => {
        known.set(text, vectors[i]);
        embedCache.set(text, vectors[i]);
      });
    }
    return texts.map((text) => known.get(text) as Float32Array);
  }

  return {
    embed,
    rerank: (query, passages) =>
      passages.length === 0 ? Promise.resolve([]) : request("rerank", { query, passages }, "reranker"),
    answer: (question, context) => request("answer", { question, context }, "reader"),
    warmup: async (key) => {
      await request("warmup", { key }, key, { force: true });
    },
    getStatus: () => status,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export const ml: MLApi = createMLClient(
  () => new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
);

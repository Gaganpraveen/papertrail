/**
 * Local persistence in IndexedDB (via idb-keyval), so the library survives
 * reloads without any server.
 *
 * One key-value store with prefixed keys:
 *   paper:<id>   PaperRecord without its trace (records are re-rendered, not re-run)
 *   pdf:<id>     raw PDF bytes, for page previews in the evidence drawer
 *   emb:<id>     passage embeddings packed into one Float32Array buffer
 *   survey:<id>  SurveyRecord without its trace
 *
 * When IndexedDB is unavailable (private browsing, tests) everything falls
 * back to an in-memory map, so the app keeps working for the session. A single
 * failed write (e.g. a PDF over quota) keeps only that item in memory.
 */
import { createStore, del, delMany, get, getMany, keys, set, type UseStore } from "idb-keyval";
import type { PaperRecord, SurveyRecord } from "./types";

const DB_NAME = "papertrail";
const STORE_NAME = "kv";

interface PackedEmbeddings {
  count: number;
  dim: number;
  data: ArrayBuffer;
}

interface Backend {
  get<T>(key: string): Promise<T | undefined>;
  getMany<T>(keys: string[]): Promise<(T | undefined)[]>;
  set(key: string, value: unknown): Promise<void>;
  del(keys: string[]): Promise<void>;
  keys(): Promise<string[]>;
}

function idbBackend(store: UseStore): Backend {
  return {
    get: (key) => get(key, store),
    getMany: (list) => getMany(list, store),
    set: (key, value) => set(key, value, store),
    del: (list) => (list.length === 1 ? del(list[0], store) : delMany(list, store)),
    keys: async () => (await keys(store)).map(String),
  };
}

function memoryBackend(): Backend {
  const map = new Map<string, unknown>();
  return {
    get: async <T>(key: string) => map.get(key) as T | undefined,
    getMany: async <T>(list: string[]) => list.map((key) => map.get(key) as T | undefined),
    set: async (key, value) => void map.set(key, value),
    del: async (list) => list.forEach((key) => map.delete(key)),
    keys: async () => [...map.keys()],
  };
}

let backend: Backend | null = null;
/** Items whose IndexedDB write failed, kept for this session only. */
const overlay = memoryBackend();

/** Opened lazily: idb-keyval opens the database as soon as the store is created. */
function db(): Backend {
  if (!backend) {
    backend = typeof indexedDB === "undefined" ? memoryBackend() : idbBackend(createStore(DB_NAME, STORE_NAME));
  }
  return backend;
}

/**
 * Writes one item; on quota or permission errors keeps just that item in memory
 * so the session continues. Swapping the whole backend would hide everything
 * already stored and send later saves and deletes nowhere.
 */
async function put(key: string, value: unknown): Promise<void> {
  try {
    await db().set(key, value);
    await overlay.del([key]);
  } catch (error) {
    console.warn("Local storage failed; keeping this item in memory for this session", error);
    await overlay.set(key, value);
  }
}

async function remove(list: string[]): Promise<void> {
  await overlay.del(list);
  try {
    await db().del(list);
  } catch (error) {
    console.warn("Local storage delete failed", error);
  }
}

/** Reads see items held in memory first, then IndexedDB. */
const layered: Pick<Backend, "get" | "getMany" | "keys"> = {
  get: async <T>(key: string) => (await overlay.get<T>(key)) ?? (await db().get<T>(key)),
  getMany: async <T>(list: string[]) => {
    const [held, stored] = await Promise.all([overlay.getMany<T>(list), db().getMany<T>(list)]);
    return list.map((_, i) => held[i] ?? stored[i]);
  },
  keys: async () => [...new Set([...(await overlay.keys()), ...(await db().keys())])],
};

async function read<T>(op: (b: typeof layered) => Promise<T>, fallback: T): Promise<T> {
  try {
    return await op(layered);
  } catch (error) {
    console.warn("Local storage read failed", error);
    // IndexedDB itself is unreadable: what this session saved is still in memory.
    return op(overlay).catch(() => fallback);
  }
}

const paperKey = (id: string) => `paper:${id}`;
const pdfKey = (id: string) => `pdf:${id}`;
const embKey = (id: string) => `emb:${id}`;
const surveyKey = (id: string) => `survey:${id}`;

async function loadByPrefix<T>(prefix: string): Promise<T[]> {
  return read(async (b) => {
    const matching = (await b.keys()).filter((key) => key.startsWith(prefix));
    const values = await b.getMany<T>(matching);
    return values.filter((v): v is Awaited<T> & T => v !== undefined);
  }, []);
}

// ───────────────────────────────── Papers ──────────────────────────────────

export function loadPapers(): Promise<PaperRecord[]> {
  return loadByPrefix<PaperRecord>("paper:");
}

/** Live-only fields (trace, pending flags) are not persisted. */
export function savePaper(record: PaperRecord): Promise<void> {
  const stored: PaperRecord = { ...record, trace: [], qaPending: false, currentNode: undefined };
  return put(paperKey(record.id), stored);
}

export function deletePaper(id: string): Promise<void> {
  return remove([paperKey(id), pdfKey(id), embKey(id)]);
}

// ─────────────────────────────── PDF bytes ─────────────────────────────────

export function savePdf(id: string, bytes: ArrayBuffer): Promise<void> {
  // Store a copy: the caller's buffer may later be transferred or detached.
  return put(pdfKey(id), bytes.slice(0));
}

export function loadPdf(id: string): Promise<ArrayBuffer | null> {
  return read(async (b) => (await b.get<ArrayBuffer>(pdfKey(id))) ?? null, null);
}

// ─────────────────────────────── Embeddings ────────────────────────────────

/** Packs equal-length vectors into one buffer: one IndexedDB value instead of hundreds. */
export function packEmbeddings(vectors: Float32Array[]): PackedEmbeddings {
  const dim = vectors[0]?.length ?? 0;
  const data = new Float32Array(vectors.length * dim);
  vectors.forEach((v, i) => data.set(v, i * dim));
  return { count: vectors.length, dim, data: data.buffer };
}

export function unpackEmbeddings(packed: PackedEmbeddings): Float32Array[] {
  const all = new Float32Array(packed.data);
  return Array.from({ length: packed.count }, (_, i) => all.subarray(i * packed.dim, (i + 1) * packed.dim));
}

export function saveEmbeddings(id: string, vectors: Float32Array[]): Promise<void> {
  return put(embKey(id), packEmbeddings(vectors));
}

export function loadEmbeddings(id: string): Promise<Float32Array[] | null> {
  return read(async (b) => {
    const packed = await b.get<PackedEmbeddings>(embKey(id));
    return packed ? unpackEmbeddings(packed) : null;
  }, null);
}

// ───────────────────────────────── Surveys ─────────────────────────────────

export function loadSurveys(): Promise<SurveyRecord[]> {
  return loadByPrefix<SurveyRecord>("survey:");
}

export function saveSurvey(record: SurveyRecord): Promise<void> {
  return put(surveyKey(record.id), { ...record, trace: [] });
}

export function deleteSurvey(id: string): Promise<void> {
  return remove([surveyKey(id)]);
}

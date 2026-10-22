/**
 * Resilient HTTP for public scholarly APIs, called straight from the browser.
 *
 * Free APIs (DataCite, OpenAlex, Crossref, Semantic Scholar) rate-limit and
 * occasionally return 5xx, so every request goes through `fetchWithRetry`:
 *  - transient failures (429, 5xx, timeouts, dropped connections) are retried
 *    with exponential backoff plus random jitter, so many tabs retrying at
 *    once do not hit the server in lock-step;
 *  - a server-provided `Retry-After` header wins over our own backoff;
 *  - permanent failures (404, other 4xx) fail fast; everything ends in a
 *    typed `SourceError` the UI can turn into a helpful message ("cors" means
 *    "this site will not let the page read it, upload the file instead").
 */

export type SourceErrorCode = "network" | "not-found" | "rate-limited" | "cors" | "invalid" | "timeout";

export class SourceError extends Error {
  constructor(
    message: string,
    readonly code: SourceErrorCode,
    readonly url?: string,
    options?: { cause?: unknown },
  ) {
    super(message, options);
    this.name = "SourceError";
  }
}

export interface FetchOptions {
  signal?: AbortSignal;
  /** Time allowed until response headers arrive (the body may stream longer). */
  timeoutMs?: number;
  /** Extra attempts after the first one, for transient failures only. */
  retries?: number;
  headers?: Record<string, string>;
}

const DEFAULT_TIMEOUT_MS = 15_000;
const DEFAULT_RETRIES = 2;
const BACKOFF_BASE_MS = 500;
const BACKOFF_CAP_MS = 8_000;
/** Waiting longer than this inside an interactive UI is worse than failing with "rate-limited". */
const MAX_RETRY_AFTER_MS = 10_000;

export function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name: unknown }).name === "AbortError"
  );
}

function abortError(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
}

/** Resolves after `ms`, or rejects as soon as `signal` aborts. */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(abortError(signal));
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError(signal as AbortSignal));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * "Equal jitter" backoff: half of the exponential delay is fixed, half random.
 * This guarantees a real pause while still de-synchronising concurrent clients.
 */
export function backoffDelay(attempt: number, random: () => number = Math.random): number {
  const exponential = Math.min(BACKOFF_CAP_MS, BACKOFF_BASE_MS * 2 ** attempt);
  return exponential / 2 + random() * (exponential / 2);
}

/** Parses `Retry-After` given either as delta-seconds or as an HTTP date. */
export function parseRetryAfter(value: string | null, now: number = Date.now()): number | null {
  if (!value) return null;
  const trimmed = value.trim();
  if (/^\d+(\.\d+)?$/.test(trimmed)) return Math.round(Number(trimmed) * 1000);
  const date = Date.parse(trimmed);
  if (Number.isNaN(date)) return null;
  return Math.max(0, date - now);
}

/**
 * Browsers report both "blocked by CORS" and "no connectivity" as a bare
 * TypeError, so we infer the likelier cause: offline means network; a
 * cross-origin request while online is almost always a missing CORS header.
 */
function classifyFetchFailure(url: string): SourceErrorCode {
  if (typeof navigator !== "undefined" && navigator.onLine === false) return "network";
  if (typeof location !== "undefined" && typeof location.origin === "string") {
    try {
      if (new URL(url, location.href).origin !== location.origin) return "cors";
    } catch {
      return "network";
    }
  }
  return "network";
}

function isTransientStatus(status: number): boolean {
  return status === 429 || status === 408 || status >= 500;
}

function statusError(response: Response, url: string): SourceError {
  const { status } = response;
  if (status === 404 || status === 410)
    return new SourceError(`Not found (HTTP ${status})`, "not-found", url);
  if (status === 429)
    return new SourceError("The service is rate-limiting requests (HTTP 429)", "rate-limited", url);
  if (status === 408) return new SourceError("The service timed out (HTTP 408)", "timeout", url);
  if (status >= 500) return new SourceError(`The service is unavailable (HTTP ${status})`, "network", url);
  return new SourceError(`The request was rejected (HTTP ${status})`, "invalid", url);
}

type AttemptOutcome =
  | { kind: "response"; response: Response }
  | { kind: "error"; error: SourceError; retryable: boolean };

/** One fetch with its own timeout, linked to the caller's abort signal. */
async function attemptFetch(url: string, opts: FetchOptions): Promise<AttemptOutcome> {
  const controller = new AbortController();
  const external = opts.signal;
  // Kept attached after headers arrive so a caller abort still cancels a streaming body.
  external?.addEventListener("abort", () => controller.abort(external.reason), { once: true });
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);

  try {
    const response = await fetch(url, { signal: controller.signal, headers: opts.headers });
    return { kind: "response", response };
  } catch (error) {
    if (external?.aborted) throw abortError(external);
    if (timedOut) {
      return {
        kind: "error",
        error: new SourceError("The request timed out", "timeout", url, { cause: error }),
        retryable: true,
      };
    }
    if (error instanceof TypeError) {
      const code = classifyFetchFailure(url);
      const message =
        code === "cors"
          ? "The server does not allow this page to read it (blocked by CORS)"
          : "Network error while contacting the server";
      // The page cannot tell a dropped connection from a CORS rejection, so both are
      // retried; for a genuine CORS block that costs at most the short backoff.
      return { kind: "error", error: new SourceError(message, code, url, { cause: error }), retryable: true };
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * `fetch` with a timeout, typed errors and retries for transient failures.
 * Resolves only with a 2xx response; every other outcome becomes a `SourceError`
 * (or the caller's abort reason when `signal` fires).
 */
export async function fetchWithRetry(url: string, opts: FetchOptions = {}): Promise<Response> {
  const retries = Math.max(0, opts.retries ?? DEFAULT_RETRIES);
  if (opts.signal?.aborted) throw abortError(opts.signal);

  for (let attempt = 0; ; attempt++) {
    const outcome = await attemptFetch(url, opts);
    const canRetry = attempt < retries;

    if (outcome.kind === "error") {
      if (!outcome.retryable || !canRetry) throw outcome.error;
      await sleep(backoffDelay(attempt), opts.signal);
      continue;
    }

    const { response } = outcome;
    if (response.ok) return response;

    const error = statusError(response, url);
    if (!isTransientStatus(response.status) || !canRetry) throw error;

    const retryAfter = parseRetryAfter(response.headers.get("retry-after"));
    if (retryAfter !== null && retryAfter > MAX_RETRY_AFTER_MS) throw error;
    // Release the connection; the body of an error response is not needed.
    void response.body?.cancel().catch(() => undefined);
    await sleep(retryAfter ?? backoffDelay(attempt), opts.signal);
  }
}

/**
 * GET a JSON document; malformed JSON becomes `SourceError("…", "invalid")`.
 * No Accept header is added: DataCite content-negotiates on it, and its
 * default (JSON:API) is the shape our mappers read.
 */
export async function fetchJson<T = unknown>(url: string, opts: FetchOptions = {}): Promise<T> {
  const response = await fetchWithRetry(url, opts);
  try {
    return (await response.json()) as T;
  } catch (error) {
    if (opts.signal?.aborted) throw abortError(opts.signal);
    throw new SourceError("The service returned malformed JSON", "invalid", url, { cause: error });
  }
}

/**
 * Picks the error that best explains why several sources all failed: a real
 * outage beats "not-found" (which only means one index lacks the record), and
 * a caller abort beats everything.
 */
export function mostInformativeError(errors: unknown[], fallback: SourceError): unknown {
  const present = errors.filter((e) => e !== undefined);
  const abort = present.find(isAbortError);
  if (abort) return abort;
  const outage = present.find((e) => e instanceof SourceError && e.code !== "not-found");
  if (outage) return outage;
  return present.find((e) => !(e instanceof SourceError)) ?? fallback;
}

// ── Narrowing helpers: API payloads arrive as `unknown` and are read defensively. ──

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function asString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() !== "" ? value : undefined;
}

export function asNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && /^\d+(\.\d+)?$/.test(value.trim())) return Number(value);
  return undefined;
}

export function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

export function asRecords(value: unknown): Record<string, unknown>[] {
  return asArray(value).filter(isRecord);
}

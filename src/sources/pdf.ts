/**
 * PDF download with streaming progress and validation.
 *
 * The body is read chunk by chunk from `fetch`'s ReadableStream, so the UI
 * gets real progress from Content-Length, the size cap is enforced while
 * streaming (a wrong link cannot fill memory before we notice), and the
 * "%PDF-" signature is checked so an HTML landing page or login wall is
 * never handed to the parser.
 */
import { fetchWithRetry, isAbortError, SourceError } from "./http";

export const DEFAULT_MAX_PDF_BYTES = 50 * 1024 * 1024;
/** Time allowed until the server starts answering; the body itself may take longer. */
const HEADERS_TIMEOUT_MS = 30_000;
/** The PDF spec tolerates bytes before the header; readers scan the first 1 KB for it. */
const SIGNATURE_WINDOW = 1024;
const PDF_SIGNATURE = [0x25, 0x50, 0x44, 0x46, 0x2d]; // "%PDF-"
/** Report progress in steps of at least 1% to avoid flooding React with updates. */
const PROGRESS_STEP = 0.01;

export interface DownloadOptions {
  signal?: AbortSignal;
  onProgress?: (fraction: number) => void;
  maxBytes?: number;
}

function megabytes(bytes: number): string {
  return `${Math.round(bytes / (1024 * 1024))} MB`;
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
}

/** A failed cross-origin fetch is indistinguishable from CORS in the browser; both mean "upload it instead". */
function toDownloadError(error: unknown, url: string): unknown {
  if (isAbortError(error)) return error;
  const isFetchFailure =
    error instanceof SourceError &&
    (error.code === "cors" || (error.code === "network" && error.cause instanceof TypeError));
  if (isFetchFailure) {
    return new SourceError(
      "Could not download the PDF: the site blocks downloads from other web pages, or the connection failed. " +
        "Download it yourself and upload the file.",
      "cors",
      url,
      { cause: error },
    );
  }
  return error;
}

export function hasPdfSignature(bytes: Uint8Array): boolean {
  const limit = Math.min(bytes.length, SIGNATURE_WINDOW) - PDF_SIGNATURE.length;
  for (let start = 0; start <= limit; start++) {
    if (PDF_SIGNATURE.every((byte, k) => bytes[start + k] === byte)) return true;
  }
  return false;
}

function looksLikeHtml(bytes: Uint8Array): boolean {
  const head = new TextDecoder().decode(bytes.subarray(0, SIGNATURE_WINDOW)).trimStart().toLowerCase();
  return (
    head.startsWith("<!doctype html") ||
    head.startsWith("<html") ||
    head.includes("<head") ||
    head.includes("<body")
  );
}

function concat(chunks: Uint8Array[], total: number): Uint8Array {
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

function tooLarge(url: string, maxBytes: number): SourceError {
  return new SourceError(`The PDF is larger than the ${megabytes(maxBytes)} limit`, "invalid", url);
}

async function readBody(
  response: Response,
  url: string,
  maxBytes: number,
  opts: DownloadOptions,
): Promise<Uint8Array> {
  const declared = Number(response.headers.get("content-length"));
  const expected = Number.isFinite(declared) && declared > 0 ? declared : null;
  const reader = response.body?.getReader();
  if (!reader) {
    const buffer = new Uint8Array(await response.arrayBuffer());
    if (buffer.byteLength > maxBytes) throw tooLarge(url, maxBytes);
    return buffer;
  }

  const chunks: Uint8Array[] = [];
  let received = 0;
  let reported = 0;
  for (;;) {
    let step: ReadableStreamReadResult<Uint8Array>;
    try {
      step = await reader.read();
    } catch (error) {
      if (opts.signal?.aborted) throw abortReason(opts.signal);
      throw new SourceError("The download was interrupted", "network", url, { cause: error });
    }
    if (step.done) break;
    received += step.value.byteLength;
    if (received > maxBytes) {
      await reader.cancel().catch(() => undefined);
      throw tooLarge(url, maxBytes);
    }
    chunks.push(step.value);
    if (expected && opts.onProgress) {
      // Content-Length can be the compressed size, so stay below 1 until the file is validated.
      const fraction = Math.min(0.99, received / expected);
      if (fraction - reported >= PROGRESS_STEP) {
        reported = fraction;
        opts.onProgress(fraction);
      }
    }
  }
  return concat(chunks, received);
}

/**
 * Downloads a PDF into memory. Rejects with SourceError "cors" when the
 * browser cannot fetch it, "invalid" when it is not a PDF or exceeds
 * `maxBytes` (default 50 MB), or with the caller's abort reason.
 */
export async function downloadPdf(url: string, opts: DownloadOptions = {}): Promise<ArrayBuffer> {
  const maxBytes = opts.maxBytes ?? DEFAULT_MAX_PDF_BYTES;
  let response: Response;
  try {
    response = await fetchWithRetry(url, { signal: opts.signal, timeoutMs: HEADERS_TIMEOUT_MS, retries: 1 });
  } catch (error) {
    throw toDownloadError(error, url);
  }

  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > maxBytes) {
    void response.body?.cancel().catch(() => undefined);
    throw tooLarge(url, maxBytes);
  }

  opts.onProgress?.(0);
  const bytes = await readBody(response, url, maxBytes, opts);
  if (!hasPdfSignature(bytes)) {
    const message = looksLikeHtml(bytes)
      ? "The link returned a web page, not a PDF"
      : "The downloaded file is not a PDF";
    throw new SourceError(message, "invalid", url);
  }
  opts.onProgress?.(1);
  // `concat` allocates exactly `received` bytes, so the backing buffer is the file.
  return bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? (bytes.buffer as ArrayBuffer)
    : (bytes.slice().buffer as ArrayBuffer);
}

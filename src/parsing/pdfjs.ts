/**
 * pdf.js loader shared by parsing and page rendering.
 *
 * The browser build runs pdf.js in a Web Worker whose script URL comes from
 * Vite (`?url`), so the main thread stays responsive while large PDFs parse.
 * Both paths use pdf.js's "legacy" build: the modern build relies on very new
 * JavaScript built-ins (e.g. Map.prototype.getOrInsertComputed) that many
 * browsers still in use do not ship.
 * Node (tests, dev scripts) uses the legacy build, which runs the worker code
 * in-process. The two paths are imported dynamically so neither bundle drags
 * in the other.
 */
import type { PDFDocumentProxy } from "pdfjs-dist";

export type PdfjsLib = typeof import("pdfjs-dist");

/** Node entry point; kept in a variable so the browser bundler skips it. */
const NODE_ENTRY = "pdfjs-dist/legacy/build/pdf.mjs";

let libPromise: Promise<PdfjsLib> | null = null;

function isNodeRuntime(): boolean {
  const proc = (globalThis as { process?: { versions?: { node?: string } } }).process;
  return typeof window === "undefined" && typeof proc?.versions?.node === "string";
}

async function loadBrowserBuild(): Promise<PdfjsLib> {
  const [lib, worker] = await Promise.all([
    import("pdfjs-dist/legacy/build/pdf.mjs"),
    import("pdfjs-dist/legacy/build/pdf.worker.min.mjs?url"),
  ]);
  lib.GlobalWorkerOptions.workerSrc = worker.default;
  return lib as unknown as PdfjsLib;
}

async function loadNodeBuild(): Promise<PdfjsLib> {
  return (await import(/* @vite-ignore */ NODE_ENTRY)) as PdfjsLib;
}

/** Loads pdf.js once per session. */
export function loadPdfjs(): Promise<PdfjsLib> {
  libPromise ??= (isNodeRuntime() ? loadNodeBuild() : loadBrowserBuild()).catch((error: unknown) => {
    libPromise = null; // allow a retry after a transient chunk-load failure
    throw error;
  });
  return libPromise;
}

export interface OpenPdfOptions {
  /**
   * Text extraction does not need real font faces, only font names (for bold
   * detection). Rendering needs the faces.
   */
  purpose: "text" | "render";
}

export interface OpenedPdf {
  doc: PDFDocumentProxy;
  close(): Promise<void>;
}

/**
 * Opens a PDF from bytes. pdf.js transfers (detaches) the buffer it is given,
 * so it always receives a private copy and the caller's buffer stays usable.
 */
export async function openPdf(bytes: ArrayBuffer, { purpose }: OpenPdfOptions): Promise<OpenedPdf> {
  const lib = await loadPdfjs();
  const task = lib.getDocument({
    data: new Uint8Array(bytes.slice(0)),
    verbosity: 0,
    disableFontFace: purpose === "text",
    fontExtraProperties: purpose === "text",
    useSystemFonts: purpose === "render",
  });
  const doc = await task.promise;
  return { doc, close: () => task.destroy() };
}

/** Font naming conventions used for bold faces (Times "Medi", CM "CMBX", Libertine "TB", ...). */
const BOLD_NAME_RE = /bold|black|heavy|semibold|demibold|medi|cmbx|sfbx|-b$|,b$|-bd$|tb$|-sb$/i;

/** Whether a PostScript font name denotes a bold face. Subset prefixes ("ABCDEF+") are ignored. */
export function isBoldFontName(name: string | undefined): boolean {
  if (!name) return false;
  return BOLD_NAME_RE.test(name.replace(/^[A-Z]{6}\+/, ""));
}

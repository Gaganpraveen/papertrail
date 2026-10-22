/**
 * Public entry points of the parsing layer.
 *
 * parsePdf: pdf.js text items -> layout lines (layout.ts) -> document
 * structure (structure.ts). Font names are resolved lazily: pdf.js only
 * reports real PostScript names ("NimbusRomNo9L-Medi") after a page's
 * operator list is loaded, so we load it only for pages that introduce fonts
 * we have not seen yet, which keeps bold detection cheap.
 *
 * renderPage: draws a page on a canvas and highlights the text items that
 * belong to a quote, matched on a letters-and-digits stream so hyphenation,
 * ligatures and spacing differences do not break the match.
 */
import type { PDFDocumentProxy, PDFPageProxy, RenderTask } from "pdfjs-dist";
import type { TextContent, TextItem } from "pdfjs-dist/types/src/display/api";
import type { ParsedDocument, TextLine } from "@/types";
import { cleanText, itemsToLines, type RawTextItem } from "./layout";
import { isBoldFontName, openPdf, type OpenedPdf } from "./pdfjs";
import { buildDocument } from "./structure";

export { itemsToLines, type RawTextItem } from "./layout";
export { buildDocument } from "./structure";

const DEFAULT_MAX_PAGES = 60;

export interface ParseOptions {
  maxPages?: number;
  onProgress?: (fraction: number) => void;
}

// ───────────────────────────── Text extraction ────────────────────────────

function isTextItem(item: unknown): item is TextItem {
  return typeof item === "object" && item !== null && "str" in item && "transform" in item;
}

/** Remembers, per pdf.js font id, whether the font is bold. */
class FontRegistry {
  private readonly bold = new Map<string, boolean>();

  isBold(fontName: string): boolean {
    return this.bold.get(fontName) ?? false;
  }

  async resolve(page: PDFPageProxy, fontNames: Iterable<string>, styles: TextContent["styles"]): Promise<void> {
    const unknown = [...new Set(fontNames)].filter((name) => !this.bold.has(name));
    if (unknown.length === 0) return;
    try {
      await page.getOperatorList(); // makes pdf.js publish the page's fonts
    } catch {
      // A broken content stream only costs us bold detection on this page.
    }
    for (const id of unknown) {
      const family = styles[id]?.fontFamily;
      this.bold.set(id, isBoldFontName(readFontName(page, id)) || isBoldFontName(family));
    }
  }
}

function readFontName(page: PDFPageProxy, fontId: string): string | undefined {
  try {
    if (!page.commonObjs.has(fontId)) return undefined;
    const font: unknown = page.commonObjs.get(fontId);
    if (typeof font === "object" && font !== null && "name" in font) {
      const name = (font as { name: unknown }).name;
      return typeof name === "string" ? name : undefined;
    }
  } catch {
    // Unresolved object: treat as an unknown font.
  }
  return undefined;
}

/** Text rotated by more than ~6 degrees (arXiv side stamps, axis labels) is not body text. */
const MAX_SKEW = 0.1;

function toRawItem(item: TextItem, fonts: FontRegistry, originX: number, originY: number): RawTextItem | null {
  const [a, b, c, d, e, f] = item.transform as number[];
  if (a <= 0 || Math.abs(b) > Math.abs(a) * MAX_SKEW) return null;
  const fontSize = Math.hypot(c, d) || Math.abs(d) || item.height;
  return {
    str: item.str,
    x: e - originX,
    y: f - originY,
    width: item.width,
    height: item.height,
    fontName: item.fontName,
    fontSize,
    bold: fonts.isBold(item.fontName),
  };
}

async function extractPageLines(page: PDFPageProxy, pageNumber: number, fonts: FontRegistry): Promise<TextLine[]> {
  const content = await page.getTextContent();
  const items = content.items.filter(isTextItem).filter((item) => item.str.trim() !== "");
  await fonts.resolve(
    page,
    items.map((item) => item.fontName),
    content.styles,
  );
  const [x0, y0, x1, y1] = page.view;
  const raw = items
    .map((item) => toRawItem(item, fonts, x0, y0))
    .filter((item): item is RawTextItem => item !== null);
  return itemsToLines(raw, pageNumber, x1 - x0, y1 - y0);
}

/** Layout lines of the first `maxPages` pages, in reading order. */
export async function extractPdfLines(
  bytes: ArrayBuffer,
  opts: ParseOptions = {},
): Promise<{ lines: TextLine[]; pageCount: number; pagesParsed: number }> {
  const maxPages = Math.max(1, opts.maxPages ?? DEFAULT_MAX_PAGES);
  const { doc, close } = await openPdf(bytes, { purpose: "text" });
  try {
    const pageCount = doc.numPages;
    const pagesParsed = Math.min(pageCount, maxPages);
    const fonts = new FontRegistry();
    const lines: TextLine[] = [];
    for (let p = 1; p <= pagesParsed; p++) {
      const page = await doc.getPage(p);
      lines.push(...(await extractPageLines(page, p, fonts)));
      page.cleanup();
      opts.onProgress?.(p / pagesParsed);
    }
    return { lines, pageCount, pagesParsed };
  } finally {
    await close();
  }
}

/** Parses a PDF into sections, references, captions and links. */
export async function parsePdf(bytes: ArrayBuffer, opts: ParseOptions = {}): Promise<ParsedDocument> {
  const { lines, pageCount, pagesParsed } = await extractPdfLines(bytes, opts);
  return buildDocument(lines, pageCount, pagesParsed);
}

// ─────────────────────────────── Rendering ────────────────────────────────

const HIGHLIGHT_FILL = "rgba(139,149,255,0.28)";
const RENDER_CACHE_SIZE = 3;

interface CachedDoc {
  key: string;
  opened: Promise<OpenedPdf>;
}

const renderCache: CachedDoc[] = [];
const keysByBuffer = new WeakMap<ArrayBuffer, string>();
const activeRenders = new WeakMap<HTMLCanvasElement, RenderTask>();

/** FNV-1a over the bytes: the drawer may hand us fresh copies of the same PDF. */
function contentKey(bytes: ArrayBuffer): string {
  const known = keysByBuffer.get(bytes);
  if (known) return known;
  const view = new Uint8Array(bytes);
  let hash = 0x811c9dc5;
  for (let i = 0; i < view.length; i++) {
    hash ^= view[i];
    hash = Math.imul(hash, 0x01000193);
  }
  const key = `${view.length}:${(hash >>> 0).toString(16)}`;
  keysByBuffer.set(bytes, key);
  return key;
}

/** Small LRU of open documents so flipping between evidence quotes is instant. */
function getRenderDoc(bytes: ArrayBuffer): Promise<PDFDocumentProxy> {
  const key = contentKey(bytes);
  const index = renderCache.findIndex((entry) => entry.key === key);
  if (index >= 0) {
    const [entry] = renderCache.splice(index, 1);
    renderCache.unshift(entry);
    return entry.opened.then((o) => o.doc);
  }
  const opened = openPdf(bytes, { purpose: "render" });
  opened.catch(() => {
    const failed = renderCache.findIndex((entry) => entry.opened === opened);
    if (failed >= 0) renderCache.splice(failed, 1);
  });
  renderCache.unshift({ key, opened });
  for (const evicted of renderCache.splice(RENDER_CACHE_SIZE)) {
    evicted.opened.then((o) => o.close()).catch(() => undefined);
  }
  return opened.then((o) => o.doc);
}

function isCancellation(error: unknown): boolean {
  return error instanceof Error && error.name === "RenderingCancelledException";
}

/**
 * Draws one page on the canvas (crisp on high-DPI screens). When `highlight`
 * is given, translucent boxes are drawn over the text that matches the quote.
 */
export async function renderPage(
  bytes: ArrayBuffer,
  pageNumber: number,
  canvas: HTMLCanvasElement,
  opts: { scale?: number; highlight?: string } = {},
): Promise<void> {
  const doc = await getRenderDoc(bytes);
  const pageIndex = Math.min(Math.max(1, Math.round(pageNumber)), doc.numPages);
  const page = await doc.getPage(pageIndex);
  const viewport = page.getViewport({ scale: opts.scale ?? 1.5 });
  const ratio = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;

  activeRenders.get(canvas)?.cancel();
  canvas.width = Math.floor(viewport.width * ratio);
  canvas.height = Math.floor(viewport.height * ratio);
  canvas.style.width = `${Math.floor(viewport.width)}px`;
  canvas.style.maxWidth = "100%";
  canvas.style.height = "auto";

  const task = page.render({
    canvas,
    viewport,
    transform: ratio !== 1 ? [ratio, 0, 0, ratio, 0, 0] : undefined,
  });
  activeRenders.set(canvas, task);
  try {
    await task.promise;
  } catch (error) {
    if (isCancellation(error)) return; // superseded by a newer render on this canvas
    throw error;
  } finally {
    if (activeRenders.get(canvas) === task) activeRenders.delete(canvas);
  }

  const quote = opts.highlight?.trim();
  if (!quote) return;
  const content = await page.getTextContent();
  const items = content.items.filter(isTextItem);
  const matched = matchQuoteItems(
    items.map((item) => item.str),
    quote,
  );
  if (matched.size === 0) return;
  const rects = mergeRects(
    [...matched]
      .map((i) => itemRect(items[i], viewport.transform, viewport.scale, content.styles))
      .filter((r): r is Rect => r !== null),
  );
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.save();
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.globalCompositeOperation = "multiply"; // keeps the glyphs dark under the marker
  ctx.fillStyle = HIGHLIGHT_FILL;
  for (const r of rects) ctx.fillRect(r.x - 1, r.y - 1, r.width + 2, r.height + 2);
  ctx.restore();
}

// ─────────────────────────── Quote highlighting ───────────────────────────

/** Lower-case letters and digits only, so layout noise cannot break a match. */
function matchKey(text: string): string {
  return cleanText(text)
    .normalize("NFKD")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]/gu, "");
}

/** Below this length a substring match is too likely to be accidental. */
const MIN_PIECE_CHARS = 14;
const PIECE_WORDS = 5;

/**
 * Returns indices of the item strings that make up the quote. The page is
 * flattened into one letters-and-digits stream with an owner index per
 * character; the whole quote is searched first, and if layout differences
 * prevent an exact hit, overlapping five-word pieces are searched instead.
 */
export function matchQuoteItems(itemTexts: string[], quote: string): Set<number> {
  const owners: number[] = [];
  let stream = "";
  itemTexts.forEach((text, index) => {
    const key = matchKey(text);
    stream += key;
    for (let i = 0; i < key.length; i++) owners.push(index);
  });

  const matched = new Set<number>();
  const mark = (needle: string): boolean => {
    if (!needle) return false;
    let found = false;
    for (let at = stream.indexOf(needle); at >= 0; at = stream.indexOf(needle, at + needle.length)) {
      for (let i = at; i < at + needle.length; i++) matched.add(owners[i]);
      found = true;
    }
    return found;
  };

  const whole = matchKey(quote);
  if (whole.length === 0) return matched;
  if (mark(whole) || whole.length < MIN_PIECE_CHARS) return matched;

  const words = quote.split(/\s+/).filter(Boolean);
  const step = Math.max(1, PIECE_WORDS - 2);
  for (let start = 0; start < words.length; start += step) {
    const piece = matchKey(words.slice(start, start + PIECE_WORDS).join(" "));
    if (piece.length >= MIN_PIECE_CHARS) mark(piece);
    if (start + PIECE_WORDS >= words.length) break;
  }
  return matched;
}

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

function multiply(m: number[], n: number[]): number[] {
  return [
    m[0] * n[0] + m[2] * n[1],
    m[1] * n[0] + m[3] * n[1],
    m[0] * n[2] + m[2] * n[3],
    m[1] * n[2] + m[3] * n[3],
    m[0] * n[4] + m[2] * n[5] + m[4],
    m[1] * n[4] + m[3] * n[5] + m[5],
  ];
}

function itemRect(
  item: TextItem,
  viewportTransform: number[],
  scale: number,
  styles: TextContent["styles"],
): Rect | null {
  const m = multiply(viewportTransform, item.transform as number[]);
  if (Math.abs(m[1]) > Math.abs(m[0]) * MAX_SKEW) return null; // rotated text: skip
  const fontHeight = Math.hypot(m[2], m[3]);
  const style = styles[item.fontName];
  const ascent = style?.ascent || 0.8;
  const descent = style?.descent || -0.2;
  const width = item.width * scale;
  if (!(width > 0) || !(fontHeight > 0)) return null;
  return { x: m[4], y: m[5] - fontHeight * ascent, width, height: fontHeight * (ascent - descent) };
}

/** Joins boxes on the same line so overlapping translucent fills do not stack. */
function mergeRects(rects: Rect[]): Rect[] {
  const sorted = [...rects].sort((a, b) => a.y - b.y || a.x - b.x);
  const merged: Rect[] = [];
  for (const r of sorted) {
    const last = merged[merged.length - 1];
    const sameLine = last && Math.abs(last.y - r.y) < r.height * 0.4;
    if (sameLine && r.x - (last.x + last.width) < r.height * 0.8) {
      const right = Math.max(last.x + last.width, r.x + r.width);
      const bottom = Math.max(last.y + last.height, r.y + r.height);
      last.x = Math.min(last.x, r.x);
      last.y = Math.min(last.y, r.y);
      last.width = right - last.x;
      last.height = bottom - last.y;
    } else {
      merged.push({ ...r });
    }
  }
  return merged;
}

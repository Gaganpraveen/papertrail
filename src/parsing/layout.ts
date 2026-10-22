/**
 * Layout reconstruction: turns the loose text fragments pdf.js reports for a
 * page into visual lines in reading order.
 *
 * A PDF has no notion of "lines" or "columns"; it only places glyph runs at
 * coordinates. We rebuild them geometrically:
 *   1. Clean each fragment (ligatures, zero-width characters).
 *   2. Cluster fragments that share a baseline into rows, and fold small
 *      raised/lowered rows (footnote markers, exponents, subscripts) into the
 *      row they decorate.
 *   3. Find columns. A page gutter is a vertical strip near the centre that
 *      body text never crosses while plenty of text sits on both sides. On
 *      single-column pages, a local split is a run of rows in which text
 *      restarts at the same x after a gap (side-by-side tables, minipages).
 *   4. Cut rows at the column split, at table-cell gaps, and where small figure
 *      labels sit beside or on top of body text.
 *   5. Order lines in "bands": full-width lines (title, wide figures) act as
 *      separators, and between two separators the left column is read before
 *      the right column.
 */
import type { TextLine } from "@/types";

/** One pdf.js text fragment, flattened to page coordinates. */
export interface RawTextItem {
  str: string;
  /** Left edge, in PDF units from the left side of the page. */
  x: number;
  /** Baseline, in PDF units from the BOTTOM of the page (PDF convention). */
  y: number;
  width: number;
  height: number;
  fontName: string;
  fontSize: number;
  bold: boolean;
}

/** A cleaned fragment in top-down coordinates. */
interface Fragment {
  text: string;
  x: number;
  right: number;
  /** Baseline measured from the top of the page. */
  top: number;
  size: number;
  bold: boolean;
}

interface Row {
  /** Baseline of the row's largest font. */
  top: number;
  size: number;
  items: Fragment[];
  column: 0 | 1 | 2;
}

interface Segment {
  column: 0 | 1 | 2;
  top: number;
  size: number;
  x: number;
  right: number;
  items: Fragment[];
}

/** Where a row is cut into a left and a right column. */
type ColumnSplit = { kind: "gutter"; left: number; right: number; center: number } | { kind: "local"; x: number };

const byX = (a: Fragment, b: Fragment): number => a.x - b.x;
const rowLeft = (row: Row): number => Math.min(...row.items.map((f) => f.x));
const rowRight = (row: Row): number => Math.max(...row.items.map((f) => f.right));

// ───────────────────────────── Text clean-up ──────────────────────────────

const LIGATURES: Record<string, string> = {
  "\uFB00": "ff",
  "\uFB01": "fi",
  "\uFB02": "fl",
  "\uFB03": "ffi",
  "\uFB04": "ffl",
  "\uFB05": "st",
  "\uFB06": "st",
};
const LIGATURE_RE = /[\uFB00-\uFB06]/g;
const ZERO_WIDTH_RE = /[\u200B-\u200D\u2060\uFEFF\u00AD]/g;
const ODD_SPACE_RE = /[\u00A0\u2000-\u200A\u202F\u205F\u3000\t]/g;

/** Expands typographic ligatures and removes invisible characters. */
export function cleanText(text: string): string {
  return text
    .replace(LIGATURE_RE, (ch) => LIGATURES[ch] ?? ch)
    .replace(ZERO_WIDTH_RE, "")
    .replace(ODD_SPACE_RE, " ");
}

function toFragments(items: RawTextItem[], pageWidth: number, pageHeight: number): Fragment[] {
  const fragments: Fragment[] = [];
  for (const item of items) {
    const text = cleanText(item.str);
    if (!text.trim()) continue;
    const size = item.fontSize > 0 ? item.fontSize : item.height;
    if (!(size > 0)) continue;
    const top = pageHeight - item.y;
    // Ignore fragments placed entirely outside the visible page box.
    if (top < 0 || top > pageHeight + size || item.x > pageWidth || item.x + item.width < 0) continue;
    fragments.push({ text, x: item.x, right: item.x + Math.max(item.width, 0), top, size, bold: item.bold });
  }
  return fragments;
}

// ───────────────────────────── Rows and scripts ───────────────────────────

/** Fragments whose baselines differ by less than this share of the font size are on one row. */
const BASELINE_TOLERANCE = 0.3;
/** A row is a script candidate when its font is clearly smaller than its host's. */
const SCRIPT_SIZE_RATIO = 0.85;
/**
 * Max baseline offset of a script, in host ems. Superscripts sit up to ~0.65em
 * above the host baseline; subscripts drop only ~0.3em, so a raised fragment is
 * not mistaken for a subscript of the line above it.
 */
const SUPERSCRIPT_OFFSET = 0.7;
const SUBSCRIPT_OFFSET = 0.35;

function groupRows(fragments: Fragment[]): Row[] {
  const sorted = [...fragments].sort((a, b) => a.top - b.top || a.x - b.x);
  const rows: Row[] = [];
  for (const f of sorted) {
    const row = rows[rows.length - 1];
    if (row && Math.abs(f.top - row.top) <= BASELINE_TOLERANCE * Math.max(f.size, row.size)) {
      row.items.push(f);
      if (f.size > row.size) {
        row.size = f.size;
        row.top = f.top;
      }
    } else {
      rows.push({ top: f.top, size: f.size, items: [f], column: 0 });
    }
  }
  return rows;
}

function scriptHost(script: Row, candidate: Row): number | null {
  if (script.size > candidate.size * SCRIPT_SIZE_RATIO) return null;
  if (script.column !== 0 && candidate.column !== 0 && script.column !== candidate.column) return null;
  const offset = Math.abs(script.top - candidate.top);
  const maxOffset = script.top < candidate.top ? SUPERSCRIPT_OFFSET : SUBSCRIPT_OFFSET;
  if (offset > candidate.size * maxOffset || offset < 0.05) return null;
  const near =
    rowLeft(script) >= rowLeft(candidate) - candidate.size && rowLeft(script) <= rowRight(candidate) + candidate.size;
  return near ? offset : null;
}

/**
 * Folds small rows sitting just above or below a larger row into it, so
 * "R" + "l×d" or "LLMs" + "1" stay on one line before rows are cut at gaps.
 */
function attachScriptRows(rows: Row[]): Row[] {
  const absorbed = new Set<Row>();
  rows.forEach((script, i) => {
    let host: Row | null = null;
    let best = Infinity;
    for (let k = Math.max(0, i - 4); k <= Math.min(rows.length - 1, i + 4); k++) {
      const candidate = rows[k];
      if (candidate === script || absorbed.has(candidate)) continue;
      const offset = scriptHost(script, candidate);
      if (offset !== null && offset < best) {
        host = candidate;
        best = offset;
      }
    }
    if (host) {
      host.items.push(...script.items);
      absorbed.add(script);
    }
  });
  return rows.filter((row) => !absorbed.has(row));
}

// ───────────────────────────── Column detection ────────────────────────────

const BIN_WIDTH = 2; // PDF units per histogram bin
/** Narrowest strip we accept as a gutter; real gutters are 12-20pt, word gaps 2-4pt. */
const MIN_GUTTER_WIDTH = 8;

/** Most common font size, weighted by characters, rounded to 0.5pt. */
function modalSize(fragments: Fragment[]): number {
  const weights = new Map<number, number>();
  for (const f of fragments) {
    const key = Math.round(f.size * 2) / 2;
    weights.set(key, (weights.get(key) ?? 0) + f.text.length);
  }
  let best = 0;
  let bestWeight = -1;
  for (const [size, weight] of weights) {
    if (weight > bestWeight) {
      best = size;
      bestWeight = weight;
    }
  }
  return best;
}

/**
 * Finds the gutter of a two-column page, or null for single-column pages.
 * Coverage of each x-bin is measured in distinct baselines of body-sized text,
 * so a handful of full-width lines (title, authors) cannot hide a gutter.
 */
function detectGutter(fragments: Fragment[], pageWidth: number): ColumnSplit | null {
  const body = modalSize(fragments);
  const bodyFragments = fragments.filter((f) => Math.abs(f.size - body) <= 1.2);
  const lineCount = new Set(bodyFragments.map((f) => Math.round(f.top))).size;
  if (lineCount < 12) return null; // too little text to tell

  const binCount = Math.ceil(pageWidth / BIN_WIDTH);
  const cover: Set<number>[] = Array.from({ length: binCount }, () => new Set<number>());
  for (const f of bodyFragments) {
    const from = Math.max(0, Math.floor(f.x / BIN_WIDTH));
    const to = Math.min(binCount - 1, Math.floor((f.right - 0.01) / BIN_WIDTH));
    for (let b = from; b <= to; b++) cover[b].add(Math.round(f.top));
  }

  // Only the middle band of the page can host a gutter.
  const lo = Math.floor((pageWidth * 0.3) / BIN_WIDTH);
  const hi = Math.min(binCount - 1, Math.ceil((pageWidth * 0.7) / BIN_WIDTH));
  let minCover = Infinity;
  for (let b = lo; b <= hi; b++) minCover = Math.min(minCover, cover[b].size);
  // Full-width lines may cross the gutter, but only a minority of them.
  if (minCover > lineCount * 0.3) return null;

  // Widest run of near-minimal bins.
  const limit = minCover + Math.max(1, lineCount * 0.03);
  let bestStart = -1;
  let bestLen = 0;
  for (let b = lo; b <= hi; b++) {
    if (cover[b].size > limit) continue;
    let e = b;
    while (e + 1 <= hi && cover[e + 1].size <= limit) e++;
    if (e - b + 1 > bestLen) {
      bestStart = b;
      bestLen = e - b + 1;
    }
    b = e;
  }
  if (bestStart < 0 || bestLen * BIN_WIDTH < MIN_GUTTER_WIDTH) return null;

  const left = bestStart * BIN_WIDTH;
  const right = (bestStart + bestLen) * BIN_WIDTH;
  const leftLines = new Set<number>();
  const rightLines = new Set<number>();
  for (const f of bodyFragments) {
    if (f.right <= left + BIN_WIDTH) leftLines.add(Math.round(f.top));
    else if (f.x >= right - BIN_WIDTH) rightLines.add(Math.round(f.top));
  }
  // Both columns must hold a real share of the page's lines.
  if (leftLines.size < lineCount * 0.25 || rightLines.size < lineCount * 0.25) return null;
  return { kind: "gutter", left, right, center: (left + right) / 2 };
}

/** Minimum whitespace before a fragment for it to count as a column start. */
const LOCAL_GAP = 3;
/** Column starts within this distance are the same left edge. */
const START_TOLERANCE = 1.5;
/** Rows that must restart at the same x before we believe in a local column. */
const MIN_LOCAL_ROWS = 4;
/** The left part must be real text, not a bullet or a line number. */
const MIN_LEFT_WIDTH = 40;

interface StartMark {
  row: number;
  x: number;
  gap: number;
  size: number;
  leftWidth: number;
}

function columnStarts(rows: Row[]): StartMark[] {
  const marks: StartMark[] = [];
  rows.forEach((row, r) => {
    const items = [...row.items].sort(byX);
    let reach = -Infinity;
    for (const f of items) {
      const gap = f.x - reach;
      if (reach > -Infinity && gap >= LOCAL_GAP) {
        marks.push({ row: r, x: f.x, gap, size: row.size, leftWidth: reach - items[0].x });
      }
      reach = Math.max(reach, f.right);
    }
  });
  return marks.sort((a, b) => a.x - b.x);
}

const median = (values: number[]): number => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

/**
 * Keeps the marks that look like the rest of their cluster: same font size and
 * a gap at least half the typical one, so a heading whose word space happens
 * to fall on a table's column edge is not split.
 */
function consistentMarks(cluster: StartMark[]): StartMark[] {
  const size = median(cluster.map((m) => m.size));
  const gap = median(cluster.map((m) => m.gap));
  return cluster.filter((m) => Math.abs(m.size - size) <= 1 && m.gap >= gap * 0.5);
}

function clusterStarts(marks: StartMark[]): StartMark[][] {
  const clusters: StartMark[][] = [];
  for (const mark of marks) {
    const last = clusters[clusters.length - 1];
    if (last && mark.x - last[0].x <= START_TOLERANCE * 2) last.push(mark);
    else clusters.push([mark]);
  }
  return clusters;
}

const crosses = (row: Row, x: number): boolean => row.items.some((f) => f.x < x - 1 && f.right > x + 1);

/**
 * Local columns on single-column pages: side-by-side tables, captions or
 * author blocks. Text that restarts at the same x after a gap on several
 * consecutive rows, with no row in between running across that x, marks a
 * left/right split for those rows. Justified prose never lines up like that.
 */
function findLocalSplits(rows: Row[]): Map<Row, ColumnSplit> {
  const splits = new Map<Row, ColumnSplit>();
  const clusters = clusterStarts(columnStarts(rows)).sort((a, b) => b.length - a.length);
  for (const rawCluster of clusters) {
    if (rawCluster.length < MIN_LOCAL_ROWS) break;
    const cluster = consistentMarks(rawCluster);
    if (cluster.length < MIN_LOCAL_ROWS) continue;
    const x = Math.min(...cluster.map((m) => m.x)) - 0.5;
    const markedRows = new Map(cluster.map((m) => [m.row, m]));
    const first = Math.min(...markedRows.keys());
    const last = Math.max(...markedRows.keys());

    let run: number[] = [];
    let marked: StartMark[] = [];
    const closeRun = () => {
      const realLeft = marked.filter((m) => m.leftWidth >= MIN_LEFT_WIDTH).length;
      if (marked.length >= MIN_LOCAL_ROWS && realLeft * 2 >= marked.length) {
        for (const r of run) if (!splits.has(rows[r])) splits.set(rows[r], { kind: "local", x });
      }
      run = [];
      marked = [];
    };
    for (let r = first; r <= last; r++) {
      const mark = markedRows.get(r);
      if (mark) {
        run.push(r);
        marked.push(mark);
      } else if (crosses(rows[r], x)) {
        closeRun();
      } else if (run.length > 0) {
        run.push(r);
      }
    }
    closeRun();
  }
  return splits;
}

// ───────────────────────────── Cutting rows ────────────────────────────────

/** Gap (in ems) that separates table cells or a margin note from the text beside it. */
const CELL_GAP = 2.5;
/** Figure labels are set clearly smaller than the text next to them. */
const LABEL_SIZE_RATIO = 0.75;
const LABEL_GAP = 0.8;

function largest(items: Fragment[]): Fragment {
  let anchor = items[0];
  for (const f of items) if (f.size > anchor.size) anchor = f;
  return anchor;
}

function makeSegment(items: Fragment[], column: 0 | 1 | 2): Segment {
  // Baseline and size come from the largest font in the segment.
  const anchor = largest(items);
  return {
    column,
    top: anchor.top,
    size: anchor.size,
    x: Math.min(...items.map((f) => f.x)),
    right: Math.max(...items.map((f) => f.right)),
    items,
  };
}

function makeRow(items: Fragment[], column: 0 | 1 | 2): Row {
  const anchor = largest(items);
  return { top: anchor.top, size: anchor.size, items, column };
}

/** Cuts a row into its left/right column parts (or keeps it whole when it spans). */
function toColumnRows(row: Row, split: ColumnSplit): Row[] {
  return splitColumns([...row.items].sort(byX), split).map((part) => makeRow(part.items, part.column));
}

function splitColumns(items: Fragment[], split: ColumnSplit): { items: Fragment[]; column: 0 | 1 | 2 }[] {
  if (split.kind === "local") {
    const left = items.filter((f) => f.x < split.x);
    const right = items.filter((f) => f.x >= split.x);
    return [
      { items: left, column: 1 as const },
      { items: right, column: 2 as const },
    ].filter((part) => part.items.length > 0);
  }

  const gutterWidth = split.right - split.left;
  // Largest gap between neighbouring fragments that overlaps the gutter strip.
  let cut = -1;
  let cutGap = 0;
  let reach = -Infinity;
  for (let i = 0; i < items.length - 1; i++) {
    reach = Math.max(reach, items[i].right);
    const gap = items[i + 1].x - reach;
    const overlapsGutter = reach <= split.right && items[i + 1].x >= split.left;
    if (overlapsGutter && gap > cutGap) {
      cut = i;
      cutGap = gap;
    }
  }
  // A true column break is about as wide as the gutter; author lists and
  // centred title lines have ordinary word gaps there.
  if (cut >= 0 && cutGap >= Math.max(MIN_GUTTER_WIDTH * 0.75, gutterWidth * 0.6)) {
    return [
      { items: items.slice(0, cut + 1), column: 1 },
      { items: items.slice(cut + 1), column: 2 },
    ];
  }
  if (Math.max(...items.map((f) => f.right)) <= split.center) return [{ items, column: 1 }];
  if (items[0].x >= split.center) return [{ items, column: 2 }];
  return [{ items, column: 0 }];
}

/**
 * Small text drawn on top of larger text (labels of a figure that overlaps
 * the text box) cannot belong to the same line; it is set aside.
 */
function separateOverlays(items: Fragment[]): { main: Fragment[]; overlays: Fragment[] } {
  const maxSize = Math.max(...items.map((f) => f.size));
  const large = items.filter((f) => f.size >= maxSize * LABEL_SIZE_RATIO);
  const overlays = items.filter(
    (f) => f.size < maxSize * LABEL_SIZE_RATIO && large.some((b) => f.x < b.right - 1 && f.right > b.x + 1),
  );
  if (overlays.length === 0) return { main: items, overlays };
  return { main: items.filter((f) => !overlays.includes(f)), overlays };
}

function isCellBreak(prev: Fragment, next: Fragment, reach: number): boolean {
  const gap = next.x - reach;
  if (gap > CELL_GAP * Math.max(prev.size, next.size)) return true;
  // Text followed by much smaller text after a real gap is a figure label. The
  // reverse order is how exponents end ("R^{l x d} is"), so it is not split.
  return next.size < prev.size * LABEL_SIZE_RATIO && gap > LABEL_GAP * prev.size;
}

/** Cuts a run of fragments at table-cell gaps and at figure labels set beside the text. */
function splitCells(items: Fragment[], column: 0 | 1 | 2): Segment[] {
  const segments: Segment[] = [];
  let current: Fragment[] = [];
  let reach = -Infinity;
  for (const f of items) {
    const prev = current[current.length - 1];
    if (prev && isCellBreak(prev, f, reach)) {
      segments.push(makeSegment(current, column));
      current = [];
      reach = -Infinity;
    }
    current.push(f);
    reach = Math.max(reach, f.right);
  }
  if (current.length > 0) segments.push(makeSegment(current, column));
  return segments;
}

function cutRow(row: Row): Segment[] {
  const { main, overlays } = separateOverlays([...row.items].sort(byX));
  return [...splitCells(main, row.column), ...(overlays.length ? splitCells(overlays, row.column) : [])];
}

// ─────────────────────────────── Line text ────────────────────────────────

/** Gap (in ems) above which two fragments are separated by a space. */
const WORD_GAP = 0.15;
const MARKER_RE = /^[\s*†‡§¶∗⋆♯#,\d]{1,6}$/u;

/** Joins a segment's fragments into text, handling script fragments. */
function segmentText(segment: Segment): string {
  const { size, top } = segment;
  let text = "";
  let prev: Fragment | null = null;
  for (const f of segment.items) {
    const isScript = f.size <= size * SCRIPT_SIZE_RATIO;
    const raised = isScript && f.top < top - size * 0.15;
    let piece = f.text;
    if (raised && MARKER_RE.test(piece)) {
      const lastChar = text.trimEnd().slice(-1);
      // Exponent after a digit ("10^9"); otherwise a footnote/affiliation marker.
      if (/\d/.test(lastChar) && /^\s*\d+\s*$/.test(piece)) piece = `^${piece.trim()}`;
      else continue; // `prev` stays put, so the marker's width still reads as a word gap
    }
    if (prev) {
      const gap = f.x - prev.right;
      const needsSpace =
        gap > WORD_GAP * size && !/\s$/.test(text) && !/^[\s,.;:)\]]/.test(piece) && !piece.startsWith("^");
      if (needsSpace) text += " ";
    }
    text += piece;
    prev = f;
  }
  return text.replace(/\s+/g, " ").trim();
}

function dominantStyle(items: Fragment[], size: number): { fontSize: number; bold: boolean } {
  let boldChars = 0;
  let chars = 0;
  for (const f of items) {
    if (f.size <= size * SCRIPT_SIZE_RATIO) continue;
    const n = f.text.replace(/\s/g, "").length;
    chars += n;
    if (f.bold) boldChars += n;
  }
  return { fontSize: round2(size), bold: chars > 0 && boldChars / chars >= 0.6 };
}

// ────────────────────────────── Reading order ─────────────────────────────

/**
 * Orders segments top-down in bands: a full-width line closes the current
 * band, and inside a band the left column is read before the right one.
 */
function readingOrder(segments: Segment[]): Segment[] {
  const sorted = [...segments].sort((a, b) => a.top - b.top || a.x - b.x);
  const ordered: Segment[] = [];
  let left: Segment[] = [];
  let right: Segment[] = [];
  const flush = () => {
    ordered.push(...left, ...right);
    left = [];
    right = [];
  };
  for (const segment of sorted) {
    if (segment.column === 0) {
      flush();
      ordered.push(segment);
    } else if (segment.column === 1) left.push(segment);
    else right.push(segment);
  }
  flush();
  return ordered;
}

/**
 * Rebuilds visual lines for one page in reading order.
 * Input y values are PDF baselines (from the bottom); output y is from the top.
 */
export function itemsToLines(
  items: RawTextItem[],
  page: number,
  pageWidth: number,
  pageHeight: number,
): TextLine[] {
  const fragments = toFragments(items, pageWidth, pageHeight);
  if (fragments.length === 0) return [];
  const gutter = detectGutter(fragments, pageWidth);
  let rows = groupRows(fragments);
  // Columns first, so a superscript is never judged against the other column's text.
  if (gutter) rows = rows.flatMap((row) => toColumnRows(row, gutter));
  rows = attachScriptRows(rows);
  if (!gutter) {
    const local = findLocalSplits(rows);
    rows = rows.flatMap((row) => {
      const split = local.get(row);
      return split ? toColumnRows(row, split) : [row];
    });
  }
  const segments = rows.flatMap(cutRow);

  const lines: TextLine[] = [];
  for (const segment of readingOrder(segments)) {
    segment.items.sort(byX);
    const text = segmentText(segment);
    if (!text) continue;
    const style = dominantStyle(segment.items, segment.size);
    lines.push({
      text,
      page,
      x: round2(segment.x),
      y: round2(segment.top),
      width: round2(segment.right - segment.x),
      fontSize: style.fontSize,
      bold: style.bold,
      column: segment.column,
    });
  }
  return lines;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

/**
 * Sentence-aligned chunking for retrieval.
 *
 * Passages are built from whole sentences until they reach a target length
 * (~170 words: long enough to hold a claim with its context, short enough
 * that the question and passage fit the reader's 512-token window). A chunk
 * never straddles two sections, so every passage has one honest section label
 * and page, and consecutive chunks share one sentence of overlap so an answer
 * that spans a boundary is still retrievable from at least one chunk.
 */
import type { Chunk, Sentence } from "@/types";

export interface ChunkOptions {
  targetWords?: number;
  maxWords?: number;
  /** Sentences repeated at the start of the next chunk. */
  overlap?: number;
}

const DEFAULTS = { targetWords: 170, maxWords: 260, overlap: 1 } satisfies Required<ChunkOptions>;

/**
 * A section tail shorter than this share of the target is folded into the
 * previous chunk (if it still fits) rather than becoming a stub passage.
 */
const MIN_TAIL_SHARE = 0.25;

export function wordCount(text: string): number {
  const trimmed = text.trim();
  return trimmed === "" ? 0 : trimmed.split(/\s+/).length;
}

/** Groups consecutive sentences that belong to the same section. */
function sectionRuns(sentences: Sentence[]): Sentence[][] {
  const runs: Sentence[][] = [];
  for (const sentence of sentences) {
    const run = runs[runs.length - 1];
    if (run && run[0].sectionId === sentence.sectionId) run.push(sentence);
    else runs.push([sentence]);
  }
  return runs;
}

/**
 * Returns the exclusive end index of the window that starts at `start`:
 * grow until the target is met, never past maxWords (a single over-long
 * sentence still forms its own window).
 */
function windowEnd(lengths: number[], start: number, targetWords: number, maxWords: number): number {
  let end = start;
  let words = 0;
  while (end < lengths.length) {
    if (end > start && words + lengths[end] > maxWords) break;
    words += lengths[end];
    end++;
    if (words >= targetWords) break;
  }

  const tail = lengths.slice(end).reduce((a, b) => a + b, 0);
  const shortTail = tail > 0 && tail < targetWords * MIN_TAIL_SHARE;
  return shortTail && words + tail <= maxWords ? lengths.length : end;
}

/** Splits one section into overlapping windows of sentence indices [start, end). */
function sectionWindows(lengths: number[], opts: Required<ChunkOptions>): [number, number][] {
  const windows: [number, number][] = [];
  let start = 0;
  while (start < lengths.length) {
    const end = windowEnd(lengths, start, opts.targetWords, opts.maxWords);
    windows.push([start, end]);
    if (end >= lengths.length) break;
    // Step back for overlap, but always advance by at least one sentence.
    start = Math.max(end - opts.overlap, start + 1);
  }
  return windows;
}

export function chunkSentences(paperId: string, sentences: Sentence[], opts: ChunkOptions = {}): Chunk[] {
  const options: Required<ChunkOptions> = { ...DEFAULTS, ...opts };
  options.overlap = Math.max(0, Math.floor(options.overlap));

  const chunks: Chunk[] = [];
  for (const run of sectionRuns(sentences)) {
    const lengths = run.map((s) => wordCount(s.text));
    for (const [start, end] of sectionWindows(lengths, options)) {
      const members = run.slice(start, end);
      const first = members[0];
      const text = members.map((s) => s.text.trim()).join(" ");
      chunks.push({
        id: `${paperId}#${chunks.length}`,
        paperId,
        index: chunks.length,
        sectionId: first.sectionId,
        sectionTitle: first.sectionTitle,
        sectionKind: first.sectionKind,
        page: first.page,
        text,
        tokenCount: wordCount(text),
      });
    }
  }
  return chunks;
}

/** Text fed to the embedder: the section title adds topical context to short passages. */
export function embeddingText(chunk: Chunk): string {
  return `${chunk.sectionTitle}. ${chunk.text}`;
}

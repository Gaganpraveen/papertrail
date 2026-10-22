/**
 * Live classification of what the user is typing, shown as a chip in the
 * command bar before anything is submitted. Several identifiers separated by
 * spaces, commas or new lines are treated as a batch of papers.
 */
import { parseInput } from "@/sources/input";
import type { InputKind } from "@/types";

export interface InputPreview {
  label: string;
  kind: InputKind | "batch";
  entries: string[];
}

const KIND_LABEL: Record<InputKind, string> = {
  arxiv: "arXiv ID",
  doi: "DOI",
  "pdf-url": "PDF link",
  url: "Web link",
  topic: "Topic search",
  upload: "PDF upload",
};

/** Splits a pasted list of identifiers; returns [] when the text is not a list of papers. */
export function splitBatch(raw: string): string[] {
  const tokens = raw
    .split(/[\s,;]+/)
    .map((t) => t.trim())
    .filter(Boolean);
  if (tokens.length < 2) return [];
  return tokens.every((t) => parseInput(t).kind !== "topic") ? [...new Set(tokens)] : [];
}

export function previewInput(raw: string): InputPreview | null {
  const text = raw.trim();
  if (!text) return null;
  const batch = splitBatch(text);
  if (batch.length > 1) return { label: `${batch.length} papers`, kind: "batch", entries: batch };
  const kind = parseInput(text).kind;
  return { label: KIND_LABEL[kind], kind, entries: [text] };
}

import { describe, expect, it } from "vitest";
import type { Chunk, Sentence, SectionKind } from "@/types";
import { chunkSentences, embeddingText, wordCount } from "@/retrieval/chunker";
import { HybridIndex, reciprocalRankFusion } from "@/retrieval/hybrid";

// ───────────────────────────────── helpers ─────────────────────────────────

let sentenceCounter = 0;

function sentence(sectionId: string, text: string, opts: { kind?: SectionKind; title?: string; page?: number } = {}): Sentence {
  const index = sentenceCounter++;
  return {
    id: `${sectionId}:${index}`,
    text,
    sectionId,
    sectionTitle: opts.title ?? sectionId,
    sectionKind: opts.kind ?? "other",
    page: opts.page ?? 1,
    index,
  };
}

/** A sentence of exactly `n` words. */
function wordsSentence(sectionId: string, n: number, tag: string): Sentence {
  return sentence(sectionId, Array.from({ length: n }, (_, i) => `${tag}${i}`).join(" "));
}

function chunk(id: string, text: string, sectionKind: SectionKind = "other"): Chunk {
  return {
    id,
    paperId: "p",
    index: Number(id.split("#")[1] ?? 0),
    sectionId: sectionKind,
    sectionTitle: sectionKind,
    sectionKind,
    page: 1,
    text,
    tokenCount: wordCount(text),
  };
}

const unit = (...xs: number[]) => {
  const v = Float32Array.from(xs);
  const n = Math.hypot(...xs);
  return v.map((x) => x / n);
};

// ──────────────────────────────── RRF ──────────────────────────────────────

describe("reciprocalRankFusion", () => {
  it("sums 1 / (k + rank) across rankings", () => {
    const fused = reciprocalRankFusion([
      ["a", "b"],
      ["b", "c"],
    ]);
    expect(fused.get("a")).toBeCloseTo(1 / 61);
    expect(fused.get("b")).toBeCloseTo(1 / 62 + 1 / 61);
    expect(fused.get("c")).toBeCloseTo(1 / 62);
  });

  it("respects a custom k and ignores empty rankings", () => {
    const fused = reciprocalRankFusion([["x"], []], 1);
    expect(fused.get("x")).toBeCloseTo(0.5);
    expect(fused.size).toBe(1);
  });
});

// ─────────────────────────────── Chunker ───────────────────────────────────

describe("chunkSentences", () => {
  it("never crosses a section boundary", () => {
    const sentences = [
      ...Array.from({ length: 4 }, (_, i) => wordsSentence("intro", 40, `i${i}_`)),
      ...Array.from({ length: 4 }, (_, i) => wordsSentence("method", 40, `m${i}_`)),
    ];
    const chunks = chunkSentences("paper", sentences);
    for (const c of chunks) {
      const prefix = c.sectionId === "intro" ? "i" : "m";
      expect(c.text.split(" ").every((w) => w.startsWith(prefix))).toBe(true);
    }
    expect(new Set(chunks.map((c) => c.sectionId))).toEqual(new Set(["intro", "method"]));
  });

  it("fills to the target, respects maxWords and overlaps by one sentence", () => {
    const sentences = Array.from({ length: 12 }, (_, i) => wordsSentence("s", 50, `w${i}_`));
    const chunks = chunkSentences("paper", sentences, { targetWords: 150, maxWords: 200, overlap: 1 });
    expect(chunks.length).toBeGreaterThan(2);
    for (const c of chunks) expect(c.tokenCount).toBeLessThanOrEqual(200);
    expect(chunks[0].tokenCount).toBe(150);
    // The last sentence of chunk n opens chunk n + 1.
    for (let i = 1; i < chunks.length; i++) {
      const previousLast = chunks[i - 1].text.split(" ").slice(-50).join(" ");
      expect(chunks[i].text.startsWith(previousLast)).toBe(true);
    }
    // Every sentence is covered.
    const all = chunks.map((c) => c.text).join(" ");
    for (const s of sentences) expect(all).toContain(s.text);
  });

  it("assigns sequential ids, section metadata and the first sentence's page", () => {
    const sentences = [
      sentence("s1", "Alpha beta gamma delta epsilon.", { kind: "method", title: "3 Method", page: 4 }),
      sentence("s1", "Zeta eta theta iota kappa.", { kind: "method", title: "3 Method", page: 5 }),
    ];
    const [only] = chunkSentences("arxiv:1", sentences);
    expect(only).toMatchObject({
      id: "arxiv:1#0",
      paperId: "arxiv:1",
      index: 0,
      sectionId: "s1",
      sectionTitle: "3 Method",
      sectionKind: "method",
      page: 4,
      text: "Alpha beta gamma delta epsilon. Zeta eta theta iota kappa.",
      tokenCount: 10,
    });
    expect(embeddingText(only)).toBe("3 Method. Alpha beta gamma delta epsilon. Zeta eta theta iota kappa.");
  });

  it("folds a short section tail into the previous chunk instead of leaving a stub", () => {
    const sentences = [wordsSentence("s", 170, "a"), wordsSentence("s", 20, "b")];
    const chunks = chunkSentences("paper", sentences);
    expect(chunks).toHaveLength(1);
    expect(chunks[0].tokenCount).toBe(190);
  });

  it("keeps an over-long sentence as its own chunk and terminates with large overlap", () => {
    const sentences = [wordsSentence("s", 400, "x"), wordsSentence("s", 30, "y"), wordsSentence("s", 30, "z")];
    const chunks = chunkSentences("paper", sentences, { overlap: 5 });
    expect(chunks[0].tokenCount).toBe(400);
    expect(chunks.length).toBeLessThanOrEqual(3);
  });

  it("returns nothing for no sentences", () => {
    expect(chunkSentences("paper", [])).toEqual([]);
  });
});

// ─────────────────────────────── HybridIndex ───────────────────────────────

describe("HybridIndex", () => {
  const corpus = [
    chunk("p#0", "We quantize the key cache per channel and the value cache per token.", "method"),
    chunk("p#1", "Experiments run on a single NVIDIA A100 GPU with batch size 64.", "experiments"),
    chunk("p#2", "Related work on pruning and distillation of transformers.", "related"),
    chunk("p#3", "Peak memory drops by 2.6x, enabling four times larger batches.", "results"),
  ];

  it("works BM25-only when embeddings are null", () => {
    const index = new HybridIndex(corpus, null);
    expect(index.hasDense).toBe(false);
    const hits = index.search("which GPU was used", null);
    expect(hits[0].chunk.id).toBe("p#1");
    expect(hits[0].dense).toBe(0);
    expect(hits[0].rank).toEqual({ bm25: 1, dense: undefined, final: 1 });
    expect(hits[0].bm25).toBeGreaterThan(0);
    // Chunks with no lexical match are not returned in lexical-only mode.
    expect(hits.every((h) => h.bm25 > 0)).toBe(true);
  });

  it("works dense-only when the query shares no terms with the paper", () => {
    const embeddings = [unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1), unit(1, 1, 0)];
    const index = new HybridIndex(corpus, embeddings);
    expect(index.hasDense).toBe(true);
    const hits = index.search("zzz qqq", unit(0.1, 1, 0));
    expect(hits[0].chunk.id).toBe("p#1");
    expect(hits[0].rank.bm25).toBeUndefined();
    expect(hits[0].rank.dense).toBe(1);
    expect(hits[0].bm25).toBe(0);
    // Orthogonal / opposite vectors carry no evidence and are left out.
    expect(hits.map((h) => h.chunk.id)).not.toContain("p#2");
  });

  it("fuses rankings: consistent second places beat a single first place", () => {
    const docs = [
      chunk("d#0", "attention attention attention sparse"), // BM25 #1, dense absent
      chunk("d#1", "attention mechanism for sequence models"), // BM25 #2, dense #2
      chunk("d#2", "state space models scale linearly"), // BM25 absent, dense #1
    ];
    const embeddings = [unit(-1, 0), unit(0.6, 0.8), unit(1, 0.2)];
    const index = new HybridIndex(docs, embeddings);
    const hits = index.search("attention", unit(1, 0));

    expect(hits.map((h) => h.chunk.id)).toEqual(["d#1", "d#0", "d#2"]);
    expect(hits[0].rrf).toBeCloseTo(2 / 62);
    expect(hits[0].rank).toEqual({ bm25: 2, dense: 2, final: 1 });
    // d#0 and d#2 tie on RRF (1/61 each); the lexical score breaks the tie.
    expect(hits[1].rrf).toBeCloseTo(hits[2].rrf);
    expect(hits[1].rank.bm25).toBe(1);
    expect(hits[2].rank.dense).toBe(1);
  });

  it("applies a filter before ranking", () => {
    const index = new HybridIndex(corpus, null);
    const hits = index.search("cache batch", null, { filter: (c) => c.sectionKind !== "method" });
    expect(hits.map((h) => h.chunk.sectionKind)).not.toContain("method");
    expect(hits[0].rank.final).toBe(1);
    expect(hits[0].rank.bm25).toBe(1);
  });

  it("limits results to k", () => {
    const embeddings = corpus.map((_, i) => unit(1, i + 1));
    const index = new HybridIndex(corpus, embeddings);
    expect(index.search("cache", unit(1, 1), { k: 2 })).toHaveLength(2);
  });

  it("falls back to lexical search when embeddings do not line up with chunks", () => {
    const index = new HybridIndex(corpus, [unit(1, 0)]);
    expect(index.hasDense).toBe(false);
    const hits = index.search("GPU", unit(1, 0));
    expect(hits[0].chunk.id).toBe("p#1");
    expect(hits[0].rank.dense).toBeUndefined();
  });

  it("returns nothing for an empty query without embeddings", () => {
    expect(new HybridIndex(corpus, null).search("the of and", null)).toEqual([]);
  });
});

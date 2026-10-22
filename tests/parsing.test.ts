import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { buildDocument, itemsToLines, matchQuoteItems, parsePdf, type RawTextItem } from "@/parsing";
import { cleanText } from "@/parsing/layout";
import { extractLinks, joinLineText, parseReference, splitReferenceEntries } from "@/parsing/structure";
import type { TextLine } from "@/types";

const PAGE_WIDTH = 612;
const PAGE_HEIGHT = 792;

/** A pdf.js-like text item; `baseline` is measured from the bottom of the page. */
function item(str: string, x: number, baseline: number, opts: Partial<RawTextItem> = {}): RawTextItem {
  const fontSize = opts.fontSize ?? 10;
  return {
    str,
    x,
    y: baseline,
    width: str.length * fontSize * 0.5,
    height: fontSize,
    fontName: "g_d0_f1",
    fontSize,
    bold: false,
    ...opts,
  };
}

/** Lays out a run of words left to right with normal word spacing. */
function words(text: string, x: number, baseline: number, opts: Partial<RawTextItem> = {}): RawTextItem[] {
  const size = opts.fontSize ?? 10;
  const out: RawTextItem[] = [];
  let cursor = x;
  for (const word of text.split(" ")) {
    out.push(item(word, cursor, baseline, opts));
    cursor += word.length * size * 0.5 + size * 0.3;
  }
  return out;
}

describe("cleanText", () => {
  it("expands ligatures and removes invisible characters", () => {
    const ffi = String.fromCharCode(0xfb03);
    const fi = String.fromCharCode(0xfb01);
    const zeroWidth = String.fromCharCode(0x200b);
    const softHyphen = String.fromCharCode(0x00ad);
    expect(cleanText(`e${ffi}cient ${fi}ne${zeroWidth}-tuning`)).toBe("efficient fine-tuning");
    expect(cleanText(`token${softHyphen}izer`)).toBe("tokenizer");
    expect(cleanText(`a${String.fromCharCode(0x00a0)}b`)).toBe("a b");
  });
});

describe("itemsToLines", () => {
  it("merges items on one baseline into a line with top-down y", () => {
    const lines = itemsToLines(
      [...words("Attention is all", 72, 700), item("you", 160, 700.4), item("need", 182, 699.8)],
      1,
      PAGE_WIDTH,
      PAGE_HEIGHT,
    );
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({ text: "Attention is all you need", page: 1, column: 0, fontSize: 10 });
    expect(lines[0].y).toBeCloseTo(PAGE_HEIGHT - 700, 0);
    expect(lines[0].x).toBe(72);
  });

  it("joins abutting fragments without a space and sorts rows top to bottom", () => {
    const lines = itemsToLines(
      [item("second line", 72, 688), item("compres", 72, 700), item("sion", 107, 700)],
      3,
      PAGE_WIDTH,
      PAGE_HEIGHT,
    );
    expect(lines.map((l) => l.text)).toEqual(["compression", "second line"]);
  });

  it("normalises ligatures and drops empty or off-page items", () => {
    const fl = String.fromCharCode(0xfb02);
    const lines = itemsToLines(
      [item(`over${fl}ow`, 72, 700), item("   ", 120, 700), item("ghost", 900, 700), item("hidden", 72, -50)],
      1,
      PAGE_WIDTH,
      PAGE_HEIGHT,
    );
    expect(lines.map((l) => l.text)).toEqual(["overflow"]);
  });

  it("reads a two-column page left column first, after the full-width title", () => {
    const items: RawTextItem[] = [
      ...words("A Study of Streaming Attention", 180, 740, { fontSize: 16, bold: true }),
    ];
    for (let row = 0; row < 24; row++) {
      const baseline = 700 - row * 12;
      items.push(...words(`left column sentence number ${row} keeps going here`, 54, baseline));
      items.push(...words(`right column sentence number ${row} keeps going on`, 318, baseline));
    }
    const lines = itemsToLines(items, 2, PAGE_WIDTH, PAGE_HEIGHT);

    expect(lines[0]).toMatchObject({ text: "A Study of Streaming Attention", column: 0, bold: true, fontSize: 16 });
    const body = lines.slice(1);
    expect(body).toHaveLength(48);
    expect(body.slice(0, 24).every((l) => l.column === 1 && l.text.startsWith("left"))).toBe(true);
    expect(body.slice(24).every((l) => l.column === 2 && l.text.startsWith("right"))).toBe(true);
    expect(body[0].text).toBe("left column sentence number 0 keeps going here");
    expect(body[24].text).toBe("right column sentence number 0 keeps going on");
  });

  it("drops raised footnote markers but keeps exponents", () => {
    const lines = itemsToLines(
      [
        item("Ada Lovelace", 72, 700, { width: 60 }),
        item("1", 132.5, 704, { fontSize: 7, width: 3.5 }),
        item("and Alan Turing", 140, 700, { width: 75 }),
        item("trained on 10", 72, 680, { width: 65 }),
        item("9", 137, 684, { fontSize: 7, width: 3.5 }),
        item("tokens", 144, 680, { width: 30 }),
      ],
      1,
      PAGE_WIDTH,
      PAGE_HEIGHT,
    );
    expect(lines.map((l) => l.text)).toEqual(["Ada Lovelace and Alan Turing", "trained on 10^9 tokens"]);
  });

  it("marks lines as bold when most of their characters are bold", () => {
    const lines = itemsToLines(
      [item("3.2 Results", 72, 700, { bold: true, width: 55 }), item("on", 130, 700, { width: 10 })],
      1,
      PAGE_WIDTH,
      PAGE_HEIGHT,
    );
    expect(lines[0].bold).toBe(true);
  });
});

// ───────────────────────────── buildDocument ─────────────────────────────

interface Spec {
  text: string;
  size?: number;
  bold?: boolean;
  /** Distance from the previous line's baseline. */
  gap?: number;
  /** Absolute y (from the top), for page furniture. */
  y?: number;
  x?: number;
}

/** Stacks lines down a single-column page the way a typeset paper would. */
function layoutPage(page: number, specs: Array<string | Spec>): TextLine[] {
  let y = 60;
  return specs.map((raw) => {
    const spec: Spec = typeof raw === "string" ? { text: raw } : raw;
    const size = spec.size ?? 10;
    y = spec.y ?? y + (spec.gap ?? 12);
    return {
      text: spec.text,
      page,
      x: spec.x ?? 72,
      y,
      width: Math.min(468, spec.text.length * size * 0.5),
      fontSize: size,
      bold: spec.bold ?? false,
      column: 0,
    };
  });
}

const RUNNING_HEADER: Spec = { text: "Preprint. Under review as a conference paper.", size: 9, y: 40 };
const pageNumber = (n: number): Spec => ({ text: String(n), y: 750 });
const heading = (text: string, size = 12): Spec => ({ text, size, bold: true, gap: 24 });

function syntheticPaper(): TextLine[] {
  const page1 = layoutPage(1, [
    RUNNING_HEADER,
    { text: "Attention Sinks for Streaming Language Models", size: 17, bold: true, y: 90 },
    { text: "Ada Lovelace Alan Turing", size: 11, gap: 24 },
    heading("Abstract"),
    "Deploying language models on long streams is hard because the key-value cache grows without",
    "bound and quality collapses once the text exceeds the training window. We observe that the",
    "first tokens act as attention sinks that absorb surplus attention. Keeping them together with a",
    "rolling window of recent tokens lets models handle millions of tokens with stable perplexity.",
    heading("1 Introduction"),
    "Long inputs dominate real deployments such as dialogue systems and agents. Prior work studies the",
    "compression of the cache, yet the compres-",
    "sion ratio alone does not explain the collapse we observe on long streams of dialogue text.",
    "Each key-value entry costs memory, so the key-",
    "value cache must stay bounded while the model keeps attending to the most relevant tokens.",
    "3 of the 32 layers attend almost exclusively to the first token, a pattern we study below.",
    "Code is available at https://github.com/example/attention-sinks for anyone to reproduce.",
    "We summarise our contributions and then describe the method in detail in the next section.",
    pageNumber(1),
  ]);
  const page2 = layoutPage(2, [
    RUNNING_HEADER,
    heading("2 Method"),
    "Our method keeps a small set of sink tokens and a rolling window of recent tokens in the cache.",
    "Positions are assigned relative to the cache rather than to the original text, which keeps every",
    "attention score inside the range seen during training and avoids any additional fine-tuning.",
    heading("2.1 Cache policy", 10),
    "When the window is full, the oldest non-sink entry is evicted and the new token is appended.",
    "The policy needs no training and adds no parameters, so it applies to any pretrained model.",
    { text: "Figure 1: Overview of the rolling cache with four attention sinks.", size: 9, gap: 24 },
    heading("3 Experiments"),
    "We evaluate perplexity on four million tokens of book text with several model families and",
    "compare the rolling cache against dense attention, window attention and recomputation baselines.",
    "Results show stable perplexity across the whole stream while dense attention runs out of memory.",
    pageNumber(2),
  ]);
  const page3 = layoutPage(3, [
    RUNNING_HEADER,
    heading("4 Conclusion"),
    "Keeping a few initial tokens as attention sinks lets language models process endless streams",
    "with constant memory. We hope the observation helps future work on efficient long contexts.",
    heading("References"),
    "[1] Ada Lovelace and Alan Turing. Sketches of an analytical engine. In Proceedings of",
    { text: "Neural Information Processing Systems, 2020.", x: 84 },
    "[2] Grace Hopper. Compiling endless streams. arXiv preprint arXiv:2309.17453, 2023.",
    "[3] Edsger Dijkstra. Go to statement considered harmful. Commun. ACM, 1968.",
    { text: "doi:10.1145/362929.362947.", x: 84 },
    pageNumber(3),
  ]);
  return [...page1, ...page2, ...page3];
}

describe("buildDocument", () => {
  const doc = buildDocument(syntheticPaper(), 3, 3);
  const section = (title: string) => doc.sections.find((s) => s.title === title);

  it("recovers the title from the largest text on page one", () => {
    expect(doc.title).toBe("Attention Sinks for Streaming Language Models");
  });

  it("detects numbered and named headings with levels and kinds", () => {
    expect(
      doc.sections.map((s) => ({ number: s.number, title: s.title, level: s.level, kind: s.kind })),
    ).toEqual([
      { number: undefined, title: "Abstract", level: 1, kind: "abstract" },
      { number: "1", title: "Introduction", level: 1, kind: "introduction" },
      { number: "2", title: "Method", level: 1, kind: "method" },
      { number: "2.1", title: "Cache policy", level: 2, kind: "method" },
      { number: "3", title: "Experiments", level: 1, kind: "experiments" },
      { number: "4", title: "Conclusion", level: 1, kind: "conclusion" },
      { number: undefined, title: "References", level: 1, kind: "references" },
    ]);
    expect(doc.sections.map((s) => s.id)).toEqual(["s0", "s1", "s2", "s3", "s4", "s5", "s6"]);
    expect(section("Method")).toMatchObject({ pageStart: 2, pageEnd: 2 });
  });

  it("classifies plural related-work headings and failure analyses by what they hold", () => {
    const page = layoutPage(1, [
      { text: "A Study of Streaming Caches", size: 17, bold: true, y: 90 },
      heading("1 Introduction"),
      "Long inputs dominate real deployments such as dialogue systems and agents in practice.",
      heading("2 Related Works"),
      "Earlier systems compress the cache with quantization, eviction and low-rank projection.",
      heading("3 Experiments"),
      "We evaluate perplexity on four million tokens of book text with several model families.",
      heading("3.1 Success and Failure Modes Analysis", 10),
      "The window baseline loses the topic once the cache slides past the first few tokens.",
    ]);
    const kinds = Object.fromEntries(buildDocument(page, 1, 1).sections.map((s) => [s.title, s.kind]));
    expect(kinds["Related Works"]).toBe("related");
    expect(kinds["Success and Failure Modes Analysis"]).toBe("results");
  });

  it("keeps a body sentence that starts with a number inside its section", () => {
    expect(section("Introduction")?.text).toContain("3 of the 32 layers attend almost exclusively");
  });

  it("uses the Abstract section as the abstract", () => {
    expect(doc.abstract).toMatch(/^Deploying language models on long streams/);
    expect(doc.abstract).toContain("stable perplexity.");
  });

  it("removes running headers and page numbers", () => {
    const allText = [...doc.sections.map((s) => s.text), ...doc.pages.map((p) => p.text)].join("\n");
    expect(allText).not.toContain("Under review");
    expect(doc.pages.every((p) => !/^\d+$/m.test(p.text))).toBe(true);
  });

  it("dehyphenates line breaks but keeps compound hyphens", () => {
    const intro = section("Introduction")?.text ?? "";
    expect(intro).toContain("yet the compression ratio alone");
    expect(intro).toContain("so the key-value cache must stay bounded");
  });

  it("splits and parses the reference list", () => {
    expect(doc.references).toHaveLength(3);
    expect(doc.references[0]).toMatchObject({
      authors: "Ada Lovelace and Alan Turing",
      title: "Sketches of an analytical engine",
      year: 2020,
    });
    expect(doc.references[0].raw).toContain("Neural Information Processing Systems");
    expect(doc.references[1]).toMatchObject({ arxivId: "2309.17453", year: 2023 });
    expect(doc.references[2]).toMatchObject({ doi: "10.1145/362929.362947", year: 1968 });
  });

  it("collects captions and appends them to their section", () => {
    expect(doc.captions).toEqual([
      { kind: "figure", label: "Figure 1", text: "Overview of the rolling cache with four attention sinks.", page: 2 },
    ]);
    expect(section("Cache policy")?.text).toContain("Figure 1: Overview of the rolling cache");
  });

  it("finds links outside the reference list", () => {
    expect(doc.links).toEqual(["https://github.com/example/attention-sinks"]);
  });

  it("reports per-page text and quality", () => {
    expect(doc.pages.map((p) => p.page)).toEqual([1, 2, 3]);
    expect(doc.pages.every((p) => p.charCount === p.text.length && p.charCount > 0)).toBe(true);
    expect(doc.quality).toBe("good");
    expect(doc.warnings).toEqual([]);
  });

  it("warns about truncated, image-only and unstructured documents", () => {
    const truncated = buildDocument(syntheticPaper(), 80, 3);
    expect(truncated.warnings).toEqual([
      "Only the first 3 of 80 pages were parsed.",
      "Very long document (80 pages); analysis may take longer.",
    ]);

    const scanned = buildDocument([], 4, 4);
    expect(scanned.quality).toBe("image-only");
    expect(scanned.sections).toEqual([]);
    expect(scanned.warnings.join(" ")).toMatch(/scanned or image-only/);

    const prose = layoutPage(1, [
      "A short memo without any headings, as a scanned form with a typed",
      "cover note would produce. Only these few lines carry real text, which",
      "is far too little for a page of a research paper.",
    ]);
    const note = buildDocument(prose, 1, 1);
    expect(note.sections).toEqual([]);
    expect(note.quality).toBe("degraded");
    expect(note.warnings.join(" ")).toMatch(/No section headings were detected/);
  });
});

describe("references", () => {
  const line = (text: string, y: number, x = 72): TextLine => ({
    text,
    page: 9,
    x,
    y,
    width: 400,
    fontSize: 9,
    bold: false,
    column: 0,
  });

  it("splits author-year entries by hanging indent", () => {
    const entries = splitReferenceEntries([
      line("Vaswani, A., Shazeer, N., and Parmar, N. Attention is all you need. In Advances in", 100),
      line("Neural Information Processing Systems, 2017.", 111, 82),
      line("Brown, T., Mann, B., and Ryder, N. Language models are few-shot learners. arXiv", 122),
      line("preprint arXiv:2005.14165, 2020.", 133, 82),
      line("Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural Computation, 1997.", 144),
    ]);
    expect(entries.map((e) => e.length)).toEqual([2, 2, 1]);
  });

  it("parses surname-first author lists", () => {
    expect(
      parseReference("Vaswani, A., Shazeer, N., and Parmar, N. Attention is all you need. In NeurIPS, 2017."),
    ).toMatchObject({
      authors: "Vaswani, A., Shazeer, N., and Parmar, N.",
      title: "Attention is all you need",
      year: 2017,
    });
  });

  it("parses numbered entries with quoted titles and arXiv ids", () => {
    const ref = parseReference(
      '[12] T. Brown, B. Mann, and N. Ryder, "Language models are few-shot learners," arXiv:2005.14165, 2020.',
    );
    expect(ref).toMatchObject({ title: "Language models are few-shot learners", arxivId: "2005.14165", year: 2020 });
    expect(ref.authors).toContain("T. Brown");
  });

  it("ignores years inside identifiers", () => {
    const ref = parseReference("Lee, K. Scaling laws. https://doi.org/10.1000/2019.12345, 2021.");
    expect(ref.year).toBe(2021);
    expect(ref.doi).toBe("10.1000/2019.12345");
  });
});

describe("text helpers", () => {
  const vocab = new Set(["compression", "state", "art", "commonly", "used"]);

  it("joins lines across hyphens, compounds and wrapped URLs", () => {
    expect(joinLineText("the compres-", "sion ratio", vocab)).toBe("the compression ratio");
    expect(joinLineText("commonly-", "used methods", vocab)).toBe("commonly-used methods");
    expect(joinLineText("we use GPT-", "4 as judge", vocab)).toBe("we use GPT-4 as judge");
    expect(joinLineText("see https://github.com/org/", "repo-name for code", vocab)).toBe(
      "see https://github.com/org/repo-name for code",
    );
    expect(joinLineText("plain line", "Next sentence", vocab)).toBe("plain line Next sentence");
  });

  it("extracts and normalises links", () => {
    expect(
      extractLinks([
        "Code: https://github.com/foo/bar). Data at huggingface.co/datasets/foo/baz, demo (www.example.org/demo).",
        "Again https://github.com/foo/bar/ and an email user@example.com.",
      ]),
    ).toEqual(["https://github.com/foo/bar", "https://huggingface.co/datasets/foo/baz", "https://www.example.org/demo"]);
  });

  it("matches a quote to the text items it spans", () => {
    const items = ["Window attention", "fails once the", "initial tokens are", "evicted from the cache.", "Table 3"];
    expect([...matchQuoteItems(items, "once the initial tokens are evicted")].sort()).toEqual([1, 2, 3]);
    expect(matchQuoteItems(items, "nothing like this appears anywhere").size).toBe(0);
  });
});

// ───────────────────────────── Real papers ─────────────────────────────

// The repository ships a few CC-BY papers; point PAPERTRAIL_SAMPLES at a folder of PDFs to run more.
const SAMPLE_DIRS = [fileURLToPath(new URL("./fixtures/pdfs", import.meta.url)), process.env.PAPERTRAIL_SAMPLES];
const samplePdfs = [
  ...new Map(
    SAMPLE_DIRS.filter((dir): dir is string => !!dir && existsSync(dir)).flatMap((dir) =>
      readdirSync(dir)
        .filter((f) => f.endsWith(".pdf"))
        .map((f) => [f, join(dir, f)] as const),
    ),
  ).entries(),
].sort(([a], [b]) => a.localeCompare(b));

describe.skipIf(samplePdfs.length === 0)("parsePdf on sample papers", () => {
  it.each(samplePdfs)("%s", async (_name, path) => {
    const buf = readFileSync(path);
    const bytes = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
    const progress: number[] = [];
    const doc = await parsePdf(bytes, { onProgress: (f) => progress.push(f) });

    expect(doc.quality).toBe("good");
    expect(doc.pagesParsed).toBe(Math.min(doc.pageCount, 60));
    expect(doc.pages).toHaveLength(doc.pagesParsed);
    expect(doc.title?.length ?? 0).toBeGreaterThan(5);
    expect(doc.abstract?.length ?? 0).toBeGreaterThan(300);
    expect(progress.at(-1)).toBe(1);

    const kinds = new Set(doc.sections.map((s) => s.kind));
    expect(doc.sections.length).toBeGreaterThanOrEqual(5);
    for (const kind of ["introduction", "references"] as const) expect(kinds).toContain(kind);
    // No section title should read like a body sentence.
    for (const s of doc.sections) {
      expect(s.title.split(/\s+/).length, s.title).toBeLessThanOrEqual(16);
      expect(s.title, s.title).not.toMatch(/[.,;]$/);
    }
    // Pages run forward through the outline.
    for (let i = 1; i < doc.sections.length; i++) {
      expect(doc.sections[i].pageStart).toBeGreaterThanOrEqual(doc.sections[i - 1].pageStart);
    }

    expect(doc.references.length).toBeGreaterThanOrEqual(20);
    const withYear = doc.references.filter((r) => r.year !== undefined).length;
    expect(withYear / doc.references.length).toBeGreaterThan(0.8);
    expect(doc.captions.length).toBeGreaterThan(0);
  }, 60_000);
});

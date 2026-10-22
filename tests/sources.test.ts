import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { EmbedFn, PaperMeta } from "@/types";
import {
  arxivAbsUrl,
  arxivPdfUrl,
  fetchArxivMeta,
  mapDataCite,
  venueFromComment,
  withArxivLinks,
} from "@/sources/arxiv";
import { findArxivByTitle, mapCrossref, resolveDoi, stripJats } from "@/sources/doi";
import { sha256Hex } from "@/sources/hash";
import {
  backoffDelay,
  fetchJson,
  fetchWithRetry,
  isAbortError,
  parseRetryAfter,
  SourceError,
} from "@/sources/http";
import {
  cleanMetaText,
  normalizeAbstract,
  normalizeDoi,
  normalizePersonName,
  parseArxivId,
  parseInput,
} from "@/sources/input";
import { mapOpenAlexWork, mergeMeta, reconstructAbstract } from "@/sources/openalex";
import { downloadPdf, hasPdfSignature } from "@/sources/pdf";
import { rankCandidates, recencyScore, RANK_WEIGHTS } from "@/sources/rank";
import { broadenQuery, searchArxiv } from "@/sources/search";

// ─────────────────────────────────── helpers ───────────────────────────────────

const FIXTURE_DIRS = [fileURLToPath(new URL("./fixtures/api", import.meta.url))];
const PDF_FIXTURE = fileURLToPath(new URL("./fixtures/pdfs/2402.02750.pdf", import.meta.url));

function fixturePath(name: string): string | undefined {
  return FIXTURE_DIRS.map((dir) => join(dir, name)).find((path) => existsSync(path));
}
function hasFixtures(...names: string[]): boolean {
  return names.every((name) => fixturePath(name) !== undefined);
}
function fixture(name: string): unknown {
  const path = fixturePath(name);
  if (!path) throw new Error(`Missing fixture ${name}`);
  return JSON.parse(readFileSync(path, "utf8"));
}

function jsonResponse(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

type Route = [pattern: RegExp, respond: (url: string) => Response | Promise<Response>];

/** Replaces global fetch with a router; unmatched URLs get a 404. */
function routeFetch(routes: Route[]) {
  const spy = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const route = routes.find(([pattern]) => pattern.test(url));
    return route ? route[1](url) : new Response("not found", { status: 404 });
  });
  vi.stubGlobal("fetch", spy);
  return spy;
}

/** A fetch that only settles when its signal aborts, to exercise timeouts and cancellation. */
function hangingFetch() {
  const spy = vi.fn(
    (_input: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }),
  );
  vi.stubGlobal("fetch", spy);
  return spy;
}

function streamOf(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  let i = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < chunks.length) controller.enqueue(chunks[i++]);
      else controller.close();
    },
  });
}

function chunked(bytes: Uint8Array, size: number): Uint8Array[] {
  const chunks: Uint8Array[] = [];
  for (let offset = 0; offset < bytes.length; offset += size) chunks.push(bytes.slice(offset, offset + size));
  return chunks;
}

async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("Expected the promise to reject");
}

afterEach(() => {
  vi.unstubAllGlobals();
});

// ───────────────────────────────── parseInput ─────────────────────────────────

describe("parseInput", () => {
  const arxivCases: [input: string, id: string, version?: string][] = [
    ["2402.02750", "2402.02750"],
    ["2402.02750v3", "2402.02750", "v3"],
    ["  2402.02750  ", "2402.02750"],
    ["2402.02750V2", "2402.02750", "v2"],
    ["arXiv:2402.02750", "2402.02750"],
    ["arxiv: 2402.02750v2", "2402.02750", "v2"],
    ["arXiv:2402.02750.", "2402.02750"],
    ["arXiv:2402.02750 [cs.CL]", "2402.02750"],
    ["2402.02750v2 [cs.LG]", "2402.02750", "v2"],
    ["1412.6980", "1412.6980"],
    ["1501.00001", "1501.00001"],
    ["hep-th/9901001", "hep-th/9901001"],
    ["math.GT/0309136v2", "math/0309136", "v2"],
    ["arxiv.org/abs/2402.02750v2", "2402.02750", "v2"],
    ["https://arxiv.org/abs/2402.02750", "2402.02750"],
    ["https://arxiv.org/pdf/2402.02750v2.pdf", "2402.02750", "v2"],
    ["https://arxiv.org/pdf/2402.02750", "2402.02750"],
    ["https://arxiv.org/html/2402.02750v2/", "2402.02750", "v2"],
    ["http://export.arxiv.org/abs/hep-th/9901001", "hep-th/9901001"],
    ["https://arxiv.org/pdf/hep-th/9901001v1.pdf", "hep-th/9901001", "v1"],
    ["https://www.arxiv.org/abs/2402.02750#S3", "2402.02750"],
    ["https://huggingface.co/papers/2402.02750", "2402.02750"],
    ["<https://arxiv.org/abs/2402.02750>", "2402.02750"],
    ["10.48550/arXiv.2402.02750", "2402.02750"],
  ];

  it.each(arxivCases)("recognises %j as arXiv %s", (input, id, version) => {
    const parsed = parseInput(input);
    expect(parsed.kind).toBe("arxiv");
    expect(parsed.arxivId).toBe(id);
    expect(parsed.version).toBe(version);
    expect(parsed.raw).toBe(input);
  });

  it("keeps the normalised URL for arXiv links", () => {
    expect(parseInput("arxiv.org/abs/2402.02750v2").url).toBe("https://arxiv.org/abs/2402.02750v2");
  });

  it("maps an arXiv DOI URL to kind arxiv and keeps the DOI", () => {
    const parsed = parseInput("https://doi.org/10.48550/arXiv.2402.02750");
    expect(parsed).toMatchObject({ kind: "arxiv", arxivId: "2402.02750", doi: "10.48550/arxiv.2402.02750" });
  });

  const doiCases: [input: string, doi: string][] = [
    ["10.1145/3292500.3330701", "10.1145/3292500.3330701"],
    ["doi:10.18653/v1/N19-1423", "10.18653/v1/n19-1423"],
    ["DOI: 10.18653/v1/N19-1423", "10.18653/v1/n19-1423"],
    ["https://doi.org/10.1038/s41586-021-03819-2", "10.1038/s41586-021-03819-2"],
    ["https://dx.doi.org/10.1016/S0140-6736(20)30183-5.", "10.1016/s0140-6736(20)30183-5"],
    ["https://doi.org/10.1002%2Fanie.201915678", "10.1002/anie.201915678"],
    ["https://dl.acm.org/doi/10.1145/3292500.3330701", "10.1145/3292500.3330701"],
    ["https://onlinelibrary.wiley.com/doi/full/10.1002/anie.201915678", "10.1002/anie.201915678"],
  ];

  it.each(doiCases)("recognises %j as DOI %s", (input, doi) => {
    const parsed = parseInput(input);
    expect(parsed.kind).toBe("doi");
    expect(parsed.doi).toBe(doi);
  });

  it.each([
    ["https://aclanthology.org/N19-1423.pdf"],
    ["https://example.com/papers/attention.PDF?download=1"],
    ["https://openreview.net/pdf?id=rkE3y85ee"],
  ])("treats %j as a direct PDF link", (input) => {
    const parsed = parseInput(input);
    expect(parsed.kind).toBe("pdf-url");
    expect(parsed.url).toBe(new URL(input).toString());
  });

  it.each([
    ["https://github.com/jy-yuan/KIVI", "https://github.com/jy-yuan/KIVI"],
    ["https://arxiv.org/list/cs.CL/recent", "https://arxiv.org/list/cs.CL/recent"],
    ["https://arxiv.org/abs/not-an-id", "https://arxiv.org/abs/not-an-id"],
    ["www.example.com", "https://www.example.com/"],
  ])("treats %j as a generic URL", (input, url) => {
    expect(parseInput(input)).toEqual({ kind: "url", raw: input, url });
  });

  it.each([
    ["kv cache compression", "kv cache compression"],
    ["  Retrieval   augmented\ngeneration ", "Retrieval augmented generation"],
    ['"attention is all you need"', "attention is all you need"],
    ["2023.12345", "2023.12345"], // month 23 does not exist
    ["1412.69801", "1412.69801"], // 5-digit serials only exist from 2015
    ["0612.1234", "0612.1234"], // new-style IDs start in April 2007
    ["hep-th/2501001", "hep-th/2501001"], // the old scheme ended in 2007
    ["foo/1234567", "foo/1234567"], // not an arXiv archive
    ["", ""],
  ])("treats %j as a topic", (input, query) => {
    expect(parseInput(input)).toEqual({ kind: "topic", raw: input, query });
  });
});

describe("identifier helpers", () => {
  it("parses arXiv IDs strictly", () => {
    expect(parseArxivId("2402.02750v2")).toEqual({ id: "2402.02750", version: "v2" });
    expect(parseArxivId("cond-mat/0207270")).toEqual({ id: "cond-mat/0207270" });
    expect(parseArxivId("2402.2750")).toBeNull();
    expect(parseArxivId("hello")).toBeNull();
  });

  it("normalises DOIs", () => {
    expect(normalizeDoi("https://doi.org/10.18653/v1/N19-1423")).toBe("10.18653/v1/n19-1423");
    expect(normalizeDoi("10.1000/xyz123,")).toBe("10.1000/xyz123");
    expect(normalizeDoi("11.1000/xyz")).toBeNull();
  });

  it("normalises person names", () => {
    expect(normalizePersonName("Liu, Zirui")).toBe("Zirui Liu");
    expect(normalizePersonName("Jacob Devlin")).toBe("Jacob Devlin");
    expect(normalizePersonName("OneRec Team")).toBe("OneRec Team");
  });
});

describe("normalizeAbstract", () => {
  const body =
    "Proteins are essential to life, and understanding their structure can facilitate a mechanistic understanding of their function in every living cell.";
  it("drops a leading label but keeps real words", () => {
    expect(normalizeAbstract(`Abstract ${body}`)).toBe(body);
    expect(normalizeAbstract(`Abstract interpretation ${body}`)).toBe(`Abstract interpretation ${body}`);
  });
  it("rejects fragments and citation strings", () => {
    expect(normalizeAbstract("Too short to be an abstract.")).toBeUndefined();
    const citation =
      "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics. 2019.";
    expect(normalizeAbstract(citation, ["Jacob Devlin", "Ming-Wei Chang"])).toBeUndefined();
    expect(normalizeAbstract(body, ["Jacob Devlin"])).toBe(body);
  });
});

describe("cleanMetaText", () => {
  it("renders the inline LaTeX found in arXiv abstracts", () => {
    expect(cleanMetaText("using $\\mathbf{2.6\\times}$ less peak memory")).toBe(
      "using 2.6× less peak memory",
    );
    expect(cleanMetaText("bringing $\\mathbf{2.35\\times \\sim 3.47\\times}$ throughput")).toBe(
      "bringing 2.35× ~ 3.47× throughput",
    );
    expect(cleanMetaText("KV$^2$: A Self-Refining KV Cache")).toBe("KV²: A Self-Refining KV Cache");
    expect(cleanMetaText("H$_2$O, \\textbf{CaliDrop}, and $d_{eff}$")).toBe("H₂O, CaliDrop, and d_eff");
    expect(cleanMetaText("reduces cost by 50\\%")).toBe("reduces cost by 50%");
    expect(cleanMetaText("Key--value caches")).toBe("Key–value caches");
  });

  it("strips markup, decodes entities and collapses whitespace", () => {
    expect(cleanMetaText("<p>Deep &amp; wide\n   networks</p> , really")).toBe(
      "Deep & wide networks, really",
    );
    expect(cleanMetaText("a < b and c > d")).toBe("a < b and c > d");
    expect(cleanMetaText(undefined)).toBe("");
  });
});

// ─────────────────────────────────── http ───────────────────────────────────

describe("fetchWithRetry", () => {
  it("retries a 5xx and honours Retry-After", async () => {
    let calls = 0;
    const spy = routeFetch([
      [
        /example/,
        () =>
          ++calls === 1
            ? new Response("", { status: 503, headers: { "retry-after": "0" } })
            : jsonResponse({ ok: true }),
      ],
    ]);
    const response = await fetchWithRetry("https://example.org/x");
    expect(response.status).toBe(200);
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("retries 429 and gives up with rate-limited", async () => {
    const spy = routeFetch([
      [/example/, () => new Response("", { status: 429, headers: { "retry-after": "0" } })],
    ]);
    const error = await rejection(fetchWithRetry("https://example.org/x", { retries: 2 }));
    expect(error).toBeInstanceOf(SourceError);
    expect((error as SourceError).code).toBe("rate-limited");
    expect(spy).toHaveBeenCalledTimes(3);
  });

  it("fails fast when Retry-After asks for an unreasonable wait", async () => {
    const spy = routeFetch([
      [/example/, () => new Response("", { status: 429, headers: { "retry-after": "120" } })],
    ]);
    const error = (await rejection(fetchWithRetry("https://example.org/x"))) as SourceError;
    expect(error.code).toBe("rate-limited");
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("does not retry 404 or other client errors", async () => {
    const spy = routeFetch([
      [/missing/, () => new Response("", { status: 404 })],
      [/bad/, () => new Response("", { status: 400 })],
    ]);
    expect(((await rejection(fetchWithRetry("https://x.org/missing"))) as SourceError).code).toBe(
      "not-found",
    );
    expect(((await rejection(fetchWithRetry("https://x.org/bad"))) as SourceError).code).toBe("invalid");
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it("maps a fetch TypeError to a network error outside the browser", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new TypeError("fetch failed"))),
    );
    const error = (await rejection(fetchWithRetry("https://example.org/x", { retries: 0 }))) as SourceError;
    expect(error.code).toBe("network");
    expect(error.url).toBe("https://example.org/x");
  });

  it("times out slow servers", async () => {
    hangingFetch();
    const error = (await rejection(
      fetchWithRetry("https://example.org/x", { timeoutMs: 20, retries: 0 }),
    )) as SourceError;
    expect(error.code).toBe("timeout");
  });

  it("propagates a caller abort as an AbortError, not a SourceError", async () => {
    hangingFetch();
    const controller = new AbortController();
    const pending = fetchWithRetry("https://example.org/x", { signal: controller.signal });
    controller.abort();
    const error = await rejection(pending);
    expect(isAbortError(error)).toBe(true);
    expect(error).not.toBeInstanceOf(SourceError);
  });

  it("computes jittered exponential backoff within bounds", () => {
    expect(backoffDelay(0, () => 0)).toBe(250);
    expect(backoffDelay(0, () => 1)).toBe(500);
    expect(backoffDelay(3, () => 0.5)).toBe(3000);
    expect(backoffDelay(20, () => 1)).toBe(8000);
  });

  it("parses Retry-After seconds and HTTP dates", () => {
    expect(parseRetryAfter("2")).toBe(2000);
    expect(parseRetryAfter(null)).toBeNull();
    expect(parseRetryAfter("soon")).toBeNull();
    const now = Date.UTC(2026, 0, 1, 0, 0, 0);
    expect(parseRetryAfter("Thu, 01 Jan 2026 00:00:05 GMT", now)).toBe(5000);
  });

  it("fetchJson reports malformed JSON as invalid", async () => {
    routeFetch([[/example/, () => new Response("<html>oops</html>", { status: 200 })]]);
    const error = (await rejection(fetchJson("https://example.org/x"))) as SourceError;
    expect(error.code).toBe("invalid");
  });
});

// ─────────────────────────────────── arXiv ───────────────────────────────────

describe("arXiv URLs", () => {
  it("builds abs and pdf links, with optional version", () => {
    expect(arxivPdfUrl("2402.02750")).toBe("https://arxiv.org/pdf/2402.02750");
    expect(arxivPdfUrl("2402.02750", "v2")).toBe("https://arxiv.org/pdf/2402.02750v2");
    expect(arxivAbsUrl("hep-th/9901001", "1")).toBe("https://arxiv.org/abs/hep-th/9901001v1");
  });

  it("extracts venues from author comments", () => {
    expect(venueFromComment("ICML2024")).toBe("ICML 2024");
    expect(venueFromComment("Accepted to Findings of EMNLP 2026, forthcoming")).toBe(
      "Findings of EMNLP 2026",
    );
    expect(venueFromComment("Accepted by The Journal of Systems Architecture")).toBe(
      "Journal of Systems Architecture",
    );
    expect(venueFromComment("Accepted as a poster at ICLR 2024")).toBe("ICLR 2024");
    expect(venueFromComment("NeurIPS'23 camera ready")).toBe("NeurIPS 2023");
    expect(venueFromComment("8 pages, 6 figures, 6 tables")).toBeUndefined();
    expect(venueFromComment("Work in progress")).toBeUndefined();
  });
});

describe.skipIf(!hasFixtures("datacite-arxiv-2402.02750.json"))("mapDataCite (real KIVI record)", () => {
  it("maps title, authors, abstract, dates, categories, version and venue", () => {
    const meta = mapDataCite(fixture("datacite-arxiv-2402.02750.json"));
    expect(meta).not.toBeNull();
    expect(meta).toMatchObject({
      id: "arxiv:2402.02750",
      arxivId: "2402.02750",
      title: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
      published: "2024-02-05",
      year: 2024,
      version: "v2",
      venue: "ICML 2024",
      doi: "10.48550/arxiv.2402.02750",
      categories: ["cs.CL", "cs.LG", "cs.PF"],
      pdfUrl: "https://arxiv.org/pdf/2402.02750",
      landingUrl: "https://arxiv.org/abs/2402.02750",
      source: "datacite",
    });
    expect(meta?.authors).toHaveLength(8);
    expect(meta?.authors[0]).toBe("Zirui Liu");
    expect(meta?.abstract).toContain("2.6× less peak memory");
    expect(meta?.abstract).not.toMatch(/[$\\]/);
  });

  it("accepts the bare data object too", () => {
    const json = fixture("datacite-arxiv-2402.02750.json") as { data: unknown };
    expect(mapDataCite(json.data)?.id).toBe("arxiv:2402.02750");
  });

  it("rejects records without a title", () => {
    expect(mapDataCite({ data: { attributes: { titles: [] } } })).toBeNull();
    expect(mapDataCite("nope")).toBeNull();
    expect(mapDataCite(null)).toBeNull();
  });
});

describe.skipIf(!hasFixtures("datacite-arxiv-1706.03762.json"))(
  "mapDataCite (Attention Is All You Need)",
  () => {
    it("uses the v1 submission date and the latest version number", () => {
      const meta = mapDataCite(fixture("datacite-arxiv-1706.03762.json"));
      expect(meta).toMatchObject({
        title: "Attention Is All You Need",
        published: "2017-06-12",
        version: "v7",
      });
      expect(meta?.authors[0]).toBe("Ashish Vaswani");
    });
  },
);

describe.skipIf(!hasFixtures("datacite-search-kv-cache.json"))("mapDataCite (search list items)", () => {
  it("maps every list item to an arXiv paper", () => {
    const json = fixture("datacite-search-kv-cache.json") as { data: unknown[] };
    const metas = json.data.map(mapDataCite);
    expect(metas.every((m) => m !== null && m.arxivId && m.id === `arxiv:${m.arxivId}`)).toBe(true);
    const retrace = metas.find((m) => m?.arxivId === "2610.25005");
    expect(retrace?.venue).toBe("Findings of EMNLP 2026");
    const kwai = metas.find((m) => m?.arxivId === "2610.18654");
    expect(kwai?.authors[0]).toBe("OneRec Team");
    expect(kwai?.venue).toBe("arXiv");
  });
});

describe("withArxivLinks", () => {
  const base: PaperMeta = {
    id: "doi:10.1/x",
    title: "T",
    authors: [],
    source: "crossref",
    landingUrl: "https://aclanthology.org/N19-1423",
  };
  it("points the PDF at arXiv and keeps a publisher landing page", () => {
    const linked = withArxivLinks(base, "1810.04805");
    expect(linked).toMatchObject({
      id: "arxiv:1810.04805",
      arxivId: "1810.04805",
      pdfUrl: "https://arxiv.org/pdf/1810.04805",
      landingUrl: "https://aclanthology.org/N19-1423",
    });
  });
  it("pins the requested version", () => {
    const linked = withArxivLinks({ ...base, landingUrl: undefined, version: "v3" }, "1810.04805", "v1");
    expect(linked).toMatchObject({
      version: "v1",
      pdfUrl: "https://arxiv.org/pdf/1810.04805v1",
      landingUrl: "https://arxiv.org/abs/1810.04805v1",
    });
  });
});

describe.skipIf(!hasFixtures("datacite-arxiv-2402.02750.json", "openalex-doi-arxiv-2402.02750.json"))(
  "fetchArxivMeta",
  () => {
    it("reads DataCite and pins the requested version", async () => {
      const spy = routeFetch([
        [
          /api\.datacite\.org\/dois\/10\.48550\/arxiv\.2402\.02750$/,
          () => jsonResponse(fixture("datacite-arxiv-2402.02750.json")),
        ],
      ]);
      const meta = await fetchArxivMeta("2402.02750", "v1");
      expect(meta).toMatchObject({
        id: "arxiv:2402.02750",
        source: "datacite",
        version: "v1",
        pdfUrl: "https://arxiv.org/pdf/2402.02750v1",
      });
      expect(spy).toHaveBeenCalledTimes(1);
    });

    it("falls back to OpenAlex when DataCite has no record", async () => {
      routeFetch([
        [
          /api\.openalex\.org\/works\/doi:10\.48550\/arxiv\.2402\.02750/,
          () => jsonResponse(fixture("openalex-doi-arxiv-2402.02750.json")),
        ],
      ]);
      const meta = await fetchArxivMeta("arXiv:2402.02750v2");
      expect(meta).toMatchObject({
        id: "arxiv:2402.02750",
        source: "openalex",
        version: "v2",
        pdfUrl: "https://arxiv.org/pdf/2402.02750v2",
        landingUrl: "https://arxiv.org/abs/2402.02750v2",
      });
      expect(meta.abstract).toContain("2.6×");
    });

    it("reports not-found when neither index knows the paper", async () => {
      routeFetch([]);
      const error = (await rejection(fetchArxivMeta("2402.99999"))) as SourceError;
      expect(error.code).toBe("not-found");
    });

    it("prefers an outage over not-found when explaining a failure", async () => {
      routeFetch([[/datacite/, () => new Response("", { status: 503, headers: { "retry-after": "0" } })]]);
      const error = (await rejection(fetchArxivMeta("2402.99999"))) as SourceError;
      expect(error.code).toBe("network");
    });

    it("rejects malformed identifiers without a request", async () => {
      const spy = routeFetch([]);
      expect(((await rejection(fetchArxivMeta("banana"))) as SourceError).code).toBe("invalid");
      expect(spy).not.toHaveBeenCalled();
    });
  },
);

// ────────────────────────────────── OpenAlex ──────────────────────────────────

describe("reconstructAbstract", () => {
  it("orders words by position, including repeats", () => {
    expect(reconstructAbstract({ b: [1], a: [0], c: [2, 4], d: [3] })).toBe("a b c d c");
  });
  it("tolerates gaps and empty input", () => {
    expect(reconstructAbstract({ x: [10], y: [2] })).toBe("y x");
    expect(reconstructAbstract({})).toBeUndefined();
    expect(reconstructAbstract(null)).toBeUndefined();
    expect(reconstructAbstract(undefined)).toBeUndefined();
  });
});

describe.skipIf(!hasFixtures("openalex-doi-arxiv-2402.02750.json"))(
  "mapOpenAlexWork (arXiv DOI record)",
  () => {
    it("detects the arXiv ID and rebuilds the abstract", () => {
      const meta = mapOpenAlexWork(fixture("openalex-doi-arxiv-2402.02750.json"));
      expect(meta).toMatchObject({
        id: "arxiv:2402.02750",
        arxivId: "2402.02750",
        published: "2024-02-05",
        year: 2024,
        citationCount: 0,
        venue: "arXiv",
        source: "openalex",
      });
      expect(meta?.authors[0]).toBe("Zirui Liu");
      expect(meta?.abstract?.startsWith("Efficiently serving large language models")).toBe(true);
      expect(meta?.abstract).toContain("2.6× less peak memory");
    });
  },
);

describe.skipIf(!hasFixtures("openalex-search-kv-cache.json"))("mapOpenAlexWork (search results)", () => {
  const results = () =>
    (fixture("openalex-search-kv-cache.json") as { results: unknown[] }).results.map(mapOpenAlexWork);

  it("finds an arXiv ID for every result hosted on arXiv", () => {
    const metas = results();
    expect(metas.every((m) => m?.arxivId)).toBe(true);
  });

  it("keeps the published DOI and venue of a conference version", () => {
    const pitfalls = results().find((m) => m?.arxivId === "2510.00231");
    expect(pitfalls).toMatchObject({
      id: "arxiv:2510.00231",
      doi: "10.18653/v1/2026.acl-long.1926",
      venue: "Annual Meeting of the Association for Computational Linguistics (ACL)",
    });
  });

  it("repairs dates that predate the arXiv submission", () => {
    // OpenAlex dates PyramidInfer (arXiv 2405.12532) to 2024-01-01, before it existed.
    const pyramid = results().find((m) => m?.arxivId === "2405.12532");
    expect(pyramid?.published).toBe("2024-05");
    expect(pyramid?.citationCount).toBe(30);
  });
});

describe.skipIf(!hasFixtures("openalex-doi-bert.json", "openalex-doi-nature.json"))(
  "mapOpenAlexWork (publisher DOIs)",
  () => {
    it("maps BERT without inventing an arXiv ID", () => {
      const meta = mapOpenAlexWork(fixture("openalex-doi-bert.json"));
      expect(meta).toMatchObject({ id: "doi:10.18653/v1/n19-1423", citationCount: 48269, year: 2019 });
      expect(meta?.arxivId).toBeUndefined();
      expect(meta?.authors).toEqual(["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"]);
    });

    it("uses the best open-access PDF", () => {
      const meta = mapOpenAlexWork(fixture("openalex-doi-nature.json"));
      expect(meta?.pdfUrl).toBe("https://www.nature.com/articles/s41586-021-03819-2.pdf");
      expect(meta?.venue).toBe("Nature");
    });
  },
);

describe("mergeMeta", () => {
  const primary: PaperMeta = {
    id: "arxiv:1",
    title: "A",
    authors: ["X"],
    venue: "arXiv",
    source: "datacite",
    doi: "10.48550/arxiv.1",
  };
  const secondary: PaperMeta = {
    id: "arxiv:1",
    title: "A (other)",
    authors: ["Y"],
    venue: "ICML",
    citationCount: 12,
    doi: "10.5555/wrong",
    abstract: "Text",
    source: "openalex",
  };
  it("fills gaps without overwriting primary facts", () => {
    const merged = mergeMeta(primary, secondary);
    expect(merged).toMatchObject({
      title: "A",
      authors: ["X"],
      doi: "10.48550/arxiv.1",
      abstract: "Text",
      citationCount: 12,
      venue: "ICML",
      source: "datacite",
    });
  });
  it("returns the primary unchanged without a secondary", () => {
    expect(mergeMeta(primary, null)).toBe(primary);
  });
});

// ─────────────────────────────────── DOIs ───────────────────────────────────

describe("stripJats", () => {
  it("drops headings and citation superscripts but keeps exponents", () => {
    const jats =
      "<jats:title>Abstract</jats:title><jats:p>Through effort\n <jats:sup>1–4</jats:sup>\n , we grew 10<jats:sup>6</jats:sup> cells.</jats:p>";
    expect(stripJats(jats)).toBe("Through effort, we grew 10^6 cells.");
  });
});

describe.skipIf(!hasFixtures("crossref-nature.json", "crossref-bert.json"))("mapCrossref", () => {
  it("cleans the JATS abstract of the AlphaFold paper", () => {
    const meta = mapCrossref(fixture("crossref-nature.json"));
    expect(meta).toMatchObject({
      id: "doi:10.1038/s41586-021-03819-2",
      title: "Highly accurate protein structure prediction with AlphaFold",
      published: "2021-07-15",
      venue: "Nature",
      source: "crossref",
    });
    expect(meta?.authors[0]).toBe("John Jumper");
    expect(meta?.abstract).toMatch(/^Proteins are essential to life/);
    expect(meta?.abstract).toContain("experimental effort, the structures of around 100,000 unique proteins");
    expect(meta?.abstract).not.toMatch(/<|jats|1–4/);
  });

  it("maps a proceedings paper without an abstract", () => {
    const meta = mapCrossref(fixture("crossref-bert.json"));
    expect(meta?.abstract).toBeUndefined();
    expect(meta?.year).toBe(2019);
    expect(meta?.venue).toMatch(/^Proceedings of the 2019 Conference of the North American Chapter/);
    expect(meta?.landingUrl).toBe("https://aclanthology.org/N19-1423");
  });
});

const BERT_ABSTRACT =
  "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder " +
  "Representations from Transformers. Unlike recent language representation models, BERT is designed to " +
  "pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and " +
  "right context in all layers.";

/** DataCite record for arXiv 1810.04805, in the shape of the real KIVI record. */
function bertDataCite(): unknown {
  const record = fixture("datacite-arxiv-2402.02750.json") as {
    data: { attributes: Record<string, unknown> };
  };
  const attributes = {
    ...record.data.attributes,
    doi: "10.48550/arxiv.1810.04805",
    identifiers: [{ identifier: "1810.04805", identifierType: "arXiv" }],
    alternateIdentifiers: [{ alternateIdentifierType: "arXiv", alternateIdentifier: "1810.04805" }],
    titles: [{ title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" }],
    creators: [{ name: "Devlin, Jacob", givenName: "Jacob", familyName: "Devlin" }],
    descriptions: [{ description: BERT_ABSTRACT, descriptionType: "Abstract" }],
    subjects: [{ subject: "Computation and Language (cs.CL)", subjectScheme: "arXiv" }],
    dates: [{ date: "2018-10-11T00:50:01Z", dateType: "Submitted", dateInformation: "v1" }],
    publicationYear: 2018,
  };
  return { data: { ...record.data, id: "10.48550/arxiv.1810.04805", attributes } };
}

describe.skipIf(
  !hasFixtures(
    "crossref-bert.json",
    "openalex-doi-bert.json",
    "s2-doi-bert.json",
    "datacite-arxiv-2402.02750.json",
  ),
)("resolveDoi", () => {
  it("drops the citation string OpenAlex stores as BERT's abstract", () => {
    expect(mapOpenAlexWork(fixture("openalex-doi-bert.json"))?.abstract).toBeUndefined();
  });

  it("merges Crossref, OpenAlex and Semantic Scholar and prefers the arXiv PDF", async () => {
    const spy = routeFetch([
      [
        /api\.crossref\.org\/works\/10\.18653\/v1\/n19-1423$/,
        () => jsonResponse(fixture("crossref-bert.json")),
      ],
      [
        /api\.openalex\.org\/works\/doi:10\.18653\/v1\/n19-1423$/,
        () => jsonResponse(fixture("openalex-doi-bert.json")),
      ],
      [
        /api\.semanticscholar\.org\/graph\/v1\/paper\/DOI:10\.18653\/v1\/n19-1423\?/,
        () => jsonResponse(fixture("s2-doi-bert.json")),
      ],
      [/api\.datacite\.org\/dois\/10\.48550\/arxiv\.1810\.04805$/, () => jsonResponse(bertDataCite())],
    ]);
    const meta = await resolveDoi("https://doi.org/10.18653/v1/N19-1423");
    expect(meta).toMatchObject({
      id: "arxiv:1810.04805",
      arxivId: "1810.04805",
      doi: "10.18653/v1/n19-1423",
      title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      citationCount: 48269,
      pdfUrl: "https://arxiv.org/pdf/1810.04805",
      landingUrl: "https://aclanthology.org/N19-1423",
      source: "crossref",
    });
    // Neither Crossref nor OpenAlex has a usable abstract, so it comes from the arXiv record.
    expect(meta.abstract).toBe(BERT_ABSTRACT);
    expect(meta.categories).toEqual(["cs.CL"]);
    expect(spy).toHaveBeenCalledTimes(4);
  });

  it("ignores a rate-limited Semantic Scholar", async () => {
    routeFetch([
      [/crossref/, () => jsonResponse(fixture("crossref-bert.json"))],
      [/openalex/, () => jsonResponse(fixture("openalex-doi-bert.json"))],
      [/semanticscholar/, () => new Response("", { status: 429 })],
    ]);
    const meta = await resolveDoi("10.18653/v1/N19-1423");
    expect(meta.id).toBe("doi:10.18653/v1/n19-1423");
    expect(meta.arxivId).toBeUndefined();
    expect(meta.citationCount).toBe(48269);
  });

  it("finds the arXiv preprint by exact title when no index links it", async () => {
    const preprint = { data: [{ ...(bertDataCite() as { data: object }).data }] };
    routeFetch([
      [/crossref/, () => jsonResponse(fixture("crossref-bert.json"))],
      [/openalex/, () => jsonResponse(fixture("openalex-doi-bert.json"))],
      [/semanticscholar/, () => new Response("", { status: 429 })],
      [/api\.datacite\.org\/dois\?/, () => jsonResponse(preprint)],
      [/api\.datacite\.org\/dois\/10\.48550\/arxiv\.1810\.04805$/, () => jsonResponse(bertDataCite())],
    ]);
    const meta = await resolveDoi("10.18653/v1/N19-1423");
    expect(meta).toMatchObject({ arxivId: "1810.04805", pdfUrl: "https://arxiv.org/pdf/1810.04805" });
  });

  it("ignores title search hits that are not exact matches", async () => {
    const other = structuredClone(bertDataCite()) as { data: { attributes: { titles: { title: string }[] } } };
    other.data.attributes.titles = [{ title: "BERT Rediscovers the Classical NLP Pipeline" }];
    routeFetch([[/datacite/, () => jsonResponse({ data: [other.data] })]]);
    expect(await findArxivByTitle("BERT: Pre-training of Deep Bidirectional Transformers")).toBeUndefined();
  });
});

describe.skipIf(!hasFixtures("crossref-nature.json", "openalex-doi-nature.json"))(
  "resolveDoi (open-access journal)",
  () => {
    it("uses the OpenAlex open-access PDF when there is no arXiv copy", async () => {
      routeFetch([
        [/crossref/, () => jsonResponse(fixture("crossref-nature.json"))],
        [/openalex/, () => jsonResponse(fixture("openalex-doi-nature.json"))],
      ]);
      const meta = await resolveDoi("doi:10.1038/s41586-021-03819-2");
      expect(meta).toMatchObject({
        id: "doi:10.1038/s41586-021-03819-2",
        pdfUrl: "https://www.nature.com/articles/s41586-021-03819-2.pdf",
        citationCount: 43495,
        venue: "Nature",
      });
    });
  },
);

describe("resolveDoi routing and failures", () => {
  it.skipIf(!hasFixtures("datacite-arxiv-2402.02750.json"))("sends arXiv DOIs to DataCite", async () => {
    const spy = routeFetch([[/datacite/, () => jsonResponse(fixture("datacite-arxiv-2402.02750.json"))]]);
    const meta = await resolveDoi("10.48550/arXiv.2402.02750");
    expect(meta.id).toBe("arxiv:2402.02750");
    expect(spy).toHaveBeenCalledTimes(1);
  });

  it("reports not-found when no index knows the DOI", async () => {
    routeFetch([]);
    const error = (await rejection(resolveDoi("10.9999/does-not-exist"))) as SourceError;
    expect(error.code).toBe("not-found");
  });

  it("rejects strings that are not DOIs", async () => {
    expect(((await rejection(resolveDoi("hello"))) as SourceError).code).toBe("invalid");
  });
});

// ─────────────────────────────────── search ───────────────────────────────────

describe.skipIf(!hasFixtures("datacite-search-kv-cache.json", "openalex-search-kv-cache.json"))(
  "searchArxiv",
  () => {
    const dataCite = () => jsonResponse(fixture("datacite-search-kv-cache.json"));
    const openAlex = () => jsonResponse(fixture("openalex-search-kv-cache.json"));

    it("queries both indexes and merges duplicates by arXiv ID", async () => {
      const spy = routeFetch([
        [/api\.datacite\.org\/dois\?/, dataCite],
        [/api\.openalex\.org\/works\?/, openAlex],
      ]);
      const metas = await searchArxiv("kv cache compression", { limit: 25 });
      const ids = metas.map((m) => m.arxivId);
      expect(new Set(ids).size).toBe(ids.length);
      expect(metas.length).toBeGreaterThan(25);
      expect(metas.every((m) => m.pdfUrl?.startsWith("https://arxiv.org/pdf/"))).toBe(true);

      const urls = spy.mock.calls.map(([input]) => String(input));
      const dataCiteUrl = new URL(urls.find((u) => u.includes("datacite")) ?? "");
      expect(dataCiteUrl.searchParams.get("client-id")).toBe("arxiv.content");
      expect(dataCiteUrl.searchParams.get("query")).toBe("titles.title:(kv cache compression)");
      expect(dataCiteUrl.searchParams.get("page[size]")).toBe("25");
      const openAlexUrl = new URL(urls.find((u) => u.includes("openalex")) ?? "");
      expect(openAlexUrl.searchParams.get("filter")).toBe(
        "fulltext.search:kv cache compression,locations.source.id:S4306400194",
      );
    });

    it("keeps DataCite metadata primary and takes citations from OpenAlex", async () => {
      routeFetch([
        [
          /datacite/,
          () =>
            jsonResponse({
              data: [(fixture("datacite-search-kv-cache.json") as { data: unknown[] }).data[0]],
            }),
        ],
        [
          /openalex/,
          () => {
            const result = (
              fixture("openalex-search-kv-cache.json") as { results: Record<string, unknown>[] }
            ).results[0];
            // Same paper as the DataCite item, as OpenAlex would describe it.
            return jsonResponse({
              results: [
                {
                  ...result,
                  doi: "https://doi.org/10.48550/arxiv.2610.29733",
                  locations: [{ landing_page_url: "https://arxiv.org/abs/2610.29733" }],
                  cited_by_count: 7,
                },
              ],
            });
          },
        ],
      ]);
      const [meta] = await searchArxiv("kv cache compression");
      expect(meta).toMatchObject({ arxivId: "2610.29733", source: "datacite", citationCount: 7 });
      expect(meta.title).toMatch(/^SAKURA/);
    });

    it("tolerates one index failing", async () => {
      routeFetch([
        [/datacite/, () => new Response("", { status: 500, headers: { "retry-after": "0" } })],
        [/openalex/, openAlex],
      ]);
      const metas = await searchArxiv("kv cache compression");
      expect(metas.length).toBe(25);
      expect(metas.every((m) => m.source === "openalex")).toBe(true);
    });

    it("fails only when both indexes fail", async () => {
      routeFetch([[/./, () => new Response("", { status: 400 })]]);
      const error = (await rejection(searchArxiv("kv cache compression"))) as SourceError;
      expect(error).toBeInstanceOf(SourceError);
    });

    it.skipIf(!hasFixtures("datacite-search-empty.json"))(
      "returns an empty list for nonsense queries",
      async () => {
        routeFetch([
          [/datacite/, () => jsonResponse(fixture("datacite-search-empty.json"))],
          [/openalex/, () => jsonResponse({ meta: { count: 0 }, results: [] })],
        ]);
        expect(await searchArxiv("qwzxv flurbination zzq")).toEqual([]);
      },
    );
  },
);

describe("search cancellation", () => {
  it("rejects with the abort reason instead of a partial result", async () => {
    hangingFetch();
    const controller = new AbortController();
    const pending = searchArxiv("kv cache compression", { signal: controller.signal });
    controller.abort();
    expect(isAbortError(await rejection(pending))).toBe(true);
  });
});

describe("broadenQuery", () => {
  it("first strips conversational filler", () => {
    expect(broadenQuery("recent work on kv cache compression", 1)).toBe("kv cache compression");
    expect(broadenQuery("Papers about speculative decoding", 1)).toBe("speculative decoding");
  });

  it("then keeps the most specific terms in their original order", () => {
    expect(broadenQuery("recent work on kv cache compression", 2)).toBe("cache compression");
    const long = "efficient methods for compressing the key value cache in large language models";
    const first = broadenQuery(long, 1)?.split(" ") ?? [];
    const second = broadenQuery(long, 2)?.split(" ") ?? [];
    expect(first).not.toContain("the");
    expect(second.length).toBeGreaterThanOrEqual(2);
    expect(second.length).toBeLessThanOrEqual(3);
    // Kept terms are a subsequence of the filler-free query, led by its most specific word.
    expect(second.every((term) => first.includes(term))).toBe(true);
    expect(second).toContain("compressing");
    expect(second).not.toContain("efficient");
  });

  it("skips straight to term reduction when there is no filler", () => {
    expect(broadenQuery("kv cache compression", 1)).toBe("cache compression");
    expect(broadenQuery("kv cache compression", 2)).toBeNull();
  });

  it("gives up when nothing can be broadened or after two attempts", () => {
    expect(broadenQuery("transformers", 1)).toBeNull();
    expect(broadenQuery("kv cache compression", 3)).toBeNull();
    expect(broadenQuery("kv cache compression", 0)).toBeNull();
  });
});

// ─────────────────────────────────── ranking ───────────────────────────────────

function paper(
  id: string,
  title: string,
  abstract: string,
  published: string,
  citationCount?: number,
): PaperMeta {
  return {
    id,
    title,
    abstract,
    published,
    year: Number(published.slice(0, 4)),
    citationCount,
    authors: ["A. Author"],
    source: "datacite",
  };
}

const NOW = new Date("2026-10-22T00:00:00Z");
const CORPUS: PaperMeta[] = [
  paper(
    "arxiv:1",
    "KV cache compression with quantization",
    "We compress the key value cache of language models by quantization.",
    "2024-02-05",
    120,
  ),
  paper(
    "arxiv:2",
    "Image segmentation with vision transformers",
    "A segmentation model for medical images.",
    "2025-06-01",
    40,
  ),
  paper(
    "arxiv:3",
    "Eviction policies for KV cache compression",
    "Token eviction keeps the cache small during long context inference.",
    "2025-09-01",
    5,
  ),
  paper(
    "arxiv:4",
    "Protein structure prediction",
    "Folding proteins with deep learning.",
    "2021-07-15",
    40000,
  ),
];

/** Deterministic bag-of-words embedding (hashed, L2-normalised) standing in for MiniLM. */
const fakeEmbed: EmbedFn = async (texts) =>
  texts.map((text) => {
    const v = new Float32Array(64);
    for (const word of text.toLowerCase().match(/[a-z]+/g) ?? []) {
      let h = 0;
      for (const c of word) h = (h * 31 + c.charCodeAt(0)) >>> 0;
      v[h % 64] += 1;
    }
    const norm = Math.hypot(...v) || 1;
    return v.map((x) => x / norm);
  });

describe("rankCandidates", () => {
  it("puts keyword matches first and explains why (lexical-only mode)", async () => {
    const ranked = await rankCandidates("kv cache compression", CORPUS, null, { now: NOW });
    expect(ranked.map((c) => c.rank)).toEqual([1, 2, 3, 4]);
    expect(
      ranked
        .slice(0, 2)
        .map((c) => c.meta.id)
        .sort(),
    ).toEqual(["arxiv:1", "arxiv:3"]);
    for (const c of ranked) {
      for (const value of Object.values(c.scores)) expect(value).toBeGreaterThanOrEqual(0);
      for (const value of Object.values(c.scores)) expect(value).toBeLessThanOrEqual(1);
      expect(c.scores.semantic).toBe(0);
      const w = RANK_WEIGHTS.lexicalOnly;
      expect(c.scores.final).toBeCloseTo(
        w.lexical * c.scores.lexical + w.recency * c.scores.recency + w.impact * c.scores.impact,
        10,
      );
      expect(c.why.length).toBeGreaterThan(0);
    }
    expect(ranked[0].why).toMatch(/title matches every query term/i);
  });

  it("is deterministic and independent of input order", async () => {
    const a = await rankCandidates("kv cache compression", CORPUS, fakeEmbed, { now: NOW });
    const b = await rankCandidates("kv cache compression", [...CORPUS].reverse(), fakeEmbed, { now: NOW });
    expect(b.map((c) => c.meta.id)).toEqual(a.map((c) => c.meta.id));
    expect(b.map((c) => c.scores)).toEqual(a.map((c) => c.scores));
  });

  it("blends in semantic similarity when embeddings are available", async () => {
    const ranked = await rankCandidates("kv cache compression", CORPUS, fakeEmbed, { now: NOW });
    const top = ranked[0];
    const w = RANK_WEIGHTS.withEmbeddings;
    expect(top.scores.semantic).toBeGreaterThan(0);
    expect(top.scores.final).toBeCloseTo(
      w.lexical * top.scores.lexical +
        w.semantic * top.scores.semantic +
        w.recency * top.scores.recency +
        w.impact * top.scores.impact,
      10,
    );
    const protein = ranked.find((c) => c.meta.id === "arxiv:4");
    expect(protein?.scores.semantic).toBeLessThan(top.scores.semantic);
  });

  it("falls back to lexical-only weights when the embedder fails", async () => {
    const broken: EmbedFn = async () => {
      throw new Error("model unavailable");
    };
    const ranked = await rankCandidates("kv cache compression", CORPUS, broken, { now: NOW });
    expect(ranked).toHaveLength(4);
    expect(ranked.every((c) => c.scores.semantic === 0)).toBe(true);
  });

  it("diversifies near-duplicates with MMR", async () => {
    const duplicate = { ...CORPUS[0], id: "arxiv:1-copy" };
    const ranked = await rankCandidates(
      "kv cache compression",
      [CORPUS[0], duplicate, CORPUS[2], CORPUS[1]],
      null,
      { now: NOW },
    );
    const order = ranked.map((c) => c.meta.id);
    // The distinct eviction paper is promoted above the exact copy of the first result.
    expect(order.indexOf("arxiv:3")).toBeLessThan(
      Math.max(order.indexOf("arxiv:1"), order.indexOf("arxiv:1-copy")),
    );
  });

  it("honours limit, drops duplicate ids and handles empty input", async () => {
    const ranked = await rankCandidates("kv cache compression", [...CORPUS, CORPUS[0]], null, {
      now: NOW,
      limit: 2,
    });
    expect(ranked).toHaveLength(2);
    expect(await rankCandidates("anything", [], null)).toEqual([]);
  });

  it("decays recency with a three-year half-life", () => {
    expect(recencyScore(paper("x", "t", "a", "2023-10-22", 0), NOW)).toBeCloseTo(0.5, 2);
    expect(recencyScore(paper("x", "t", "a", "2026-10-22", 0), NOW)).toBeCloseTo(1, 5);
    expect(recencyScore({ id: "x", title: "t", authors: [], source: "upload" }, NOW)).toBe(0);
  });
});

// ──────────────────────────────────── PDFs ────────────────────────────────────

const FAKE_PDF = new TextEncoder().encode("%PDF-1.7\n" + "x".repeat(5000) + "\n%%EOF");

describe("downloadPdf", () => {
  it("streams the body and reports monotonic progress ending at 1", async () => {
    routeFetch([
      [
        /paper\.pdf/,
        () =>
          new Response(streamOf(chunked(FAKE_PDF, 512)), {
            headers: { "content-length": String(FAKE_PDF.length) },
          }),
      ],
    ]);
    const progress: number[] = [];
    const buffer = await downloadPdf("https://example.org/paper.pdf", {
      onProgress: (f) => progress.push(f),
    });
    expect(new Uint8Array(buffer)).toEqual(FAKE_PDF);
    expect(progress[0]).toBe(0);
    expect(progress.at(-1)).toBe(1);
    expect(progress.length).toBeGreaterThan(3);
    for (let i = 1; i < progress.length; i++) expect(progress[i]).toBeGreaterThan(progress[i - 1]);
  });

  it.skipIf(!existsSync(PDF_FIXTURE))("downloads a real arXiv PDF byte for byte", async () => {
    const bytes = new Uint8Array(readFileSync(PDF_FIXTURE));
    routeFetch([
      [
        /arxiv/,
        () =>
          new Response(streamOf(chunked(bytes, 64 * 1024)), {
            headers: { "content-length": String(bytes.length) },
          }),
      ],
    ]);
    const buffer = await downloadPdf("https://arxiv.org/pdf/2402.02750");
    expect(buffer.byteLength).toBe(bytes.length);
    expect(hasPdfSignature(new Uint8Array(buffer))).toBe(true);
  });

  it("reports only start and end when the size is unknown", async () => {
    routeFetch([[/paper/, () => new Response(streamOf(chunked(FAKE_PDF, 512)))]]);
    const progress: number[] = [];
    await downloadPdf("https://example.org/paper", { onProgress: (f) => progress.push(f) });
    expect(progress).toEqual([0, 1]);
  });

  it("rejects an HTML page served instead of a PDF", async () => {
    routeFetch([[/./, () => new Response("<!DOCTYPE html><html><body>Sign in</body></html>")]]);
    const error = (await rejection(downloadPdf("https://example.org/paper.pdf"))) as SourceError;
    expect(error.code).toBe("invalid");
    expect(error.message).toMatch(/web page/);
  });

  it("enforces maxBytes from Content-Length and while streaming", async () => {
    routeFetch([
      [/declared/, () => new Response(FAKE_PDF, { headers: { "content-length": "999999999" } })],
      [/streamed/, () => new Response(streamOf(chunked(FAKE_PDF, 512)))],
    ]);
    expect(((await rejection(downloadPdf("https://example.org/declared.pdf"))) as SourceError).code).toBe(
      "invalid",
    );
    const error = (await rejection(
      downloadPdf("https://example.org/streamed.pdf", { maxBytes: 1024 }),
    )) as SourceError;
    expect(error.code).toBe("invalid");
    expect(error.message).toMatch(/larger than/);
  });

  it("turns a failed fetch into a cors error with upload advice", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Promise.reject(new TypeError("Failed to fetch"))),
    );
    const error = (await rejection(downloadPdf("https://publisher.example/paper.pdf"))) as SourceError;
    expect(error.code).toBe("cors");
    expect(error.message).toMatch(/upload/i);
  });

  it("passes through not-found", async () => {
    routeFetch([]);
    expect(((await rejection(downloadPdf("https://example.org/missing.pdf"))) as SourceError).code).toBe(
      "not-found",
    );
  });

  it("can be cancelled", async () => {
    hangingFetch();
    const controller = new AbortController();
    const pending = downloadPdf("https://example.org/paper.pdf", { signal: controller.signal });
    controller.abort();
    expect(isAbortError(await rejection(pending))).toBe(true);
  });
});

// ──────────────────────────────────── hash ────────────────────────────────────

describe("sha256Hex", () => {
  it("matches the FIPS 180-2 test vectors", async () => {
    const abc = new TextEncoder().encode("abc");
    expect(await sha256Hex(abc.buffer.slice(0, abc.byteLength))).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    expect(await sha256Hex(new ArrayBuffer(0))).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
  });
});

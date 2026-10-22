/**
 * The paper agent: one explicit state graph that turns any input (arXiv id,
 * DOI, PDF link or uploaded file) into a full analysis.
 *
 *   understand -> resolve ─┬─> fetch ─┬─> parse ─┬─> segment -> zone -> chunk -> embed -> analyze
 *                          │          │          │      ^
 *                          └──────────┴──────────┴─> fallback (abstract-only)
 *
 * Shared state flows through every node; each node returns only what it adds.
 * Conditional edges decide the route at runtime:
 *  - no PDF to read (no link, no file)          -> fallback
 *  - download blocked or failed (SourceError)   -> fallback
 *  - PDF without a usable text layer            -> fallback
 * The fallback continues from the metadata abstract, so the user still gets a
 * (clearly labelled) analysis; only when there is no abstract either does the
 * run fail, with a message that says what to do next.
 *
 * Model failures never stop a run: zoning falls back to cue phrases and
 * section priors, retrieval falls back to BM25 keyword search.
 */
import { CompiledGraph, END, START, StateGraph, type NodeContext } from "@/agent/graph";
import { analyzePaper } from "@/analysis";
import { MLUnavailableError, type MLApi } from "@/ml/client";
import { segmentAbstract, segmentDocument } from "@/nlp/segment";
import { zoneSentences } from "@/nlp/zoning";
import { parsePdf } from "@/parsing";
import { arxivAbsUrl, arxivPdfUrl, fetchArxivMeta } from "@/sources/arxiv";
import { resolveDoi } from "@/sources/doi";
import { sha256Hex } from "@/sources/hash";
import { isAbortError, SourceError } from "@/sources/http";
import { downloadPdf } from "@/sources/pdf";
import { chunkSentences, embeddingText } from "@/retrieval/chunker";
import type {
  Chunk,
  EmbedFn,
  PaperAnalysis,
  PaperMeta,
  ParsedDocument,
  ParsedInput,
  Sentence,
  Zone,
  ZonedSentence,
} from "@/types";

export interface PaperPipelineState {
  input: ParsedInput;
  file?: File;
  paperId?: string;
  meta?: PaperMeta;
  pdf?: ArrayBuffer;
  doc?: ParsedDocument | null;
  sentences?: ZonedSentence[];
  chunks?: Chunk[];
  embeddings?: Float32Array[] | null;
  analysis?: PaperAnalysis;
  mode: "full" | "abstract-only";
  warnings: string[];
  fetchError?: string;
}

export const PAPER_NODES: { name: string; label: string; description: string }[] = [
  { name: "understand", label: "Understand", description: "Classify the input: arXiv id, DOI, PDF link or uploaded file" },
  { name: "resolve", label: "Resolve", description: "Look up title, authors, abstract and an open-access PDF link" },
  { name: "fetch", label: "Fetch PDF", description: "Download the PDF with progress, verify it is really a PDF" },
  { name: "parse", label: "Parse", description: "Rebuild lines, columns, sections, references and captions from the PDF" },
  { name: "fallback", label: "Abstract fallback", description: "No readable full text: continue from the abstract only" },
  { name: "segment", label: "Segment", description: "Split the text into clean sentences with section and page" },
  { name: "zone", label: "Zone", description: "Label each sentence's rhetorical role (problem, method, result, limitation…)" },
  { name: "chunk", label: "Chunk", description: "Group sentences into ~170-word passages that never cross sections" },
  { name: "embed", label: "Embed", description: "Encode passages with MiniLM for semantic search" },
  { name: "analyze", label: "Analyze", description: "Briefing, critique, rigor profile and visual analytics" },
];

/** Nodes on the normal full-text path; used for coarse progress. */
export const PAPER_PATH_LENGTH = 9;

/** Below this many characters of text the parse is not worth analysing (about a quarter page). */
const MIN_FULL_TEXT_CHARS = 1500;
/** Chunks per embedding request: big enough to amortise messaging, small enough for steady progress. */
const EMBED_BATCH = 32;

export const SEMANTIC_SEARCH_WARNING = "Semantic search unavailable, using keyword retrieval.";
const ZONING_WARNING = "Embedding model unavailable: sentence roles come from cue phrases and section position only.";

// ─────────────────────────────── Identities ────────────────────────────────

async function shortHash(bytes: ArrayBuffer): Promise<string> {
  return (await sha256Hex(bytes)).slice(0, 12);
}

/** Content-addressed id, so re-uploading the same file reuses its analysis. */
export async function uploadId(bytes: ArrayBuffer): Promise<string> {
  return `upload:${await shortHash(bytes)}`;
}

/**
 * Stable record id for a typed input, known before any network call.
 * Uploads need their bytes (see `uploadId`); topics are not papers.
 */
export async function paperIdFor(input: ParsedInput): Promise<string | null> {
  switch (input.kind) {
    case "arxiv":
      return input.arxivId ? `arxiv:${input.arxivId}` : null;
    case "doi":
      return input.doi ? `doi:${input.doi}` : null;
    case "pdf-url":
    case "url":
      return input.url ? `url:${await shortHash(new TextEncoder().encode(input.url).buffer as ArrayBuffer)}` : null;
    default:
      return null;
  }
}

/** "…/papers/attention.pdf?dl=1" -> "attention" */
export function titleFromFileName(name: string): string {
  const base = decodeURIComponent(name.split(/[?#]/)[0].split("/").filter(Boolean).pop() ?? name);
  const stem = base.replace(/\.pdf$/i, "").replace(/[_-]+/g, " ").trim();
  return stem || "Untitled PDF";
}

function describeInput(input: ParsedInput): string {
  switch (input.kind) {
    case "arxiv":
      return `arXiv paper ${input.arxivId}${input.version ?? ""}`;
    case "doi":
      return `DOI ${input.doi}`;
    case "pdf-url":
      return `PDF link ${input.url}`;
    case "url":
      return `Web link ${input.url} (will be read as a PDF)`;
    case "upload":
      return `Uploaded file ${input.fileName ?? "document.pdf"}`;
    case "topic":
      return `Topic "${input.query}"`;
  }
}

// ──────────────────────────────── Helpers ──────────────────────────────────

function requireMeta(state: PaperPipelineState): PaperMeta {
  if (!state.meta) throw new Error("Paper metadata is missing");
  return state.meta;
}

function requirePaperId(state: PaperPipelineState): string {
  return state.paperId ?? requireMeta(state).id;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Metadata only names the paper when it came from a scholarly index. */
function hasPlaceholderTitle(meta: PaperMeta): boolean {
  return meta.source === "upload" || meta.source === "pdf";
}

/** Why a parsed document cannot carry a full-text analysis, or null when it can. */
export function unusableReason(doc: ParsedDocument): string | null {
  const chars = doc.pages.reduce((sum, p) => sum + p.charCount, 0);
  if (doc.quality === "image-only") return "the PDF has no text layer (scanned or image-only)";
  if (chars < MIN_FULL_TEXT_CHARS) return "the PDF contains very little extractable text";
  if (doc.sections.length === 0) return "no section structure could be recovered from the PDF";
  return null;
}

const NO_ZONE_SCORES: Record<Zone, number> = {
  background: 0,
  problem: 0,
  contribution: 0,
  method: 0,
  result: 0,
  limitation: 0,
  future: 0,
  other: 0,
};

/** A sentence before zoning: role "other" with zero confidence. */
function unzoned(sentence: Sentence): ZonedSentence {
  return { ...sentence, zone: "other", confidence: 0, scores: { ...NO_ZONE_SCORES } };
}

/** Wraps an embedder so a node can tell afterwards whether the model actually worked. */
function trackFailures(embed: EmbedFn): { embed: EmbedFn; failed: () => boolean } {
  let failed = false;
  return {
    embed: async (texts) => {
      try {
        return await embed(texts);
      } catch (error) {
        failed = true;
        throw error;
      }
    },
    failed: () => failed,
  };
}

/** Logs download progress at 25% steps so the trace stays readable. */
function progressLogger(ctx: NodeContext): (fraction: number) => void {
  let nextStep = 0.25;
  return (fraction) => {
    if (fraction < nextStep) return;
    ctx.log(`Downloaded ${Math.round(fraction * 100)}%`);
    while (nextStep <= fraction) nextStep += 0.25;
  };
}

// ───────────────────────────────── Nodes ───────────────────────────────────

function understand(state: PaperPipelineState, ctx: NodeContext): Partial<PaperPipelineState> {
  const { input } = state;
  if (input.kind === "topic") {
    throw new Error("Topic queries are handled by the search agent, not the paper agent");
  }
  if (input.kind === "upload" && !state.file && !state.pdf) {
    throw new Error("No file was provided for the upload");
  }
  ctx.log(describeInput(input));
  return {};
}

async function resolveUpload(state: PaperPipelineState): Promise<Partial<PaperPipelineState>> {
  const pdf = state.pdf ?? (await state.file!.arrayBuffer());
  const id = state.paperId ?? (await uploadId(pdf));
  const name = state.file?.name ?? state.input.fileName ?? "document.pdf";
  return {
    pdf,
    paperId: id,
    meta: { id, title: titleFromFileName(name), authors: [], source: "upload" },
  };
}

async function resolveArxiv(state: PaperPipelineState, ctx: NodeContext): Promise<Partial<PaperPipelineState>> {
  const { arxivId, version } = state.input;
  if (!arxivId) throw new Error("The arXiv identifier is missing");
  try {
    return { meta: await fetchArxivMeta(arxivId, version, ctx.signal) };
  } catch (error) {
    // An unreachable or lagging index is not fatal: brand-new submissions reach
    // DataCite a little after arXiv, and the PDF itself carries title and abstract.
    // If the PDF is missing too, the fallback node reports the paper as not found.
    if (isAbortError(error) || !(error instanceof SourceError) || error.code === "invalid") {
      throw error;
    }
    ctx.log(`Metadata lookup failed (${error.code}); reading details from the PDF instead`);
    return {
      warnings: [...state.warnings, "Metadata services were unreachable; title and abstract were read from the PDF."],
      meta: {
        id: `arxiv:${arxivId}`,
        title: `arXiv:${arxivId}`,
        authors: [],
        arxivId,
        version,
        pdfUrl: arxivPdfUrl(arxivId, version),
        landingUrl: arxivAbsUrl(arxivId, version),
        source: "pdf",
      },
    };
  }
}

function resolveLink(state: PaperPipelineState, id: string): Partial<PaperPipelineState> {
  const url = state.input.url ?? state.input.raw;
  return { meta: { id, title: titleFromFileName(url), authors: [], pdfUrl: url, landingUrl: url, source: "pdf" } };
}

async function resolve(state: PaperPipelineState, ctx: NodeContext): Promise<Partial<PaperPipelineState>> {
  let update: Partial<PaperPipelineState>;
  if (state.meta) {
    ctx.log("Metadata already known from search; skipping lookup");
    update = {};
  } else if (state.input.kind === "upload") {
    update = await resolveUpload(state);
  } else if (state.input.kind === "arxiv") {
    update = await resolveArxiv(state, ctx);
  } else if (state.input.kind === "doi") {
    if (!state.input.doi) throw new Error("The DOI is missing");
    update = { meta: await resolveDoi(state.input.doi, ctx.signal) };
  } else {
    update = resolveLink(state, state.paperId ?? (await paperIdFor(state.input)) ?? state.input.raw);
  }

  const meta = update.meta ?? requireMeta(state);
  const paperId = update.paperId ?? state.paperId ?? meta.id;
  // The record id is the single key for chunks, evidence and storage.
  const resolved: PaperMeta = { ...meta, id: paperId };
  ctx.log(resolved.pdfUrl || state.file || update.pdf ? `Resolved "${resolved.title}"` : `Resolved "${resolved.title}" (no open PDF found)`);
  return { ...update, paperId, meta: resolved };
}

function routeAfterResolve(state: PaperPipelineState): string {
  return state.pdf || state.file || state.meta?.pdfUrl ? "fetch" : "fallback";
}

async function fetchPdf(state: PaperPipelineState, ctx: NodeContext): Promise<Partial<PaperPipelineState>> {
  if (state.pdf) {
    ctx.log(`Using the local file (${(state.pdf.byteLength / 1048576).toFixed(1)} MB)`);
    return {};
  }
  if (state.file) return { pdf: await state.file.arrayBuffer() };

  const url = requireMeta(state).pdfUrl!;
  try {
    const pdf = await downloadPdf(url, { signal: ctx.signal, onProgress: progressLogger(ctx) });
    ctx.log(`Downloaded ${(pdf.byteLength / 1048576).toFixed(1)} MB`);
    return { pdf };
  } catch (error) {
    // Source errors are final (the HTTP layer already retried transient ones):
    // route to the abstract fallback. Anything unexpected is retried by the graph.
    if (error instanceof SourceError) {
      ctx.log(`Download failed (${error.code}): ${error.message}`);
      return { fetchError: error.message };
    }
    throw error;
  }
}

function routeAfterFetch(state: PaperPipelineState): string {
  return state.pdf ? "parse" : "fallback";
}

/** Uploads and bare PDF links only learn their title and abstract from the document itself. */
function enrichMeta(meta: PaperMeta, doc: ParsedDocument): PaperMeta {
  const title = hasPlaceholderTitle(meta) && doc.title ? doc.title : meta.title;
  return { ...meta, title, abstract: meta.abstract ?? doc.abstract };
}

async function parse(state: PaperPipelineState, ctx: NodeContext): Promise<Partial<PaperPipelineState>> {
  const meta = requireMeta(state);
  let doc: ParsedDocument;
  try {
    doc = await parsePdf(state.pdf!);
  } catch (error) {
    ctx.log(`The PDF could not be parsed: ${errorMessage(error)}`);
    return { doc: null, fetchError: `The PDF could not be parsed (${errorMessage(error)})` };
  }

  ctx.log(
    `${doc.pagesParsed}/${doc.pageCount} pages, ${doc.sections.length} sections, ` +
      `${doc.references.length} references, quality ${doc.quality}`,
  );
  const reason = unusableReason(doc);
  return {
    doc,
    meta: enrichMeta(meta, doc),
    warnings: [...state.warnings, ...doc.warnings],
    ...(reason ? { fetchError: reason } : {}),
  };
}

function routeAfterParse(state: PaperPipelineState): string {
  return state.doc && !unusableReason(state.doc) ? "segment" : "fallback";
}

function fallback(state: PaperPipelineState, ctx: NodeContext): Partial<PaperPipelineState> {
  const meta = requireMeta(state);
  const usableDocAbstract = state.doc && state.doc.quality !== "image-only" ? state.doc.abstract : undefined;
  const abstract = meta.abstract?.trim() || usableDocAbstract?.trim();
  const reason = state.fetchError ?? "no open-access PDF was found";

  if (!abstract) {
    if (meta.source === "pdf" && meta.arxivId) {
      throw new Error(`No arXiv paper found for ${meta.arxivId}: neither its metadata nor its PDF could be retrieved.`);
    }
    const isUpload = meta.source === "upload";
    throw new Error(
      isUpload
        ? `Nothing to analyse: ${reason}. Try a PDF with selectable text.`
        : `Could not read the full text (${reason}) and no abstract is available. ` +
            "Download the PDF and upload it to analyse the full paper.",
    );
  }

  ctx.log(`Continuing from the abstract: ${reason}`);
  return {
    mode: "abstract-only",
    doc: null,
    meta: { ...meta, abstract },
    warnings: [
      ...state.warnings,
      `Abstract-only analysis: ${reason}. Upload the PDF for a full-text analysis.`,
    ],
  };
}

function segment(state: PaperPipelineState, ctx: NodeContext): Partial<PaperPipelineState> {
  const sentences =
    state.mode === "full" && state.doc ? segmentDocument(state.doc) : segmentAbstract(requireMeta(state));
  if (sentences.length === 0) throw new Error("No readable sentences were found in this paper");
  ctx.log(`${sentences.length} sentences`);
  return { sentences: sentences.map(unzoned) };
}

function makeZone(deps: { ml: MLApi }) {
  return async (state: PaperPipelineState, ctx: NodeContext): Promise<Partial<PaperPipelineState>> => {
    const sentences = state.sentences ?? [];
    const tracked = trackFailures(deps.ml.embed);
    let zoned: ZonedSentence[];
    try {
      zoned = await zoneSentences(sentences, tracked.embed);
    } catch (error) {
      if (!(error instanceof MLUnavailableError)) throw error;
      zoned = await zoneSentences(sentences, null);
    }
    const counts = new Map<string, number>();
    for (const s of zoned) counts.set(s.zone, (counts.get(s.zone) ?? 0) + 1);
    ctx.log([...counts].map(([zone, n]) => `${zone} ${n}`).join(", "));
    if (!tracked.failed()) return { sentences: zoned };
    ctx.log("Embedding model unavailable; used cue phrases and section priors");
    return { sentences: zoned, warnings: [...state.warnings, ZONING_WARNING] };
  };
}

function chunk(state: PaperPipelineState, ctx: NodeContext): Partial<PaperPipelineState> {
  const chunks = chunkSentences(requirePaperId(state), state.sentences ?? []);
  ctx.log(`${chunks.length} passages`);
  return { chunks };
}

function makeEmbed(deps: { ml: MLApi }) {
  return async (state: PaperPipelineState, ctx: NodeContext): Promise<Partial<PaperPipelineState>> => {
    const texts = (state.chunks ?? []).map(embeddingText);
    const vectors: Float32Array[] = [];
    try {
      for (let start = 0; start < texts.length; start += EMBED_BATCH) {
        vectors.push(...(await deps.ml.embed(texts.slice(start, start + EMBED_BATCH))));
        ctx.log(`Embedded ${Math.min(start + EMBED_BATCH, texts.length)}/${texts.length} passages`);
      }
    } catch (error) {
      if (isAbortError(error)) throw error;
      ctx.log(`Embedding failed: ${errorMessage(error)}`);
      return { embeddings: null, warnings: [...state.warnings, SEMANTIC_SEARCH_WARNING] };
    }
    return { embeddings: vectors.length === texts.length ? vectors : null };
  };
}

function makeAnalyze(deps: { ml: MLApi }) {
  return async (state: PaperPipelineState, ctx: NodeContext): Promise<Partial<PaperPipelineState>> => {
    const sentences = state.sentences ?? [];
    const analysis = await analyzePaper({
      meta: requireMeta(state),
      doc: state.mode === "full" ? (state.doc ?? null) : null,
      sentences,
      chunks: state.chunks ?? [],
      // Only hand the embedder on when it demonstrably works for this run.
      embed: state.embeddings ? deps.ml.embed : null,
      sentenceEmbeddings: state.embeddings ? await sentenceVectors(deps.ml, sentences) : undefined,
    });
    ctx.log(
      `Rigor ${Math.round(analysis.rigor.overall)} (${analysis.rigor.grade}), ` +
        `${analysis.critique.findings.length} critique findings`,
    );
    return { analysis, mode: analysis.mode };
  };
}

// ───────────────────────────────── Graph ───────────────────────────────────

function spec(name: string): { label: string; description: string } {
  const node = PAPER_NODES.find((n) => n.name === name);
  if (!node) throw new Error(`Unknown paper node "${name}"`);
  return { label: node.label, description: node.description };
}

export function buildPaperGraph(deps: { ml: MLApi }): CompiledGraph<PaperPipelineState> {
  return new StateGraph<PaperPipelineState>()
    .addNode("understand", understand, spec("understand"))
    .addNode("resolve", resolve, spec("resolve"))
    .addNode("fetch", fetchPdf, { ...spec("fetch"), retries: 1 })
    .addNode("parse", parse, spec("parse"))
    .addNode("fallback", fallback, spec("fallback"))
    .addNode("segment", segment, spec("segment"))
    .addNode("zone", makeZone(deps), spec("zone"))
    .addNode("chunk", chunk, spec("chunk"))
    .addNode("embed", makeEmbed(deps), spec("embed"))
    .addNode("analyze", makeAnalyze(deps), spec("analyze"))
    .addEdge(START, "understand")
    .addEdge("understand", "resolve")
    .addConditionalEdges("resolve", routeAfterResolve, ["fetch", "fallback"])
    .addConditionalEdges("fetch", routeAfterFetch, ["parse", "fallback"])
    .addConditionalEdges("parse", routeAfterParse, ["segment", "fallback"])
    .addEdge("fallback", "segment")
    .addEdge("segment", "zone")
    .addEdge("zone", "chunk")
    .addEdge("chunk", "embed")
    .addEdge("embed", "analyze")
    .addEdge("analyze", END)
    .compile();
}

/**
 * Sentence vectors for dense LexRank. Zoning already embedded every sentence,
 * so these come straight from the embedding cache; on any failure the analysis
 * simply falls back to TF-IDF similarity.
 */
async function sentenceVectors(
  ml: MLApi,
  sentences: ZonedSentence[],
): Promise<Map<string, Float32Array> | undefined> {
  if (sentences.length === 0) return undefined;
  try {
    const vectors = await ml.embed(sentences.map((s) => s.text));
    if (vectors.length !== sentences.length) return undefined;
    return new Map(sentences.map((s, i) => [s.id, vectors[i]]));
  } catch {
    return undefined;
  }
}

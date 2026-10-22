/**
 * Offline evaluation of retrieval and grounded question answering.
 *
 *   npm run models && npm run eval
 *
 * Each paper in eval/questions.json goes through the app's own paper graph
 * (parse -> segment -> zone -> chunk -> embed -> analyze), driven by the same
 * three quantised ONNX models, run here in Node instead of a Web Worker. Then:
 *
 *  1. Retrieval ablation. For every answerable question, where does the first
 *     gold passage (one containing an expected answer string) rank under BM25
 *     alone, dense cosine alone, hybrid RRF, and hybrid RRF + cross-encoder?
 *     Reported as Recall@1/3/5 and mean reciprocal rank.
 *  2. Question answering. `answerQuestion` runs exactly as in the Ask tab. An
 *     answer counts as correct when the span or its quoted evidence contains an
 *     expected string; unanswerable questions should abstain. The threshold
 *     only gates the last step, so other thresholds can be replayed on the
 *     recorded confidences without re-running the models.
 *  3. Latency of each stage on this machine.
 *
 * Tables go to stdout and into docs/evaluation.md between the eval markers,
 * so the hand-written text around them survives a rerun. Source modules load
 * through Vite's SSR runner, as in scripts/inspect-parse.mts.
 */
import { createServer, createServerModuleRunner } from "vite";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { join } from "node:path";
import type {
  FeatureExtractionPipeline,
  PretrainedModelOptions,
  QuestionAnsweringPipeline,
  Tensor,
} from "@huggingface/transformers";
import type { MLApi } from "../src/ml/client";
import type { PaperPipelineState } from "../src/pipeline/paper";
import type { Answer, Chunk, MLStatus, TraceEvent } from "../src/types";

const QUESTIONS_PATH = "eval/questions.json";
const DOC_PATH = "docs/evaluation.md";
const MODEL_ROOT = join(process.cwd(), "public", "models");
const MARK_START = "<!-- eval:start -->";
const MARK_END = "<!-- eval:end -->";

/** ONNX Runtime threads per model: low, so the run shares the machine politely. */
const THREADS = Number(process.env.EVAL_THREADS ?? 2);
/** Passages the cross-encoder rescores and the reader reads; the same depths as src/qa/answer.ts. */
const RERANK_DEPTH = 8;
const READ_DEPTH = 3;
const RECALL_AT = [1, 3, 5];
const THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5];

// ───────────────────────────────── Types ───────────────────────────────────

interface LabelledQuestion {
  id: string;
  category: string;
  question: string;
  /** Acceptable answer substrings (case-insensitive); null when the paper does not answer it. */
  expected: string[] | null;
}

interface PaperSpec {
  id: string;
  title: string;
  pdf: string;
  questions: LabelledQuestion[];
}

interface App {
  paper: typeof import("../src/pipeline/paper");
  hybrid: typeof import("../src/retrieval/hybrid");
  chunker: typeof import("../src/retrieval/chunker");
  qa: typeof import("../src/qa/answer");
  vector: typeof import("../src/nlp/vector");
  models: typeof import("../src/ml/models");
}

const RETRIEVERS = ["BM25 only", "Dense only", "Hybrid (RRF)", "Hybrid + cross-encoder"] as const;
type Retriever = (typeof RETRIEVERS)[number];

/** Which path answerQuestion took: a reader span, a quoted sentence, the briefing, or nothing retrieved. */
type AnswerMode = "reader" | "sentence" | "composed" | "none";

interface QuestionResult {
  paperId: string;
  label: LabelledQuestion;
  /** 1-based rank of the first gold passage per retriever (null: not retrieved); empty when unanswerable. */
  goldRank: Partial<Record<Retriever, number | null>>;
  /** Rank of the first gold passage in the hits answerQuestion actually used (top 8 after reranking). */
  qaGoldRank: number | null;
  answer: Answer;
  mode: AnswerMode;
  /** Whether the best candidate is right, judged even when the system abstained (for the sweep). */
  candidateCorrect: boolean;
}

interface PaperStats {
  id: string;
  pages: number;
  sentences: number;
  chunks: number;
  stageMs: Record<string, number>;
}

// ──────────────────────────────── Loading ──────────────────────────────────

function readBytes(path: string): ArrayBuffer {
  const buf = readFileSync(path);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

async function loadApp(runner: { import(url: string): Promise<unknown> }): Promise<App> {
  return {
    paper: (await runner.import("/src/pipeline/paper.ts")) as App["paper"],
    hybrid: (await runner.import("/src/retrieval/hybrid.ts")) as App["hybrid"],
    chunker: (await runner.import("/src/retrieval/chunker.ts")) as App["chunker"],
    qa: (await runner.import("/src/qa/answer.ts")) as App["qa"],
    vector: (await runner.import("/src/nlp/vector.ts")) as App["vector"],
    models: (await runner.import("/src/ml/models.ts")) as App["models"],
  };
}

/** The worker's three inference paths (src/ml/worker.ts) plus the client's embedding cache, in-process. */
async function loadModels({ models }: App): Promise<MLApi> {
  const tf = await import("@huggingface/transformers");
  tf.env.localModelPath = `${MODEL_ROOT}/`;
  tf.env.allowLocalModels = true;
  tf.env.allowRemoteModels = false; // offline: a missing file is an error, never a download
  const options: PretrainedModelOptions = {
    dtype: models.MODEL_DTYPE,
    session_options: { intraOpNumThreads: THREADS, interOpNumThreads: 1 },
  };

  const load = tf.pipeline as (task: string, model: string, opts: object) => Promise<unknown>;
  const extractor = (await load("feature-extraction", models.MODELS.embedder.id, options)) as FeatureExtractionPipeline;
  const reader = (await load("question-answering", models.MODELS.reader.id, options)) as QuestionAnsweringPipeline;
  const tokenizer = await tf.AutoTokenizer.from_pretrained(models.MODELS.reranker.id);
  const crossEncoder = await tf.AutoModelForSequenceClassification.from_pretrained(models.MODELS.reranker.id, options);

  const cache = new Map<string, Float32Array>();
  const embed = async (texts: string[]): Promise<Float32Array[]> => {
    const missing = [...new Set(texts.filter((t) => !cache.has(t)))];
    for (let i = 0; i < missing.length; i += models.INFERENCE_BATCH_SIZE) {
      const batch = missing.slice(i, i + models.INFERENCE_BATCH_SIZE);
      const tensor = await extractor(batch, { pooling: "mean", normalize: true });
      const dim = tensor.dims[1];
      const data = tensor.data as Float32Array;
      batch.forEach((text, row) => cache.set(text, data.slice(row * dim, (row + 1) * dim)));
    }
    return texts.map((t) => cache.get(t) as Float32Array);
  };

  const ready: MLStatus = {
    embedder: { state: "ready", progress: 1 },
    reranker: { state: "ready", progress: 1 },
    reader: { state: "ready", progress: 1 },
  };
  return {
    embed,
    async rerank(query, passages) {
      if (passages.length === 0) return [];
      const inputs = tokenizer(new Array<string>(passages.length).fill(query), { text_pair: passages, padding: true, truncation: true });
      const { logits } = (await crossEncoder(inputs)) as { logits: Tensor };
      return Array.from(logits.data as Float32Array);
    },
    async answer(question, context) {
      const output = await reader(question, context);
      const best = Array.isArray(output) ? output[0] : output;
      if (!best) return { answer: "", score: 0, start: -1, end: -1 };
      const span = models.alignSpan(best.answer, context);
      return span
        ? { answer: context.slice(span.start, span.end), score: best.score, ...span }
        : { answer: best.answer, score: best.score, start: -1, end: -1 };
    },
    warmup: async () => undefined,
    getStatus: () => ready,
    subscribe: () => () => undefined,
  };
}

/** Runs the app's paper graph on a local PDF, as if it had been uploaded. */
async function runPaperGraph(app: App, ml: MLApi, paper: PaperSpec): Promise<{ state: PaperPipelineState; stageMs: Record<string, number> }> {
  const stageMs: Record<string, number> = {};
  const state = await app.paper.buildPaperGraph({ ml }).run(
    {
      input: { kind: "upload", raw: paper.pdf, fileName: `${paper.id}.pdf` },
      pdf: readBytes(paper.pdf),
      paperId: `arxiv:${paper.id}`,
      mode: "full",
      warnings: [],
    },
    {
      onEvent: (e) => {
        if (e.type === "node:end" && e.durationMs !== undefined) stageMs[e.node] = e.durationMs;
      },
    },
  );
  if (state.mode !== "full" || !state.chunks || !state.sentences || !state.embeddings) {
    throw new Error(`${paper.id}: the pipeline fell back to ${state.mode} (${state.warnings.join("; ")})`);
  }
  return { state, stageMs };
}

// ─────────────────────────────── Measuring ─────────────────────────────────

function contains(text: string | undefined, expected: string[]): boolean {
  const haystack = (text ?? "").toLowerCase();
  return expected.some((e) => haystack.includes(e.toLowerCase()));
}

function firstGoldRank(ranking: string[], gold: Set<string>): number | null {
  const at = ranking.findIndex((id) => gold.has(id));
  return at < 0 ? null : at + 1;
}

/** Chunk ids, best first, under each retriever. */
async function rankings(
  app: App,
  ml: MLApi,
  question: string,
  chunks: Chunk[],
  embeddings: Float32Array[],
  indexes: { lexical: InstanceType<App["hybrid"]["HybridIndex"]>; hybrid: InstanceType<App["hybrid"]["HybridIndex"]> },
): Promise<Record<Retriever, string[]>> {
  const [queryVector] = await ml.embed([question]);
  const all = { k: chunks.length };

  const dense = chunks
    .map((chunk, i) => ({ id: chunk.id, score: app.vector.cosine(queryVector, embeddings[i]) }))
    .sort((a, b) => b.score - a.score)
    .map((d) => d.id);

  const hybridHits = indexes.hybrid.search(question, queryVector, all);
  const head = hybridHits.slice(0, RERANK_DEPTH);
  const logits = await ml.rerank(question, head.map((h) => app.chunker.embeddingText(h.chunk)));
  const reranked = head.map((h, i) => ({ id: h.chunk.id, logit: logits[i] })).sort((a, b) => b.logit - a.logit);

  return {
    "BM25 only": indexes.lexical.search(question, null, all).map((h) => h.chunk.id),
    "Dense only": dense,
    "Hybrid (RRF)": hybridHits.map((h) => h.chunk.id),
    "Hybrid + cross-encoder": [...reranked.map((r) => r.id), ...hybridHits.slice(RERANK_DEPTH).map((h) => h.chunk.id)],
  };
}

/** The read node logs which path it took; no read node means nothing was retrieved. */
function answerMode(events: TraceEvent[]): AnswerMode {
  if (events.some((e) => e.node === "compose_from_analysis")) return "composed";
  const reads = events.filter((e) => e.node === "read" && e.type === "log");
  if (reads.some((e) => e.message?.startsWith("Reader proposed"))) return "reader";
  return reads.length > 0 ? "sentence" : "none";
}

async function evaluatePaper(app: App, ml: MLApi, paper: PaperSpec): Promise<{ stats: PaperStats; results: QuestionResult[] }> {
  const { state, stageMs } = await runPaperGraph(app, ml, paper);
  const chunks = state.chunks as Chunk[];
  const embeddings = state.embeddings as Float32Array[];
  const indexes = {
    lexical: new app.hybrid.HybridIndex(chunks, null),
    hybrid: new app.hybrid.HybridIndex(chunks, embeddings),
  };
  // The same context the Ask tab builds (src/store/library.ts).
  const ctx = {
    paperId: state.paperId as string,
    index: indexes.hybrid,
    sentences: state.sentences ?? [],
    analysis: state.analysis,
    ml: { embed: ml.embed, rerank: ml.rerank, answer: ml.answer },
  };

  const results: QuestionResult[] = [];
  for (const label of paper.questions) {
    const expected = label.expected ?? [];
    const gold = new Set(chunks.filter((c) => contains(c.text, expected)).map((c) => c.id));
    if (label.expected && gold.size === 0) throw new Error(`${label.id}: no passage contains ${JSON.stringify(expected)}`);

    // QA first, so its timings include embedding the question (the ablation reuses the cached vector).
    const events: TraceEvent[] = [];
    const answer = await app.qa.answerQuestion(label.question, ctx, (e) => events.push(e));

    const goldRank: QuestionResult["goldRank"] = {};
    if (label.expected) {
      const ranked = await rankings(app, ml, label.question, chunks, embeddings, indexes);
      for (const retriever of RETRIEVERS) goldRank[retriever] = firstGoldRank(ranked[retriever], gold);
    }

    const candidateCorrect =
      label.expected !== null && (contains(answer.answer, expected) || answer.evidence.some((e) => contains(e.quote, expected)));
    results.push({
      paperId: paper.id,
      label,
      goldRank,
      qaGoldRank: firstGoldRank(answer.hits.map((h) => h.chunk.id), gold),
      answer,
      mode: answerMode(events),
      candidateCorrect,
    });
  }

  const pages = state.doc?.pagesParsed ?? 0;
  return { stats: { id: paper.id, pages, sentences: state.sentences?.length ?? 0, chunks: chunks.length, stageMs }, results };
}

// ─────────────────────────────── Statistics ────────────────────────────────

const answerable = (r: QuestionResult) => r.label.expected !== null;

/** Would this question be answered if the reader threshold were `t`? Other paths keep their own bar. */
function answeredAt(r: QuestionResult, t: number): boolean {
  return r.mode === "reader" ? r.answer.confidence >= t : r.answer.status === "answered";
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1)];
}

function recallAt(ranks: (number | null)[], k: number): number {
  return ranks.filter((r) => r !== null && r <= k).length / ranks.length;
}

function meanReciprocalRank(ranks: (number | null)[]): number {
  return ranks.reduce<number>((sum, r) => sum + (r === null ? 0 : 1 / r), 0) / ranks.length;
}

// ─────────────────────────────── Reporting ─────────────────────────────────

const fixed = (x: number, digits = 2) => (Number.isFinite(x) ? x.toFixed(digits) : "—");
const pct = (n: number, d: number) => (d === 0 ? "—" : `${Math.round((100 * n) / d)}%`);
const share = (n: number, d: number) => `${n}/${d} (${pct(n, d)})`;
const seconds = (ms: number | undefined) => (ms === undefined ? "—" : `${(ms / 1000).toFixed(2)} s`);
const cell = (text: string) => text.replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();

function table(header: string[], rows: (string | number)[][]): string {
  const line = (cells: (string | number)[]) => `| ${cells.join(" | ")} |`;
  return [line(header), line(header.map(() => "---")), ...rows.map(line)].join("\n");
}

function retrievalSection(results: QuestionResult[], papers: PaperSpec[]): string {
  const scored = results.filter(answerable);
  const ranksOf = (retriever: Retriever, rs: QuestionResult[]) => rs.map((r) => r.goldRank[retriever] ?? null);
  const overall = table(
    ["Retriever", ...RECALL_AT.map((k) => `Recall@${k}`), "MRR"],
    RETRIEVERS.map((retriever) => {
      const ranks = ranksOf(retriever, scored);
      return [retriever, ...RECALL_AT.map((k) => fixed(recallAt(ranks, k))), fixed(meanReciprocalRank(ranks))];
    }),
  );
  const byPaper = table(
    ["MRR by paper", ...papers.map((p) => `${p.id} (${scored.filter((r) => r.paperId === p.id).length} q)`)],
    RETRIEVERS.map((retriever) => [
      retriever,
      ...papers.map((p) => fixed(meanReciprocalRank(ranksOf(retriever, scored.filter((r) => r.paperId === p.id))))),
    ]),
  );
  return `### Retrieval (${scored.length} answerable questions)\n\n${overall}\n\n${byPaper}`;
}

function qaSection(results: QuestionResult[], threshold: number): string {
  const yes = results.filter(answerable);
  const no = results.filter((r) => !answerable(r));
  const answered = (r: QuestionResult) => r.answer.status === "answered";
  const abstained = results.filter((r) => !answered(r));
  const correctAbstentions = abstained.filter((r) => !answerable(r)).length;

  const summary = table(
    ["Measure", "Result"],
    [
      ["Answerable: answered correctly", share(yes.filter((r) => answered(r) && r.candidateCorrect).length, yes.length)],
      ["Answerable: answered wrongly", share(yes.filter((r) => answered(r) && !r.candidateCorrect).length, yes.length)],
      ["Answerable: abstained", share(yes.filter((r) => !answered(r)).length, yes.length)],
      ["Unanswerable: abstained", share(no.filter((r) => !answered(r)).length, no.length)],
      ["Abstention precision (abstentions that were right)", share(correctAbstentions, abstained.length)],
      ["Abstention recall (unanswerable questions caught)", share(correctAbstentions, no.length)],
    ],
  );

  const categories = [...new Set(results.map((r) => r.label.category))];
  const byCategory = table(
    ["Category", "Questions", "Answered correctly", "Answered wrongly", "Abstained"],
    categories.map((category) => {
      const rs = results.filter((r) => r.label.category === category);
      return [
        category,
        rs.length,
        rs.filter((r) => answered(r) && r.candidateCorrect).length,
        rs.filter((r) => answered(r) && !r.candidateCorrect).length,
        rs.filter((r) => !answered(r)).length,
      ];
    }),
  );

  // A failure is a retrieval miss when the gold passage never reached the reader, otherwise a reading miss.
  const failures = yes.filter((r) => !(answered(r) && r.candidateCorrect));
  const read = (r: QuestionResult) => r.qaGoldRank !== null && r.qaGoldRank <= READ_DEPTH;
  const sources = table(
    ["Answerable failures", "Gold passage among the passages read", "Gold passage not read"],
    [
      ["Answered wrongly", ...[true, false].map((seen) => failures.filter((r) => answered(r) && read(r) === seen).length)],
      ["Abstained", ...[true, false].map((seen) => failures.filter((r) => !answered(r) && read(r) === seen).length)],
    ],
  );

  const modes = (["reader", "sentence", "composed", "none"] as const)
    .map((mode) => `${results.filter((r) => r.mode === mode).length} ${mode}`)
    .join(", ");
  return `### Question answering at the shipped threshold (${threshold})\n\nPaths taken: ${modes}.\n\n${summary}\n\n${byCategory}\n\n${sources}`;
}

function sweepSection(results: QuestionResult[]): string {
  const yes = results.filter(answerable);
  const no = results.filter((r) => !answerable(r));
  const rows = THRESHOLDS.map((t) => {
    const answered = results.filter((r) => answeredAt(r, t));
    const correct = answered.filter((r) => answerable(r) && r.candidateCorrect).length;
    const falseAnswers = answered.filter((r) => !answerable(r)).length;
    return [
      fixed(t),
      pct(answered.length, results.length),
      pct(correct, yes.length),
      pct(correct, answered.length),
      pct(falseAnswers, no.length),
    ];
  });
  const readerCount = results.filter((r) => r.mode === "reader").length;
  return (
    `### Threshold sweep (replayed on ${readerCount} reader-span questions; the other paths keep their own bar)\n\n` +
    table(["Threshold", "Coverage (answered / all)", "Accuracy (correct / answerable)", "Precision (correct / answered)", "False answers (/ unanswerable)"], rows)
  );
}

function separationSection(results: QuestionResult[]): string {
  const groups: [string, (r: QuestionResult) => boolean][] = [
    ["Answerable, right candidate", (r) => answerable(r) && r.candidateCorrect],
    ["Answerable, wrong candidate", (r) => answerable(r) && !r.candidateCorrect],
    ["Unanswerable", (r) => !answerable(r)],
  ];
  const rows: (string | number)[][] = [];
  for (const mode of ["reader", "sentence"] as const) {
    for (const [name, test] of groups) {
      const scores = results.filter((r) => r.mode === mode && test(r)).map((r) => r.answer.confidence);
      if (scores.length === 0) continue;
      rows.push([mode, name, scores.length, fixed(Math.min(...scores)), fixed(percentile(scores, 0.5)), fixed(Math.max(...scores))]);
    }
  }
  return `### Confidence by outcome\n\n${table(["Path", "Group", "n", "Min", "Median", "Max"], rows)}`;
}

function latencySection(stats: PaperStats[], results: QuestionResult[], loadMs: number): string {
  const perPaper = table(
    ["Paper", "Pages", "Sentences", "Chunks", "Parse", "Zone (embeds sentences)", "Embed chunks", "Analyze"],
    stats.map((s) => [
      s.id,
      s.pages,
      s.sentences,
      s.chunks,
      seconds(s.stageMs.parse),
      seconds(s.stageMs.zone),
      seconds(s.stageMs.embed),
      seconds(s.stageMs.analyze),
    ]),
  );
  const steps = ["retrieve", "rerank", "read", "verify", "total"];
  const perQuestion = table(
    ["Per question", ...steps],
    [0.5, 0.9].map((p) => [
      p === 0.5 ? "median" : "p90",
      ...steps.map((step) => `${Math.round(percentile(results.map((r) => r.answer.timingsMs[step] ?? 0), p))} ms`),
    ]),
  );
  const cpu = cpus()[0]?.model.trim() ?? "unknown CPU";
  return (
    `### Latency\n\nMeasured on ${cpu} (${cpus().length} cores), Node ${process.version}, ${THREADS} ONNX threads per model. ` +
    `Loading the three models took ${seconds(loadMs)}.\n\n${perPaper}\n\n${perQuestion}`
  );
}

function detailsSection(results: QuestionResult[]): string {
  const rows = results.map((r) => {
    const verdict = !answerable(r)
      ? r.answer.status === "abstained" ? "ok" : "false answer"
      : r.answer.status === "abstained" ? "missed" : r.candidateCorrect ? "ok" : "wrong";
    const shown = r.answer.answer ?? r.answer.evidence[0]?.quote ?? "";
    return [
      r.label.id,
      cell(r.label.question),
      r.label.expected ? cell(r.label.expected.join(" / ")) : "(none)",
      r.mode,
      fixed(r.answer.confidence),
      r.label.expected ? (r.qaGoldRank ?? "—") : "",
      r.answer.status,
      cell(shown.length > 90 ? `${shown.slice(0, 87)}...` : shown),
      verdict,
    ];
  });
  return (
    "<details>\n<summary>Every question, with the answer or (when abstaining) the closest evidence</summary>\n\n" +
    "Gold rank: position of the first gold passage among the hits the QA graph used (— = not in its top 8).\n\n" +
    table(["Id", "Question", "Expected", "Path", "Conf.", "Gold rank", "Status", "Answer / closest evidence", "Verdict"], rows) +
    "\n\n</details>"
  );
}

function writeResults(markdown: string): void {
  const block = `${MARK_START}\n${markdown}\n${MARK_END}`;
  const current = existsSync(DOC_PATH) ? readFileSync(DOC_PATH, "utf8") : "# Evaluation\n";
  const start = current.indexOf(MARK_START);
  const end = current.indexOf(MARK_END);
  const next =
    start >= 0 && end > start
      ? current.slice(0, start) + block + current.slice(end + MARK_END.length)
      : `${current.trimEnd()}\n\n${block}\n`;
  writeFileSync(DOC_PATH, next);
}

// ───────────────────────────────── Main ────────────────────────────────────

async function main(app: App): Promise<void> {
  const missing = Object.values(app.models.MODELS).filter((m) => !existsSync(join(MODEL_ROOT, m.id, "onnx", "model_quantized.onnx")));
  if (missing.length > 0) {
    throw new Error(`Missing models (${missing.map((m) => m.id).join(", ")}); run \`npm run models\` first.`);
  }
  const papers = (JSON.parse(readFileSync(QUESTIONS_PATH, "utf8")) as { papers: PaperSpec[] }).papers;

  const loadStarted = performance.now();
  const ml = await loadModels(app);
  const loadMs = performance.now() - loadStarted;

  const stats: PaperStats[] = [];
  const results: QuestionResult[] = [];
  for (const paper of papers) {
    console.error(`Evaluating ${paper.id} (${paper.questions.length} questions)...`);
    const run = await evaluatePaper(app, ml, paper);
    stats.push(run.stats);
    results.push(...run.results);
  }

  const markdown = [
    `_Generated by \`npm run eval\` from ${QUESTIONS_PATH}: ${results.length} questions over ${papers.length} papers._`,
    retrievalSection(results, papers),
    qaSection(results, app.qa.ABSTAIN_THRESHOLD),
    sweepSection(results),
    separationSection(results),
    latencySection(stats, results, loadMs),
    detailsSection(results),
  ].join("\n\n");
  console.log(markdown);
  writeResults(markdown);
  console.error(`\nWrote ${DOC_PATH}`);
}

const server = await createServer({
  root: process.cwd(),
  server: { middlewareMode: true, hmr: false, ws: false },
  appType: "custom",
  logLevel: "error",
});
const runner = createServerModuleRunner(server.environments.ssr, { hmr: false });
try {
  await main(await loadApp(runner));
} finally {
  await runner.close();
  await server.close();
}

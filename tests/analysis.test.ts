import { fileURLToPath } from "node:url";
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { analyzePaper, extractSignals, stripCitations, type AnalyzeInput } from "@/analysis";
import { CODE_STATEMENT_RE, COMPUTE_RE, HUMAN_EVAL_RE, LIMITATIONS_TITLE_RE } from "@/analysis/lexicons";
import { buildRigor, credit, grade } from "@/analysis/rigor";
import { proportionMargin } from "@/analysis/critique";
import { segmentAbstract, segmentDocument } from "@/nlp/segment";
import { zoneSentences } from "@/nlp/zoning";
import { parsePdf } from "@/parsing";
import { chunkSentences } from "@/retrieval/chunker";
import type { Caption, PaperAnalysis, PaperMeta, ParsedDocument, Reference, Section, SectionKind } from "@/types";

// ─────────────────────────────── Synthetic paper ──────────────────────────────

const ABSTRACT =
  "Large language models require a key-value (KV) cache whose memory grows linearly with context length, which has become the main bottleneck for long-context inference. " +
  "Existing compression methods either require retraining or suffer from severe accuracy degradation at high compression ratios. " +
  "We propose SparseKV, a training-free method that evicts cache entries using accumulated attention sparsity. " +
  "SparseKV dramatically reduces peak memory by 4.2× and improves throughput by 2.8× on LongBench while matching the accuracy of the full cache. " +
  "Our method achieves state-of-the-art results and clearly outperforms all previous methods by a large margin.";

const INTRODUCTION =
  "Serving large language models at long context lengths is increasingly important for applications such as document question answering and code completion [1, 2]. " +
  "However, the KV cache grows linearly with sequence length and batch size, and it quickly exceeds the memory of a single accelerator (Liu et al., 2024). " +
  "Long-context inference is therefore limited by memory rather than by compute. " +
  "Prior heuristics keep a fixed budget of tokens, but they remain challenging to tune and suffer from accuracy loss on retrieval-heavy tasks [3]. " +
  "In this paper, we introduce SparseKV, which scores each cached token by its accumulated attention mass and evicts the least useful entries. " +
  "Our contributions are as follows. " +
  "First, we show that attention in long-context models is highly sparse across heads and layers. " +
  "Second, we design an eviction policy that requires no training and no calibration data. " +
  "Third, we demonstrate that SparseKV achieves an unprecedented 4.2× memory reduction on LongBench with negligible accuracy loss.";

const RELATED =
  "KV cache compression has been studied through quantization, eviction and low-rank projection [4, 5]. " +
  "Quantization methods such as KIVI reduce the precision of cached keys and values to two bits (Liu et al., 2024). " +
  "Eviction methods such as H2O retain heavy-hitter tokens selected by cumulative attention scores [6]. " +
  "StreamingLLM keeps attention sinks together with a sliding window of recent tokens [7]. " +
  "Several of these methods report ablation studies over three random seeds with standard deviations [8].";

const METHOD =
  "SparseKV maintains a running score for every cached token equal to the sum of attention weights it has received. " +
  "At each decoding step, the attention weights of the current query are added to the scores of all cached tokens. " +
  "When the cache exceeds the budget, the tokens with the lowest scores are evicted, except for the most recent window of tokens. " +
  "The recent window protects local context that has not yet accumulated attention mass. " +
  "Scores are computed independently for each attention head, which lets different heads keep different tokens. " +
  "The eviction step adds only a top-k selection over the cache, whose cost is small compared with the attention computation itself. " +
  "The budget is expressed as a fraction of the prompt length so that long and short inputs are compressed proportionally. " +
  "We apply the same budget to every layer to keep the implementation simple.";

const EXPERIMENTS =
  "We evaluate SparseKV on LongBench and Needle-in-a-Haystack using Llama-2-7B and Mistral-7B. " +
  "We compare against the full cache, H2O, StreamingLLM and SnapKV under the same memory budget. " +
  "On LongBench, SparseKV reaches an average score of 41.3 compared with 41.5 for the full cache. " +
  "SparseKV outperforms H2O by 3.4 points and StreamingLLM by 5.1 points at a 20% cache budget. " +
  "On Needle-in-a-Haystack, SparseKV retrieves the needle with 96.2% accuracy at 32K context. " +
  "Peak memory falls from 28.4 GB to 6.8 GB, a reduction of 4.2× in memory usage. " +
  "Throughput increases by 2.8× because larger batches fit on the device. " +
  "The improvements are significantly better than all baselines across every task. " +
  "Accuracy degrades gracefully as the budget shrinks, remaining above 39.0 at a 10% budget. " +
  "All experiments use greedy decoding with a maximum generation length of 128 tokens. " +
  "Results for Mistral-7B follow the same trend as those for Llama-2-7B.";

const CONCLUSION =
  "We presented SparseKV, a training-free KV cache eviction method based on attention sparsity. " +
  "SparseKV reduces memory by 4.2× and improves throughput by 2.8× with minimal accuracy loss. " +
  "In future work, we plan to combine eviction with quantization and to extend the approach to multimodal models.";

const ABLATION =
  "We ablate each component of SparseKV on LongBench with a 20% budget. " +
  "Removing the recent window component lowers the average score from 41.3 to 37.9. " +
  "Without the per-head scoring module, accuracy drops by 2.1 points on retrieval tasks. " +
  "We vary the window size from 16 to 256 tokens and find that 64 tokens works best. " +
  "All results are averaged over 5 random seeds and reported as mean ± standard deviation, for example 41.3 ± 0.4 on LongBench. " +
  "All experiments run on 8 NVIDIA A100 GPUs with a batch size of 16 and a learning rate of 1e-5 for the calibration-free baselines.";

const LIMITATIONS =
  "SparseKV has several limitations. " +
  "Eviction is irreversible, so a token that becomes relevant late in generation cannot be recovered. " +
  "We only evaluate models up to 7B parameters, and the behaviour on larger models remains untested. " +
  "The method fails to preserve accuracy below a 5% budget on retrieval-heavy tasks.";

const STRONG_EXTRA_RESULTS = "Code is publicly available at https://github.com/example/sparsekv.";

interface SectionSpec {
  title: string;
  number?: string;
  kind: SectionKind;
  page: number;
  text: string;
}

function referenceList(): Reference[] {
  const years = [2017, 2019, 2020, 2020, 2021, 2022, 2023, 2023, 2023, 2024, 2024, 2024];
  return years.map((year, i) => ({
    raw: `[${i + 1}] A. Author${i}, B. Writer. A paper about caches. arXiv, ${year}.`,
    authors: `A. Author${i}, B. Writer`,
    title: "A paper about caches",
    year,
  }));
}

function buildDoc(specs: SectionSpec[], links: string[]): ParsedDocument {
  const sections: Section[] = specs.map((s, i) => ({
    id: `s${i}`,
    title: s.number ? `${s.number} ${s.title}` : s.title,
    number: s.number,
    level: s.number?.includes(".") ? 2 : 1,
    kind: s.kind,
    pageStart: s.page,
    pageEnd: s.page,
    text: s.text,
  }));
  const pageCount = Math.max(...specs.map((s) => s.page));
  const pages = Array.from({ length: pageCount }, (_, i) => {
    const text = specs.filter((s) => s.page === i + 1).map((s) => `${s.title}\n${s.text}`).join("\n");
    return { page: i + 1, text, charCount: text.length };
  });
  const captions: Caption[] = [
    { kind: "figure", label: "Figure 1", text: "Figure 1: Attention sparsity across layers of Llama-2-7B.", page: 2 },
    { kind: "table", label: "Table 1", text: "Table 1: LongBench results at a 20% cache budget.", page: 3 },
    { kind: "table", label: "Table 2", text: "Table 2: Needle-in-a-Haystack retrieval accuracy.", page: 3 },
  ];
  return {
    pageCount,
    pagesParsed: pageCount,
    pages,
    title: META.title,
    abstract: ABSTRACT,
    sections,
    references: referenceList(),
    captions,
    links,
    warnings: [],
    quality: "good",
  };
}

const META: PaperMeta = {
  id: "arxiv:2499.00001",
  title: "SparseKV: Training-Free KV Cache Compression via Attention Sparsity",
  authors: ["Jane Doe", "Wei Chen"],
  abstract: ABSTRACT,
  year: 2025,
  arxivId: "2499.00001",
  source: "datacite",
};

const WEAK_SECTIONS: SectionSpec[] = [
  { title: "Abstract", kind: "abstract", page: 1, text: ABSTRACT },
  { title: "Introduction", number: "1", kind: "introduction", page: 1, text: INTRODUCTION },
  { title: "Related Work", number: "2", kind: "related", page: 2, text: RELATED },
  { title: "Method", number: "3", kind: "method", page: 2, text: METHOD },
  { title: "Experiments", number: "4", kind: "experiments", page: 3, text: EXPERIMENTS },
  { title: "Conclusion", number: "5", kind: "conclusion", page: 4, text: CONCLUSION },
];

const STRONG_SECTIONS: SectionSpec[] = [
  ...WEAK_SECTIONS.slice(0, 5),
  { title: "Ablation Study", number: "4.1", kind: "experiments", page: 3, text: ABLATION },
  { title: "Limitations", number: "5", kind: "limitations", page: 4, text: LIMITATIONS },
  { title: "Conclusion", number: "6", kind: "conclusion", page: 4, text: `${CONCLUSION} ${STRONG_EXTRA_RESULTS}` },
];

/** Deterministic bag-of-words hashing "embedder" so the dense code paths run without a model. */
async function hashEmbed(texts: string[]): Promise<Float32Array[]> {
  return texts.map((t) => {
    const v = new Float32Array(64);
    for (const w of t.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
      let h = 0;
      for (const c of w) h = (h * 31 + c.charCodeAt(0)) >>> 0;
      v[h % 64] += 1;
    }
    const norm = Math.hypot(...v) || 1;
    return v.map((x) => x / norm);
  });
}

async function fullInput(specs: SectionSpec[], links: string[] = [], withEmbeddings = false): Promise<AnalyzeInput> {
  const doc = buildDoc(specs, links);
  const sentences = await zoneSentences(segmentDocument(doc), null);
  const chunks = chunkSentences(META.id, sentences);
  const input: AnalyzeInput = { meta: META, doc, sentences, chunks, embed: withEmbeddings ? hashEmbed : null };
  if (withEmbeddings) {
    const vectors = await hashEmbed(sentences.map((s) => s.text));
    input.sentenceEmbeddings = new Map(sentences.map((s, i) => [s.id, vectors[i]]));
  }
  return input;
}

// ─────────────────────────────────── Checks ─────────────────────────────────

/** Every Evidence must point at this paper and quote a real sentence or caption verbatim. */
function expectValidEvidence(analysis: PaperAnalysis, input: AnalyzeInput): void {
  const byId = new Map(input.sentences.map((s) => [s.id, s]));
  const all = [
    analysis.briefing.tldr.evidence,
    ...analysis.briefing.problem.map((i) => i.evidence),
    ...analysis.briefing.method.map((i) => i.evidence),
    ...analysis.briefing.results.map((i) => i.evidence),
    ...analysis.briefing.limitations.authorStated.map((i) => i.evidence),
    ...analysis.briefing.limitations.detected.map((i) => i.evidence),
    analysis.briefing.keyNumbers.map((k) => k.evidence),
    ...analysis.critique.findings.map((f) => f.evidence),
    analysis.rigor.axes.flatMap((a) => a.signals.flatMap((s) => (s.evidence ? [s.evidence] : []))),
  ].flat();
  expect(all.length).toBeGreaterThan(0);
  for (const e of all) {
    expect(e.paperId).toBe(input.meta.id);
    expect(e.page).toBeGreaterThanOrEqual(1);
    expect(e.quote.length).toBeGreaterThan(0);
    expect(e.section.length).toBeGreaterThan(0);
    if (e.sentenceId) expect(byId.get(e.sentenceId)?.text).toBe(e.quote);
  }
}

function expectCommonInvariants(analysis: PaperAnalysis, input: AnalyzeInput): void {
  const { briefing, critique, rigor, visuals } = analysis;
  // Briefing
  expect(briefing.tldr.text.length).toBeGreaterThan(40);
  expect(briefing.tldr.text).not.toMatch(/\[\d+|et al\., \d{4}/);
  expect(briefing.tldr.evidence.length).toBeGreaterThanOrEqual(1);
  expect(briefing.limitations.detected.length).toBeGreaterThan(0);
  expect(briefing.followUps.length).toBeGreaterThanOrEqual(5);
  expect(briefing.followUps.length).toBeLessThanOrEqual(7);
  for (const q of briefing.followUps) expect(q).toMatch(/\?$/);
  expect(new Set(briefing.followUps).size).toBe(briefing.followUps.length);

  // Critique: every finding cites evidence or states what was searched for and not found.
  for (const f of critique.findings) {
    expect(f.title.length).toBeGreaterThan(5);
    if (f.evidence.length === 0) expect(f.rationale).toMatch(/\bno\b|\bnot\b|\bnone\b|\bonly\b|searched/i);
  }
  expect(new Set(critique.findings.map((f) => f.id)).size).toBe(critique.findings.length);
  expect(critique.reviewerQuestions.length).toBeGreaterThanOrEqual(5);
  expect(critique.reviewerQuestions.length).toBeLessThanOrEqual(8);
  const verdictSentences = critique.verdict.split(/(?<=[.!?])\s+(?=[A-Z])/).length;
  expect(verdictSentences).toBeGreaterThanOrEqual(2);
  expect(verdictSentences).toBeLessThanOrEqual(3);

  // Rigor
  expect(rigor.axes.map((a) => a.key)).toEqual(["empirical", "baselines", "ablation", "statistics", "reproducibility", "transparency", "scholarship"]);
  expect(rigor.axes.reduce((s, a) => s + a.weight, 0)).toBeCloseTo(1, 10);
  for (const a of rigor.axes) {
    expect(a.score).toBeGreaterThanOrEqual(0);
    expect(a.score).toBeLessThanOrEqual(100);
    expect(a.signals.length).toBeGreaterThan(0);
    expect(a.explanation).toMatch(new RegExp(`^${a.score}/100 = `));
  }
  const weighted = Math.round(rigor.axes.reduce((s, a) => s + a.score * a.weight, 0));
  expect(rigor.overall).toBe(weighted);
  expect(rigor.grade).toBe(grade(rigor.overall));

  // Visuals
  expect(visuals.rhetoricalMap).toHaveLength(input.sentences.length);
  const positions = visuals.rhetoricalMap.map((p) => p.position);
  expect(positions[0]).toBe(0);
  for (let i = 1; i < positions.length; i++) expect(positions[i]).toBeGreaterThan(positions[i - 1]);
  expect(visuals.zoneDistribution.reduce((s, z) => s + z.count, 0)).toBe(input.sentences.length);
  const nodeIds = new Set(visuals.keyphraseGraph.nodes.map((n) => n.id));
  expect(nodeIds.size).toBeLessThanOrEqual(18);
  for (const l of visuals.keyphraseGraph.links) {
    expect(nodeIds.has(l.source) && nodeIds.has(l.target)).toBe(true);
    expect(l.weight).toBeGreaterThan(0);
    expect(l.weight).toBeLessThanOrEqual(1);
  }
  expect(visuals.readability.readingMinutes).toBeGreaterThanOrEqual(1);
  expect(visuals.readability.jargonDensity).toBeGreaterThanOrEqual(0);
  expect(visuals.readability.jargonDensity).toBeLessThanOrEqual(1);

  expectValidEvidence(analysis, input);
}

const ids = (a: PaperAnalysis) => a.critique.findings.map((f) => f.id);
const severityOf = (a: PaperAnalysis, id: string) => a.critique.findings.find((f) => f.id === id)?.severity;

// ─────────────────────────────────── Tests ──────────────────────────────────

describe("analyzePaper on a full paper with typical weaknesses", async () => {
  const input = await fullInput(WEAK_SECTIONS);
  const analysis = await analyzePaper(input);
  const signals = extractSignals(input);

  it("uses a realistic number of zoned sentences across sections", () => {
    expect(input.sentences.length).toBeGreaterThanOrEqual(40);
    expect(new Set(input.sentences.map((s) => s.sectionKind)).size).toBeGreaterThanOrEqual(5);
    expect(analysis.mode).toBe("full");
    expect(analysis.stats).toMatchObject({ pages: 4, sentences: input.sentences.length, references: 12, figures: 1, tables: 2 });
  });

  it("satisfies every structural invariant", () => {
    expectCommonInvariants(analysis, input);
  });

  it("extracts signals from the paper's own experiments, not from related work", () => {
    expect(signals.datasets.map((d) => d.name)).toEqual(expect.arrayContaining(["LongBench", "Needle-in-a-Haystack"]));
    // Seeds and standard deviations are only mentioned for *other* papers in Related Work.
    expect(signals.seedsOrRuns).toHaveLength(0);
    expect(signals.statistics).toHaveLength(0);
    expect(signals.ablation).toHaveLength(0);
    expect(signals.methodName).toBe("SparseKV");
    expect(signals.comparedSystems).toEqual(expect.arrayContaining(["H2O", "StreamingLLM"]));
    expect(signals.sotaClaims.length).toBeGreaterThan(0);
    expect(signals.boosters).toBeGreaterThanOrEqual(3);
    expect(signals.codeLinks).toEqual([]);
    expect(signals.limitationsSection).toBe(false);
    expect(signals.referenceYears).toHaveLength(12);
  });

  it("raises the findings a careful reviewer would raise, with specific rationales", () => {
    expect(severityOf(analysis, "no-variance")).toBe("major");
    const variance = analysis.critique.findings.find((f) => f.id === "no-variance");
    expect(variance?.rationale).toMatch(/result sentences? and 2 tables/);
    expect(variance?.rationale).toMatch(/smallest reported gain is 3\.4 points/);
    expect(ids(analysis)).toEqual(expect.arrayContaining(["no-ablation", "no-code", "no-limitations", "sota-without-stats", "overclaiming", "significance-wording"]));
    // Findings are ordered by severity, strengths last.
    const order = { major: 0, moderate: 1, minor: 2, strength: 3 };
    const ranks = analysis.critique.findings.map((f) => order[f.severity]);
    expect([...ranks].sort((a, b) => a - b)).toEqual(ranks);
    expect(analysis.critique.verdict).toMatch(/SparseKV/);
    expect(analysis.critique.reviewerQuestions.join(" ")).toMatch(/seeds|ablat|removed/i);
  });

  it("builds an extractive briefing with readable, cited content", () => {
    const { briefing } = analysis;
    expect(briefing.tldr.text).toMatch(/SparseKV/);
    expect(briefing.tldr.text).not.toContain("(Liu et al., 2024)");
    expect(briefing.method.length).toBeGreaterThanOrEqual(2);
    expect(briefing.results.length).toBeGreaterThanOrEqual(2);
    expect(briefing.results.some((r) => /\d/.test(r.text))).toBe(true);
    expect(briefing.keyNumbers.map((k) => k.value)).toEqual(expect.arrayContaining(["4.2×"]));
    expect(briefing.keyNumbers.every((k) => k.label.length > 0)).toBe(true);
    expect(briefing.keyphrases.length).toBeGreaterThan(5);
    // Detected limitations are the critique's major/moderate findings phrased as limitations.
    expect(briefing.limitations.detected[0].text).toMatch(/^Results reported as single numbers/);
    expect(briefing.followUps.some((q) => q.includes("SparseKV"))).toBe(true);
  });
});

describe("analyzePaper on a full paper with good practice", async () => {
  const weak = await analyzePaper(await fullInput(WEAK_SECTIONS));
  const input = await fullInput(STRONG_SECTIONS, ["https://github.com/example/sparsekv"], true);
  const analysis = await analyzePaper(input);

  it("satisfies every structural invariant on the dense-embedding path", () => {
    expectCommonInvariants(analysis, input);
  });

  it("credits strengths and drops the corresponding weaknesses", () => {
    expect(ids(analysis)).toEqual(expect.arrayContaining(["variance-reported", "ablations", "code-released", "candid-limitations"]));
    expect(ids(analysis)).not.toContain("no-variance");
    expect(ids(analysis)).not.toContain("no-ablation");
    expect(ids(analysis)).not.toContain("no-limitations");
    expect(analysis.critique.findings.find((f) => f.id === "code-released")?.rationale).toContain("github.com/example/sparsekv");
  });

  it("scores higher rigor than the weak version, axis by axis where it should", () => {
    const axis = (a: PaperAnalysis, key: string) => a.rigor.axes.find((x) => x.key === key)?.score ?? 0;
    expect(analysis.rigor.overall).toBeGreaterThan(weak.rigor.overall + 10);
    for (const key of ["ablation", "statistics", "reproducibility", "transparency"]) {
      expect(axis(analysis, key)).toBeGreaterThan(axis(weak, key));
    }
  });

  it("collects the authors' own limitations, not other papers' problems", () => {
    const stated = analysis.briefing.limitations.authorStated.map((i) => i.text).join(" ");
    expect(stated).toMatch(/irreversible|up to 7B|fails to preserve/);
    expect(stated).not.toMatch(/Prior heuristics/);
  });
});

describe("analyzePaper in abstract-only mode", async () => {
  const sentences = await zoneSentences(segmentAbstract(META), null);
  const input: AnalyzeInput = { meta: META, doc: null, sentences, chunks: chunkSentences(META.id, sentences), embed: null };
  const analysis = await analyzePaper(input);

  it("produces a coherent, hedged analysis", () => {
    expect(analysis.mode).toBe("abstract-only");
    expect(analysis.stats.pages).toBe(0);
    expectCommonInvariants(analysis, input);
    expect(ids(analysis)).toContain("abstract-only");
    // Absence-based checks must not fire when the body was never seen.
    for (const id of ["no-variance", "no-ablation", "no-code", "no-limitations", "no-compute", "no-related-work"]) {
      expect(ids(analysis)).not.toContain(id);
    }
    expect(analysis.critique.findings.every((f) => f.severity !== "major")).toBe(true);
    expect(analysis.critique.verdict).toMatch(/abstract/i);
    expect(analysis.critique.findings.find((f) => f.id === "abstract-only")?.rationale).toMatch(/could not run/);
    for (const axis of analysis.rigor.axes) expect(axis.explanation).toMatch(/Only the abstract was available/);
    expect(analysis.briefing.limitations.detected.length).toBeGreaterThan(0);
    expect(analysis.briefing.keyNumbers.length).toBeGreaterThan(0);
  });

  it("degrades gracefully with no text at all", async () => {
    const empty = await analyzePaper({ meta: { ...META, abstract: undefined }, doc: null, sentences: [], chunks: [], embed: null });
    expect(empty.mode).toBe("abstract-only");
    expect(empty.briefing.limitations.detected.length).toBeGreaterThan(0);
    expect(empty.critique.reviewerQuestions.length).toBeGreaterThanOrEqual(5);
    expect(empty.visuals.rhetoricalMap).toEqual([]);
  });
});

describe("helpers", () => {
  it("strips bracket and author-year citations", () => {
    expect(stripCitations("Attention is sparse [3, 5-7] across heads (Liu et al., 2024; Wei, 2022).")).toBe("Attention is sparse across heads.");
    expect(stripCitations("As shown by Wei et al. (2022), chain of thought helps [12].")).toBe("As shown by Wei et al., chain of thought helps.");
    expect(stripCitations("Accuracy (Table 2) improves.")).toBe("Accuracy (Table 2) improves.");
  });

  it("uses saturating, explainable scoring", () => {
    expect(credit(0, 2)).toBe(0);
    expect(credit(2, 2)).toBeCloseTo(0.5);
    expect(credit(100, 2)).toBeLessThan(1);
    expect([grade(80), grade(65), grade(50), grade(49)]).toEqual(["A", "B", "C", "D"]);
    expect(proportionMargin(100)).toBeCloseTo(9.8, 1);
  });

  it("rigor of an empty signal set is a D with explanations", async () => {
    const sentences = await zoneSentences(segmentAbstract(META), null);
    const signals = extractSignals({ meta: META, doc: null, sentences, chunks: [], embed: null });
    const rigor = buildRigor({ ...signals, datasets: [], metrics: [], resultSentences: 0 });
    expect(rigor.grade).toBe("D");
    expect(rigor.axes.every((a) => a.explanation.length > 20)).toBe(true);
  });
});

describe("text that must not count as the paper's own evidence", () => {
  it("treats a section titled Related Works as other people's work", async () => {
    const specs = WEAK_SECTIONS.map((s) => (s.kind === "related" ? { ...s, title: "Extended Related Works", kind: "appendix" as const } : s));
    const signals = extractSignals(await fullInput(specs));
    expect(signals.relatedWorkSection).toBe(true);
    // "three random seeds with standard deviations" there describes other papers.
    expect(signals.seedsOrRuns).toHaveLength(0);
    expect(signals.statistics).toHaveLength(0);
  });

  it("ignores reproducibility-checklist questions and guidelines", async () => {
    const checklist =
      "(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes] " +
      "If you used crowdsourcing or conducted research with human subjects, include the instructions given to participants.";
    const guidelines = "Guidelines: The authors should report standard deviations and the number of random seeds.";
    const signals = extractSignals(
      await fullInput([
        ...WEAK_SECTIONS,
        { title: "Checklist", kind: "other", page: 4, text: checklist },
        { title: "Experiment Statistical Significance", number: "7", kind: "other", page: 4, text: guidelines },
      ]),
    );
    expect(signals.statistics).toHaveLength(0);
    expect(signals.seedsOrRuns).toHaveLength(0);
    expect(signals.humanEval).toHaveLength(0);
  });

  it("does not read negated availability as a code release", () => {
    expect(CODE_STATEMENT_RE.test("In this setting, no pre-trained SFT model is available, so we fine-tune one.")).toBe(false);
    expect(CODE_STATEMENT_RE.test("We cannot compare with it, since the model is not publicly available.")).toBe(false);
    expect(CODE_STATEMENT_RE.test("Our code and models are publicly available at https://github.com/example/repo.")).toBe(true);
    expect(CODE_STATEMENT_RE.test("The source code is available at https://github.com/jy-yuan/KIVI.")).toBe(true);
  });

  it("needs an evaluation, not preference training data, to count as human evaluation", () => {
    expect(HUMAN_EVAL_RE.test("The policy is then aligned with human preferences to produce helpful outputs.")).toBe(false);
    expect(HUMAN_EVAL_RE.test("Comparisons were collected from human labelers for reward modelling.")).toBe(false);
    expect(HUMAN_EVAL_RE.test("A human preference study confirms the win rates.")).toBe(true);
    expect(HUMAN_EVAL_RE.test("We also run a human evaluation of 200 summaries.")).toBe(true);
  });

  it("does not take a failure-modes analysis for a limitations section", () => {
    expect(LIMITATIONS_TITLE_RE.test("E.1 Success and Failure Modes Analysis")).toBe(false);
    expect(LIMITATIONS_TITLE_RE.test("6 Limitations and Failure Modes")).toBe(true);
  });

  it("counts training time, not epochs, as compute", () => {
    expect(COMPUTE_RE.test("The reward model is trained for 3 epochs on the preference datasets.")).toBe(false);
    expect(COMPUTE_RE.test("The largest model was trained for about 2 days.")).toBe(true);
    expect(COMPUTE_RE.test("Each model is trained for roughly a week.")).toBe(true);
  });
});

// ───────────────────────────── Real papers (optional) ───────────────────────────

const SAMPLES = fileURLToPath(new URL("./fixtures/pdfs", import.meta.url));

async function realInput(arxivId: string, title: string): Promise<AnalyzeInput> {
  const bytes = readFileSync(`${SAMPLES}/${arxivId}.pdf`);
  const doc = await parsePdf(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
  const meta: PaperMeta = { id: `arxiv:${arxivId}`, title, authors: [], abstract: doc.abstract, year: 2024, arxivId, source: "datacite" };
  const sentences = await zoneSentences(segmentDocument(doc), null);
  return { meta, doc, sentences, chunks: chunkSentences(meta.id, sentences), embed: null };
}

describe.skipIf(!existsSync(`${SAMPLES}/2402.02750.pdf`))("analyzePaper on a real paper (KIVI, arXiv 2402.02750)", async () => {
  const input = await realInput("2402.02750", "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache");
  const analysis = await analyzePaper(input);
  const signals = extractSignals(input);

  it("keeps every invariant on parsed PDF text", () => {
    expectCommonInvariants(analysis, input);
  });

  it("finds what a human reviewer would find", () => {
    expect(signals.methodName).toBe("KIVI");
    expect(signals.codeLinks.join(" ")).toContain("github.com/jy-yuan/KIVI");
    expect(signals.datasets.map((d) => d.name)).toEqual(expect.arrayContaining(["GSM8K", "LongBench"]));
    expect(ids(analysis)).toEqual(expect.arrayContaining(["no-variance", "code-released", "ablations"]));
    expect(analysis.briefing.keyNumbers.map((k) => k.value)).toContain("2.6×");
    expect(analysis.briefing.tldr.text).not.toMatch(/\(\w+ et al\.?,? \d{4}\)/);
  });
});

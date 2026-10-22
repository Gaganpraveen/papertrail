import { describe, expect, it } from "vitest";
import type {
  Chunk,
  CritiqueFinding,
  EmbedFn,
  PaperAnalysis,
  PaperMeta,
  Reference,
  SectionKind,
  SurveyPaperInput,
  Zone,
  ZonedSentence,
} from "@/types";
import { buildSurvey } from "@/analysis/survey";
import { averageLinkage, clusterPapers, cutTree, silhouette, toDistance } from "@/analysis/survey/clustering";
import { cleanQuote, clutterPenalty, compareRow, isQuotable, truncate } from "@/analysis/survey/comparison";
import { bibliographicCoupling, directCitations } from "@/analysis/survey/coupling";
import { countTerms, labelClusters } from "@/analysis/survey/ctfidf";
import { sharedWeaknesses } from "@/analysis/survey/gaps";
import { attribute, joinList } from "@/analysis/survey/narrative";
import { classicalMds, mulberry32 } from "@/analysis/survey/projection";
import { tokenize } from "@/nlp/text";

// ───────────────────────────────── Fixtures ─────────────────────────────────

/** Deterministic stand-in for the sentence encoder: hashed bag of stems, L2-normalised. */
function hashingEmbed(dimensions = 128): EmbedFn {
  return async (texts) =>
    texts.map((text) => {
      const v = new Float32Array(dimensions);
      for (const token of tokenize(text)) {
        let h = 2166136261;
        for (let i = 0; i < token.length; i++) h = Math.imul(h ^ token.charCodeAt(i), 16777619) >>> 0;
        v[h % dimensions] += 1;
      }
      const norm = Math.hypot(...v);
      return norm > 0 ? v.map((x) => x / norm) : v;
    });
}

type Line = [Zone, string, SectionKind?];

interface PaperSpec {
  id: string;
  title: string;
  authors: string[];
  year: number;
  lines: Line[];
  references?: Reference[];
  findings?: CritiqueFinding[];
  rigor?: number;
}

function analysisFor(spec: PaperSpec, sentences: ZonedSentence[]): PaperAnalysis {
  const evidence = (s: ZonedSentence) => ({ paperId: spec.id, page: s.page, section: s.sectionTitle, quote: s.text });
  const item = (s: ZonedSentence) => ({ text: s.text, evidence: [evidence(s)] });
  const zone = (z: Zone) => sentences.filter((s) => s.zone === z);
  return {
    briefing: {
      tldr: item(sentences[0]),
      problem: zone("problem").map(item),
      method: zone("method").map(item),
      results: zone("result").map(item),
      limitations: {
        authorStated: zone("limitation").map(item),
        detected: [{ text: "No variance across seeds is reported.", evidence: [] }],
      },
      followUps: [],
      keyphrases: tokenize(spec.title)
        .slice(0, 4)
        .map((phrase, i) => ({ phrase, score: 1 - i * 0.1 })),
      keyNumbers: [],
    },
    critique: { findings: spec.findings ?? [], reviewerQuestions: [], verdict: "" },
    rigor: { overall: spec.rigor ?? 60, grade: "C", axes: [] },
    visuals: {
      rhetoricalMap: [],
      sectionComposition: [],
      zoneDistribution: [],
      keyphraseGraph: { nodes: [], links: [] },
      citationYears: [],
      readability: { fleschReadingEase: 30, avgSentenceLength: 20, readingMinutes: 10, jargonDensity: 0.1 },
    },
    stats: { pages: 10, words: 5000, sentences: sentences.length, chunks: 1, references: 0, figures: 0, tables: 0 },
    mode: "full",
    computedMs: 1,
  };
}

async function makePaper(spec: PaperSpec, embed: EmbedFn | null): Promise<SurveyPaperInput> {
  const sentences: ZonedSentence[] = spec.lines.map(([zone, text, kind = "abstract"], index) => ({
    id: `s${index}:${index}`,
    text,
    sectionId: `s${index}`,
    sectionTitle: kind,
    sectionKind: kind,
    page: 1 + Math.floor(index / 3),
    index,
    zone,
    confidence: 0.8,
    scores: { background: 0, problem: 0, contribution: 0, method: 0, result: 0, limitation: 0, future: 0, other: 0, [zone]: 1 },
  }));
  // Two chunks per paper, each covering half of its sentences.
  const half = Math.ceil(sentences.length / 2);
  const chunks: Chunk[] = [sentences.slice(0, half), sentences.slice(half)].map((part, index) => ({
    id: `${spec.id}#${index}`,
    paperId: spec.id,
    index,
    sectionId: part[0].sectionId,
    sectionTitle: part[0].sectionTitle,
    sectionKind: part[0].sectionKind,
    page: part[0].page,
    text: part.map((s) => s.text).join(" "),
    tokenCount: part.reduce((n, s) => n + s.text.split(" ").length, 0),
  }));
  const meta: PaperMeta = {
    id: spec.id,
    title: spec.title,
    authors: spec.authors,
    year: spec.year,
    published: `${spec.year}-01-15`,
    arxivId: spec.id.replace("arxiv:", ""),
    source: "datacite",
  };
  return {
    meta,
    analysis: analysisFor(spec, sentences),
    sentences,
    chunks,
    chunkEmbeddings: embed ? await embed(chunks.map((c) => c.text)) : null,
    references: spec.references ?? [],
  };
}

const KV_SHARED_RESULT =
  "Quantizing the key cache per channel and the value cache per token reduces peak memory of the KV cache by 2.6x with almost no accuracy loss.";

const SPECS: PaperSpec[] = [
  {
    id: "arxiv:2402.02750",
    title: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache",
    authors: ["Zirui Liu", "Jiayi Yuan", "Hongye Jin", "Shaochen Zhong"],
    year: 2024,
    lines: [
      ["problem", "The KV cache memory of large language models grows linearly with batch size and context length during decoding."],
      ["contribution", "We propose KIVI, a tuning-free 2bit KV cache quantization algorithm for the key cache and value cache."],
      ["method", "We quantize the key cache per channel and the value cache per token to keep quantization error of the KV cache small.", "method"],
      ["result", KV_SHARED_RESULT, "results"],
      ["result", "KIVI enables up to 4x larger batch size and improves throughput on GSM8K with Llama models, measured by accuracy.", "results"],
      ["future", "Extending KV cache quantization to multimodal vision encoders is left for future work.", "conclusion"],
      ["other", "Our code is available at https://github.com/jy-yuan/KIVI for the KV cache quantization kernels.", "conclusion"],
    ],
    references: [
      { raw: "Vaswani et al. Attention is all you need. 2017", title: "Attention Is All You Need", year: 2017, arxivId: "1706.03762" },
      { raw: "Xiao et al. Efficient streaming language models with attention sinks", arxivId: "2309.17453v2" },
      { raw: "Frantar et al. GPTQ", title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" },
    ],
    findings: [
      { id: "f1", category: "statistics", severity: "major", title: "No variance across runs", rationale: "", evidence: [] },
      { id: "code-released", category: "reproducibility", severity: "strength", title: "Code is linked", rationale: "", evidence: [] },
    ],
    rigor: 72,
  },
  {
    id: "arxiv:2401.18079",
    title: "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization",
    authors: ["Coleman Hooper", "Sehoon Kim"],
    year: 2024,
    lines: [
      ["problem", "Long context inference is bottlenecked by the memory footprint of the KV cache in large language models."],
      ["contribution", "We introduce KVQuant, which quantizes the KV cache with per channel key quantization and non uniform datatypes."],
      ["method", "Our method applies pre RoPE key quantization and dense and sparse quantization of the value cache.", "method"],
      ["result", "Quantizing the key cache per channel and the value cache per token reduces peak memory of the KV cache by 4.8x with almost no accuracy loss.", "results"],
      ["result", "KVQuant reaches lower perplexity on WikiText-2 than prior KV cache quantization methods.", "results"],
    ],
    references: [
      { raw: "Attention is all you need", title: "Attention is all you need.", year: 2017 },
      { raw: "StreamingLLM", doi: "10.48550/arXiv.2309.17453" },
      { raw: "KIVI", arxivId: "2402.02750" },
    ],
    findings: [{ id: "f9", category: "statistics", severity: "major", title: "No variance across runs", rationale: "", evidence: [] }],
    rigor: 64,
  },
  {
    id: "arxiv:2201.11903",
    title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
    authors: ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans"],
    year: 2022,
    lines: [
      ["problem", "Scaling up model size alone has not proved sufficient for arithmetic reasoning and commonsense reasoning benchmarks."],
      ["contribution", "We explore chain of thought prompting, where few shot exemplars contain intermediate reasoning steps for arithmetic reasoning."],
      ["method", "Each exemplar in the prompt pairs a math word problem with a chain of thought reasoning rationale and the answer.", "method"],
      ["result", "Chain of thought prompting with PaLM 540B achieves state of the art accuracy on the GSM8K math word problem benchmark.", "results"],
      ["limitation", "Chain of thought reasoning does not guarantee correct reasoning paths and the prompting emerges only at large model scale.", "discussion"],
    ],
    findings: [{ id: "x", category: "statistics", severity: "major", title: "No variance across runs", rationale: "", evidence: [] }],
    rigor: 55,
  },
  {
    id: "arxiv:2203.11171",
    title: "Self-Consistency Improves Chain of Thought Reasoning in Language Models",
    authors: ["Xuezhi Wang", "Jason Wei"],
    year: 2023,
    lines: [
      ["problem", "Greedy decoding of a single chain of thought reasoning path is brittle on arithmetic reasoning benchmarks."],
      ["contribution", "We propose self consistency, a decoding strategy that samples diverse chain of thought reasoning paths and takes a majority vote."],
      ["method", "Self consistency marginalizes over sampled reasoning paths from chain of thought prompting instead of greedy decoding.", "method"],
      ["result", "Self consistency boosts chain of thought prompting accuracy on GSM8K by 17.9 points and on the SVAMP math word problem benchmark.", "results"],
    ],
    references: [{ raw: "Wei et al. Chain of thought prompting", arxivId: "2201.11903" }],
    rigor: 58,
  },
  {
    id: "arxiv:2005.11401",
    title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    authors: ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
    year: 2020,
    lines: [
      ["problem", "Parametric language models cannot easily update their world knowledge or provide provenance for retrieved documents."],
      ["contribution", "We present retrieval augmented generation models that combine a dense passage retriever over Wikipedia documents with a seq2seq generator."],
      ["method", "The retriever returns top k Wikipedia passages and the generator marginalizes over retrieved documents during generation.", "method"],
      ["result", "Retrieval augmented generation sets the state of the art on Natural Questions and TriviaQA open domain question answering, measured by exact match.", "results"],
    ],
    rigor: 66,
  },
  {
    id: "arxiv:2310.11511",
    title: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
    authors: ["Akari Asai", "Zeqiu Wu"],
    year: 2023,
    lines: [
      ["problem", "Retrieval augmented generation retrieves a fixed number of passages regardless of whether retrieval of documents is necessary."],
      ["contribution", "We introduce Self-RAG, which trains a language model to retrieve passages on demand and critique retrieved documents with reflection tokens."],
      ["method", "Reflection tokens decide when to retrieve Wikipedia passages and grade the relevance of each retrieved document during generation.", "method"],
      ["result", "Self-RAG outperforms retrieval augmented generation baselines on open domain question answering and fact verification on TriviaQA.", "results"],
      ["future", "Future work should study retrieval on demand for longer document generation with multimodal retrieval corpora.", "conclusion"],
    ],
    references: [{ raw: "Lewis et al. Retrieval-augmented generation for knowledge-intensive NLP tasks", title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" }],
    rigor: 70,
  },
];

const THEME_OF: Record<string, number> = {
  "arxiv:2402.02750": 0,
  "arxiv:2401.18079": 0,
  "arxiv:2201.11903": 1,
  "arxiv:2203.11171": 1,
  "arxiv:2005.11401": 2,
  "arxiv:2310.11511": 2,
};

async function corpus(embed: EmbedFn | null, specs = SPECS): Promise<SurveyPaperInput[]> {
  return Promise.all(specs.map((s) => makePaper(s, embed)));
}

/** Same grouping regardless of label numbering. */
function samePartition(labels: number[], expected: number[]): boolean {
  for (let i = 0; i < labels.length; i++) {
    for (let j = 0; j < labels.length; j++) {
      if ((labels[i] === labels[j]) !== (expected[i] === expected[j])) return false;
    }
  }
  return true;
}

// ──────────────────────────────── Clustering ────────────────────────────────

describe("clustering", () => {
  it("average linkage recovers well separated groups and silhouette picks their number", () => {
    // Three tight groups on a line: {0,1}, {2,3,4}, {5,6}.
    const xs = [0, 0.1, 5, 5.1, 5.2, 10, 10.1];
    const similarity = xs.map((a) => xs.map((b) => 1 - Math.abs(a - b) / 10));
    const result = clusterPapers(similarity);
    expect(result.k).toBe(3);
    expect(samePartition(result.labels, [0, 0, 1, 1, 1, 2, 2])).toBe(true);
    expect(result.silhouette).toBeGreaterThan(0.8);
    expect(result.candidates.map((c) => c.k)).toEqual([2, 3, 4, 5, 6]);
  });

  it("silhouette stays within [-1, 1] for arbitrary labelings", () => {
    const random = mulberry32(7);
    for (let trial = 0; trial < 20; trial++) {
      const n = 3 + Math.floor(random() * 6);
      const points = Array.from({ length: n }, () => [random(), random()]);
      const distance = points.map((p) => points.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])));
      const labels = points.map(() => Math.floor(random() * 3));
      const s = silhouette(distance, labels);
      expect(s).toBeGreaterThanOrEqual(-1);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it("cutting the dendrogram yields exactly k clusters", () => {
    const distance = toDistance([
      [1, 0.9, 0.1, 0.1],
      [0.9, 1, 0.1, 0.1],
      [0.1, 0.1, 1, 0.8],
      [0.1, 0.1, 0.8, 1],
    ]);
    const merges = averageLinkage(distance);
    expect(merges).toHaveLength(3);
    expect(new Set(cutTree(merges, 4, 2)).size).toBe(2);
    expect(cutTree(merges, 4, 1)).toEqual([0, 0, 0, 0]);
    expect(cutTree(merges, 4, 4)).toEqual([0, 1, 2, 3]);
  });

  it("gives every paper its own cluster when there are only two", () => {
    expect(clusterPapers([[1, 0.4], [0.4, 1]])).toMatchObject({ k: 2, labels: [0, 1], silhouette: 0 });
    expect(clusterPapers([[1]])).toMatchObject({ k: 1, labels: [0] });
  });
});

// ────────────────────────────── Projection ──────────────────────────────────

describe("projection", () => {
  it("classical MDS reproduces distances of points that lie on a plane", () => {
    const points = [[0, 0], [3, 0], [0, 4], [3, 4]];
    const distance = points.map((p) => points.map((q) => Math.hypot(p[0] - q[0], p[1] - q[1])));
    const layout = classicalMds(distance);
    for (let i = 0; i < points.length; i++) {
      for (let j = 0; j < points.length; j++) {
        expect(Math.hypot(layout[i][0] - layout[j][0], layout[i][1] - layout[j][1])).toBeCloseTo(distance[i][j], 4);
      }
    }
  });

  it("mulberry32 is deterministic", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
});

// ─────────────────────────────── c-TF-IDF ───────────────────────────────────

describe("c-TF-IDF", () => {
  it("labels each cluster with its distinctive terms and keeps acronyms", () => {
    const counts = [
      countTerms("KV cache quantization reduces KV cache memory. The KV cache stores keys."),
      countTerms("Quantization of the KV cache and the value cache saves memory."),
      countTerms("Chain-of-thought prompting improves arithmetic reasoning. Reasoning chains help."),
    ];
    const [kv, cot] = labelClusters(counts, [0, 0, 1], 2);
    expect(kv.terms[0]).toBe("KV cache");
    expect(kv.terms).not.toContain("cache"); // absorbed by the bigram
    expect(cot.terms).toContain("reasoning");
    expect(kv.label).toContain("KV cache");
  });
});

// ─────────────────────────────── Coupling ───────────────────────────────────

describe("bibliographic coupling", () => {
  it("matches shared references across arXiv ids, versions, DOIs and titles", async () => {
    const papers = await corpus(null);
    const linked = papers.map((p) => ({ id: p.meta.id, meta: p.meta, references: p.references }));
    const edges = bibliographicCoupling(linked);
    const kv = edges.find((e) => e.source === "arxiv:2402.02750" && e.target === "arxiv:2401.18079");
    // Shared: "Attention Is All You Need" (by title) and StreamingLLM (arXiv id with version vs arXiv DOI).
    expect(kv).toMatchObject({ shared: 2 });
    expect(kv?.jaccard).toBeCloseTo(2 / 4, 5);
    expect(edges.every((e) => e.shared > 0 && e.jaccard > 0 && e.jaccard <= 1)).toBe(true);
    expect(edges.some((e) => e.source === "arxiv:2201.11903" || e.target === "arxiv:2201.11903")).toBe(false);
  });

  it("detects direct citations between surveyed papers", async () => {
    const papers = await corpus(null);
    const citations = directCitations(papers.map((p) => ({ id: p.meta.id, meta: p.meta, references: p.references })));
    expect(citations).toContainEqual({ citing: "arxiv:2401.18079", cited: "arxiv:2402.02750" });
    expect(citations).toContainEqual({ citing: "arxiv:2203.11171", cited: "arxiv:2201.11903" });
    expect(citations).toContainEqual({ citing: "arxiv:2310.11511", cited: "arxiv:2005.11401" });
  });
});

// ─────────────────────────────── Quoting ────────────────────────────────────

describe("quoting and attribution", () => {
  const meta: PaperMeta = { id: "p", title: "T", authors: ["Zirui Liu", "Jiayi Yuan", "Hongye Jin"], source: "upload" };

  it("strips source citations and context connectives", () => {
    expect(cleanQuote("However, attention is quadratic [12, 14] in length (Vaswani et al., 2017)")).toBe(
      "Attention is quadratic in length.",
    );
    expect(truncate("one two three four five six", 14)).toBe("one two three…");
    expect(cleanQuote("As shown in Table 2 (bottom), our model outperforms RAG (Figure 3).")).toBe("Our model outperforms RAG.");
    expect(cleanQuote("However, based on these observations, we propose KIVI (B. Peng et al. 2023).")).toBe("We propose KIVI.");
  });

  it("rejects checklist items, questions, table residue and figure pointers", () => {
    expect(isQuotable("[Yes] (b) Did you describe the limitations of your work in the paper")).toBe(false);
    expect(isQuotable("How should chain of thought annotations be generated in a robust fashion?")).toBe(false);
    expect(clutterPenalty("SELF-RAG also wins with PQA (acc) SELF-RAG (50k) 45.5 Training No Retriever R 43.6 overall.")).toBeGreaterThan(0);
    expect(clutterPenalty("We use the prompt on the left for Act and the prompt on the right for ReAct.")).toBeGreaterThan(0);
    expect(clutterPenalty("Each new key tK is appended to XKr until it reaches R tokens.")).toBeGreaterThan(0);
    expect(clutterPenalty("KIVI enables 4x larger batch sizes for LLMs on GPUs with no tuning.")).toBe(0);
  });

  it("rewrites first-person sentences into attributed third person", () => {
    expect(attribute("We propose KIVI, which halves our memory.", meta, "[1]")).toBe(
      "Liu et al. [1] propose KIVI, which halves their memory.",
    );
    expect(attribute("Our method reaches 2.6x lower memory.", meta, "[1]")).toBe("Liu et al.'s [1] method reaches 2.6x lower memory.");
    expect(attribute("This paper studies KV caches.", meta, "[1]")).toBe("The paper by Liu et al. [1] studies KV caches.");
    expect(attribute("Existing methods fail on long inputs.", meta, "[1]", "problem")).toBe(
      "Liu et al. [1] argue that existing methods fail on long inputs.",
    );
    expect(attribute("Existing methods fail.", { ...meta, authors: ["Ada Lovelace"] }, "[2]", "problem")).toBe(
      "Lovelace [2] argues that existing methods fail.",
    );
    expect(attribute("To address this, we design a tree of summaries.", meta, "[1]")).toBe("Liu et al. [1] design a tree of summaries.");
    expect(attribute("We propose a new cache.", { ...meta, authors: ["Ada Lovelace"] }, "[2]")).toBe(
      "As Lovelace [2] put it, “We propose a new cache.”",
    );
    expect(joinList(["[1]", "[2]", "[3]"])).toBe("[1], [2] and [3]");
  });
});

// ─────────────────────────────── Comparison ─────────────────────────────────

describe("comparison", () => {
  it("reads benchmarks, metrics, code and the headline result from the paper", async () => {
    const [kivi, kvquant] = await corpus(null);
    const row = compareRow(kivi);
    expect(row.datasets).toContain("GSM8K");
    expect(row.metrics).toContain("accuracy");
    expect(row.code).toBe(true);
    expect(row.rigor).toBe(72);
    expect(row.year).toBe(2024);
    expect(row.keyResult).toContain("2.6x");
    expect(row.limitations).toBe(1);
    const other = compareRow(kvquant);
    expect(other.code).toBe(false);
    expect(other.datasets).toContain("WikiText-2");
    expect(other.metrics).toContain("perplexity");
  });

  it("agrees with the paper's own critique about code, whatever links the text holds", async () => {
    const [kvquant] = await corpus(null, [
      {
        ...SPECS[1],
        lines: [...SPECS[1].lines, ["method", "We run the GPTQ baseline from https://github.com/IST-DASLab/gptq with its defaults.", "experiments"]],
        findings: [{ id: "no-code", category: "reproducibility", severity: "moderate", title: "No code or model release found", rationale: "", evidence: [] }],
      },
    ]);
    expect(compareRow(kvquant).code).toBe(false);
  });
});

// ─────────────────────────────── Gaps ───────────────────────────────────────

describe("shared weaknesses", () => {
  it("surfaces critique findings that recur in at least half of the papers", async () => {
    const papers = await corpus(null);
    const shared = sharedWeaknesses(papers);
    expect(shared).toHaveLength(1);
    expect(shared[0].paperIds).toHaveLength(3);
    expect(shared[0].statement).toBe("3 of 6 papers share this weakness: No variance across runs.");
  });
});

// ─────────────────────────────── End to end ─────────────────────────────────

describe("buildSurvey", () => {
  it("clusters six papers into their three themes with embeddings", async () => {
    const embed = hashingEmbed();
    const papers = await corpus(embed);
    const steps: string[] = [];
    const survey = await buildSurvey(papers, embed, (s) => steps.push(s));

    expect(survey.paperIds).toEqual(SPECS.map((s) => s.id));
    expect(survey.clusters).toHaveLength(3);
    const labels = survey.paperIds.map((id) => survey.clusters.find((c) => c.paperIds.includes(id))?.id ?? -1);
    expect(samePartition(labels, survey.paperIds.map((id) => THEME_OF[id]))).toBe(true);
    expect(survey.silhouette).toBeGreaterThan(0);
    expect(survey.silhouette).toBeLessThanOrEqual(1);
    expect(survey.clusters.map((c) => c.color)).toEqual([0, 1, 2]);
    expect(survey.clusters.every((c) => c.terms.length > 0 && c.label.length > 0)).toBe(true);
    expect(survey.clusters.find((c) => c.paperIds.includes("arxiv:2402.02750"))?.label).toMatch(/KV cache/);

    // Similarity matrix: symmetric, unit diagonal.
    expect(survey.similarity).toHaveLength(6);
    survey.similarity.forEach((row, i) => {
      expect(row[i]).toBeCloseTo(1, 5);
      row.forEach((v, j) => expect(v).toBeCloseTo(survey.similarity[j][i], 6));
    });

    // 12 chunks + 6 centroids >= 8 points: UMAP.
    expect(survey.map.method).toBe("umap");
    expect(survey.map.points).toHaveLength(18);
    expect(survey.map.points.filter((p) => p.isCentroid)).toHaveLength(6);
    expect(survey.map.points.every((p) => Math.abs(p.x) <= 1 && Math.abs(p.y) <= 1)).toBe(true);

    expect(steps).toContain("Writing the review");
    expect(survey.timeline.map((t) => t.paperId)[0]).toBe("arxiv:2005.11401");
    expect(survey.comparison).toHaveLength(6);
  });

  it("finds cross-paper consensus and open problems", async () => {
    const embed = hashingEmbed();
    const survey = await buildSurvey(await corpus(embed), embed);
    const kv = survey.consensus.find((c) => c.paperIds.includes("arxiv:2402.02750"));
    expect(kv?.paperIds.sort()).toEqual(["arxiv:2401.18079", "arxiv:2402.02750"]);
    expect(kv?.support).toHaveLength(2);
    expect(kv?.support.every((e) => e.quote.length > 0 && e.sentenceId)).toBe(true);

    const statements = survey.gaps.map((g) => g.statement);
    expect(statements.some((s) => s.includes("multimodal vision encoders"))).toBe(true);
    expect(statements.some((s) => s.includes("No variance across runs"))).toBe(true);
  });

  it("writes a cited review with every section and a reference for every paper", async () => {
    const embed = hashingEmbed();
    const survey = await buildSurvey(await corpus(embed), embed);
    const text = survey.narrative;
    for (const heading of ["## Overview", "## Themes", "## Methods compared", "## Results and consensus", "## Open problems and gaps", "## References"]) {
      expect(text).toContain(heading);
    }
    const body = text.slice(0, text.indexOf("## References"));
    const references = text.slice(text.indexOf("## References"));
    SPECS.forEach((spec, i) => {
      expect(body).toContain(`[${i + 1}]`);
      expect(references).toContain(`[${i + 1}]`);
      expect(references).toContain(`arXiv:${spec.id.replace("arxiv:", "")}`);
      expect(references).toContain(spec.title.replace(/\.$/, ""));
    });
    expect(text).toContain("Liu et al. [1] propose KIVI");
    expect(text).toContain("Building directly on [1]");
    expect(text).toMatch(/GSM8K is the most common testbed, used by \[1\], \[3\] and \[4\]/);
    expect(text).toContain("| Ref | Year |");
    expect(text).toContain("whereas");
    // Source citation markers and raw first person never leak into attributed prose.
    expect(body).not.toMatch(/\bWe propose\b/);
  });

  it("is deterministic", async () => {
    const embed = hashingEmbed();
    const papers = await corpus(embed);
    const a = await buildSurvey(papers, embed);
    const b = await buildSurvey(papers, embed);
    expect(a.map).toEqual(b.map);
    expect(a.narrative).toEqual(b.narrative);
    expect(a.clusters).toEqual(b.clusters);
  });

  it("falls back to TF-IDF and MDS when embeddings are missing", async () => {
    const papers = await corpus(null);
    const survey = await buildSurvey(papers, null);
    expect(survey.map.method).toBe("mds");
    expect(survey.map.points).toHaveLength(18);
    expect(survey.clusters.length).toBeGreaterThanOrEqual(2);
    const labels = survey.paperIds.map((id) => survey.clusters.find((c) => c.paperIds.includes(id))?.id ?? -1);
    expect(samePartition(labels, survey.paperIds.map((id) => THEME_OF[id]))).toBe(true);
    expect(survey.narrative).toContain("TF-IDF");
    SPECS.forEach((_, i) => expect(survey.narrative).toContain(`[${i + 1}]`));
  });

  it("uses TF-IDF for every paper when only some have embeddings", async () => {
    const embed = hashingEmbed();
    const papers = await corpus(embed);
    papers[2] = { ...papers[2], chunkEmbeddings: null };
    const survey = await buildSurvey(papers, embed);
    expect(survey.narrative).toContain("TF-IDF representations");
    expect(survey.map.method).toBe("mds");
  });

  it("works with two papers", async () => {
    const embed = hashingEmbed();
    const papers = await corpus(embed, [SPECS[0], SPECS[2]]);
    const survey = await buildSurvey(papers, embed);
    expect(survey.clusters).toHaveLength(2);
    expect(survey.silhouette).toBe(0);
    expect(survey.map.points).toHaveLength(6);
    expect(survey.map.method).toBe("mds");
    expect(survey.narrative).toContain("[1]");
    expect(survey.narrative).toContain("[2]");
    expect(survey.narrative).toContain("This review covers two papers published between 2022 and 2024.");
  });

  it("works with two papers, no embeddings and no embedder", async () => {
    const papers = await corpus(null, [SPECS[4], SPECS[5]]);
    const survey = await buildSurvey(papers, null);
    expect(survey.paperIds).toHaveLength(2);
    expect(survey.coupling).toHaveLength(0); // Self-RAG cites RAG by title, but RAG cites nothing.
    expect(survey.narrative).toContain("[2] cites [1] directly.");
  });

  it("rejects an empty selection", async () => {
    await expect(buildSurvey([], null)).rejects.toThrow(/at least one/);
  });
});

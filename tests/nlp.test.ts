import { describe, expect, it } from "vitest";
import type { EmbedFn, PaperMeta, ParsedDocument, Sentence, SectionKind } from "@/types";
import { BM25 } from "@/nlp/bm25";
import { candidatePhrases, extractKeyphrases } from "@/nlp/keyphrases";
import { lexRank, similarityMatrix } from "@/nlp/lexrank";
import { mmrSelect } from "@/nlp/mmr";
import { extractMetrics, type Metric } from "@/nlp/numbers";
import { countSyllables, readability } from "@/nlp/readability";
import { looksLikeProse, segmentAbstract, segmentDocument } from "@/nlp/segment";
import { STOPWORDS, normalizeText, splitSentences, stem, tokenize, words } from "@/nlp/text";
import { cosineSparse, tfidf } from "@/nlp/tfidf";
import { cosine, dot, meanVector, normalize } from "@/nlp/vector";
import { ZONE_PROTOTYPES, cueScores, zoneSentences } from "@/nlp/zoning";

// ───────────────────────────────── Helpers ─────────────────────────────────

/** Deterministic stand-in for the sentence encoder: hashed bag of tokens, L2-normalised. */
function hashingEmbed(dimensions = 96): EmbedFn & { calls: string[][] } {
  const calls: string[][] = [];
  const fn = async (texts: string[]): Promise<Float32Array[]> => {
    calls.push(texts);
    return texts.map((text) => {
      const v = new Float32Array(dimensions);
      for (const token of tokenize(text)) {
        let h = 2166136261;
        for (let i = 0; i < token.length; i++) h = Math.imul(h ^ token.charCodeAt(i), 16777619);
        v[Math.abs(h) % dimensions] += 1;
      }
      return normalize(v);
    });
  };
  return Object.assign(fn, { calls });
}

function sentence(text: string, sectionKind: SectionKind, n: number, sectionId = `s-${sectionKind}`): Sentence {
  return { id: `${sectionId}:${n}`, text, sectionId, sectionTitle: sectionKind, sectionKind, page: 1, index: n };
}

const metricsOf = (s: string): string[] => extractMetrics(s).map((m) => m.raw);
const find = (metrics: Metric[], raw: string): Metric | undefined => metrics.find((m) => m.raw === raw);

const KIVI_ABSTRACT = `Efficiently serving large language models (LLMs) requires batching many requests together to reduce the cost per request. Yet, with larger batch sizes and longer context lengths, the key-value (KV) cache, which stores attention keys and values to avoid re-computations, significantly increases the memory demands and becomes the new bottleneck in speed and memory usage. A straightforward and effective solution to reduce KV cache size is quantization, which decreases the total bytes taken by KV cache. However, there is a lack of in-depth studies that explore the element distribution of KV cache to understand the hardness and limitation of KV cache quantization. To fill the gap, we conducted a comprehensive study on the element distribution in KV cache of popular LLMs. Our findings indicate that the key cache should be quantized per-channel, i.e., group elements along the channel dimension and quantize them together. In contrast, the value cache should be quantized per-token. From this analysis, we developed a tuning-free 2bit KV cache quantization algorithm named KIVI. With hardware-friendly implementation, KIVI can enable Llama, Falcon, and Mistral models to maintain almost the same quality while using 2.6× less peak memory. This reduction in memory usage enables up to 4× larger batch size, bringing 2.35× to 3.47× throughput on real LLM inference workloads. Extensive experiments show that the proposed method works well, and the experimental results confirm that per-channel key cache quantization and per-token value cache quantization reduce the quantization error of the KV cache (Liu et al., 2024).`;

// ─────────────────────────────────── text ──────────────────────────────────

describe("normalizeText", () => {
  it("expands ligatures, removes invisible characters and collapses whitespace", () => {
    expect(normalizeText("  ﬁne-tuning the​ model\n\twith ﬂash attention ")).toBe(
      "fine-tuning the model with flash attention",
    );
  });
});

describe("splitSentences", () => {
  it("handles citations, abbreviations, figure and equation references, decimals, URLs and initials", () => {
    const text =
      "Large language models have shown strong results on many benchmarks (Brown et al., 2020). " +
      "However, serving them is costly, e.g. on GPUs with limited memory. " +
      "As shown in Fig. 2a, the KV cache grows linearly with the sequence length. " +
      "Following Vaswani et al. (2017), we compute attention as in Eq. (3) and Sec. 4. " +
      "The accuracy reaches 93.2% on GSM8K, approx. 3.5 points above the baseline. " +
      "Our code is available at https://github.com/jy-yuan/KIVI. " +
      "We compare against PaLM [12]. " +
      "Results are reported in Appendix B. " +
      "We also thank J. Smith and A. B. Jones for helpful comments on this draft.";
    expect(splitSentences(text)).toEqual([
      "Large language models have shown strong results on many benchmarks (Brown et al., 2020).",
      "However, serving them is costly, e.g. on GPUs with limited memory.",
      "As shown in Fig. 2a, the KV cache grows linearly with the sequence length.",
      "Following Vaswani et al. (2017), we compute attention as in Eq. (3) and Sec. 4.",
      "The accuracy reaches 93.2% on GSM8K, approx. 3.5 points above the baseline.",
      "Our code is available at https://github.com/jy-yuan/KIVI.",
      "We compare against PaLM [12].",
      "Results are reported in Appendix B.",
      "We also thank J. Smith and A. B. Jones for helpful comments on this draft.",
    ]);
  });

  it("ends a sentence after 'et al.' only when a typical sentence opener follows", () => {
    expect(
      splitSentences(
        "This approach was first introduced by Smith et al. We extend their analysis to much larger models. " +
          "The comparison vs. BERT uses the same U.S. English corpus throughout.",
      ),
    ).toEqual([
      "This approach was first introduced by Smith et al.",
      "We extend their analysis to much larger models.",
      "The comparison vs. BERT uses the same U.S. English corpus throughout.",
    ]);
  });

  it("drops footnote markers glued to the period but keeps decimals intact", () => {
    expect(
      splitSentences("The code and models are publicly available.2 We release all training scripts as well."),
    ).toEqual(["The code and models are publicly available.", "We release all training scripts as well."]);
    expect(splitSentences("The model gains 0.12 BLEU over the baseline system on WMT.")).toHaveLength(1);
  });

  it("ends sentences after reference numbers, but not after list markers", () => {
    expect(
      splitSentences("Details of the derivation are given in Eq. 3. Then we discuss the quantization error."),
    ).toEqual(["Details of the derivation are given in Eq. 3.", "Then we discuss the quantization error."]);
    expect(splitSentences("Our contributions are as follows: 1. We propose a new quantization scheme.")).toEqual([
      "Our contributions are as follows: 1. We propose a new quantization scheme.",
    ]);
  });

  it("treats blank lines and bullets as hard boundaries and drops short fragments", () => {
    const text =
      "Our contributions are summarized as follows:\n\n" +
      "• We propose a new quantization scheme for key caches\n" +
      "• We show it halves the memory footprint of all models\n\n" +
      "3.2 Setup\n\nWhy does grouping help so much in practice? It does.";
    expect(splitSentences(text)).toEqual([
      "Our contributions are summarized as follows:",
      "We propose a new quantization scheme for key caches",
      "We show it halves the memory footprint of all models",
      "Why does grouping help so much in practice?",
    ]);
  });
});

describe("tokenize, stem and words", () => {
  it("lowercases, keeps hyphenated compounds, drops stopwords and stems lightly", () => {
    expect(tokenize("The embeddings of key-value caches were trained on GPT-4's outputs")).toEqual([
      "embed",
      "key-value",
      "cache",
      "train",
      "gpt-4",
      "output",
    ]);
  });

  it("maps inflections of the same word to one readable stem", () => {
    expect(stem("studies")).toBe("study");
    expect(stem("quantized")).toBe("quantize");
    expect(stem("quantizing")).toBe("quantize");
    expect(stem("proposed")).toBe("propose");
    expect(stem("encoding")).toBe(stem("encoded"));
    expect(stem("embedding")).toBe("embed");
    expect(stem("biases")).toBe("bias");
    expect(stem("analysis")).toBe("analysis");
    expect(stem("used")).toBe("used"); // too short to strip safely
    expect(stem("treated")).toBe("treat");
  });

  it("returns raw word tokens with stopwords and case preserved", () => {
    expect(words("It's a GPT-4 model.")).toEqual(["It's", "a", "GPT-4", "model"]);
    expect(STOPWORDS.has("the")).toBe(true);
    expect(STOPWORDS.has("model")).toBe(false);
  });
});

// ────────────────────────────── vectors & ranking ──────────────────────────

describe("vector helpers", () => {
  it("computes dot products, cosines, unit vectors and centroids", () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
    expect(cosine([1, 0], [0, 1])).toBe(0);
    expect(cosine([1, 1], [2, 2])).toBeCloseTo(1, 6);
    expect(cosine([0, 0], [1, 1])).toBe(0);
    const unit = normalize(Float32Array.from([3, 4]));
    expect(Math.hypot(unit[0], unit[1])).toBeCloseTo(1, 6);
    const mean = meanVector([Float32Array.from([1, 0]), Float32Array.from([0, 1])]);
    expect(mean[0]).toBeCloseTo(Math.SQRT1_2, 5);
    expect(meanVector([])).toHaveLength(0);
  });
});

describe("BM25", () => {
  const docs = [
    ["kv", "cache", "quantization"],
    ["image", "classification", "resnet"],
    ["kv", "cache", "eviction", "policy", "kv"],
  ];

  it("ranks matching documents and gives zero to unrelated ones", () => {
    const bm25 = new BM25(docs);
    const scores = bm25.score(["kv", "cache"]);
    expect(scores[0]).toBeGreaterThan(0);
    expect(scores[2]).toBeGreaterThan(0);
    expect(scores[1]).toBe(0);
    expect(bm25.avgdl).toBeCloseTo(11 / 3, 6);
    expect(bm25.score(["quantization"])[0]).toBeGreaterThan(bm25.score(["quantization"])[2]);
  });

  it("normalises by length and survives empty corpora", () => {
    const bm25 = new BM25([["a", "b"], ["a", "b", "c", "d", "e", "f"]]);
    const [short, long] = bm25.score(["a"]);
    expect(short).toBeGreaterThan(long);
    expect(new BM25([[], []]).score(["a"])).toEqual([0, 0]);
  });
});

describe("TF-IDF", () => {
  it("builds unit-length sparse vectors with higher idf for rarer terms", () => {
    const { vectors, idf } = tfidf([["cache", "memory"], ["cache", "speed"], ["cache", "memory", "memory"]]);
    const norm = Math.sqrt([...vectors[0].values()].reduce((s, w) => s + w * w, 0));
    expect(norm).toBeCloseTo(1, 6);
    expect(idf.get("speed") ?? 0).toBeGreaterThan(idf.get("cache") ?? 0);
    expect(cosineSparse(vectors[0], vectors[0])).toBeCloseTo(1, 6);
    expect(cosineSparse(new Map([["a", 1]]), new Map([["b", 1]]))).toBe(0);
  });
});

describe("LexRank", () => {
  it("scores sum to one and the most connected sentence wins", () => {
    const vectors = [
      [1, 1, 1, 0],
      [1, 1, 0, 0],
      [0, 1, 1, 0],
      [1, 0, 1, 0],
      [0, 0, 0, 1],
    ];
    const sim = similarityMatrix(vectors);
    expect(sim[1][2]).toBe(sim[2][1]);
    expect(sim[3][3]).toBe(1);
    const scores = lexRank(sim);
    expect(scores.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    expect(scores.indexOf(Math.max(...scores))).toBe(0);
    expect(scores[4]).toBeLessThan(scores[0]);
  });

  it("is uniform when no pair reaches the threshold", () => {
    const scores = lexRank(similarityMatrix([[1, 0], [0, 1]]), { threshold: 0.5 });
    expect(scores[0]).toBeCloseTo(0.5, 9);
    expect(lexRank([])).toEqual([]);
  });
});

describe("MMR", () => {
  const items = [
    { id: "a", rel: 1, v: [1, 0] },
    { id: "a2", rel: 0.95, v: [0.99, 0.1] },
    { id: "b", rel: 0.6, v: [0, 1] },
  ];
  const sim = (x: (typeof items)[number], y: (typeof items)[number]) => cosine(x.v, y.v);

  it("trades relevance for diversity", () => {
    expect(mmrSelect(items, (t) => t.rel, sim, 2, 0.5).map((t) => t.id)).toEqual(["a", "b"]);
    expect(mmrSelect(items, (t) => t.rel, sim, 2, 1).map((t) => t.id)).toEqual(["a", "a2"]);
    expect(mmrSelect(items, (t) => t.rel, sim, 10)).toHaveLength(3);
    expect(mmrSelect(items, (t) => t.rel, sim, 0)).toEqual([]);
  });
});

// ──────────────────────────────── keyphrases ───────────────────────────────

describe("keyphrases", () => {
  it("proposes noun-phrase-like candidates without digits-only or stopword spans", () => {
    const candidates = candidatePhrases(KIVI_ABSTRACT);
    const phrases = candidates.map((c) => c.phrase);
    expect(phrases).toContain("KV cache");
    expect(phrases.some((p) => /^\d+$/.test(p))).toBe(false);
    expect(phrases.some((p) => /^(?:the|of|and|we)\b/i.test(p))).toBe(false);
    const kv = candidates.find((c) => c.phrase === "KV cache");
    expect(kv?.count).toBeGreaterThanOrEqual(5);
    // Sorted by first appearance.
    expect(candidates.every((c, i) => i === 0 || candidates[i - 1].firstPos <= c.firstPos)).toBe(true);
  });

  it("extracts clean scientific phrases and filters boilerplate (statistical mode)", async () => {
    const keyphrases = await extractKeyphrases(KIVI_ABSTRACT, { embed: null });
    const phrases = keyphrases.map((k) => k.phrase.toLowerCase());
    expect(keyphrases.length).toBeGreaterThan(4);
    expect(keyphrases.length).toBeLessThanOrEqual(12);
    expect(phrases.some((p) => p.includes("kv cache"))).toBe(true);
    expect(phrases.some((p) => /value cache|key cache/.test(p))).toBe(true);
    for (const generic of ["proposed method", "experimental results", "et al", "extensive experiments"]) {
      expect(phrases).not.toContain(generic);
    }
    expect(phrases.every((p) => !/\bliu\b/.test(p))).toBe(true); // citation names are stripped
    expect(keyphrases.every((k) => k.score >= 0 && k.score <= 1)).toBe(true);
  });

  it("merges near-duplicates: inflections, bare heads and acronym expansions", async () => {
    const text =
      "The key-value cache dominates memory. We compress the KV cache with quantization. " +
      "Quantization of the KV cache reduces memory. Attention heads attend to the key-value cache. " +
      "Each attention head reads the KV cache. Quantization error grows for outlier channels. " +
      "The quantization error of outlier channels is large. Cache compression helps attention heads.";
    const phrases = (await extractKeyphrases(text, { top: 8 })).map((k) => k.phrase.toLowerCase());
    const kvVariants = phrases.filter((p) => p === "kv cache" || p === "key-value cache" || p === "cache");
    expect(kvVariants).toHaveLength(1);
    expect(phrases.filter((p) => /^attention heads?$/.test(p))).toHaveLength(1);
    expect(phrases).toContain("quantization error");
  });

  it("blends embedding relevance when a model is available and falls back when it fails", async () => {
    const embed = hashingEmbed();
    const withEmbed = await extractKeyphrases(KIVI_ABSTRACT, { embed, top: 6 });
    expect(withEmbed).toHaveLength(6);
    expect(embed.calls.length).toBe(1);
    const failing: EmbedFn = async () => {
      throw new Error("model unavailable");
    };
    const fallback = await extractKeyphrases(KIVI_ABSTRACT, { embed: failing });
    expect(fallback).toEqual(await extractKeyphrases(KIVI_ABSTRACT, { embed: null }));
    expect(await extractKeyphrases("", {})).toEqual([]);
  });
});

// ──────────────────────────────── readability ──────────────────────────────

describe("readability", () => {
  it("counts syllables with the usual English heuristics", () => {
    expect(countSyllables("cat")).toBe(1);
    expect(countSyllables("table")).toBe(2);
    expect(countSyllables("large")).toBe(1);
    expect(countSyllables("quantization")).toBe(4);
    expect(countSyllables("trained")).toBe(1);
    expect(countSyllables("added")).toBe(2);
  });

  it("scores plain prose as easier than dense academic prose", () => {
    const plain = readability("The cat sat on the mat. It was a warm day and the sun was out. The dog ran to the park.");
    const dense = readability(KIVI_ABSTRACT);
    expect(plain.fleschReadingEase).toBeGreaterThan(dense.fleschReadingEase);
    expect(dense.avgSyllablesPerWord).toBeGreaterThan(plain.avgSyllablesPerWord);
    expect(plain.words).toBe(22);
    expect(readability("")).toEqual({ fleschReadingEase: 0, avgSentenceLength: 0, avgSyllablesPerWord: 0, words: 0 });
  });
});

// ───────────────────────────────── metrics ─────────────────────────────────

describe("extractMetrics", () => {
  it("reads percentages, multipliers and named scores with direction and metric", () => {
    const kivi = extractMetrics("KIVI reduces peak memory usage by 2.6× and increases throughput by 3.47× (Liu et al., 2024).");
    expect(kivi.map((m) => m.raw)).toEqual(["2.6×", "3.47×"]);
    expect(kivi[0]).toMatchObject({ value: 2.6, unit: "x", metric: "memory", direction: "down" });
    expect(kivi[1]).toMatchObject({ value: 3.47, unit: "x", metric: "throughput", direction: "up" });

    const gsm = extractMetrics("Our model achieves 93.2% accuracy on GSM8K, outperforming GPT-3 by 4.1 points [12].");
    expect(find(gsm, "93.2%")).toMatchObject({ value: 93.2, unit: "%", metric: "accuracy" });
    expect(find(gsm, "4.1 points")).toMatchObject({ unit: "points", direction: "up" });

    const bleu = extractMetrics("We obtain +4.1 BLEU over the baseline and an F1 of 88.5 on SQuAD.");
    expect(find(bleu, "+4.1 BLEU")).toMatchObject({ value: 4.1, metric: "BLEU", direction: "up" });
    expect(find(bleu, "88.5")).toMatchObject({ metric: "F1" });

    expect(extractMetrics("It uses 2.6× less memory than PPO.")[0]).toMatchObject({ metric: "memory", direction: "down" });
    expect(extractMetrics("DPO achieves a win rate of 61.0 against the reference.")[0]).toMatchObject({
      value: 61,
      metric: "win rate",
    });
  });

  it("handles signs, uncertainty, percentage points, time and memory units", () => {
    const drop = extractMetrics("Accuracy drops by −2.3% while latency falls from 120 ms to 45 ms.");
    expect(drop.map((m) => m.raw)).toEqual(["−2.3%", "120 ms", "45 ms"]);
    expect(drop[0].value).toBe(-2.3);
    expect(drop[1]).toMatchObject({ unit: "ms", metric: "latency", direction: "down" });

    const std = extractMetrics("The model reaches 85.3 ± 0.4 accuracy, an improvement of 3.2 pp over the baseline.");
    expect(std[0]).toMatchObject({ value: 85.3, metric: "accuracy" });
    expect(std[0].raw).toContain("± 0.4");
    expect(std[1]).toMatchObject({ value: 3.2, unit: "pp", direction: "up" });
    expect(std).toHaveLength(2);

    expect(extractMetrics("Peak usage stays at 1.5 GB of memory.")[0]).toMatchObject({ unit: "GB", value: 1.5, metric: "memory" });
    expect(metricsOf("It is 3 times faster and costs 10-20% less.")).toEqual(["3 times", "10-20%"]);
    expect(extractMetrics("Perplexity improves from 5.68 to 5.12 on WikiText.").map((m) => m.metric)).toEqual([
      "perplexity",
      "perplexity",
    ]);
  });

  it("reads benchmark-anchored scores and model or context sizes", () => {
    const mistral = extractMetrics("It achieves 60.1% on MMLU and 52.2 on HumanEval with 32K context length.");
    expect(find(mistral, "60.1%")?.metric).toBe("MMLU");
    expect(find(mistral, "52.2")?.metric).toBe("HumanEval");
    expect(find(mistral, "32K")).toMatchObject({ unit: "K", value: 32 });
    expect(find(extractMetrics("A 13B model with 175B parameters."), "175B")).toMatchObject({ unit: "B", metric: "parameters" });
  });

  it("ignores years, citations, cross-references, identifiers and settings", () => {
    expect(extractMetrics("As shown in Table 2 and Figure 3, results in Section 4.2 confirm Eq. (3) from 2023.")).toEqual([]);
    expect(extractMetrics("We train for 3 epochs with a batch size of 512 on 8 GPUs using 4-bit quantization.")).toEqual([]);
    expect(extractMetrics("The 1990s saw 5-fold growth; see Appendix B.1 and arXiv:2402.02750 [12, 13].")).toEqual([]);
    expect(extractMetrics("A point cloud with 1000 points and a 3 × 3 kernel, as in Smith et al. (2021).")).toEqual([]);
    expect(extractMetrics("GPT-4 and Llama-2-7B use pass@1 and top-5 decoding, cf. Fig. 2a.")).toEqual([]);
    expect(metricsOf("Mistral 7B outperforms Llama 2 13B by 4.1 points.")).toEqual(["4.1 points"]);
    expect(metricsOf("In 2023, accuracy rose to 85% (Brown et al., 2020; Zhang, 2022).")).toEqual(["85%"]);
  });
});

// ───────────────────────────────── segment ─────────────────────────────────

describe("segmentation", () => {
  const doc: ParsedDocument = {
    pageCount: 3,
    pagesParsed: 3,
    pages: [
      {
        page: 1,
        text: "Abstract We study quantization of the key-value cache in large models. Introduction Serving language models is expensive because the cache grows with every token.",
        charCount: 0,
      },
      {
        page: 2,
        text: "Batching many requests makes the memory bottleneck even worse in practice. Model 7B 13B 70B 35.1 46.9 63.4 0.5 0.7 0.9",
        charCount: 0,
      },
      { page: 3, text: "References [1] A. Author. A paper title. 2020.", charCount: 0 },
    ],
    sections: [
      { id: "s0", title: "Abstract", level: 1, kind: "abstract", pageStart: 1, pageEnd: 1, text: "We study quantization of the key-value cache in large models." },
      {
        id: "s1",
        title: "Introduction",
        level: 1,
        kind: "introduction",
        pageStart: 1,
        pageEnd: 2,
        text:
          "Serving language models is expensive because the cache grows with every token. " +
          "Batching many requests makes the memory bottleneck even worse in practice.\n\n" +
          "Model 7B 13B 70B 35.1 46.9 63.4 0.5 0.7 0.9",
      },
      { id: "s2", title: "References", level: 1, kind: "references", pageStart: 3, pageEnd: 3, text: "[1] A. Author. A paper title published somewhere. 2020." },
    ],
    references: [],
    captions: [],
    links: [],
    warnings: [],
    quality: "good",
  };

  it("splits sections into located sentences, skipping references and non-prose", () => {
    const sentences = segmentDocument(doc);
    expect(sentences.map((s) => s.id)).toEqual(["s0:0", "s1:0", "s1:1"]);
    expect(sentences.map((s) => s.index)).toEqual([0, 1, 2]);
    expect(sentences.map((s) => s.page)).toEqual([1, 1, 2]);
    expect(sentences[2]).toMatchObject({ sectionKind: "introduction", sectionTitle: "Introduction" });
    expect(looksLikeProse("Model 7B 13B 70B 35.1 46.9 63.4 0.5 0.7 0.9")).toBe(false);
  });

  it("falls back to pages when no sections were detected", () => {
    const sentences = segmentDocument({ ...doc, sections: [] });
    expect(sentences[0]).toMatchObject({ sectionId: "p1", sectionKind: "other", page: 1 });
  });

  it("segments a metadata abstract in abstract-only mode", () => {
    const meta: PaperMeta = {
      id: "arxiv:2402.02750",
      title: "KIVI",
      authors: [],
      abstract: KIVI_ABSTRACT,
      source: "datacite",
    };
    const sentences = segmentAbstract(meta);
    expect(sentences.length).toBeGreaterThan(8);
    expect(sentences[0]).toMatchObject({ id: "abstract:0", sectionKind: "abstract", page: 1, index: 0 });
    expect(segmentAbstract({ ...meta, abstract: undefined })).toEqual([]);
  });
});

// ────────────────────────────────── zoning ─────────────────────────────────

describe("argumentative zoning", () => {
  const top = (text: string): string => {
    const scores = cueScores(text);
    return Object.entries(scores).sort((a, b) => b[1] - a[1])[0][0];
  };

  it("ships 5–8 prototype sentences for each zone", () => {
    const zones = Object.keys(ZONE_PROTOTYPES);
    expect(zones).toHaveLength(7);
    expect(zones).not.toContain("other");
    for (const prototypes of Object.values(ZONE_PROTOTYPES)) {
      expect(prototypes.length).toBeGreaterThanOrEqual(5);
      expect(prototypes.length).toBeLessThanOrEqual(8);
    }
  });

  it("recognises the stock phrases of each rhetorical move", () => {
    expect(top("In this paper, we propose a novel framework for streaming inference.")).toBe("contribution");
    expect(top("Our method outperforms all baselines and improves accuracy by 4.1 points.")).toBe("result");
    expect(top("A limitation of our approach is that it is limited to decoder-only models.")).toBe("limitation");
    expect(top("In future work, we plan to extend the method to multimodal models.")).toBe("future");
    expect(top("However, existing methods suffer from high memory cost and remain challenging to deploy.")).toBe("problem");
    expect(top("Prior work has extensively studied attention sparsity (Child et al., 2019; Beltagy et al., 2020).")).toBe("background");
    expect(top("We set the learning rate to 1e-4 and the batch size is set to 64.")).toBe("method");
    expect(top("The rest of this paper is organized as follows.")).toBe("other");
    const scores = cueScores("We propose KIVI.");
    expect(Object.values(scores).every((v) => v >= 0 && v < 1)).toBe(true);
  });

  it("turns deficiency language about the authors' own work into a limitation", () => {
    const own = cueScores("Our method fails to handle very long inputs.");
    const others = cueScores("Existing methods fail to handle very long inputs.");
    expect(own.limitation).toBeGreaterThan(own.problem);
    expect(others.problem).toBeGreaterThan(others.limitation);
  });

  const paper: Sentence[] = [
    sentence("Large language models have been widely adopted for many applications.", "introduction", 0, "s1"),
    sentence("However, serving them remains challenging because the cache is a bottleneck.", "introduction", 1, "s1"),
    sentence("Our main contributions are summarized as follows:", "introduction", 2, "s1"),
    sentence("A tuning-free two-bit quantization algorithm for the cache.", "introduction", 3, "s1"),
    sentence("The cache tensors are quantized per-channel along the hidden dimension.", "method", 4, "s2"),
    sentence("KIVI outperforms all baselines while using 2.6× less memory.", "results", 5, "s3"),
    sentence("The rest of this paper is organized as follows.", "other", 6, "s4"),
    sentence("In future work, we plan to support other model families.", "conclusion", 7, "s5"),
  ];

  it("labels sentences with calibrated probabilities (lexical only)", async () => {
    const zoned = await zoneSentences(paper, null);
    expect(zoned.map((z) => z.zone)).toEqual([
      "background",
      "problem",
      "contribution",
      "contribution",
      "method",
      "result",
      "other",
      "future",
    ]);
    for (const z of zoned) {
      const total = Object.values(z.scores).reduce((a, b) => a + b, 0);
      expect(total).toBeCloseTo(1, 6);
      expect(z.confidence).toBeGreaterThanOrEqual(0);
      expect(z.confidence).toBeLessThanOrEqual(1);
    }
    expect(zoned[3].id).toBe(paper[3].id);
  });

  it("uses prototype embeddings when available, embedding prototypes once per model", async () => {
    const embed = hashingEmbed();
    const first = await zoneSentences(paper, embed);
    await zoneSentences(paper.slice(0, 3), embed);
    const prototypeCalls = embed.calls.filter((texts) => texts.includes(ZONE_PROTOTYPES.result[0]));
    expect(prototypeCalls).toHaveLength(1);
    expect(first).toHaveLength(paper.length);
    expect(first[5].zone).toBe("result");
    expect(first[7].zone).toBe("future");
  });

  it("degrades to lexical zoning when the embedding model fails", async () => {
    const failing: EmbedFn = async () => {
      throw new Error("offline");
    };
    const zoned = await zoneSentences(paper, failing);
    expect(zoned.map((z) => z.zone)).toEqual((await zoneSentences(paper, null)).map((z) => z.zone));
    expect(await zoneSentences([], null)).toEqual([]);
  });
});

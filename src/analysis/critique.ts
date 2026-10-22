/**
 * Rule-based peer review.
 *
 * Each rule encodes one question an experienced reviewer asks ("is there
 * variance?", "which component drives the gain?", "is the code available?")
 * as an explicit test over the extracted signals. A rule either cites the
 * sentences that triggered it or states exactly what was searched for and
 * not found, so every finding is auditable. Severity follows how much the
 * gap undermines the paper's central claims: missing variance or ablations
 * on an empirical method paper is major, a thin bibliography is minor.
 * Strengths are first-class findings, because a fair review says what a
 * paper does well too.
 *
 * In abstract-only mode the body checks cannot run, so the critique says so
 * up front and only judges what the abstract itself claims.
 */
import type { Critique, CritiqueFinding, Evidence, Severity, ZonedSentence } from "@/types";
import { extractMetrics } from "@/nlp/numbers";
import { BOOSTERS, DATASETS, METRICS, datasetKey } from "./lexicons";
import type { AnalyzeInput, PaperSignals, SampleSize } from "./signals";
import { sentenceEvidence } from "./signals";

/** A finding plus the short noun phrase the verdict uses to mention it. */
interface Draft extends CritiqueFinding {
  gist: string;
}

const SEVERITY_ORDER: Record<Severity, number> = { major: 0, moderate: 1, minor: 2, strength: 3 };

// ───────────────────────────────── Phrasing ─────────────────────────────────

export function plural(n: number, word: string, many = `${word}s`): string {
  return `${n} ${n === 1 ? word : many}`;
}

/** Verb agreement for counted subjects: "1 passage reports", "3 passages report". */
const agree = (n: number, one: string, many: string) => (n === 1 ? one : many);

/** "A", "A and B", "A, B and C", "A, B, C and 2 more". */
export function listNames(names: string[], max = 4): string {
  if (names.length === 0) return "";
  const shown = names.slice(0, max);
  const rest = names.length - shown.length;
  if (rest > 0) return `${shown.join(", ")} and ${rest} more`;
  if (shown.length === 1) return shown[0];
  return `${shown.slice(0, -1).join(", ")} and ${shown[shown.length - 1]}`;
}

function shortQuote(text: string, maxWords = 16): string {
  const tokens = text.trim().split(/\s+/);
  return tokens.length <= maxWords ? text.trim() : `${tokens.slice(0, maxWords).join(" ")}…`;
}

const methodLabel = (s: PaperSignals) => s.methodName ?? "the proposed method";

// ───────────────────────────────── Context ──────────────────────────────────

interface ReviewContext {
  input: AnalyzeInput;
  s: PaperSignals;
  /** The paper reports its own quantitative results. */
  empirical: boolean;
  /** The paper proposes something whose parts can be ablated. */
  proposesMethod: boolean;
  resultExamples: ZonedSentence[];
}

function reviewContext(input: AnalyzeInput, s: PaperSignals): ReviewContext {
  const withNumbers = input.sentences
    .filter((x) => x.zone === "result")
    .map((x) => ({ x, n: extractMetrics(x.text).length }))
    .filter(({ n }) => n > 0)
    .sort((a, b) => b.n - a.n)
    .map(({ x }) => x);
  const empirical = s.fullText
    ? s.resultSentences >= 3 || (s.metrics.length > 0 && s.datasets.length > 0)
    : s.resultSentences >= 1 || s.datasets.length > 0 || withNumbers.length > 0;
  return {
    input,
    s,
    empirical,
    proposesMethod: s.contributionSentences > 0 || s.methodName !== undefined,
    resultExamples: withNumbers.slice(0, 2),
  };
}

const evidenceOf = (ctx: ReviewContext, sentences: ZonedSentence[]) => sentences.map((x) => sentenceEvidence(ctx.input, x));

/** Smallest positive improvement stated in points or percentage points, to show what variance would need to rule out. */
function smallestGain(ctx: ReviewContext): { raw: string; sentence: ZonedSentence } | null {
  let best: { raw: string; value: number; sentence: ZonedSentence } | null = null;
  for (const sentence of ctx.input.sentences) {
    if (sentence.zone !== "result") continue;
    for (const m of extractMetrics(sentence.text)) {
      const isDelta = m.unit === "pp" || m.unit === "points" || (m.direction === "up" && /^\+/.test(m.raw));
      if (!isDelta || m.value <= 0) continue;
      if (!best || m.value < best.value) best = { raw: m.raw, value: m.value, sentence };
    }
  }
  return best;
}

// ─────────────────────────────── Full-text rules ────────────────────────────

type Rule = (ctx: ReviewContext) => Draft[];

const varianceRule: Rule = (ctx) => {
  const { s } = ctx;
  if (s.statistics.length > 0) {
    return [{
      id: "variance-reported",
      category: "statistics",
      severity: "strength",
      title: "Reports variance or significance",
      rationale: `${plural(s.statistics.length, "passage")} ${agree(s.statistics.length, "reports", "report")} spread or significance (±, standard deviations, confidence intervals or tests), so readers can judge whether differences exceed run-to-run noise.`,
      evidence: s.statistics.slice(0, 3),
      gist: "reported variance",
    }];
  }
  if (!s.fullText || !ctx.empirical) return [];
  if (s.seedsOrRuns.length > 0) {
    return [{
      id: "variance-hidden",
      category: "statistics",
      severity: "moderate",
      title: "Repeated runs, but their spread is not reported",
      rationale: `The paper mentions repeated runs or seeds (${plural(s.seedsOrRuns.length, "sentence")}) but no standard deviation, confidence interval or significance test was found, so the variability across those runs stays invisible.`,
      evidence: s.seedsOrRuns.slice(0, 2),
      gist: "unreported spread across runs",
    }];
  }
  const gain = smallestGain(ctx);
  const tables = s.tables > 0 ? ` and ${plural(s.tables, "table")}` : "";
  const margin = gain
    ? ` The smallest reported gain is ${gain.raw}; without run-to-run spread there is no way to tell whether it exceeds training noise.`
    : " Without run-to-run spread there is no way to tell whether the reported margins exceed training noise.";
  return [{
    id: "no-variance",
    category: "statistics",
    severity: "major",
    title: "Results reported as single numbers, without variance",
    rationale: `No standard deviations, confidence intervals, significance tests or seed counts were found in ${plural(s.resultSentences, "result sentence")}${tables}.${margin}`,
    evidence: evidenceOf(ctx, gain ? [gain.sentence, ...ctx.resultExamples.filter((x) => x !== gain.sentence)].slice(0, 2) : ctx.resultExamples),
    gist: "unreported variance",
  }];
};

const ablationRule: Rule = (ctx) => {
  const { s } = ctx;
  if (s.ablation.length >= 3 || s.ablationSection) {
    return [{
      id: "ablations",
      category: "evaluation",
      severity: "strength",
      title: "Design choices are ablated",
      rationale: `${plural(s.ablation.length, "passage")} ${agree(s.ablation.length, "isolates", "isolate")} components or sweep settings${s.ablationSection ? ", including a dedicated ablation section" : ""}, so the contribution of individual design choices is measured rather than assumed.`,
      evidence: s.ablation.slice(0, 3),
      gist: "ablations of its design choices",
    }];
  }
  if (!s.fullText || !ctx.empirical || !ctx.proposesMethod) return [];
  if (s.ablation.length > 0) {
    return [{
      id: "thin-ablation",
      category: "evaluation",
      severity: "minor",
      title: "Ablations are thin",
      rationale: `Only ${plural(s.ablation.length, "sentence")} ${agree(s.ablation.length, "describes", "describe")} an ablation or sensitivity analysis; most design choices of ${methodLabel(s)} appear untested in isolation.`,
      evidence: s.ablation,
      gist: "thin ablations",
    }];
  }
  return [{
    id: "no-ablation",
    category: "evaluation",
    severity: s.resultSentences >= 6 ? "major" : "moderate",
    title: "No ablation of the method's components",
    rationale: `No ablation study, "w/o X" variant, component removal or sensitivity sweep was found in ${plural(ctx.input.sentences.length, "sentence")} or in any section title. It remains unclear which part of ${methodLabel(s)} is responsible for the reported gains.`,
    evidence: [],
    gist: "missing ablations",
  }];
};

/** Benchmarks built from web text, where pre-training contamination is a real risk. */
const TEXT_DOMAINS = new Set(["language", "reasoning", "code", "qa", "dialogue", "safety", "long-context", "science"]);

function domainOf(name: string): string | undefined {
  const key = datasetKey(name);
  return DATASETS.find((d) => datasetKey(d.name) === key)?.domain;
}

function datasetDomains(names: string[]): string[] {
  const byKey = new Map(DATASETS.map((d) => [datasetKey(d.name), d.domain]));
  return [...new Set(names.map((n) => byKey.get(datasetKey(n))).filter((d): d is NonNullable<typeof d> => d !== undefined))];
}

const datasetRule: Rule = (ctx) => {
  const { s } = ctx;
  const names = s.datasets.map((d) => d.name);
  const domains = datasetDomains(names);
  if (names.length >= 4 || (names.length >= 3 && domains.length >= 2)) {
    return [{
      id: "broad-evaluation",
      category: "scope",
      severity: "strength",
      title: "Broad evaluation across benchmarks",
      rationale: `Evaluated on ${plural(names.length, "recognised benchmark")}${domains.length > 1 ? ` spanning ${listNames(domains)}` : ""}: ${listNames(names, 6)}.`,
      evidence: s.datasets.slice(0, 3).map((d) => d.evidence),
      gist: `broad benchmark coverage (${plural(names.length, "benchmark")})`,
    }];
  }
  if (!s.fullText || !ctx.empirical) return [];
  const general = s.generalityClaims[0];
  if (names.length === 0) {
    return [{
      id: "no-benchmark",
      category: "scope",
      severity: "moderate",
      title: "No standard benchmark recognised",
      rationale: `None of ~290 widely used benchmarks, and no "<Name> dataset/benchmark" phrase, was found outside related work. The evaluation may rely on custom or unnamed data, which makes the results hard to compare with other work.`,
      evidence: general ? [general] : [],
      gist: "the lack of a recognisable benchmark",
    }];
  }
  if (names.length === 1 || (names.length === 2 && general)) {
    const claim = general ? ` Yet the paper claims broad applicability (“${shortQuote(general.quote)}”).` : "";
    return [{
      id: "narrow-evaluation",
      category: "scope",
      severity: names.length === 1 && general ? "major" : "moderate",
      title: names.length === 1 ? "Evaluation rests on a single benchmark" : "Generality claimed from two benchmarks",
      rationale: `All recognised quantitative evidence comes from ${listNames(names)}.${claim} Results on so few benchmarks can reflect their idiosyncrasies rather than a general improvement.`,
      evidence: [...s.datasets.map((d) => d.evidence), ...(general ? [general] : [])].slice(0, 3),
      gist: names.length === 1 ? "a single-benchmark evaluation" : "generality claims on two benchmarks",
    }];
  }
  return [];
};

const baselineRule: Rule = (ctx) => {
  const { s } = ctx;
  const systems = s.comparedSystems;
  if (systems.length >= 4 && s.baselineMentions.length >= 3) {
    return [{
      id: "baselines",
      category: "evaluation",
      severity: "strength",
      title: "Compared against several systems",
      rationale: `${plural(s.baselineMentions.length, "comparison sentence")} ${agree(s.baselineMentions.length, "mentions", "mention")} ${systems.length} named systems or models (${listNames(systems, 5)}; detected heuristically), so the results are measured against a field of alternatives.`,
      evidence: s.baselineMentions.slice(0, 2),
      gist: "comparisons against several systems",
    }];
  }
  if (!s.fullText || !ctx.empirical) return [];
  if (s.baselineMentions.length === 0) {
    return [{
      id: "no-baselines",
      category: "evaluation",
      severity: "major",
      title: "No comparison against baselines",
      rationale: `No sentence outside related work compares the method with a baseline or prior system (searched for "baseline", "compared with/to/against", "outperforms", "vs."), so the results lack a reference point.`,
      evidence: [],
      gist: "the absence of baselines",
    }];
  }
  if (s.baselineMentions.length <= 2 && systems.length <= 1) {
    return [{
      id: "few-baselines",
      category: "evaluation",
      severity: "moderate",
      title: "Few baselines",
      rationale: `Only ${plural(s.baselineMentions.length, "sentence")} ${agree(s.baselineMentions.length, "compares", "compare")} against other systems${systems.length ? ` (named: ${listNames(systems)})` : ""}. Without the strongest recent alternatives under the same budget, the size of the improvement is hard to judge.`,
      evidence: s.baselineMentions,
      gist: "thin baselines",
    }];
  }
  return [];
};

const claimsRule: Rule = (ctx) => {
  const { s } = ctx;
  const out: Draft[] = [];
  if (s.sotaClaims.length > 0 && s.statistics.length === 0) {
    out.push({
      id: "sota-without-stats",
      category: "claims",
      severity: s.fullText ? "moderate" : "minor",
      title: s.fullText ? "State-of-the-art claims without significance testing" : "State-of-the-art claim to verify in the full text",
      rationale: s.fullText
        ? `The paper claims state-of-the-art or superiority over prior work ${plural(s.sotaClaims.length, "time")} but reports no variance or significance test, so it is unclear whether the margins are larger than run-to-run noise.`
        : `The abstract claims state-of-the-art or superiority over prior work; whether the margins are backed by variance or significance tests can only be checked in the full text.`,
      evidence: s.sotaClaims.slice(0, 3),
      gist: "state-of-the-art claims without significance tests",
    });
  }
  if (s.unbackedSignificance.length > 0 && s.statistics.length === 0 && s.fullText) {
    out.push({
      id: "significance-wording",
      category: "statistics",
      severity: "minor",
      title: "“Significantly” used without a statistical test",
      rationale: `${plural(s.unbackedSignificance.length, "claim sentence")} ${agree(s.unbackedSignificance.length, "describes", "describe")} improvements as significant or substantial, but no p-value, confidence interval or test is reported anywhere; readers will take the word as a statistical claim.`,
      evidence: s.unbackedSignificance.slice(0, 2),
      gist: "untested “significant” claims",
    });
  }
  const rate = s.boosters / Math.max(1, s.claimSentences);
  // About one intensifier in eight claim sentences is already noticeably promotional.
  if (s.boosters >= 2 && (rate >= 0.12 || s.boosters >= 5)) {
    const wordsUsed = [...new Set(s.boosterExamples.flatMap((e) => [...e.quote.matchAll(BOOSTERS)].map((m) => m[0].toLowerCase())))];
    const examples = wordsUsed.length ? ` (e.g. ${wordsUsed.slice(0, 3).map((w) => `“${w}”`).join(", ")})` : "";
    out.push({
      id: "overclaiming",
      category: "claims",
      severity: s.boosters >= 3 && rate >= 0.12 && s.fullText ? "moderate" : "minor",
      title: "Promotional language in claims",
      rationale: `${plural(s.boosters, "intensifier")}${examples} ${agree(s.boosters, "appears", "appear")} across ${plural(s.claimSentences, "claim sentence")} (${Math.round(rate * 100)}%), against ${plural(s.hedges, "hedge")} in the whole text. Strong wording should be matched by correspondingly strong evidence.`,
      evidence: s.boosterExamples.slice(0, 3),
      gist: "promotional wording",
    });
  }
  return out;
};

/** Evidence for a link that sits in a footnote rather than a sentence: its page and the URL itself. */
function linkOnPage(input: AnalyzeInput, url: string): Evidence[] {
  const doc = input.doc;
  const core = url.replace(/^https?:\/\//, "").toLowerCase();
  const page = doc?.pages.find((p) => p.text.toLowerCase().includes(core));
  if (!doc || !page) return [];
  const section = doc.sections.find((x) => x.pageStart <= page.page && page.page <= x.pageEnd && x.kind !== "references");
  return [{ paperId: input.meta.id, page: page.page, section: section?.title ?? `Page ${page.page}`, quote: url }];
}

const codeRule: Rule = (ctx) => {
  const { s, input } = ctx;
  if (s.codeLinks.length > 0) {
    const linkSentence = input.sentences.find((x) => s.codeLinks.some((u) => x.text.includes(u.replace(/^https?:\/\//, ""))));
    const linkEvidence = linkSentence ? [sentenceEvidence(input, linkSentence)] : linkOnPage(input, s.codeLinks[0]);
    return [{
      id: "code-released",
      category: "reproducibility",
      severity: "strength",
      title: "Code is linked",
      rationale: `Repository links found in the paper body (bibliography links excluded): ${listNames(s.codeLinks, 3)}.`,
      evidence: linkEvidence.length ? linkEvidence : s.codeStatements.slice(0, 1),
      gist: "released code",
    }];
  }
  if (!s.fullText) return [];
  if (s.codeStatements.length > 0) {
    return [{
      id: "code-promised",
      category: "reproducibility",
      severity: "minor",
      title: "Code release promised, but no link found",
      rationale: "The authors state that code or models are (or will be) released, but no repository URL was found in the text, so availability cannot be verified.",
      evidence: s.codeStatements.slice(0, 2),
      gist: "an unverified code release",
    }];
  }
  if (!ctx.empirical) return [];
  return [{
    id: "no-code",
    category: "reproducibility",
    severity: "moderate",
    title: "No code or model release found",
    rationale: `No link to the authors' own code (GitHub, GitLab, Hugging Face, …) and no release statement was found outside the bibliography${s.externalCodeLinks.length ? `; the ${plural(s.externalCodeLinks.length, "repository link")} present ${agree(s.externalCodeLinks.length, "points", "point")} to third-party code or data` : ""}.${s.training.length ? " Because the method involves training, re-implementing it from the text alone is costly and error-prone." : ""}`,
    evidence: [],
    gist: "the absence of released code",
  }];
};

const computeRule: Rule = (ctx) => {
  const { s } = ctx;
  if (!s.fullText) return [];
  const out: Draft[] = [];
  if (s.compute.length >= 1 && s.hyperparameters.length >= 3) {
    out.push({
      id: "setup-documented",
      category: "reproducibility",
      severity: "strength",
      title: "Experimental setup is documented",
      rationale: `Hardware or compute is stated (${plural(s.compute.length, "passage")}) together with optimisation or decoding settings (${plural(s.hyperparameters.length, "passage")}), which is what a replication needs.`,
      evidence: [s.compute[0], s.hyperparameters[0]],
      gist: "a documented experimental setup",
    });
    return out;
  }
  const efficiency = s.efficiencyClaims.length > 0;
  if ((s.training.length > 0 || efficiency) && s.compute.length === 0) {
    out.push({
      id: "no-compute",
      category: "reproducibility",
      severity: "moderate",
      title: efficiency ? "Efficiency claims without hardware details" : "Training compute not reported",
      rationale: efficiency
        ? `The paper makes speed or memory claims (${plural(s.efficiencyClaims.length, "claim sentence")}) but no GPU/TPU model, device count, accelerator-hours or timing setup was found, so the numbers cannot be interpreted or reproduced.`
        : `Models are trained or fine-tuned, yet no hardware, accelerator-hours or training time was found (searched for GPU/TPU models, device counts, "GPU hours", wall-clock).`,
      evidence: (efficiency ? s.efficiencyClaims : s.training).slice(0, 2),
      gist: efficiency ? "efficiency numbers without hardware" : "unreported compute",
    });
  }
  if (s.training.length > 0 && s.hyperparameters.length === 0) {
    out.push({
      id: "no-hyperparameters",
      category: "reproducibility",
      severity: "moderate",
      title: "Training hyperparameters not reported",
      rationale: "The method is trained, but no learning rate, batch size, optimiser, schedule or epoch count was found, so the reported numbers cannot be reproduced from the paper.",
      evidence: s.training.slice(0, 2),
      gist: "unreported hyperparameters",
    });
  }
  return out;
};

const limitationsRule: Rule = (ctx) => {
  const { s } = ctx;
  if (s.limitationsSection && s.authorLimitations.length >= 2) {
    return [{
      id: "candid-limitations",
      category: "scope",
      severity: "strength",
      title: "Candid limitations discussion",
      rationale: `A dedicated limitations section with ${plural(s.authorLimitations.length, "admission")} of the method's own weaknesses tells readers where not to rely on it.`,
      evidence: s.authorLimitations.slice(0, 2),
      gist: "a candid limitations section",
    }];
  }
  if (!s.fullText) return [];
  if (s.authorLimitations.length === 0 && !s.limitationsSection) {
    return [{
      id: "no-limitations",
      category: "scope",
      severity: "moderate",
      title: "No discussion of limitations",
      rationale: `No limitations section and no sentence where the authors acknowledge a weakness of their own method was found (searched for "limitation", "fails to", "we only evaluate", "beyond the scope", …). Readers are left to infer where the method breaks.`,
      evidence: [],
      gist: "the absence of a limitations discussion",
    }];
  }
  if (s.authorLimitations.length === 1 && !s.limitationsSection) {
    return [{
      id: "limitations-in-passing",
      category: "scope",
      severity: "minor",
      title: "Limitations acknowledged only in passing",
      rationale: "A single sentence concedes a weakness and there is no limitations section, so the boundaries of the method are only sketched.",
      evidence: s.authorLimitations,
      gist: "limitations mentioned only in passing",
    }];
  }
  return [];
};

/** Half-width of a 95% confidence interval for a proportion near 50%, in percentage points. */
export function proportionMargin(n: number): number {
  return Math.round(1.96 * Math.sqrt(0.25 / n) * 1000) / 10;
}

const sampleRule: Rule = (ctx) => {
  const { s } = ctx;
  // Fewer than 30 people, or 100 items, makes a few-point difference indistinguishable from sampling noise.
  const small = (x: SampleSize) => (x.human ? x.n < 30 : x.n < 100);
  const smallest = s.sampleSizes.filter(small).sort((a, b) => a.n - b.n)[0];
  if (!smallest) return [];
  // Below 50 items the interval is wider than ±14 points, which undermines most comparisons.
  const severe = smallest.human || smallest.n < 50;
  const what = smallest.unit === "n" ? "samples" : smallest.unit;
  return [{
    id: "small-sample",
    category: "statistics",
    severity: s.fullText && severe ? "moderate" : "minor",
    title: smallest.human ? "Small human study" : "Small evaluation set",
    rationale: `An evaluation reports n = ${smallest.n} ${what}. At this size a 95% confidence interval on a rate near 50% spans roughly ±${proportionMargin(smallest.n)} points, wider than many of the differences such papers report.`,
    evidence: [smallest.evidence],
    gist: "a small evaluation sample",
  }];
};

const humanEvalRule: Rule = (ctx) => {
  const { s } = ctx;
  const generationMetrics = METRICS.filter((m) => m.generation && s.metrics.includes(m.name)).map((m) => m.name);
  if (s.humanEval.length > 0) {
    return [{
      id: "human-evaluation",
      category: "evaluation",
      severity: "strength",
      title: "Includes human evaluation",
      rationale: `${plural(s.humanEval.length, "passage")} ${agree(s.humanEval.length, "describes", "describe")} human judgements (annotators, raters or a user study), which complement automatic metrics.`,
      evidence: s.humanEval.slice(0, 2),
      gist: "human evaluation",
    }];
  }
  // One passing mention of summarisation is not a generation paper; require repeated evidence.
  if (!s.fullText || s.generationTask.length < 2) return [];
  if (s.llmJudge.length > 0) {
    return [{
      id: "llm-judge-only",
      category: "evaluation",
      severity: "moderate",
      title: "Output quality judged only by a language model",
      rationale: "Generated text is scored by an LLM judge and no human evaluation was found. LLM judges have documented position, verbosity and self-preference biases, so agreement with human raters should be shown at least on a sample.",
      evidence: s.llmJudge.slice(0, 2),
      gist: "LLM-judge-only evaluation",
    }];
  }
  if (generationMetrics.length === 0) return [];
  return [{
    id: "no-human-eval",
    category: "evaluation",
    severity: "moderate",
    title: "Generation quality measured only by automatic metrics",
    rationale: `The paper evaluates text generation using ${listNames(generationMetrics)}, but no human evaluation was found. These metrics correlate only loosely with human judgements of fluency, faithfulness and usefulness.`,
    evidence: s.generationTask.slice(0, 2),
    gist: "metric-only evaluation of generated text",
  }];
};

const scholarshipRule: Rule = (ctx) => {
  const { s } = ctx;
  if (!s.fullText) return [];
  const out: Draft[] = [];
  if (!s.relatedWorkSection) {
    const background = ctx.input.doc?.sections.some((x) => x.kind === "background") ?? false;
    out.push({
      id: "no-related-work",
      category: "related-work",
      severity: background ? "minor" : "moderate",
      title: "No dedicated related-work section",
      rationale: background
        ? "Prior work is covered only in a background section; there is no explicit comparison of how this paper differs from the closest existing methods."
        : "No section titled Related Work, Prior Work or Background was found, so positioning against prior work is left to the introduction.",
      evidence: [],
      gist: "the missing related-work section",
    });
  }
  const years = s.referenceYears;
  const anchor = s.paperYear ?? (years.length ? Math.max(...years) : undefined);
  // Ten dated references are the minimum for a meaningful age distribution.
  if (anchor && years.length >= 10) {
    const ages = years.map((y) => anchor - y).sort((a, b) => a - b);
    const median = ages[Math.floor(ages.length / 2)];
    const recent = ages.filter((a) => a <= 3).length / ages.length;
    if (median > 8 && recent < 0.2) {
      out.push({
        id: "dated-references",
        category: "related-work",
        severity: "minor",
        title: "Bibliography skews old",
        rationale: `The median cited work is ${median} years older than the paper and only ${Math.round(recent * 100)}% of ${years.length} dated references are from the preceding three years, so recent competing work may be missing.`,
        evidence: [],
        gist: "a dated bibliography",
      });
    }
  }
  const words = s.words;
  if (s.references > 0 && s.references < 15 && words > 3000) {
    out.push({
      id: "few-references",
      category: "related-work",
      severity: "minor",
      title: "Thin bibliography",
      rationale: `Only ${plural(s.references, "reference")} ${agree(s.references, "was", "were")} parsed for a ${Math.round(words / 100) * 100}-word paper, which suggests the related literature is covered selectively.`,
      evidence: [],
      gist: "a thin bibliography",
    });
  }
  if (s.selfCitationRate !== undefined && s.selfCitationRate > 0.3) {
    out.push({
      id: "self-citation",
      category: "related-work",
      severity: "minor",
      title: "High self-citation rate",
      rationale: `${Math.round(s.selfCitationRate * 100)}% of parsed references share an author surname with this paper; the related-work picture may be skewed towards the authors' own line of work.`,
      evidence: [],
      gist: "heavy self-citation",
    });
  }
  return out;
};

const evaluationExtrasRule: Rule = (ctx) => {
  const { s } = ctx;
  if (!s.fullText) return [];
  const out: Draft[] = [];
  const textBenchmarks = s.datasets.map((d) => d.name).filter((n) => TEXT_DOMAINS.has(domainOf(n) ?? ""));
  if (s.webScalePretraining.length > 0 && textBenchmarks.length > 0 && s.contamination.length === 0) {
    out.push({
      id: "no-contamination-check",
      category: "evaluation",
      severity: "minor",
      title: "No test-set contamination check",
      rationale: `Models are pre-trained on web-scale text and evaluated on public text benchmarks (${listNames(textBenchmarks, 3)}) whose test sets may appear on the web, but no contamination or n-gram-overlap analysis was found.`,
      evidence: s.webScalePretraining.slice(0, 1),
      gist: "the missing contamination check",
    });
  }
  if (s.errorAnalysis.length >= 2) {
    out.push({
      id: "error-analysis",
      category: "evaluation",
      severity: "strength",
      title: "Analyses failure cases",
      rationale: `${plural(s.errorAnalysis.length, "passage")} ${agree(s.errorAnalysis.length, "examines", "examine")} errors or qualitative examples, showing where the method breaks rather than only aggregate scores.`,
      evidence: s.errorAnalysis.slice(0, 2),
      gist: "an error analysis",
    });
  } else if (ctx.empirical && s.errorAnalysis.length === 0 && s.resultSentences >= 5) {
    out.push({
      id: "no-error-analysis",
      category: "evaluation",
      severity: "minor",
      title: "No error analysis",
      rationale: "No failure-case, error or qualitative analysis was found; aggregate scores alone do not show where or why the method fails.",
      evidence: [],
      gist: "the missing error analysis",
    });
  }
  return out;
};

const abstractOnlyRule: Rule = (ctx) => {
  if (ctx.s.fullText) return [];
  return [{
    id: "abstract-only",
    category: "scope",
    severity: "minor",
    title: "Full text unavailable: review limited to the abstract",
    rationale: `Only the abstract (${plural(ctx.input.sentences.length, "sentence")}) could be analysed. Checks that need the body (variance and seeds, ablations, baselines, compute, hyperparameters, code links, limitations, bibliography) could not run, so their absence here is not evidence of a problem.`,
    evidence: [],
    gist: "an abstract-only view",
  }];
};

const RULES: Rule[] = [
  abstractOnlyRule,
  varianceRule,
  ablationRule,
  datasetRule,
  baselineRule,
  claimsRule,
  codeRule,
  computeRule,
  limitationsRule,
  sampleRule,
  humanEvalRule,
  scholarshipRule,
  evaluationExtrasRule,
];

// ───────────────────────────── Reviewer questions ───────────────────────────

const QUESTION_BY_FINDING: Record<string, (ctx: ReviewContext) => string> = {
  "no-variance": ({ s }) =>
    `How much do the ${s.datasets[0]?.name ?? "main"} results vary across random seeds, and is the margin over the baselines larger than that spread?`,
  "variance-hidden": () => "What are the standard deviations across the repeated runs, and do the main comparisons remain significant?",
  "no-ablation": ({ s }) => `Which component of ${methodLabel(s)} accounts for most of the gain, and what happens when each is removed in isolation?`,
  "thin-ablation": ({ s }) => `How sensitive is ${methodLabel(s)} to its main hyperparameters and design choices beyond the settings reported?`,
  "narrow-evaluation": ({ s }) => `Does the improvement hold on a second benchmark with a different distribution than ${s.datasets[0]?.name ?? "the one used"}?`,
  "no-benchmark": () => "Can the method be evaluated on an established public benchmark so results are comparable with prior work?",
  "no-baselines": ({ s }) => `How does ${methodLabel(s)} compare with the strongest existing methods under the same compute budget?`,
  "few-baselines": ({ s }) => `Why were other recent alternatives not compared, and how does ${methodLabel(s)} fare against them under equal budgets?`,
  "sota-without-stats": () => "Are the state-of-the-art margins statistically significant, for example under a paired bootstrap or a t-test over seeds?",
  "significance-wording": () => "Which statistical test supports the claims described as significant?",
  overclaiming: () => "Which experiment supports the strongest claims, and would more measured wording describe the evidence more accurately?",
  "no-code": () => "Will the code, configurations and checkpoints be released so the results can be reproduced?",
  "code-promised": () => "Where can the promised code be found, and does it reproduce the headline numbers?",
  "no-compute": ({ s }) =>
    s.efficiencyClaims.length > 0
      ? "On which hardware, batch sizes and sequence lengths were the speed and memory figures measured?"
      : "What hardware, accelerator-hours and wall-clock time did training require?",
  "no-hyperparameters": () => "Which hyperparameters were used and tuned, on which data, and how sensitive are the results to them?",
  "no-limitations": ({ s }) => `In which settings does ${methodLabel(s)} fail or underperform, and why?`,
  "limitations-in-passing": ({ s }) => `Beyond the limitation mentioned, where else might ${methodLabel(s)} break down?`,
  "small-sample": () => "With such a small evaluation set, what are the confidence intervals on the main comparisons?",
  "no-human-eval": () => "Do human judgements agree with the automatic metrics on output quality?",
  "llm-judge-only": () => "How well does the LLM judge agree with human raters, and is it robust to response order and length?",
  "no-contamination-check": ({ s }) =>
    `Could ${listNames(s.datasets.map((d) => d.name).filter((n) => TEXT_DOMAINS.has(domainOf(n) ?? "")), 2)} have leaked into the pre-training data, and was the overlap measured?`,
  "no-related-work": ({ s }) => `How does ${methodLabel(s)} differ from the closest prior work, and why was that work not compared directly?`,
  "dated-references": () => "Which work from the last two years addresses the same problem, and how does this paper compare with it?",
  "no-error-analysis": () => "What do the failure cases look like, and are the errors systematic?",
  "abstract-only": () => "In the full paper, how are the headline numbers measured, and against which baselines?",
};

function contextualQuestions(ctx: ReviewContext): string[] {
  const { s } = ctx;
  const name = methodLabel(s);
  const domains = datasetDomains(s.datasets.map((d) => d.name));
  return [
    `What assumptions does ${name} rely on, and when are they likely to be violated in practice?`,
    `How does ${name} behave as model size, data or input length grow beyond the settings tested?`,
    `What is the computational overhead of ${name} relative to the baselines at training and inference time?`,
    domains.length
      ? `Would the approach transfer beyond ${listNames(domains)} tasks, and what would need to change?`
      : `Would the approach transfer to other domains or tasks, and what would need to change?`,
  ];
}

function reviewerQuestions(ctx: ReviewContext, findings: Draft[]): string[] {
  const fromFindings = findings
    .filter((f) => f.severity !== "strength")
    .map((f) => QUESTION_BY_FINDING[f.id]?.(ctx))
    .filter((q): q is string => Boolean(q));
  const questions = [...new Set([...fromFindings, ...contextualQuestions(ctx)])];
  return questions.slice(0, Math.max(5, Math.min(8, fromFindings.length + 2)));
}

// ───────────────────────────────── Verdict ──────────────────────────────────

function capitalise(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function profileSentence(ctx: ReviewContext): string {
  const { s } = ctx;
  const names = s.datasets.map((d) => d.name);
  if (!s.fullText) {
    const what = s.methodName ? `the paper introduces ${s.methodName}` : "the paper presents its approach";
    const scope = names.length ? ` and reports results on ${listNames(names, 3)}` : "";
    return `Based on the abstract alone, ${what}${scope}.`;
  }
  if (!ctx.empirical) {
    const what = s.methodName ? `This paper introduces ${s.methodName} and` : "This paper";
    return `${what} is primarily conceptual or descriptive, with ${plural(s.resultSentences, "quantitative result sentence")} detected.`;
  }
  const what = s.methodName ? `This paper introduces ${s.methodName} and evaluates it` : "This paper evaluates its approach";
  const scope = names.length ? `${plural(names.length, "recognised benchmark")} (${listNames(names, 3)})` : "data that matches no standard benchmark";
  const variance = s.statistics.length > 0 ? "with reported variance or significance" : "without reported variance";
  return `${what} on ${scope}, ${variance}.`;
}

/** The two most important items, joined for prose ("A and B"). */
function topTwo(gists: string[]): string {
  return gists.length <= 1 ? (gists[0] ?? "") : `${gists[0]} and ${gists[1]}`;
}

function verdict(ctx: ReviewContext, findings: Draft[]): string {
  const strengths = findings.filter((f) => f.severity === "strength");
  const concerns = findings.filter((f) => f.severity === "major" || f.severity === "moderate");
  const majors = findings.filter((f) => f.severity === "major").length;
  const sentences = [profileSentence(ctx)];

  if (!ctx.s.fullText) {
    if (strengths.length) sentences.push(`The abstract points to ${topTwo(strengths.map((f) => f.gist))}.`);
    sentences.push("Variance, ablations, baselines and reproducibility can only be judged from the full text, so treat this as a preliminary read.");
    return sentences.join(" ");
  }
  const praise = strengths.length ? `Its strongest ${strengths.length === 1 ? "point is" : "points are"} ${topTwo(strengths.map((f) => f.gist))}` : "";
  if (concerns.length === 0) {
    sentences.push(`${praise || "No rule-detected strengths stand out"}, and none of the review checks flagged a major or moderate issue.`);
    return sentences.join(" ");
  }
  const worry = topTwo(concerns.map((f) => f.gist));
  const verb = concerns.length === 1 ? "concern is" : "concerns are";
  const consequence =
    majors >= 2
      ? "which together limit confidence that the reported gains are robust"
      : majors === 1
        ? "which should be addressed before the central claim is taken at face value"
        : concerns.length === 1
          ? "which weakens, but does not overturn, the main claims"
          : "which weaken, but do not overturn, the main claims";
  const also = concerns.length > 2 ? ` (plus ${plural(concerns.length - 2, "further issue")})` : "";
  const clause = `the main ${verb} ${worry}${also}`;
  sentences.push(praise ? `${praise}; ${clause}, ${consequence}.` : `${capitalise(clause)}, ${consequence}.`);
  return sentences.join(" ");
}

// ────────────────────────────────── Entry ───────────────────────────────────

export function buildCritique(input: AnalyzeInput, signals: PaperSignals): Critique {
  const ctx = reviewContext(input, signals);
  const drafts = RULES.flatMap((rule) => rule(ctx))
    .map((f, order) => ({ f, order }))
    .sort((a, b) => SEVERITY_ORDER[a.f.severity] - SEVERITY_ORDER[b.f.severity] || a.order - b.order)
    .map(({ f }) => f);
  const findings: CritiqueFinding[] = drafts.map(({ gist: _gist, ...finding }) => ({
    ...finding,
    evidence: dedupeEvidence(finding.evidence),
  }));
  return { findings, reviewerQuestions: reviewerQuestions(ctx, drafts), verdict: verdict(ctx, drafts) };
}

function dedupeEvidence(evidence: Evidence[]): Evidence[] {
  const seen = new Set<string>();
  return evidence.filter((e) => {
    const key = e.sentenceId ?? `${e.page}:${e.quote}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

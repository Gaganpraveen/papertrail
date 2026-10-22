/**
 * Rigor profile: seven weighted axes, each a transparent formula over the
 * extracted signals.
 *
 * Counts are turned into points with a saturating curve,
 *   credit(x, half) = 1 - 2^(-x / half),
 * which gives half credit at `half` occurrences and approaches full credit
 * smoothly. Saturation matters: the tenth ablation sentence says much less
 * than the first, and a paper should not win by repetition. Each axis is a
 * sum of named components (their maxima add up to 100); the overall score
 * is the weighted mean of the axes, graded A >= 80, B >= 65, C >= 50, D below.
 * Every axis lists the signals it used and a plain-English explanation of
 * how its number was computed.
 */
import type { RigorAxis, RigorProfile, RigorSignal } from "@/types";
import { DATASETS, datasetKey } from "./lexicons";
import type { PaperSignals } from "./signals";

export const RIGOR_WEIGHTS: Record<RigorAxis["key"], number> = {
  empirical: 0.2,
  baselines: 0.15,
  ablation: 0.15,
  statistics: 0.15,
  reproducibility: 0.15,
  transparency: 0.1,
  scholarship: 0.1,
};

/** Saturating credit in [0, 1): half credit at `half`. */
export function credit(x: number, half: number): number {
  if (x <= 0 || half <= 0) return 0;
  return 1 - 2 ** (-x / half);
}

export function grade(score: number): RigorProfile["grade"] {
  if (score >= 80) return "A";
  if (score >= 65) return "B";
  if (score >= 50) return "C";
  return "D";
}

interface Component {
  label: string;
  max: number;
  /** Share of `max` earned, 0..1. */
  earned: number;
  /** How the share was computed, in words. */
  rule: string;
}

function axis(
  key: RigorAxis["key"],
  label: string,
  components: Component[],
  signals: RigorSignal[],
  fullText: boolean,
): RigorAxis {
  const score = Math.round(components.reduce((sum, c) => sum + c.max * Math.min(1, Math.max(0, c.earned)), 0));
  const parts = components.map((c) => `${c.label} ${Math.round(c.max * c.earned)}/${c.max} (${c.rule})`);
  const caveat = fullText ? "" : " Only the abstract was available, so absent signals mean 'not visible', not 'missing'.";
  return {
    key,
    label,
    score,
    weight: RIGOR_WEIGHTS[key],
    signals,
    explanation: `${score}/100 = ${parts.join(" + ")}.${caveat}`,
  };
}

const first = <T>(xs: T[]): T | undefined => xs[0];

function empirical(s: PaperSignals): RigorAxis {
  const datasets = s.datasets.length;
  return axis(
    "empirical",
    "Empirical breadth",
    [
      { label: "benchmarks", max: 50, earned: credit(datasets, 2), rule: `${datasets} recognised, half credit at 2` },
      { label: "metrics", max: 25, earned: credit(s.metrics.length, 2), rule: `${s.metrics.length} distinct, half credit at 2` },
      { label: "quantitative results", max: 25, earned: credit(s.resultSentences, 4), rule: `${s.resultSentences} result sentences, half credit at 4` },
    ],
    [
      { label: "Benchmarks", value: s.datasets.map((d) => d.name).join(", ") || "none recognised", evidence: first(s.datasets)?.evidence },
      { label: "Metrics", value: s.metrics.join(", ") || "none recognised" },
      { label: "Result sentences", value: s.resultSentences },
    ],
    s.fullText,
  );
}

function baselines(s: PaperSignals): RigorAxis {
  const mentions = s.baselineMentions.length;
  const systems = s.comparedSystems.length;
  return axis(
    "baselines",
    "Baselines",
    [
      { label: "comparison sentences", max: 60, earned: credit(mentions, 3), rule: `${mentions} found, half credit at 3` },
      { label: "named systems", max: 40, earned: credit(systems, 3), rule: `${systems} detected, half credit at 3` },
    ],
    [
      { label: "Comparison sentences", value: mentions, evidence: first(s.baselineMentions) },
      { label: "Systems named in comparisons", value: s.comparedSystems.join(", ") || "none detected" },
    ],
    s.fullText,
  );
}

function ablation(s: PaperSignals): RigorAxis {
  const n = s.ablation.length;
  return axis(
    "ablation",
    "Ablations",
    [
      { label: "ablation passages", max: 70, earned: credit(n, 2), rule: `${n} found, half credit at 2` },
      { label: "dedicated section", max: 30, earned: s.ablationSection ? 1 : 0, rule: s.ablationSection ? "present" : "absent" },
    ],
    [
      { label: "Ablation or sensitivity passages", value: n, evidence: first(s.ablation) },
      { label: "Ablation section", value: s.ablationSection },
    ],
    s.fullText,
  );
}

function statistics(s: PaperSignals): RigorAxis {
  const spread = s.statistics.length;
  const runs = s.seedsOrRuns.length;
  const sizes = s.sampleSizes.length;
  return axis(
    "statistics",
    "Statistical care",
    [
      { label: "variance or tests", max: 60, earned: credit(spread, 1.5), rule: `${spread} passages with ±, std, CI or tests, half credit at 1.5` },
      { label: "repeated runs", max: 25, earned: credit(runs, 1), rule: `${runs} mentions of seeds or runs, half credit at 1` },
      { label: "sample sizes stated", max: 15, earned: sizes > 0 ? 1 : 0, rule: sizes > 0 ? `${sizes} stated` : "none stated" },
    ],
    [
      { label: "Variance or significance", value: spread, evidence: first(s.statistics) },
      { label: "Seeds or repeated runs", value: runs, evidence: first(s.seedsOrRuns) },
      { label: "Evaluation sizes", value: s.sampleSizes.map((x) => `n=${x.n} ${x.unit}`).join(", ") || "none stated", evidence: first(s.sampleSizes)?.evidence },
    ],
    s.fullText,
  );
}

function reproducibility(s: PaperSignals): RigorAxis {
  const code = s.codeLinks.length > 0 ? 1 : s.codeStatements.length > 0 ? 0.5 : 0;
  const publicData = publicBenchmarkShare(s);
  const data = s.dataLinks.length > 0 ? 1 : publicData;
  return axis(
    "reproducibility",
    "Reproducibility",
    [
      { label: "code", max: 40, earned: code, rule: code === 1 ? "repository linked" : code > 0 ? "release promised, no link" : "no link or statement" },
      { label: "hyperparameters", max: 25, earned: credit(s.hyperparameters.length, 2), rule: `${s.hyperparameters.length} passages, half credit at 2` },
      { label: "compute", max: 20, earned: credit(s.compute.length, 1), rule: `${s.compute.length} passages, half credit at 1` },
      { label: "data access", max: 15, earned: data, rule: s.dataLinks.length > 0 ? "data linked" : `${Math.round(publicData * 100)}% of data are public benchmarks` },
    ],
    [
      { label: "Code links", value: s.codeLinks.join(", ") || (s.codeStatements.length ? "release statement only" : "none") , evidence: first(s.codeStatements) },
      { label: "Hyperparameter passages", value: s.hyperparameters.length, evidence: first(s.hyperparameters) },
      { label: "Compute passages", value: s.compute.length, evidence: first(s.compute) },
      { label: "Data links", value: s.dataLinks.join(", ") || "none" },
    ],
    s.fullText,
  );
}

/** Known public benchmarks are reproducible data even without a link. */
function publicBenchmarkShare(s: PaperSignals): number {
  if (s.datasets.length === 0) return 0;
  const known = new Set(DATASETS.map((d) => datasetKey(d.name)));
  return s.datasets.filter((d) => known.has(datasetKey(d.name))).length / s.datasets.length;
}

function transparency(s: PaperSignals): RigorAxis {
  const rate = s.boosters / Math.max(1, s.claimSentences);
  // One intensifier per four claim sentences or more earns no calibration credit.
  const calibration = s.claimSentences === 0 ? 0.5 : Math.max(0, 1 - rate / 0.25);
  return axis(
    "transparency",
    "Transparency",
    [
      { label: "limitations section", max: 30, earned: s.limitationsSection ? 1 : 0, rule: s.limitationsSection ? "present" : "absent" },
      { label: "admitted limitations", max: 30, earned: credit(s.authorLimitations.length, 2), rule: `${s.authorLimitations.length} sentences, half credit at 2` },
      { label: "future work", max: 10, earned: credit(s.futureWork.length, 1), rule: `${s.futureWork.length} sentences, half credit at 1` },
      { label: "calibrated claims", max: 30, earned: calibration, rule: `${s.boosters} intensifiers in ${s.claimSentences} claim sentences; zero credit at 1 in 4` },
    ],
    [
      { label: "Limitations section", value: s.limitationsSection },
      { label: "Author-stated limitations", value: s.authorLimitations.length, evidence: first(s.authorLimitations) },
      { label: "Intensifiers in claims", value: s.boosters, evidence: first(s.boosterExamples) },
      { label: "Hedges", value: s.hedges },
    ],
    s.fullText,
  );
}

function scholarship(s: PaperSignals): RigorAxis {
  const years = s.referenceYears;
  const anchor = s.paperYear ?? (years.length ? Math.max(...years) : undefined);
  const recent = anchor && years.length ? years.filter((y) => anchor - y <= 3).length / years.length : 0;
  // A quarter of references from the last three years is typical of an up-to-date ML bibliography.
  const recency = Math.min(1, recent / 0.25);
  return axis(
    "scholarship",
    "Scholarship",
    [
      { label: "references", max: 50, earned: credit(s.references, 20), rule: `${s.references} parsed, half credit at 20` },
      { label: "recency", max: 30, earned: recency, rule: `${Math.round(recent * 100)}% from the 3 years before the paper; full credit at 25%` },
      { label: "related-work section", max: 20, earned: s.relatedWorkSection ? 1 : 0, rule: s.relatedWorkSection ? "present" : "absent" },
    ],
    [
      { label: "References", value: s.references },
      { label: "Median reference year", value: years.length ? [...years].sort((a, b) => a - b)[Math.floor(years.length / 2)] : "unknown" },
      { label: "Related-work section", value: s.relatedWorkSection },
      ...(s.selfCitationRate !== undefined ? [{ label: "Self-citation rate", value: `${Math.round(s.selfCitationRate * 100)}%` }] : []),
    ],
    s.fullText,
  );
}

export function buildRigor(signals: PaperSignals): RigorProfile {
  const axes = [
    empirical(signals),
    baselines(signals),
    ablation(signals),
    statistics(signals),
    reproducibility(signals),
    transparency(signals),
    scholarship(signals),
  ];
  const overall = Math.round(axes.reduce((sum, a) => sum + a.score * a.weight, 0));
  return { overall, grade: grade(overall), axes };
}

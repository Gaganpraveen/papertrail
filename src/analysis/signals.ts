/**
 * Signal extraction: the observable facts a reviewer checks before forming
 * an opinion.
 *
 * Each signal is a pattern from `lexicons.ts` applied to the zoned sentences
 * (plus section titles, captions, links and references from the parsed
 * document). Nothing here judges the paper; it only records *what was found
 * and where*, as Evidence pointing at a page and a verbatim sentence. The
 * critique and rigor modules turn these counts into findings and scores, so
 * every judgement can be traced back to the sentences that triggered it.
 *
 * Context matters: a sentence about ablations in Related Work describes
 * someone else's paper, so experimental signals ignore related-work and
 * background sections, and boosters ("dramatically", "unprecedented") are
 * only counted in claim sentences (abstract, contributions, results,
 * conclusion) where they inflate what the evidence shows.
 */
import type { Chunk, EmbedFn, Evidence, PaperMeta, ParsedDocument, Section, Sentence, ZonedSentence } from "@/types";
import { words } from "@/nlp/text";
import {
  ABLATION_RE,
  ABLATION_TITLE_RE,
  BASELINE_RE,
  BOOSTERS,
  CODE_HOST_RE,
  CODE_STATEMENT_RE,
  COMPUTE_RE,
  CONCESSION_RE,
  CONTAMINATION_RE,
  DATA_HOST_RE,
  DATASET_MATCHERS,
  EFFICIENCY_RE,
  ERROR_ANALYSIS_RE,
  EVALUATION_CONTEXT_RE,
  GENERALITY_RE,
  GENERATION_TASK_RE,
  HEDGES,
  HUMAN_EVAL_RE,
  HUMAN_SUBJECT_NOUNS,
  HYPERPARAMETER_RE,
  LIMITATION_CUE_RE,
  LIMITATIONS_TITLE_RE,
  LLM_JUDGE_RE,
  METRICS,
  NAMED_DATASET_RE,
  NOT_A_DATASET,
  OWN_WORK_RE,
  RELATED_TITLE_RE,
  SAMPLE_SIZE_RE,
  SEEDS_RE,
  SIGNIFICANTLY_RE,
  SOTA_RE,
  STATISTICS_RE,
  TRAINING_RE,
  URL_RE,
  WEB_SCALE_RE,
  datasetKey,
} from "./lexicons";

export interface AnalyzeInput {
  meta: PaperMeta;
  doc: ParsedDocument | null;
  sentences: ZonedSentence[];
  chunks: Chunk[];
  embed: EmbedFn | null;
  sentenceEmbeddings?: Map<string, Float32Array>;
}

export interface SampleSize {
  n: number;
  evidence: Evidence;
  /** What was counted ("participants", "examples"); "n" for "n = 50". */
  unit: string;
  /** People (annotators, participants) rather than items. */
  human: boolean;
}

export interface PaperSignals {
  datasets: { name: string; evidence: Evidence }[];
  metrics: string[];
  baselineMentions: Evidence[];
  ablation: Evidence[];
  statistics: Evidence[];
  seedsOrRuns: Evidence[];
  /** Links to the authors' own code (release sentence, first-page footnote or the method's name in the URL). */
  codeLinks: string[];
  dataLinks: string[];
  compute: Evidence[];
  hyperparameters: Evidence[];
  limitationsSection: boolean;
  authorLimitations: Evidence[];
  futureWork: Evidence[];
  hedges: number;
  boosters: number;
  boosterExamples: Evidence[];
  sotaClaims: Evidence[];
  sampleSizes: SampleSize[];
  figures: number;
  tables: number;
  references: number;
  referenceYears: number[];
  selfCitationRate?: number;
  humanEval: Evidence[];
  words: number;

  // Context the reviewer rules need beyond the core signals.
  /** False in abstract-only mode: absences then mean "not visible", not "missing". */
  fullText: boolean;
  paperYear?: number;
  /** The proposed system's name when the paper gives one ("KIVI", "Mamba"). */
  methodName?: string;
  /** Training corpora are recorded apart from evaluation benchmarks. */
  corpora: { name: string; evidence: Evidence }[];
  /** Capitalised system names appearing in comparison sentences (heuristic). */
  comparedSystems: string[];
  relatedWorkSection: boolean;
  ablationSection: boolean;
  codeStatements: Evidence[];
  /** Repository links that point at other people's code or data (e.g. datasets used). */
  externalCodeLinks: string[];
  generationTask: Evidence[];
  llmJudge: Evidence[];
  contamination: Evidence[];
  webScalePretraining: Evidence[];
  generalityClaims: Evidence[];
  training: Evidence[];
  efficiencyClaims: Evidence[];
  errorAnalysis: Evidence[];
  /** "Significantly better" with no statistical test in the same sentence. */
  unbackedSignificance: Evidence[];
  resultSentences: number;
  claimSentences: number;
  contributionSentences: number;
}

// ───────────────────────────────── Evidence ─────────────────────────────────

const chunkIndexCache = new WeakMap<Chunk[], Map<string, string>>();

/** Maps sentence ids to the chunk that contains them (same section, verbatim text). */
function chunkIndex(sentences: Sentence[], chunks: Chunk[]): Map<string, string> {
  const cached = chunkIndexCache.get(chunks);
  if (cached) return cached;
  const bySection = new Map<string, Chunk[]>();
  for (const chunk of chunks) bySection.set(chunk.sectionId, [...(bySection.get(chunk.sectionId) ?? []), chunk]);
  const index = new Map<string, string>();
  for (const s of sentences) {
    const hit = bySection.get(s.sectionId)?.find((c) => c.text.includes(s.text));
    if (hit) index.set(s.id, hit.id);
  }
  chunkIndexCache.set(chunks, index);
  return index;
}

/** Evidence for one sentence: page, section title and the verbatim quote. */
export function sentenceEvidence(input: AnalyzeInput, s: Sentence): Evidence {
  const chunkId = chunkIndex(input.sentences, input.chunks).get(s.id);
  return {
    paperId: input.meta.id,
    page: s.page,
    section: s.sectionTitle,
    quote: s.text,
    sentenceId: s.id,
    ...(chunkId ? { chunkId } : {}),
  };
}

// ──────────────────────────────── Contexts ──────────────────────────────────

/** Sections that describe other people's work. */
const PRIOR_WORK_KINDS = new Set(["related", "background"]);

const isPriorWork = (s: Sentence) => PRIOR_WORK_KINDS.has(s.sectionKind) || RELATED_TITLE_RE.test(s.sectionTitle);

/** Sentences where the authors state what they achieved. */
export function isClaimSentence(s: ZonedSentence): boolean {
  if (isPriorWork(s)) return false;
  return (
    s.zone === "contribution" ||
    s.zone === "result" ||
    s.sectionKind === "abstract" ||
    s.sectionKind === "conclusion"
  );
}

/** Mentions of other methods' flaws ("existing methods fail to ...") are problems, not admissions. */
const PRIOR_WORK_SUBJECT_RE =
  /\b(?:existing|prior|previous|current|conventional|traditional|standard|most) (?:methods|approaches|work|works|models|systems|techniques|solutions|baselines)\b|\b(?:they|these methods|such methods|these approaches)\b/i;

const FUTURE_RE = /\bfuture (?:work|research|directions?|studies|investigation)\b|\bwe (?:plan|hope|intend|leave)\b|\b(?:is|are) left (?:for|to) future\b|\bopen (?:question|problem|challenge)s?\b/i;

/** Reproducibility-checklist lines ("[Yes] Did you describe the limitations?") are forms, not admissions. */
export const CHECKLIST_RE = /\[(?:Yes|No|N\/A|TODO)\]|\?\s*$/i;

/** Words that make a "limitation"-zoned sentence an actual deficiency rather than an assumption or definition. */
const DEFICIENCY_RE =
  /\b(?:limit\w*|cannot|can't|unable|only|restrict\w*|fail\w*|lack\w*|not|no longer|difficult|costly|expensive|slower|sensitiv\w*|degrad\w*|unreliable|weakness\w*|drawbacks?)\b/i;

function isAuthorLimitation(s: ZonedSentence, fullText: boolean): boolean {
  if (CHECKLIST_RE.test(s.text)) return false;
  const inSection = s.sectionKind === "limitations" || LIMITATIONS_TITLE_RE.test(s.sectionTitle);
  if (inSection) return s.text.length >= 40 && s.zone !== "other";
  // With full text, the abstract and introduction describe the problem, not the method's own flaws.
  if (isPriorWork(s) || s.sectionKind === "introduction" || (fullText && s.sectionKind === "abstract")) return false;
  if (PRIOR_WORK_SUBJECT_RE.test(s.text) && !OWN_WORK_RE.test(s.text)) return false;
  if (s.zone === "future") return CONCESSION_RE.test(s.text);
  // Method sections are full of "we do not need X"; there only an explicit admission counts.
  if (s.sectionKind === "method") return LIMITATION_CUE_RE.test(s.text) && /\blimitation|\bfails? to\b|\bonly\b/i.test(s.text);
  if (s.zone === "limitation") return DEFICIENCY_RE.test(s.text);
  return LIMITATION_CUE_RE.test(s.text) && OWN_WORK_RE.test(s.text) && s.zone !== "background";
}

// ───────────────────────────────── Datasets ─────────────────────────────────

interface Source {
  text: string;
  evidence: () => Evidence;
}

function captionSources(input: AnalyzeInput): Source[] {
  return (input.doc?.captions ?? []).map((c) => ({
    text: c.text,
    evidence: () => ({ paperId: input.meta.id, page: c.page, section: c.label, quote: c.text }),
  }));
}

function sentenceSources(input: AnalyzeInput, sentences: ZonedSentence[]): Source[] {
  return sentences.map((s) => ({ text: s.text, evidence: () => sentenceEvidence(input, s) }));
}

/** Benchmarks ordered by how many passages mention them: the main evaluation comes first. */
function findDatasets(sources: Source[]): { benchmarks: PaperSignals["datasets"]; corpora: PaperSignals["corpora"] } {
  const found: { name: string; corpus: boolean; mentions: number; evidence: Evidence }[] = [];
  const seen = new Set<string>();
  for (const { entry, re } of DATASET_MATCHERS) {
    const hits = sources.filter((src) => re.test(src.text));
    if (hits.length === 0) continue;
    seen.add(datasetKey(entry.name));
    for (const alias of entry.aliases ?? []) seen.add(datasetKey(alias));
    found.push({ name: entry.name, corpus: entry.corpus === true, mentions: hits.length, evidence: hits[0].evidence() });
  }
  found.sort((a, b) => b.mentions - a.mentions);
  const benchmarks: PaperSignals["datasets"] = found.filter((d) => !d.corpus).map(({ name, evidence }) => ({ name, evidence }));
  const corpora: PaperSignals["corpora"] = found.filter((d) => d.corpus).map(({ name, evidence }) => ({ name, evidence }));
  // Benchmarks outside the curated list, recognised by how they are introduced.
  for (const source of sources) {
    for (const match of source.text.matchAll(NAMED_DATASET_RE)) {
      const name = match[1];
      const key = datasetKey(name);
      if (seen.has(key) || NOT_A_DATASET.has(name) || name.length < 3) continue;
      seen.add(key);
      benchmarks.push({ name, evidence: source.evidence() });
    }
  }
  return { benchmarks, corpora };
}

// ─────────────────────────────── Compared systems ───────────────────────────

const SYSTEM_TOKEN_RE = /(?<![\p{L}\p{N}-])[A-Z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*(?![\p{L}\p{N}])/gu;
/** Generic acronyms and units that look like system names but are not. */
const NOT_A_SYSTEM = new Set(
  `LLM LLMs NLP AI ML RL API APIs SOTA OOD IID QA KV GPU GPUs TPU TPUs CPU CPUs FP16 FP32 BF16 INT8 INT4 FLOPs
  NLU NLG ASR TTS OCR CV MoE SFT RLHF PhD US UK EU URL HTML PDF JSON CoT ICL MLP CNN RNN LSTM GAN VAE ViT TFLOPs
  GB MB TB KB ms MHz GHz ID IDs NLI SQL OOM Out-of-Memory BLEU-4 ROUGE-L ROUGE-1 ROUGE-2 KL SSM LTI SwiGLU GeLU
  GELU ReLU SiLU RMSNorm LayerNorm BatchNorm MHA MQA FFN RoPE ALiBi FlashAttention FlashAttention-2 PaLM-based`
    .split(/\s+/)
    .filter(Boolean),
);
const METRIC_TOKENS = new Set(["BLEU", "ROUGE", "METEOR", "CIDEr", "SPICE", "BERTScore", "MAUVE", "F1", "EM", "AUC", "AUROC", "mAP", "IoU", "mIoU", "FID", "WER", "CER", "MRR", "NDCG", "MSE", "MAE", "RMSE", "PPL", "ECE"]);

/** "Direct Preference Optimization" -> "DPO". */
function acronymOf(name: string): string | null {
  const parts = name.split(/[\s-]+/u).filter((w) => /^\p{Lu}/u.test(w));
  return parts.length >= 2 ? parts.map((w) => w.charAt(0)).join("") : null;
}

/** Author-year citations: their surnames must not be mistaken for systems. */
const AUTHOR_YEAR_RE = /\([^()]*\b(?:19|20)\d{2}[a-z]?\)|\b[\p{Lu}][\p{L}'’-]+ (?:et al\.?|and [\p{Lu}][\p{L}'’-]+)/gu;

function comparedSystems(sentences: ZonedSentence[], exclude: Set<string>, methodName?: string): string[] {
  const own = methodName && methodName.length >= 3 ? datasetKey(methodName) : null;
  const ownAcronym = methodName ? acronymOf(methodName) : null;
  const counts = new Map<string, number>();
  for (const s of sentences) {
    for (const [token] of s.text.replace(AUTHOR_YEAR_RE, " ").matchAll(SYSTEM_TOKEN_RE)) {
      // A system name has an inner capital or digit ("StreamingLLM", "H2O", "GPT-4"); "Table" does not.
      if (!/[A-Z0-9]/.test(token.slice(1)) || token.length < 3) continue;
      if (/^[A-Z]+s$/.test(token)) continue; // plural acronyms ("KVs", "LLMs")
      if (ownAcronym && (token === ownAcronym || token.startsWith(`${ownAcronym}-`))) continue; // the method itself
      if (token.split("-").slice(1).some((part) => /^[a-z]{3,}$/.test(part))) continue; // "PPO-based", "KL-divergence"
      if (NOT_A_SYSTEM.has(token) || METRIC_TOKENS.has(token) || exclude.has(datasetKey(token))) continue;
      if (own && datasetKey(token).startsWith(own)) continue; // "KIVI-2" is a variant of the method, not a baseline
      if (/^[A-Z]\d+$/.test(token)) continue; // "A1", "Q3", "T5" are labels or too ambiguous
      counts.set(token, (counts.get(token) ?? 0) + 1);
    }
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([name]) => name).slice(0, 12);
}

// ───────────────────────────────── Method name ──────────────────────────────

const GENERIC_TITLE_HEADS = /^(?:towards?|on|a|an|the|beyond|rethinking|revisiting|understanding|learning|improving)\b/i;

/** The proposed system's name: "KIVI: A Tuning-Free ..." or "... a method called X" or "(DPO)". */
export function detectMethodName(meta: PaperMeta, sentences: ZonedSentence[]): string | undefined {
  const head = meta.title.split(/:\s/)[0]?.trim();
  if (head && head !== meta.title && head.split(/\s+/).length <= 4 && !GENERIC_TITLE_HEADS.test(head)) return head;

  const claims = sentences.filter((s) => s.zone === "contribution" || s.sectionKind === "abstract").slice(0, 40);
  for (const s of claims) {
    const named = /\b(?:called|named|dubbed|termed|coined)\s+([A-Z][\w-]*[A-Z0-9][\w-]*|[A-Z][a-z]+[A-Z]\w*)/.exec(s.text);
    if (named) return named[1];
  }
  for (const s of claims) {
    const introduced = /\bwe (?:propose|introduce|present|develop)\s+([A-Z][A-Za-z]*[A-Z0-9][\w-]*)/.exec(s.text);
    if (introduced && !NOT_A_SYSTEM.has(introduced[1])) return introduced[1];
  }
  for (const s of claims) {
    if (!/\bwe (?:propose|introduce|present|develop)\b/i.test(s.text)) continue;
    const acronym = /\(([A-Z][A-Za-z0-9-]*[A-Z][A-Za-z0-9-]*)\)/.exec(s.text);
    if (acronym && !NOT_A_SYSTEM.has(acronym[1])) return acronym[1];
  }
  return undefined;
}

// ───────────────────────────────── Links ────────────────────────────────────

function cleanUrl(url: string): string {
  return url.replace(/[.,;:)\]}>'"]+$/u, "");
}

function urlCore(url: string): string {
  return url.toLowerCase().replace(/^https?:\/\//u, "").replace(/^www\./u, "").replace(/\/+$/u, "");
}

function collectLinks(input: AnalyzeInput, ownSentences: ZonedSentence[]): string[] {
  const inText = ownSentences.flatMap((s) => [...s.text.matchAll(URL_RE)].map((m) => cleanUrl(m[0])));
  const fromAbstract = [...(input.meta.abstract ?? "").matchAll(URL_RE)].map((m) => cleanUrl(m[0]));
  const referenceText = (input.doc?.references ?? []).map((r) => r.raw.toLowerCase()).join(" ");
  // Links that only appear inside the bibliography point at other people's work.
  const fromDoc = (input.doc?.links ?? []).map(cleanUrl).filter((u) => !referenceText.includes(urlCore(u)));
  const unique = new Map<string, string>();
  for (const url of [...inText, ...fromAbstract, ...fromDoc]) {
    const core = urlCore(url);
    if (core.length > 8 && !unique.has(core)) unique.set(core, url);
  }
  return [...unique.values()];
}

/**
 * A repository is the authors' own when a sentence presents it as theirs
 * ("our code is available at ..."), it sits on the first page (where
 * release footnotes go), or its path carries the method's name.
 */
function isOwnCodeLink(url: string, input: AnalyzeInput, own: ZonedSentence[], methodName?: string): boolean {
  const core = urlCore(url);
  const sentence = own.find((s) => s.text.toLowerCase().includes(core));
  const ownWording = /\bour (?:code|implementation|repository|project page|models?|library|toolkit|framework)\b/i;
  if (sentence && (CODE_STATEMENT_RE.test(sentence.text) || ownWording.test(sentence.text))) return true;
  if ((input.meta.abstract ?? "").toLowerCase().includes(core)) return true;
  const firstPage = input.doc?.pages.find((p) => p.page === 1)?.text.toLowerCase() ?? "";
  if (firstPage.includes(core)) return true;
  return methodName !== undefined && methodName.length >= 3 && core.includes(methodName.toLowerCase());
}

// ──────────────────────────────── Statistics ────────────────────────────────

/** "71.2 ± 0.4" inside table bodies, which sentence splitting drops as non-prose. */
const TABLE_SPREAD_RE = /\d+(?:\.\d+)?\s?(?:±|\+\/-|\+-)\s?\d+(?:\.\d+)?/u;

function tableSpread(input: AnalyzeInput, alreadyFound: Evidence[]): Evidence[] {
  const doc = input.doc;
  if (!doc) return [];
  const out: Evidence[] = [];
  for (const section of doc.sections) {
    if (section.kind === "references" || section.kind === "acknowledgments" || out.length >= 3) continue;
    const match = TABLE_SPREAD_RE.exec(section.text);
    if (!match || alreadyFound.some((e) => e.quote.includes(match[0]))) continue;
    const at = match.index;
    const snippet = section.text.slice(Math.max(0, at - 50), at + match[0].length + 30).replace(/\s+/gu, " ").trim();
    out.push({ paperId: input.meta.id, page: pageOf(doc, section, match[0]), section: section.title, quote: snippet });
  }
  return out;
}

function pageOf(doc: ParsedDocument, section: Section, needle: string): number {
  const page = doc.pages.find((p) => p.page >= section.pageStart && p.page <= section.pageEnd && p.text.includes(needle));
  return page?.page ?? section.pageStart;
}

// ──────────────────────────────── References ────────────────────────────────

const MAX_EVIDENCE = 8;

/** Sections where a paper reports its own experiments. */
const EXPERIMENT_KINDS = new Set(["experiments", "results", "discussion", "appendix", "other"]);

function referenceYears(doc: ParsedDocument | null): number[] {
  const latest = new Date().getFullYear() + 1;
  return (doc?.references ?? [])
    .map((r) => r.year)
    .filter((y): y is number => typeof y === "number" && y >= 1900 && y <= latest);
}

/** Share of references that list one of this paper's authors (surname match). */
function selfCitationRate(meta: PaperMeta, doc: ParsedDocument | null): number | undefined {
  const withAuthors = (doc?.references ?? []).filter((r) => r.authors && r.authors.length > 0);
  // Below ~10 parsed author lists the rate is too noisy to report.
  if (withAuthors.length < 10 || meta.authors.length === 0) return undefined;
  const surnames = meta.authors
    .map((a) => a.trim().split(/\s+/).pop() ?? "")
    .filter((s) => s.length >= 3)
    .map((s) => new RegExp(`(?<![\\p{L}])${s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![\\p{L}])`, "u"));
  if (surnames.length === 0) return undefined;
  const self = withAuthors.filter((r) => surnames.some((re) => re.test(r.authors ?? ""))).length;
  return Math.round((self / withAuthors.length) * 1000) / 1000;
}

function countLabels(doc: ParsedDocument | null, sentences: Sentence[], kind: "figure" | "table"): number {
  const captions = new Set((doc?.captions ?? []).filter((c) => c.kind === kind).map((c) => c.label.toLowerCase()));
  if (captions.size > 0) return captions.size;
  // Without parsed captions, count the distinct numbers the text refers to.
  const re = kind === "figure" ? /\b(?:Figure|Fig\.)\s?(\d+)/g : /\bTable\s?(\d+)/g;
  const numbers = new Set(sentences.flatMap((s) => [...s.text.matchAll(re)].map((m) => m[1])));
  return numbers.size;
}

// ───────────────────────────────── Extraction ───────────────────────────────

function matching(input: AnalyzeInput, sentences: ZonedSentence[], re: RegExp, limit = MAX_EVIDENCE): Evidence[] {
  return sentences.filter((s) => re.test(s.text)).slice(0, limit).map((s) => sentenceEvidence(input, s));
}

function countMatches(text: string, re: RegExp): number {
  return [...text.matchAll(re)].length;
}

const COUNTED_THING_RE = /\b(?:participants?|subjects?|samples?|examples?|items?|questions?|respondents?|users?|annotat\w+|test cases?)\b/i;

/** People who label data: a handful of them is normal and says nothing about the evaluation size. */
const LABELLER_NOUNS = new Set(["annotators", "raters", "judges", "evaluators", "experts"]);

function sampleSizes(input: AnalyzeInput, sentences: ZonedSentence[]): SampleSize[] {
  const out: SampleSize[] = [];
  for (const s of sentences) {
    if (!EVALUATION_CONTEXT_RE.test(s.text)) continue;
    for (const m of s.text.matchAll(SAMPLE_SIZE_RE)) {
      const n = Number((m[1] ?? m[2] ?? "").replace(/,/g, ""));
      const unit = m[3]?.toLowerCase() ?? "n";
      const human = HUMAN_SUBJECT_NOUNS.has(unit);
      // A bare "N = 16" is usually a model dimension; only trust it next to a word for what was counted.
      if (unit === "n" && !COUNTED_THING_RE.test(s.text)) continue;
      // "4 prompts" is usually a set of templates and "3 annotators" a labelling team, not an evaluation size.
      if (!Number.isFinite(n) || n < (human ? 2 : 10) || LABELLER_NOUNS.has(unit)) continue;
      out.push({ n, unit, human, evidence: sentenceEvidence(input, s) });
    }
    if (out.length >= MAX_EVIDENCE) break;
  }
  return out;
}

export function extractSignals(input: AnalyzeInput): PaperSignals {
  const { doc, meta } = input;
  const all = input.sentences;
  const own = all.filter((s) => !isPriorWork(s));
  const claims = own.filter(isClaimSentence);
  const sections = doc?.sections ?? [];

  const { benchmarks, corpora } = findDatasets([...sentenceSources(input, own), ...captionSources(input)]);
  const metricSources = [...own.map((s) => s.text), ...(doc?.captions ?? []).map((c) => c.text)];
  const metrics = METRICS.filter((m) => metricSources.some((t) => m.re.test(t))).map((m) => m.name);

  const methodName = detectMethodName(meta, all);
  const exclude = new Set([...benchmarks, ...corpora].map((d) => datasetKey(d.name)));
  if (methodName) exclude.add(datasetKey(methodName));
  const experimental = new Set(own.filter((s) => EXPERIMENT_KINDS.has(s.sectionKind) || s.sectionKind === "abstract"));
  const baselineSentences = own.filter((s) => BASELINE_RE.test(s.text));

  const ablationSections = sections.filter((s) => ABLATION_TITLE_RE.test(s.title));
  const explicitAblation = own.filter((s) => ABLATION_RE.test(s.text));
  const ablationSentences = [
    ...explicitAblation,
    ...own.filter((s) => ABLATION_TITLE_RE.test(s.sectionTitle) && !explicitAblation.includes(s)),
  ];

  const sentenceStats = matching(input, own, STATISTICS_RE);
  const links = collectLinks(input, own);
  const codeStatements = matching(input, own, CODE_STATEMENT_RE, 3);
  const codeLinks = links.filter((u) => CODE_HOST_RE.test(u));
  const isOwnCode = (url: string) => isOwnCodeLink(url, input, own, methodName);

  const boosterSentences = claims.filter((s) => countMatches(s.text, BOOSTERS) > 0);
  const unbacked = claims.filter((s) => countMatches(s.text, SIGNIFICANTLY_RE) > 0 && !STATISTICS_RE.test(s.text));
  const boosterCount =
    claims.reduce((sum, s) => sum + countMatches(s.text, BOOSTERS), 0) + unbacked.length;

  return {
    datasets: benchmarks,
    metrics,
    baselineMentions: baselineSentences.slice(0, 12).map((s) => sentenceEvidence(input, s)),
    ablation: ablationSentences.slice(0, MAX_EVIDENCE).map((s) => sentenceEvidence(input, s)),
    statistics: [...sentenceStats, ...tableSpread(input, sentenceStats)],
    seedsOrRuns: matching(input, own, SEEDS_RE),
    codeLinks: codeLinks.filter((u) => isOwnCode(u)),
    externalCodeLinks: codeLinks.filter((u) => !isOwnCode(u)),
    dataLinks: links.filter((u) => DATA_HOST_RE.test(u) && !CODE_HOST_RE.test(u)),
    compute: matching(input, own, COMPUTE_RE),
    hyperparameters: matching(input, own, HYPERPARAMETER_RE),
    limitationsSection: sections.some((s) => s.kind === "limitations" || LIMITATIONS_TITLE_RE.test(s.title)),
    authorLimitations: all.filter((s) => isAuthorLimitation(s, doc !== null)).slice(0, 10).map((s) => sentenceEvidence(input, s)),
    futureWork: all
      .filter((s) => s.zone === "future" || FUTURE_RE.test(s.text))
      .filter((s) => !isPriorWork(s))
      .slice(0, 6)
      .map((s) => sentenceEvidence(input, s)),
    hedges: own.reduce((sum, s) => sum + countMatches(s.text, HEDGES), 0),
    boosters: boosterCount,
    boosterExamples: [...unbacked, ...boosterSentences]
      .filter((s, i, list) => list.indexOf(s) === i)
      .slice(0, 4)
      .map((s) => sentenceEvidence(input, s)),
    sotaClaims: claims.filter((s) => SOTA_RE.test(s.text) && OWN_WORK_RE.test(s.text)).slice(0, 6).map((s) => sentenceEvidence(input, s)),
    sampleSizes: sampleSizes(input, own),
    figures: countLabels(doc, all, "figure"),
    tables: countLabels(doc, all, "table"),
    references: doc?.references.length ?? 0,
    referenceYears: referenceYears(doc),
    selfCitationRate: selfCitationRate(meta, doc),
    humanEval: matching(input, own, HUMAN_EVAL_RE),
    words: all.reduce((sum, s) => sum + words(s.text).length, 0),

    fullText: doc !== null,
    paperYear: meta.year ?? (meta.published ? Number(meta.published.slice(0, 4)) || undefined : undefined),
    methodName,
    corpora,
    comparedSystems: comparedSystems(baselineSentences.filter((s) => experimental.has(s)), exclude, methodName),
    relatedWorkSection: sections.some((s) => s.kind === "related" || RELATED_TITLE_RE.test(s.title)),
    ablationSection: ablationSections.length > 0,
    codeStatements,
    generationTask: matching(input, own, GENERATION_TASK_RE, 4),
    llmJudge: matching(input, own, LLM_JUDGE_RE, 4),
    contamination: matching(input, own, CONTAMINATION_RE, 4),
    webScalePretraining: matching(input, own, WEB_SCALE_RE, 3),
    generalityClaims: matching(input, claims, GENERALITY_RE, 4),
    training: matching(input, own, TRAINING_RE, 4),
    efficiencyClaims: matching(input, claims, EFFICIENCY_RE, 4),
    errorAnalysis: matching(input, own, ERROR_ANALYSIS_RE, 4),
    unbackedSignificance: unbacked.slice(0, 4).map((s) => sentenceEvidence(input, s)),
    resultSentences: own.filter((s) => s.zone === "result").length,
    claimSentences: claims.length,
    contributionSentences: own.filter((s) => s.zone === "contribution").length,
  };
}

/**
 * Suggested follow-up questions for the Ask tab.
 *
 * Templates are filled from what the paper actually contains (its method
 * name, top keyphrases, benchmarks, compared systems, compute and ablation
 * passages), so each suggestion is specific to this paper and answerable
 * from its text by the extractive QA pipeline. A template is only used when
 * the signal it depends on was found, which keeps the questions honest.
 */
import type { Zone } from "@/types";
import type { AnalyzeInput, PaperSignals } from "./signals";

const MIN_QUESTIONS = 5;
const MAX_QUESTIONS = 7;

/** Keyphrases that read as a technique rather than a dataset or the method's own name. */
function conceptPhrases(signals: PaperSignals, keyphrases: { phrase: string }[]): string[] {
  const datasets = new Set(signals.datasets.map((d) => d.name.toLowerCase()));
  const own = signals.methodName?.toLowerCase();
  return keyphrases
    .map((k) => k.phrase)
    .filter((p) => p.length > 3 && !datasets.has(p.toLowerCase()) && !(own && p.toLowerCase().includes(own)));
}

const hasZone = (input: AnalyzeInput, zone: Zone) => input.sentences.some((s) => s.zone === zone);

export function generateFollowUps(
  input: AnalyzeInput,
  signals: PaperSignals,
  keyphrases: { phrase: string; score: number }[],
): string[] {
  const concepts = conceptPhrases(signals, keyphrases);
  // Without a named system, the paper's top keyphrase ("chain of thought") is the most specific subject.
  const name = signals.methodName ?? concepts[0] ?? "the proposed method";
  // The next keyphrase after the subject is usually a key ingredient of the approach.
  const ingredient = signals.methodName ? concepts[0] : concepts[1];
  const candidates: (string | null)[] = [
    hasZone(input, "problem") ? "What problem does the paper address, and why is it hard?" : null,
    `How does ${name} work?`,
    ingredient ? `What is the role of ${ingredient} in this work?` : null,
    signals.datasets[0] ? `What results does ${name} achieve on ${signals.datasets[0].name}?` : hasZone(input, "result") ? "What are the main quantitative results?" : null,
    signals.baselineMentions.length > 0 ? `Which baselines is ${name} compared against?` : null,
    signals.ablation.length > 0 ? "What do the ablation experiments show?" : null,
    signals.compute.length > 0 ? "What hardware and compute were used in the experiments?" : signals.hyperparameters.length > 0 ? "Which hyperparameters were used?" : null,
    "What are the limitations of this work?",
    signals.futureWork.length > 0 ? "What future work do the authors suggest?" : null,
    signals.datasets.length > 1 ? `Which datasets are used for evaluation?` : null,
  ];
  const unique = [...new Set(candidates.filter((q): q is string => q !== null))];
  const fallbacks = ["What is the main contribution of the paper?", "What are the key findings?", "How is the method evaluated?"];
  for (const q of fallbacks) if (unique.length < MIN_QUESTIONS && !unique.includes(q)) unique.push(q);
  return unique.slice(0, MAX_QUESTIONS);
}

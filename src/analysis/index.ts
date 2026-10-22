/**
 * Single-paper analysis: signals -> critique -> briefing -> rigor -> visuals.
 *
 * Signals are extracted once and shared, so every view of the paper (the
 * briefing's detected limitations, the critique, the rigor axes) is built
 * from the same evidence. Without full text (`doc` is null) the same steps
 * run on the abstract and every judgement is hedged accordingly.
 */
import type { PaperAnalysis } from "@/types";
import { buildBriefing } from "./briefing";
import { buildCritique } from "./critique";
import { buildRigor } from "./rigor";
import { type AnalyzeInput, extractSignals } from "./signals";
import { buildVisuals } from "./visuals";

export type { AnalyzeInput, PaperSignals } from "./signals";
export { extractSignals } from "./signals";
export { buildBriefing, stripCitations } from "./briefing";
export { buildCritique } from "./critique";
export { buildRigor } from "./rigor";
export { buildVisuals } from "./visuals";
export { generateFollowUps } from "./followups";

export async function analyzePaper(input: AnalyzeInput): Promise<PaperAnalysis> {
  const started = performance.now();
  const signals = extractSignals(input);
  const critique = buildCritique(input, signals);
  const briefing = await buildBriefing(input, signals, critique);
  const rigor = buildRigor(signals);
  const visuals = buildVisuals(input, briefing.keyphrases);
  return {
    briefing,
    critique,
    rigor,
    visuals,
    stats: {
      pages: input.doc?.pageCount ?? 0,
      words: signals.words,
      sentences: input.sentences.length,
      chunks: input.chunks.length,
      references: signals.references,
      figures: signals.figures,
      tables: signals.tables,
    },
    mode: input.doc ? "full" : "abstract-only",
    computedMs: Math.round(performance.now() - started),
  };
}

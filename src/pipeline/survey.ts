/**
 * The survey agent: turns several analysed papers into a literature review.
 *
 *   collect -> embed -> synthesize -> END
 *
 * `collect` validates the set, `embed` makes sure every paper has passage
 * embeddings (reusing cached ones, computing missing ones, or falling back to
 * TF-IDF vectors when the model is unavailable), and `synthesize` runs the
 * survey algorithms (clustering, semantic map, c-TF-IDF themes, bibliographic
 * coupling, consensus and gap mining, narrative) in analysis/survey.
 */
import { END, START, StateGraph, type NodeContext } from "@/agent/graph";
import { buildSurvey } from "@/analysis/survey";
import type { MLApi } from "@/ml/client";
import { isAbortError } from "@/sources/http";
import { embeddingText } from "@/retrieval/chunker";
import type { EmbedFn, SurveyPaperInput, SurveyResult, TraceEvent } from "@/types";

export const SURVEY_NODES: { name: string; label: string; description: string }[] = [
  { name: "collect", label: "Collect", description: "Gather analysed papers, their passages and references" },
  { name: "embed", label: "Embed", description: "Reuse cached passage embeddings, compute any that are missing" },
  {
    name: "synthesize",
    label: "Synthesize",
    description: "Cluster themes, map papers, compare methods, mine consensus and gaps, write the review",
  },
];

export const MIN_SURVEY_PAPERS = 2;
const EMBED_BATCH = 32;

interface SurveyState {
  papers: SurveyPaperInput[];
  embed: EmbedFn | null;
  result?: SurveyResult;
}

function spec(name: string): { label: string; description: string } {
  const node = SURVEY_NODES.find((n) => n.name === name)!;
  return { label: node.label, description: node.description };
}

function collect(state: SurveyState, ctx: NodeContext): Partial<SurveyState> {
  const unique = [...new Map(state.papers.map((p) => [p.meta.id, p])).values()];
  if (unique.length < MIN_SURVEY_PAPERS) {
    throw new Error(`A survey needs at least ${MIN_SURVEY_PAPERS} analysed papers`);
  }
  const abstractOnly = unique.filter((p) => p.analysis.mode === "abstract-only").length;
  const passages = unique.reduce((sum, p) => sum + p.chunks.length, 0);
  ctx.log(
    `${unique.length} papers, ${passages} passages` + (abstractOnly ? `, ${abstractOnly} abstract-only` : ""),
  );
  return { papers: unique };
}

function hasEmbeddings(paper: SurveyPaperInput): boolean {
  return paper.chunkEmbeddings !== null && paper.chunkEmbeddings.length === paper.chunks.length;
}

async function embedChunks(embed: EmbedFn, paper: SurveyPaperInput): Promise<Float32Array[]> {
  const texts = paper.chunks.map(embeddingText);
  const vectors: Float32Array[] = [];
  for (let start = 0; start < texts.length; start += EMBED_BATCH) {
    vectors.push(...(await embed(texts.slice(start, start + EMBED_BATCH))));
  }
  return vectors;
}

function makeEmbed(deps: { ml: MLApi }) {
  return async (state: SurveyState, ctx: NodeContext): Promise<Partial<SurveyState>> => {
    const missing = state.papers.filter((p) => !hasEmbeddings(p));
    if (missing.length === 0) {
      ctx.log("Reusing cached passage embeddings for every paper");
      return { embed: deps.ml.embed };
    }
    try {
      const filled = new Map<string, Float32Array[]>();
      for (const paper of missing) {
        filled.set(paper.meta.id, await embedChunks(deps.ml.embed, paper));
        ctx.log(`Embedded ${paper.chunks.length} passages of "${paper.meta.title}"`);
      }
      return {
        embed: deps.ml.embed,
        papers: state.papers.map((p) => (filled.has(p.meta.id) ? { ...p, chunkEmbeddings: filled.get(p.meta.id)! } : p)),
      };
    } catch (error) {
      if (isAbortError(error)) throw error;
      // Mixed representations are not comparable, so without the model every
      // paper falls back to TF-IDF vectors inside buildSurvey.
      ctx.log("Embedding model unavailable; comparing papers with TF-IDF vectors");
      return { embed: null, papers: state.papers.map((p) => ({ ...p, chunkEmbeddings: null })) };
    }
  };
}

async function synthesize(state: SurveyState, ctx: NodeContext): Promise<Partial<SurveyState>> {
  const result = await buildSurvey(state.papers, state.embed, (step: string) => ctx.log(step));
  ctx.log(`${result.clusters.length} themes, ${result.consensus.length} consensus points, ${result.gaps.length} gaps`);
  return { result };
}

export function buildSurveyGraph(deps: { ml: MLApi }) {
  return new StateGraph<SurveyState>()
    .addNode("collect", collect, spec("collect"))
    .addNode("embed", makeEmbed(deps), spec("embed"))
    .addNode("synthesize", synthesize, spec("synthesize"))
    .addEdge(START, "collect")
    .addEdge("collect", "embed")
    .addEdge("embed", "synthesize")
    .addEdge("synthesize", END)
    .compile();
}

export async function runSurveyPipeline(
  inputs: SurveyPaperInput[],
  deps: { ml: MLApi },
  onEvent?: (e: TraceEvent) => void,
): Promise<SurveyResult> {
  const final = await buildSurveyGraph(deps).run({ papers: inputs, embed: null }, { onEvent });
  if (!final.result) throw new Error("The survey finished without a result");
  return final.result;
}

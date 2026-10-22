/**
 * Ask tab: grounded, extractive Q&A over the paper. Each answer shows the
 * span highlighted inside its source sentence, a confidence meter, page
 * evidence, and a retrieval inspector exposing every ranking signal
 * (BM25, dense cosine, reciprocal-rank fusion, cross-encoder rerank).
 */
import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { ArrowUp, ChevronRight, CircleSlash, MessageSquareText, Microscope, Quote } from "lucide-react";
import { ABSTAIN_THRESHOLD, QA_NODES, SENTENCE_ABSTAIN_THRESHOLD } from "@/qa/answer";
import type { Answer, QuestionIntent, RetrievalHit } from "@/types";
import { Badge, Card, CardBody, CardHeader, EmptyState, Spinner, cn } from "@/ui/kit";
import { ModelDownload, warmModels } from "@/ui/layout/ModelDownload";
import { EvidenceChip, evidenceKey, uniqueEvidence } from "./EvidenceChips";

const INTENT_LABEL: Record<QuestionIntent, string> = {
  factual: "Factual",
  summary: "Summary",
  limitations: "Limitations",
  method: "Method",
  results: "Results",
  comparison: "Comparison",
};

const INSPECTOR_ROWS = 8;

/** Splits `context` around the first occurrence of `span` (case-insensitive fallback). */
export function splitAroundSpan(context: string, span: string): [string, string, string] | null {
  if (!span) return null;
  let at = context.indexOf(span);
  if (at < 0) at = context.toLowerCase().indexOf(span.toLowerCase());
  if (at < 0) return null;
  return [context.slice(0, at), context.slice(at, at + span.length), context.slice(at + span.length)];
}

function ConfidenceMeter({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const color = value >= 0.6 ? "var(--good)" : value >= 0.35 ? "var(--accent)" : "var(--warn)";
  return (
    <div className="flex items-center gap-2" title="Combined confidence: passage relevance × reader probability × sentence support">
      <span className="text-[11px] text-ink-3">Confidence</span>
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-panel-2">
        <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="tabular text-[11px] font-medium text-ink-2">{pct.toFixed(0)}%</span>
    </div>
  );
}

const fmt = (n: number | undefined, digits = 3) => (n === undefined || !Number.isFinite(n) ? "—" : n.toFixed(digits));

function RetrievalInspector({ answer }: { answer: Answer }) {
  const [open, setOpen] = useState(false);
  if (answer.hits.length === 0) return null;
  const cited = new Set(answer.evidence.map((e) => e.chunkId).filter(Boolean));
  const hits: RetrievalHit[] = [...answer.hits].sort((a, b) => a.rank.final - b.rank.final).slice(0, INSPECTOR_ROWS);
  const timings = Object.entries(answer.timingsMs).filter(([step, ms]) => step !== "total" && Number.isFinite(ms));

  return (
    <div className="border-t border-line">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 px-4 py-2.5 text-xs font-medium text-ink-3 transition-colors hover:text-ink"
      >
        <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} />
        <Microscope className="size-3.5" />
        Retrieval inspector
        <span className="font-normal">· {answer.hits.length} passages scored</span>
      </button>
      {open && (
        <div className="animate-in px-4 pb-4">
          <div className="overflow-x-auto rounded-lg border border-line">
            <table className="w-full min-w-[560px] text-left text-xs">
              <thead className="bg-panel-2 text-[10.5px] tracking-wider text-ink-3 uppercase">
                <tr>
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Section</th>
                  <th className="px-2 py-2 text-right font-medium">Page</th>
                  <th className="px-2 py-2 text-right font-medium" title="Okapi BM25 lexical score (rank)">BM25</th>
                  <th className="px-2 py-2 text-right font-medium" title="Cosine similarity of MiniLM embeddings (rank)">Dense</th>
                  <th className="px-2 py-2 text-right font-medium" title="Reciprocal rank fusion, k = 60">RRF</th>
                  <th className="px-3 py-2 text-right font-medium" title="Cross-encoder logit">Rerank</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line tabular">
                {hits.map((hit) => {
                  const isCited = cited.has(hit.chunk.id);
                  return (
                    <tr key={hit.chunk.id} className={cn(isCited && "bg-accent-soft/50")} title={hit.chunk.text.slice(0, 300)}>
                      <td className="px-3 py-1.5 text-ink-3">{hit.rank.final}</td>
                      <td className="max-w-[220px] truncate px-3 py-1.5 text-ink-2">
                        {isCited && <span className="mr-1.5 rounded bg-accent px-1 text-[9px] font-semibold text-accent-ink">CITED</span>}
                        {hit.chunk.sectionTitle}
                      </td>
                      <td className="px-2 py-1.5 text-right text-ink-3">{hit.chunk.page}</td>
                      <td className="px-2 py-1.5 text-right text-ink-2">
                        {fmt(hit.bm25, 2)}
                        {hit.rank.bm25 !== undefined && <span className="ml-1 text-ink-3">({hit.rank.bm25})</span>}
                      </td>
                      <td className="px-2 py-1.5 text-right text-ink-2">
                        {fmt(hit.dense)}
                        {hit.rank.dense !== undefined && <span className="ml-1 text-ink-3">({hit.rank.dense})</span>}
                      </td>
                      <td className="px-2 py-1.5 text-right text-ink-2">{fmt(hit.rrf, 4)}</td>
                      <td className="px-3 py-1.5 text-right text-ink-2">{fmt(hit.rerank, 2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {timings.length > 0 && (
            <div className="mt-2.5 flex flex-wrap gap-1.5">
              {timings.map(([step, ms]) => (
                <span key={step} className="rounded-md bg-panel-2 px-1.5 py-0.5 font-mono text-[10px] text-ink-3">
                  {step} {ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function AnswerBody({ answer }: { answer: Answer }) {
  const parts = answer.context && answer.answer ? splitAroundSpan(answer.context, answer.answer) : null;
  if (parts) {
    const [before, span, after] = parts;
    return (
      <blockquote className="border-l-2 border-accent pl-3 text-sm leading-relaxed text-ink-2">
        {before}
        <mark className="rounded bg-accent-soft px-0.5 font-medium text-ink">{span}</mark>
        {after}
      </blockquote>
    );
  }
  return <p className="text-sm leading-relaxed whitespace-pre-line text-ink">{answer.answer}</p>;
}

function AnswerCard({ answer }: { answer: Answer }) {
  const abstained = answer.status === "abstained";
  const evidence = uniqueEvidence(answer.evidence);
  const total = answer.timingsMs.total ?? 0;

  return (
    <div className="animate-in space-y-2">
      <div className="flex justify-end">
        <p className="max-w-[85%] rounded-2xl rounded-br-md bg-panel-2 px-3.5 py-2 text-sm text-ink">{answer.question}</p>
      </div>
      <Card className={cn("overflow-hidden", abstained && "border-dashed bg-transparent shadow-none")}>
        <div className="flex flex-wrap items-center gap-2 px-4 pt-3.5">
          {abstained ? (
            <Badge tone="warn">
              <CircleSlash className="size-3" />
              Not found in the paper
            </Badge>
          ) : (
            <Badge tone="good">
              <Quote className="size-3" />
              Answer from the paper
            </Badge>
          )}
          <Badge>{INTENT_LABEL[answer.intent]}</Badge>
          <div className="ml-auto flex items-center gap-3">
            <ConfidenceMeter value={answer.confidence} />
            {total > 0 && <span className="tabular text-[11px] text-ink-3">{(total / 1000).toFixed(2)} s</span>}
          </div>
        </div>
        <div className="space-y-3 px-4 pt-3 pb-4">
          {abstained ? (
            <p className="text-sm leading-relaxed text-ink-2">{answer.reason ?? "The paper does not appear to answer this question."}</p>
          ) : (
            <AnswerBody answer={answer} />
          )}
          {evidence.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 text-[11px] text-ink-3">
              <span>{abstained ? "Closest passage:" : "Source:"}</span>
              {evidence.map((e) => (
                <span key={evidenceKey(e)} className="inline-flex items-center gap-1">
                  <EvidenceChip evidence={e} />
                  <span className="max-w-[220px] truncate">{e.section}</span>
                </span>
              ))}
            </div>
          )}
          {!abstained && answer.reason && <p className="text-[11px] leading-relaxed text-ink-3">{answer.reason}</p>}
        </div>
        <RetrievalInspector answer={answer} />
      </Card>
    </div>
  );
}

function PendingCard({ question }: { question: string | null }) {
  return (
    <div className="animate-in space-y-2">
      {question && (
        <div className="flex justify-end">
          <p className="max-w-[85%] rounded-2xl rounded-br-md bg-panel-2 px-3.5 py-2 text-sm text-ink">{question}</p>
        </div>
      )}
      <ModelDownload keys={["embedder", "reranker", "reader"]} />
      <Card className="flex items-center gap-2.5 px-4 py-3.5 text-sm text-ink-3">
        <Spinner />
        Finding the best passages and reading them for an answer…
      </Card>
    </div>
  );
}

function HowItWorks() {
  const steps = QA_NODES.filter((n) => n.name !== "abstain" && n.name !== "compose_from_analysis");
  return (
    <Card className="h-fit xl:sticky xl:top-4">
      <CardHeader icon={<Microscope className="size-4" />} title="How answers are found" subtitle="A small agent graph, run in your browser." />
      <CardBody>
        <ol className="relative space-y-3 before:absolute before:top-2 before:bottom-2 before:left-[9px] before:w-px before:bg-line">
          {steps.map((step, i) => (
            <li key={step.name} className="relative flex gap-3">
              <span className="z-10 flex size-[19px] shrink-0 items-center justify-center rounded-full border border-line bg-panel font-mono text-[9px] text-ink-3">
                {i + 1}
              </span>
              <div className="min-w-0">
                <p className="text-xs font-semibold text-ink">{step.label}</p>
                <p className="text-[11px] leading-relaxed text-ink-3">{step.description}</p>
              </div>
            </li>
          ))}
        </ol>
        <p className="mt-4 rounded-lg bg-panel-2 px-3 py-2 text-[11px] leading-relaxed text-ink-3">
          Abstains when confidence is below <span className="tabular text-ink-2">{ABSTAIN_THRESHOLD.toFixed(2)}</span> for
          an extracted span, or <span className="tabular text-ink-2">{SENTENCE_ABSTAIN_THRESHOLD.toFixed(2)}</span> for a quoted
          sentence.
        </p>
      </CardBody>
    </Card>
  );
}

export function AskTab({
  answers,
  pending,
  pendingQuestion,
  suggestions,
  onAsk,
}: {
  answers: Answer[];
  pending: boolean;
  pendingQuestion: string | null;
  suggestions: string[];
  onAsk: (question: string) => void;
}) {
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const asked = new Set(answers.map((a) => a.question));
  const freshSuggestions = suggestions.filter((q) => !asked.has(q)).slice(0, 4);

  // Fetch the answering models as soon as the tab opens, so the first question is quick.
  useEffect(() => warmModels(["reranker", "reader"]), []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [answers.length, pending]);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    const question = draft.trim();
    if (!question || pending) return;
    onAsk(question);
    setDraft("");
  };

  return (
    <div className="grid gap-4 animate-in xl:grid-cols-[minmax(0,1fr)_300px]">
      <div className="min-w-0 space-y-5">
        {answers.length === 0 && !pending && (
          <Card>
            <EmptyState
              icon={<MessageSquareText className="size-5" />}
              title="Ask anything about this paper"
              body="Questions are answered by locating the exact passage in the paper. If the paper does not say it, PaperTrail says so."
            />
          </Card>
        )}
        {answers.map((answer) => (
          <AnswerCard key={answer.id} answer={answer} />
        ))}
        {pending && <PendingCard question={pendingQuestion} />}
        <div ref={endRef} />

        {/* Opaque, with a fade above it, so earlier answers scroll cleanly underneath. */}
        <div className="sticky bottom-0 z-10 -mx-1 space-y-2 bg-bg px-1 pt-2 pb-3 before:pointer-events-none before:absolute before:inset-x-0 before:-top-6 before:h-6 before:bg-gradient-to-t before:from-bg before:to-transparent">
          {freshSuggestions.length > 0 && !pending && (
            <div className="flex flex-wrap gap-1.5">
              {freshSuggestions.map((q) => (
                <button
                  key={q}
                  type="button"
                  onClick={() => onAsk(q)}
                  className="max-w-full truncate rounded-full border border-line bg-panel px-3 py-1 text-xs text-ink-2 transition-colors hover:border-line-strong hover:text-ink"
                >
                  {q}
                </button>
              ))}
            </div>
          )}
          <form
            onSubmit={submit}
            className="flex items-center gap-2 rounded-2xl border border-line-strong bg-panel p-1.5 pl-4 shadow-card focus-within:border-accent"
          >
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Ask about the method, datasets, results, limitations…"
              aria-label="Ask a question about this paper"
              className="h-9 min-w-0 flex-1 bg-transparent text-sm text-ink outline-none placeholder:text-ink-3"
            />
            <button
              type="submit"
              disabled={!draft.trim() || pending}
              aria-label="Ask"
              className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent text-accent-ink transition-all hover:brightness-110 disabled:opacity-40"
            >
              {pending ? <Spinner className="text-accent-ink" /> : <ArrowUp className="size-4" />}
            </button>
          </form>
          <p className="px-1 text-[11px] text-ink-3">
            Answers are extracted verbatim from the paper; PaperTrail abstains when the evidence is weak.
          </p>
        </div>
      </div>
      <HowItWorks />
    </div>
  );
}

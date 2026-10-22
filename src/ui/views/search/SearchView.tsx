/**
 * Topic search results: ranked candidates with a transparent score
 * breakdown, multi-select (up to 10) and one-click batch analysis.
 */
import { useState } from "react";
import { CircleAlert, Info, Layers, SearchX, Sparkles } from "lucide-react";
import { RANK_WEIGHTS } from "@/sources/rank";
import { MAX_SURVEY_PAPERS } from "@/store/types";
import type { Candidate } from "@/types";
import { Button, Card, EmptyState, Skeleton } from "@/ui/kit";
import { store, useStore } from "@/ui/hooks/store";
import { focusCommandBar } from "@/ui/hooks/useHotkeys";
import { CandidateCard } from "./CandidateCard";

/** The ranking formula, read from the ranker itself so the UI cannot drift from it. */
function weightFormula(semantic: boolean): string {
  const w = semantic ? RANK_WEIGHTS.withEmbeddings : RANK_WEIGHTS.lexicalOnly;
  const parts = [
    [w.lexical, "BM25"],
    [w.semantic, "semantic"],
    [w.recency, "recency"],
    [w.impact, "impact"],
  ] as const;
  const terms = parts.filter(([weight]) => weight > 0).map(([weight, name]) => `${weight.toFixed(2)} ${name}`);
  return `Score = ${terms.join(" + ")}, diversified with MMR`;
}

function LoadingList() {
  return (
    <div className="space-y-3">
      {[0, 1, 2, 3].map((i) => (
        <Card key={i} className="p-5">
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="mt-2 h-3 w-1/3" />
          <Skeleton className="mt-4 h-3 w-full" />
          <Skeleton className="mt-2 h-3 w-5/6" />
          <div className="mt-4 grid grid-cols-4 gap-4">
            {[0, 1, 2, 3].map((j) => (
              <Skeleton key={j} className="h-1.5" />
            ))}
          </div>
        </Card>
      ))}
    </div>
  );
}

function Attempts({ query, attempts }: { query: string; attempts: string[] }) {
  const broadened = attempts.filter((a) => a !== query);
  if (broadened.length === 0) return null;
  return (
    <p className="mt-2 flex flex-wrap items-center gap-1.5 text-xs text-ink-3">
      <Sparkles className="size-3.5 text-accent" />
      No exact hits, so the query was broadened:
      {attempts.map((a, i) => (
        <span key={i} className="inline-flex items-center gap-1.5">
          {i > 0 && <span className="text-ink-3">→</span>}
          <code className="rounded bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-ink-2">{a}</code>
        </span>
      ))}
    </p>
  );
}

export default function SearchView() {
  const search = useStore((s) => s.search);
  const [selected, setSelected] = useState<string[]>([]);
  // The view stays mounted from one search to the next, so start every new
  // query with an empty selection instead of inheriting the last one's ids.
  const [selectionQuery, setSelectionQuery] = useState(search?.query);
  if (search && search.query !== selectionQuery) {
    setSelectionQuery(search.query);
    setSelected([]);
  }

  if (!search) {
    return (
      <EmptyState
        icon={<SearchX className="size-5" />}
        title="No search yet"
        body="Type a research topic in the command bar, for example “kv cache compression”."
        action={<Button onClick={focusCommandBar}>Start a search</Button>}
      />
    );
  }

  const { candidates } = search;
  const keyOf = (c: Candidate) => c.meta.id;
  const picked = candidates.filter((c) => selected.includes(keyOf(c)));
  const semanticAvailable = candidates.some((c) => c.scores.semantic > 0);
  const toggle = (c: Candidate) =>
    setSelected((prev) => {
      // Only ids among the current candidates count toward the cap, matching `picked`.
      const live = prev.filter((k) => candidates.some((x) => keyOf(x) === k));
      if (live.includes(keyOf(c))) return live.filter((k) => k !== keyOf(c));
      return live.length >= MAX_SURVEY_PAPERS ? live : [...live, keyOf(c)];
    });

  return (
    <div className="mx-auto max-w-4xl px-4 pt-8 pb-28 sm:px-8">
      <header>
        <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">Topic search</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">“{search.query}”</h1>
        <Attempts query={search.query} attempts={search.attempts} />
        {search.status === "ready" && (
          <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-3">
            <span className="tabular">{candidates.length} candidates from arXiv (via DataCite + OpenAlex)</span>
            <span className="inline-flex items-center gap-1.5" title="Final score blend; MMR then re-orders for diversity">
              <Info className="size-3.5" />
              {semanticAvailable ? "" : "Semantic model unavailable · "}
              {weightFormula(semanticAvailable)}
            </span>
          </div>
        )}
      </header>

      <div className="mt-6">
        {search.status === "searching" && <LoadingList />}
        {search.status === "failed" && (
          <Card className="p-2">
            <EmptyState
              icon={<CircleAlert className="size-5 text-bad" />}
              title="Search failed"
              body={search.error ?? "The paper indexes could not be reached. Check your connection and try again."}
              action={<Button onClick={() => void store().submit(search.query)}>Try again</Button>}
            />
          </Card>
        )}
        {search.status === "empty" && (
          <Card className="p-2">
            <EmptyState
              icon={<SearchX className="size-5" />}
              title="No matching papers"
              body="Even the broadened queries returned nothing. Try different keywords, or paste an arXiv ID directly."
              action={<Button onClick={focusCommandBar}>Edit search</Button>}
            />
          </Card>
        )}
        {search.status === "ready" && (
          <div className="space-y-3">
            {candidates.map((c) => (
              <CandidateCard
                key={keyOf(c)}
                candidate={c}
                selected={selected.includes(keyOf(c))}
                disabled={picked.length >= MAX_SURVEY_PAPERS}
                semanticAvailable={semanticAvailable}
                onToggle={() => toggle(c)}
                onAnalyze={() => void store().analyzeCandidates([c])}
              />
            ))}
          </div>
        )}
      </div>

      {search.status === "ready" && candidates.length > 0 && (
        <div className="pointer-events-none sticky bottom-4 z-10 mt-6 flex justify-center">
          <div className="pointer-events-auto flex items-center gap-3 rounded-2xl border border-line-strong bg-panel/95 py-2 pr-2 pl-4 shadow-card backdrop-blur">
            <span className="text-xs tabular text-ink-2">
              {picked.length === 0 ? "Select papers to analyse" : `${picked.length} of ${MAX_SURVEY_PAPERS} selected`}
            </span>
            {picked.length === 0 ? (
              <Button
                size="sm"
                onClick={() => setSelected(candidates.slice(0, 5).map(keyOf))}
              >
                Select top 5
              </Button>
            ) : (
              <Button size="sm" variant="ghost" onClick={() => setSelected([])}>
                Clear
              </Button>
            )}
            <Button
              size="sm"
              variant="primary"
              disabled={picked.length === 0}
              onClick={() => {
                void store().analyzeCandidates(picked);
                setSelected([]);
              }}
            >
              <Layers className="size-3.5" />
              Analyze {picked.length || ""} {picked.length === 1 ? "paper" : "papers"}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

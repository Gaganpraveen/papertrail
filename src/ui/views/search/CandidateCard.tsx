import { useState } from "react";
import { Check, ExternalLink, Quote } from "lucide-react";
import type { Candidate, CandidateScores } from "@/types";
import { cn, safeHref } from "@/ui/kit";

const SCORE_PARTS: { key: keyof Omit<CandidateScores, "final">; label: string; color: string }[] = [
  { key: "lexical", label: "Lexical", color: "var(--c-0)" },
  { key: "semantic", label: "Semantic", color: "var(--c-1)" },
  { key: "recency", label: "Recency", color: "var(--c-2)" },
  { key: "impact", label: "Impact", color: "var(--c-3)" },
];

function ScoreBar({ label, value, color, disabled }: { label: string; value: number; color: string; disabled?: boolean }) {
  return (
    <div className={cn("min-w-0", disabled && "opacity-40")} title={`${label}: ${disabled ? "n/a" : value.toFixed(2)}`}>
      <div className="flex items-baseline justify-between text-[10px] text-ink-3">
        <span>{label}</span>
        <span className="tabular">{disabled ? "n/a" : value.toFixed(2)}</span>
      </div>
      <div className="mt-1 h-1 overflow-hidden rounded-full bg-panel-2">
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: color }}
        />
      </div>
    </div>
  );
}

function authorLine(authors: string[]): string {
  if (authors.length === 0) return "Unknown authors";
  if (authors.length <= 3) return authors.join(", ");
  return `${authors.slice(0, 3).join(", ")} +${authors.length - 3}`;
}

export function CandidateCard({
  candidate,
  selected,
  disabled,
  semanticAvailable,
  onToggle,
  onAnalyze,
}: {
  candidate: Candidate;
  selected: boolean;
  disabled: boolean;
  semanticAvailable: boolean;
  onToggle: () => void;
  onAnalyze: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const { meta, scores } = candidate;
  const link = meta.landingUrl ?? (meta.arxivId ? `https://arxiv.org/abs/${meta.arxivId}` : undefined);

  return (
    <article
      className={cn(
        "animate-in rounded-2xl border bg-panel p-4 transition-all sm:p-5",
        selected ? "border-accent/70 shadow-[0_0_0_3px_var(--accent-soft)]" : "border-line hover:border-line-strong",
      )}
    >
      <div className="flex items-start gap-3">
        <button
          role="checkbox"
          aria-checked={selected}
          aria-label={selected ? "Deselect paper" : "Select paper"}
          disabled={disabled && !selected}
          onClick={onToggle}
          className={cn(
            "mt-0.5 flex size-[18px] shrink-0 items-center justify-center rounded-md border transition-colors disabled:opacity-40",
            selected ? "border-accent bg-accent text-accent-ink" : "border-line-strong hover:border-accent",
          )}
        >
          {selected && <Check className="size-3" strokeWidth={3} />}
        </button>

        <div className="min-w-0 flex-1">
          <div className="flex items-start gap-3">
            <h3 className="min-w-0 flex-1 text-[15px] font-semibold leading-snug tracking-tight text-ink">
              <span className="mr-2 font-mono text-[11px] font-normal tabular text-ink-3">#{candidate.rank}</span>
              {meta.title}
            </h3>
            <span
              className="shrink-0 rounded-md bg-accent-soft px-1.5 py-0.5 text-[11px] font-semibold tabular text-accent"
              title="Final blended score"
            >
              {scores.final.toFixed(2)}
            </span>
          </div>
          <p className="mt-1 truncate text-xs text-ink-3">
            {authorLine(meta.authors)}
            {meta.year ? ` · ${meta.year}` : ""}
            {meta.citationCount !== undefined ? ` · ${meta.citationCount.toLocaleString()} citations` : ""}
            {meta.arxivId ? ` · arXiv:${meta.arxivId}` : ""}
          </p>

          {meta.abstract && (
            <button
              onClick={() => setExpanded(!expanded)}
              className={cn(
                "mt-2.5 w-full text-left text-[13px] leading-relaxed text-ink-2",
                // line-clamp needs its own display value, so "block" only applies when expanded.
                expanded ? "block" : "line-clamp-3",
              )}
            >
              {meta.abstract}
            </button>
          )}

          <div className="mt-3.5 grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-4">
            {SCORE_PARTS.map((p) => (
              <ScoreBar
                key={p.key}
                label={p.label}
                value={scores[p.key]}
                color={p.color}
                disabled={p.key === "semantic" && !semanticAvailable}
              />
            ))}
          </div>

          <div className="mt-3.5 flex flex-wrap items-center gap-2">
            <span className="inline-flex min-w-0 items-center gap-1.5 text-[11px] text-ink-3">
              <Quote className="size-3 shrink-0" />
              <span className="truncate">{candidate.why}</span>
            </span>
            <span className="ml-auto flex items-center gap-1">
              {safeHref(link) && (
                <a
                  href={safeHref(link)}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="inline-flex h-7 items-center gap-1 rounded-md px-2 text-xs text-ink-3 hover:bg-panel-2 hover:text-ink"
                >
                  Source <ExternalLink className="size-3" />
                </a>
              )}
              <button
                onClick={onAnalyze}
                className="inline-flex h-7 items-center rounded-md border border-line px-2.5 text-xs font-medium text-ink-2 hover:border-line-strong hover:text-ink"
              >
                Analyze
              </button>
            </span>
          </div>
        </div>
      </div>
    </article>
  );
}

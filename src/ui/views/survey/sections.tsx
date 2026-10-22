/** Survey building blocks: themes, comparison matrix, consensus and gaps. */
import { Check, Minus } from "lucide-react";
import type { Evidence, SurveyResult } from "@/types";
import { cn, seriesColor } from "@/ui/kit";
import { store } from "@/ui/hooks/store";
import type { SeriesPaper } from "@/ui/charts/SemanticMap";

export function PaperTag({ paper, className }: { paper?: SeriesPaper; className?: string }) {
  if (!paper) return null;
  return (
    <span
      title={paper.title}
      className={cn("inline-flex items-center rounded-md px-1.5 py-0.5 text-[10.5px] font-semibold tabular", className)}
      style={{ color: paper.color, background: `color-mix(in oklab, ${paper.color} 15%, transparent)` }}
    >
      {paper.short}
    </span>
  );
}

function EvidenceQuote({ evidence, paper }: { evidence: Evidence; paper?: SeriesPaper }) {
  return (
    <button
      onClick={() => store().openEvidence(evidence)}
      className="group flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-panel-2"
    >
      <PaperTag paper={paper} className="mt-0.5 shrink-0" />
      <span className="min-w-0 flex-1 text-xs leading-relaxed text-ink-2">
        <span className="line-clamp-2">“{evidence.quote}”</span>
      </span>
      <span className="mt-0.5 shrink-0 text-[10px] tabular text-ink-3 group-hover:text-accent">p.{evidence.page}</span>
    </button>
  );
}

export function Themes({
  clusters,
  silhouette,
  byId,
}: {
  clusters: SurveyResult["clusters"];
  silhouette: number;
  byId: Map<string, SeriesPaper>;
}) {
  return (
    <div>
      <p className="mb-3 text-xs text-ink-3 tabular">
        {clusters.length} {clusters.length === 1 ? "theme" : "themes"} · average-linkage clustering ·
        silhouette {silhouette.toFixed(2)}
      </p>
      <div className="grid gap-3 md:grid-cols-2">
        {clusters.map((c) => (
          <div key={c.id} className="relative overflow-hidden rounded-xl border border-line bg-panel-2/40 p-4">
            <span className="absolute inset-y-0 left-0 w-1" style={{ background: seriesColor(c.color) }} />
            <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">Theme {c.id + 1}</p>
            <p className="mt-1 text-sm font-semibold tracking-tight text-ink">{c.label}</p>
            <div className="mt-2.5 flex flex-wrap gap-1">
              {c.terms.map((t) => (
                <span key={t} className="rounded-md border border-line bg-panel px-1.5 py-0.5 text-[11px] text-ink-2">
                  {t}
                </span>
              ))}
            </div>
            <ul className="mt-3 space-y-1">
              {c.paperIds.map((id) => (
                <li key={id} className="flex items-center gap-2 text-xs text-ink-2">
                  <PaperTag paper={byId.get(id)} />
                  <span className="truncate">{byId.get(id)?.title ?? id}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

function List({ items }: { items: string[] }) {
  if (items.length === 0) return <span className="text-ink-3">—</span>;
  return <span>{items.slice(0, 5).join(", ")}{items.length > 5 ? ` +${items.length - 5}` : ""}</span>;
}

export function ComparisonMatrix({
  rows,
  byId,
}: {
  rows: SurveyResult["comparison"];
  byId: Map<string, SeriesPaper>;
}) {
  const head = "px-3 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-ink-3 whitespace-nowrap";
  const cell = "px-3 py-3 align-top text-xs leading-relaxed text-ink-2";
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full min-w-[1180px] border-collapse">
        <thead className="bg-panel-2/60">
          <tr>
            <th className={cn(head, "sticky left-0 z-10 w-56 bg-panel-2")}>Paper</th>
            <th className={head}>Year</th>
            <th className={cn(head, "w-64")}>Problem</th>
            <th className={head}>Approach</th>
            <th className={head}>Datasets</th>
            <th className={head}>Metrics</th>
            <th className={cn(head, "w-64")}>Key result</th>
            <th className={head}>Limits</th>
            <th className={head}>Rigor</th>
            <th className={head}>Code</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const paper = byId.get(row.paperId);
            return (
              <tr key={row.paperId} className="border-t border-line">
                <td className={cn(cell, "sticky left-0 z-10 bg-panel")}>
                  <div className="flex items-start gap-2">
                    <PaperTag paper={paper} className="shrink-0" />
                    <button
                      onClick={() => store().setView({ kind: "paper", id: row.paperId, tab: "briefing" })}
                      className="line-clamp-3 text-left font-medium text-ink hover:text-accent"
                    >
                      {paper?.title ?? row.paperId}
                    </button>
                  </div>
                </td>
                <td className={cn(cell, "tabular")}>{row.year ?? "—"}</td>
                <td className={cell}>
                  <span className="line-clamp-4">{row.problem || "—"}</span>
                </td>
                <td className={cell}>
                  <div className="flex w-52 flex-wrap gap-1">
                    {row.approach.length === 0 && <span className="text-ink-3">—</span>}
                    {row.approach.slice(0, 5).map((a) => (
                      <span key={a} className="rounded border border-line bg-panel-2 px-1.5 py-0.5 text-[11px] whitespace-nowrap">
                        {a}
                      </span>
                    ))}
                  </div>
                </td>
                <td className={cell}>
                  <List items={row.datasets} />
                </td>
                <td className={cell}>
                  <List items={row.metrics} />
                </td>
                <td className={cell}>
                  <span className="line-clamp-4">{row.keyResult || "—"}</span>
                </td>
                <td className={cn(cell, "tabular")}>{row.limitations}</td>
                <td className={cell}>
                  <div className="flex items-center gap-2">
                    <span className="h-1.5 w-12 overflow-hidden rounded-full bg-panel-2">
                      <span
                        className="block h-full rounded-full"
                        style={{
                          width: `${row.rigor}%`,
                          background: row.rigor >= 75 ? "var(--good)" : row.rigor >= 55 ? "var(--accent)" : row.rigor >= 40 ? "var(--warn)" : "var(--bad)",
                        }}
                      />
                    </span>
                    <span className="tabular text-ink">{Math.round(row.rigor)}</span>
                  </div>
                </td>
                <td className={cell}>
                  {row.code ? (
                    <Check className="size-4 text-good" aria-label="Code released" />
                  ) : (
                    <Minus className="size-4 text-ink-3" aria-label="No code link found" />
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Consensus({
  items,
  byId,
}: {
  items: SurveyResult["consensus"];
  byId: Map<string, SeriesPaper>;
}) {
  if (items.length === 0) {
    return (
      <p className="text-sm text-ink-3">
        No result or contribution statements were semantically close enough across papers to count as consensus.
      </p>
    );
  }
  return (
    <ul className="space-y-3">
      {items.map((item, i) => (
        <li key={i} className="rounded-xl border border-line bg-panel-2/30 p-4">
          <div className="flex items-start gap-3">
            <p className="min-w-0 flex-1 text-[13px] leading-relaxed font-medium text-ink">{item.statement}</p>
            <span className="flex shrink-0 gap-1">
              {item.paperIds.map((id) => (
                <PaperTag key={id} paper={byId.get(id)} />
              ))}
            </span>
          </div>
          {item.support.length > 0 && (
            <div className="mt-2 -mx-2 space-y-0.5">
              {item.support.slice(0, 4).map((ev, j) => (
                <EvidenceQuote key={j} evidence={ev} paper={byId.get(ev.paperId)} />
              ))}
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

export function Gaps({ items, byId }: { items: SurveyResult["gaps"]; byId: Map<string, SeriesPaper> }) {
  if (items.length === 0) {
    return <p className="text-sm text-ink-3">No future-work or limitation statements were found to derive gaps from.</p>;
  }
  return (
    <ol className="space-y-3">
      {items.map((gap, i) => (
        <li key={i} className="flex gap-3 rounded-xl border border-line bg-panel-2/30 p-4">
          <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-warn-soft text-[11px] font-semibold tabular text-warn">
            {i + 1}
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-[13px] leading-relaxed font-medium text-ink">{gap.statement}</p>
            {gap.evidence.length > 0 && (
              <div className="mt-2 -mx-2 space-y-0.5">
                {gap.evidence.slice(0, 3).map((ev, j) => (
                  <EvidenceQuote key={j} evidence={ev} paper={byId.get(ev.paperId)} />
                ))}
              </div>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}

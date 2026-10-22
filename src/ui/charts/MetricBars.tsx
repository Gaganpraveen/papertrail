/**
 * Key numbers extracted from the paper (accuracy, speed-ups, memory savings).
 * Values are parsed back into a number and unit so each bar is scaled against
 * comparable values only: percentages on a 0–100 scale, every other unit
 * relative to the largest value with the same unit. Mixing units on one axis
 * would be misleading, so we never do it.
 */
import type { Evidence, KeyNumber } from "@/types";
import { cn } from "@/ui/kit";
import { ChartEmpty } from "./common";

interface ParsedValue {
  magnitude: number;
  unit: string;
}

/** "93.2%" -> {93.2, "%"}, "2.6×" -> {2.6, "x"}, "+4.1 BLEU" -> {4.1, "bleu"}. */
export function parseValue(value: string): ParsedValue | null {
  const match = value.replace(/,/g, "").match(/[-+−]?\d+(?:\.\d+)?/);
  if (!match) return null;
  const magnitude = Math.abs(Number.parseFloat(match[0].replace("−", "-")));
  if (!Number.isFinite(magnitude)) return null;
  const rest = value.slice((match.index ?? 0) + match[0].length).trim().toLowerCase();
  const unit = rest.startsWith("%")
    ? "%"
    : /^[×x](?![a-z])/.test(rest)
      ? "x"
      : rest.split(/\s+/)[0] || "";
  return { magnitude, unit };
}

function barFraction(parsed: ParsedValue | null, maxByUnit: Map<string, number>): number {
  if (!parsed) return 0;
  if (parsed.unit === "%" && parsed.magnitude <= 100) return parsed.magnitude / 100;
  const max = maxByUnit.get(parsed.unit) ?? 0;
  return max > 0 ? parsed.magnitude / max : 0;
}

export function MetricBars({
  numbers,
  onSelect,
  limit = 10,
}: {
  numbers: KeyNumber[];
  onSelect?: (evidence: Evidence) => void;
  limit?: number;
}) {
  if (numbers.length === 0) {
    return <ChartEmpty>No quantitative results were found in the text.</ChartEmpty>;
  }

  const rows = numbers.slice(0, limit).map((n) => ({ ...n, parsed: parseValue(n.value) }));
  const maxByUnit = new Map<string, number>();
  for (const r of rows) {
    if (!r.parsed) continue;
    maxByUnit.set(r.parsed.unit, Math.max(maxByUnit.get(r.parsed.unit) ?? 0, r.parsed.magnitude));
  }

  return (
    <ul className="divide-y divide-line" aria-label="Key numbers reported in the paper">
      {rows.map((row, i) => {
        const fraction = barFraction(row.parsed, maxByUnit);
        const scaleNote = row.parsed?.unit === "%" ? "on a 0–100% scale" : "relative to the largest value with the same unit";
        return (
          <li key={`${row.value}-${i}`}>
            <button
              type="button"
              disabled={!onSelect}
              onClick={() => onSelect?.(row.evidence)}
              title={`“${row.evidence.quote}” (bar ${scaleNote})`}
              className={cn(
                "group grid w-full grid-cols-[88px_1fr] items-center gap-3 py-2.5 text-left",
                onSelect && "cursor-pointer",
              )}
            >
              <span className="tabular text-[15px] font-semibold tracking-tight text-ink">{row.value}</span>
              <span className="min-w-0">
                <span className="flex items-baseline justify-between gap-2">
                  <span className="truncate text-xs text-ink-2 group-hover:text-ink">{row.label || "Reported value"}</span>
                  <span className="shrink-0 font-mono text-[10px] text-ink-3 group-hover:text-accent">
                    {row.evidence.page > 0 ? `p.${row.evidence.page}` : "abs"}
                  </span>
                </span>
                <span className="mt-1.5 block h-1.5 w-full overflow-hidden rounded-full bg-panel-2" aria-hidden>
                  <span
                    className="block h-full rounded-full opacity-85 transition-[width] duration-500"
                    style={{ width: `${Math.max(2, fraction * 100)}%`, background: "var(--z-result)" }}
                  />
                </span>
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * Histogram of the publication years of a paper's references, with the
 * median year marked. A median far behind the paper's own year can signal a
 * dated literature review; a tight recent cluster suggests a fast-moving area.
 */
import { useState } from "react";
import type { VisualData } from "@/types";
import { ChartEmpty, ChartTooltip, niceTicks, useSize } from "./common";

type YearCount = VisualData["citationYears"][number];

interface Bin {
  start: number;
  end: number; // inclusive
  count: number;
}

const MARGIN = { top: 22, right: 8, bottom: 22, left: 28 };
// Above ~30 distinct years single-year bars get too thin, so we bin by 5 years.
const MAX_SINGLE_YEAR_SPAN = 30;

export function medianYear(years: YearCount[]): number | null {
  const total = years.reduce((s, y) => s + y.count, 0);
  if (total === 0) return null;
  const sorted = [...years].sort((a, b) => a.year - b.year);
  let seen = 0;
  for (const y of sorted) {
    seen += y.count;
    if (seen >= total / 2) return y.year;
  }
  return sorted[sorted.length - 1].year;
}

function toBins(years: YearCount[]): Bin[] {
  const min = Math.min(...years.map((y) => y.year));
  const max = Math.max(...years.map((y) => y.year));
  const width = max - min + 1 > MAX_SINGLE_YEAR_SPAN ? 5 : 1;
  const first = width === 1 ? min : Math.floor(min / 5) * 5;
  const bins: Bin[] = [];
  for (let start = first; start <= max; start += width) bins.push({ start, end: start + width - 1, count: 0 });
  for (const y of years) bins[Math.floor((y.year - first) / width)].count += y.count;
  return bins;
}

export function CitationAges({
  years,
  paperYear,
  height = 220,
}: {
  years: YearCount[];
  paperYear?: number;
  height?: number;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const valid = years.filter((y) => y.count > 0 && y.year > 1900);

  if (valid.length === 0) {
    return <ChartEmpty>No reference years could be parsed from the bibliography.</ChartEmpty>;
  }

  const bins = toBins(valid);
  const median = medianYear(valid);
  const total = valid.reduce((s, y) => s + y.count, 0);
  const maxCount = Math.max(...bins.map((b) => b.count));
  const ticks = niceTicks(maxCount, 3);
  const yMax = ticks[ticks.length - 1];
  const plotW = Math.max(0, width - MARGIN.left - MARGIN.right);
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const band = plotW / bins.length;
  const barW = Math.max(1, band - Math.min(4, band * 0.2));
  const xOfYear = (year: number) => {
    const span = bins[0].end - bins[0].start + 1;
    return MARGIN.left + ((year - bins[0].start + 0.5) / span) * band;
  };
  const y = (count: number) => MARGIN.top + plotH - (count / yMax) * plotH;
  const labelEvery = Math.max(1, Math.ceil(36 / Math.max(1, band)));
  const active = hover !== null ? bins[hover] : null;
  const recent = paperYear ? valid.filter((v) => paperYear - v.year <= 5).reduce((s, v) => s + v.count, 0) : null;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-ink-3">
        <span>
          <span className="tabular font-semibold text-ink">{total}</span> dated references
        </span>
        {median && (
          <span>
            median <span className="tabular font-semibold text-ink">{median}</span>
            {paperYear ? <span className="tabular"> ({Math.max(0, paperYear - median)} yrs before paper)</span> : null}
          </span>
        )}
        {recent !== null && (
          <span>
            <span className="tabular font-semibold text-ink">{Math.round((recent / total) * 100)}%</span> from the last 5 years
          </span>
        )}
      </div>
      <div ref={ref} className="relative w-full" style={{ height }} onMouseLeave={() => setHover(null)}>
        {width > 0 && (
          <svg width={width} height={height} role="img" aria-labelledby="cite-age-title">
            <title id="cite-age-title">{`Histogram of reference publication years${median ? `, median ${median}` : ""}`}</title>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={MARGIN.left} x2={width - MARGIN.right} y1={y(t)} y2={y(t)} stroke="var(--line)" strokeDasharray={t === 0 ? undefined : "2 3"} />
                <text x={MARGIN.left - 6} y={y(t) + 3.5} textAnchor="end" fontSize={10} fill="var(--ink-3)" className="tabular">
                  {t}
                </text>
              </g>
            ))}
            {bins.map((b, i) => (
              <g key={b.start} onMouseEnter={() => setHover(i)}>
                <rect x={MARGIN.left + i * band} y={MARGIN.top} width={band} height={plotH} fill="transparent" />
                <rect
                  x={MARGIN.left + i * band + (band - barW) / 2}
                  y={y(b.count)}
                  width={barW}
                  height={Math.max(0, MARGIN.top + plotH - y(b.count))}
                  rx={Math.min(3, barW / 3)}
                  fill={hover === i ? "var(--accent)" : "var(--c-0)"}
                  fillOpacity={hover === null || hover === i ? 0.85 : 0.45}
                />
                {i % labelEvery === 0 && (
                  <text
                    x={MARGIN.left + i * band + band / 2}
                    y={height - 6}
                    textAnchor="middle"
                    fontSize={10}
                    fill="var(--ink-3)"
                    className="tabular"
                  >
                    {b.start === b.end ? b.start : `${b.start}–${String(b.end).slice(2)}`}
                  </text>
                )}
              </g>
            ))}
            {median && (
              <g pointerEvents="none">
                <line x1={xOfYear(median)} x2={xOfYear(median)} y1={MARGIN.top - 6} y2={MARGIN.top + plotH} stroke="var(--warn)" strokeWidth={1.5} strokeDasharray="4 3" />
                <text x={xOfYear(median)} y={MARGIN.top - 10} textAnchor="middle" fontSize={10} fontWeight={600} fill="var(--warn)" className="tabular">
                  median {median}
                </text>
              </g>
            )}
          </svg>
        )}
        {active && hover !== null && (
          <ChartTooltip x={MARGIN.left + hover * band + band / 2} y={y(active.count)} containerWidth={width}>
            <div className="font-semibold text-ink tabular">
              {active.start === active.end ? active.start : `${active.start}–${active.end}`}
            </div>
            <div className="tabular">
              {active.count} reference{active.count === 1 ? "" : "s"} ({Math.round((active.count / total) * 100)}%)
            </div>
          </ChartTooltip>
        )}
      </div>
    </div>
  );
}

/**
 * Publication timeline. Papers sit on a time axis; labels alternate above
 * and below the axis and are pushed apart when they would collide.
 */
import { useMemo, useState } from "react";
import { ChartEmpty, ChartTooltip, useSize } from "./common";
import type { SeriesPaper } from "./SemanticMap";

const HEIGHT = 200;
const AXIS_Y = HEIGHT / 2;
const PAD_X = 28;
const LABEL_W = 132;

interface Item {
  paperId: string;
  date: string;
  title: string;
}

function toTime(date: string): number {
  const t = Date.parse(date.length === 4 ? `${date}-07-01` : date);
  return Number.isFinite(t) ? t : NaN;
}

const YEAR_MS = 365.25 * 24 * 3600 * 1000;

export function Timeline({ items, papers }: { items: Item[]; papers: SeriesPaper[] }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const byId = useMemo(() => new Map(papers.map((p) => [p.id, p])), [papers]);

  const dated = useMemo(
    () =>
      items
        .map((item) => ({ ...item, t: toTime(item.date) }))
        .filter((item) => Number.isFinite(item.t))
        .sort((a, b) => a.t - b.t),
    [items],
  );

  if (dated.length === 0) return <ChartEmpty>No publication dates available.</ChartEmpty>;

  // Pad the range by a few months so the first and last papers are not on the edge.
  const t0 = dated[0].t - YEAR_MS / 4;
  const t1 = dated[dated.length - 1].t + YEAR_MS / 4;
  const x = (t: number) => PAD_X + ((t - t0) / (t1 - t0)) * Math.max(1, width - PAD_X * 2);
  const years: number[] = [];
  for (let y = new Date(t0).getFullYear() + 1; y <= new Date(t1).getFullYear(); y++) years.push(y);

  // Alternate sides, and stack further out when a label would overlap the previous one on its side.
  const lastRight = [-Infinity, -Infinity];
  const placed = dated.map((item, i) => {
    const side = i % 2;
    const cx = x(item.t);
    const level = cx - LABEL_W / 2 < lastRight[side] ? 1 : 0;
    lastRight[side] = cx + LABEL_W / 2;
    const offset = 34 + level * 30;
    return { ...item, cx, labelY: side === 0 ? AXIS_Y - offset : AXIS_Y + offset };
  });

  return (
    <div ref={ref} className="relative w-full" style={{ height: HEIGHT }} onMouseLeave={() => setHover(null)}>
      {width > 0 && (
        <svg width={width} height={HEIGHT} role="img" aria-label="Publication timeline">
          <line x1={PAD_X} x2={width - PAD_X} y1={AXIS_Y} y2={AXIS_Y} stroke="var(--line-strong)" />
          {years.map((year) => {
            const yx = x(Date.parse(`${year}-01-01`));
            return (
              <g key={year}>
                <line x1={yx} x2={yx} y1={AXIS_Y - 4} y2={AXIS_Y + 4} stroke="var(--line-strong)" />
                <text x={yx} y={AXIS_Y + 16} textAnchor="middle" fontSize={10} fill="var(--ink-3)" className="tabular">
                  {year}
                </text>
              </g>
            );
          })}
          {placed.map((item, i) => {
            const paper = byId.get(item.paperId);
            const color = paper?.color ?? "var(--ink-3)";
            const above = item.labelY < AXIS_Y;
            return (
              <g key={item.paperId} onMouseEnter={() => setHover(i)} className="cursor-default">
                <line
                  x1={item.cx}
                  x2={item.cx}
                  y1={AXIS_Y}
                  y2={item.labelY + (above ? 6 : -12)}
                  stroke={color}
                  strokeOpacity={0.5}
                  strokeDasharray="2 3"
                />
                <circle cx={item.cx} cy={AXIS_Y} r={hover === i ? 7 : 5.5} fill={color} stroke="var(--panel)" strokeWidth={2} />
                <foreignObject x={item.cx - LABEL_W / 2} y={item.labelY - (above ? 18 : 6)} width={LABEL_W} height={22}>
                  <div className="truncate text-center text-[11px] leading-[20px] text-ink-2">
                    <span className="font-semibold" style={{ color }}>
                      {paper?.short}
                    </span>{" "}
                    {paper?.title ?? item.title}
                  </div>
                </foreignObject>
              </g>
            );
          })}
        </svg>
      )}
      {hover !== null && placed[hover] && (
        <ChartTooltip x={placed[hover].cx} y={AXIS_Y - 8} containerWidth={width}>
          <p className="font-medium text-ink">{placed[hover].title}</p>
          <p className="mt-0.5 tabular">{placed[hover].date}</p>
        </ChartTooltip>
      )}
    </div>
  );
}

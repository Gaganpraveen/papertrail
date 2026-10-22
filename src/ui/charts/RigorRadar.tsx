/**
 * Radar chart of the seven rigor axes. Each spoke is one axis scored 0..100;
 * hovering a spoke shows the score, its weight in the overall grade and the
 * plain-English explanation produced by the rigor model.
 */
import { useState } from "react";
import type { RigorProfile } from "@/types";
import { ChartEmpty, ChartTooltip, useSize } from "./common";

const RINGS = [25, 50, 75, 100];
// Room reserved for axis labels: vertical for two text lines, horizontal for
// the longest label ("Reproducibility" is ~90px at 11px Inter).
const LABEL_MARGIN_Y = 42;
const LABEL_MARGIN_X = 104;

export function RigorRadar({ rigor, height = 300 }: { rigor: RigorProfile; height?: number }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [active, setActive] = useState<number | null>(null);
  const axes = rigor.axes;

  if (axes.length < 3) return <ChartEmpty>Not enough rigor axes to draw a radar.</ChartEmpty>;

  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.max(40, Math.min(width / 2 - LABEL_MARGIN_X, height / 2 - LABEL_MARGIN_Y));
  const angle = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / axes.length;
  const polar = (i: number, r: number) =>
    [cx + r * Math.cos(angle(i)), cy + r * Math.sin(angle(i))] as const;
  const point = (i: number, value: number) => polar(i, (Math.max(0, Math.min(100, value)) / 100) * radius);
  const ringPath = (value: number) =>
    axes.map((_, i) => point(i, value).join(",")).join(" ");
  const dataPath = axes.map((a, i) => point(i, a.score).join(",")).join(" ");
  const activeAxis = active !== null ? axes[active] : null;
  const activePoint = active !== null ? point(active, axes[active].score) : null;

  return (
    <div ref={ref} className="relative w-full" style={{ height }} onMouseLeave={() => setActive(null)}>
      {width > 0 && (
        <svg width={width} height={height} role="img" overflow="visible" aria-labelledby="rigor-radar-title">
          <title id="rigor-radar-title">
            {`Rigor profile: overall ${Math.round(rigor.overall)} of 100, grade ${rigor.grade}`}
          </title>
          {RINGS.map((ring) => (
            <polygon key={ring} points={ringPath(ring)} fill="none" stroke="var(--line)" strokeWidth={1} />
          ))}
          {RINGS.map((ring) => (
            <text
              key={`l${ring}`}
              x={cx + 3}
              y={cy - (ring / 100) * radius - 2}
              fontSize={9}
              fill="var(--ink-3)"
              className="tabular"
            >
              {ring}
            </text>
          ))}
          {axes.map((axis, i) => {
            const [x, y] = point(i, 100);
            return (
              <line
                key={axis.key}
                x1={cx}
                y1={cy}
                x2={x}
                y2={y}
                stroke={active === i ? "var(--accent)" : "var(--line)"}
                strokeWidth={active === i ? 1.5 : 1}
              />
            );
          })}
          <polygon
            points={dataPath}
            fill="var(--accent)"
            fillOpacity={0.16}
            stroke="var(--accent)"
            strokeWidth={1.75}
            strokeLinejoin="round"
          />
          {axes.map((axis, i) => {
            const [x, y] = point(i, axis.score);
            const [lx, ly] = polar(i, radius + 14);
            const cos = Math.cos(angle(i));
            const sin = Math.sin(angle(i));
            const baseY = ly + (sin > 0.3 ? 10 : sin < -0.3 ? -6 : 0);
            const anchor = Math.abs(cos) < 0.2 ? "middle" : cos > 0 ? "start" : "end";
            return (
              <g key={axis.key} onMouseEnter={() => setActive(i)} className="cursor-default">
                <line x1={cx} y1={cy} x2={point(i, 100)[0]} y2={point(i, 100)[1]} stroke="transparent" strokeWidth={18} />
                <circle cx={x} cy={y} r={active === i ? 5 : 3.5} fill="var(--panel)" stroke="var(--accent)" strokeWidth={2} />
                <text x={lx} y={baseY - 5} textAnchor={anchor} fontSize={11} fontWeight={500} fill={active === i ? "var(--ink)" : "var(--ink-2)"}>
                  {axis.label}
                </text>
                <text x={lx} y={baseY + 9} textAnchor={anchor} fontSize={10.5} fill="var(--ink-3)" className="tabular">
                  {Math.round(axis.score)}
                </text>
              </g>
            );
          })}
        </svg>
      )}
      {activeAxis && activePoint && (
        <ChartTooltip x={activePoint[0]} y={activePoint[1]} containerWidth={width}>
          <div className="flex items-baseline justify-between gap-3">
            <span className="font-semibold text-ink">{activeAxis.label}</span>
            <span className="tabular text-ink">{Math.round(activeAxis.score)}/100</span>
          </div>
          <div className="mt-0.5 text-[11px] text-ink-3 tabular">
            {Math.round(activeAxis.weight * 100)}% of overall score
          </div>
          <p className="mt-1.5">{activeAxis.explanation}</p>
        </ChartTooltip>
      )}
    </div>
  );
}

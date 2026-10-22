/**
 * How many sentences fall into each argumentative zone. A paper heavy on
 * "method" but light on "result" or "limitation" reads very differently from
 * one that spends most of its words on evidence.
 */
import { useState } from "react";
import type { VisualData } from "@/types";
import { ZONES, zoneColor } from "@/ui/kit";
import { ChartEmpty, ChartTooltip, useSize } from "./common";

const LABEL_WIDTH = 92;
const VALUE_WIDTH = 72;
const ROW_HEIGHT = 26;

export function ZoneDistribution({ data }: { data: VisualData["zoneDistribution"] }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);

  const total = data.reduce((sum, d) => sum + d.count, 0);
  const order = new Map(ZONES.map((z, i) => [z.zone, i]));
  const rows = data
    .filter((d) => d.count > 0)
    .sort((a, b) => b.count - a.count || (order.get(a.zone) ?? 0) - (order.get(b.zone) ?? 0));

  if (total === 0) return <ChartEmpty>No zoned sentences yet.</ChartEmpty>;

  const max = Math.max(...rows.map((r) => r.count));
  const barSpace = Math.max(0, width - LABEL_WIDTH - VALUE_WIDTH);
  const height = rows.length * ROW_HEIGHT;
  const active = hover !== null ? rows[hover] : null;
  const meta = active ? ZONES.find((z) => z.zone === active.zone) : undefined;

  return (
    <div ref={ref} className="relative w-full" style={{ height }} onMouseLeave={() => setHover(null)}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-labelledby="zone-dist-title">
          <title id="zone-dist-title">{`Sentence count per argumentative zone, ${total} sentences in total`}</title>
          {rows.map((row, i) => {
            const y = i * ROW_HEIGHT;
            const w = Math.max(2, (row.count / max) * barSpace);
            const label = ZONES.find((z) => z.zone === row.zone)?.label ?? row.zone;
            return (
              <g key={row.zone} onMouseEnter={() => setHover(i)} opacity={hover === null || hover === i ? 1 : 0.55}>
                <rect x={0} y={y} width={width} height={ROW_HEIGHT} fill="transparent" />
                <circle cx={5} cy={y + ROW_HEIGHT / 2} r={3.5} fill={zoneColor(row.zone)} />
                <text x={15} y={y + ROW_HEIGHT / 2 + 4} fontSize={12} fill="var(--ink-2)">
                  {label}
                </text>
                <rect x={LABEL_WIDTH} y={y + 7} width={barSpace} height={ROW_HEIGHT - 14} rx={3} fill="var(--panel-2)" />
                <rect x={LABEL_WIDTH} y={y + 7} width={w} height={ROW_HEIGHT - 14} rx={3} fill={zoneColor(row.zone)} />
                <text
                  x={width}
                  y={y + ROW_HEIGHT / 2 + 4}
                  fontSize={11.5}
                  textAnchor="end"
                  fill="var(--ink-2)"
                  className="tabular"
                >
                  {row.count}
                  <tspan fill="var(--ink-3)">{`  ${Math.round((row.count / total) * 100)}%`}</tspan>
                </text>
              </g>
            );
          })}
        </svg>
      )}
      {active && hover !== null && (
        <ChartTooltip x={LABEL_WIDTH + (active.count / max) * barSpace} y={hover * ROW_HEIGHT + 6} containerWidth={width}>
          <div className="font-semibold text-ink">{meta?.label ?? active.zone}</div>
          <div className="text-ink-3">{meta?.description}</div>
          <div className="mt-1 tabular text-ink-2">
            {active.count} of {total} sentences ({((active.count / total) * 100).toFixed(1)}%)
          </div>
        </ChartTooltip>
      )}
    </div>
  );
}

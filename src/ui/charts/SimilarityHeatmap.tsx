/**
 * Paper-by-paper cosine similarity as a heatmap. Colour intensity is scaled
 * between the lowest and highest off-diagonal value so that differences
 * remain visible even when every pair is fairly similar; the legend shows
 * the actual range.
 */
import { useState } from "react";
import { ChartTooltip, useSize } from "./common";
import type { SeriesPaper } from "./SemanticMap";

const LABEL_W = 34;
const MAX_CELL = 64;

function offDiagonalRange(matrix: number[][]): [number, number] {
  const values = matrix.flatMap((row, i) => row.filter((_, j) => j !== i));
  if (values.length === 0) return [0, 1];
  return [Math.min(...values), Math.max(...values)];
}

export function SimilarityHeatmap({ matrix, papers }: { matrix: number[][]; papers: SeriesPaper[] }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const n = matrix.length;
  const cell = Math.max(18, Math.min(MAX_CELL, (width - LABEL_W) / Math.max(1, n)));
  const size = LABEL_W + cell * n;
  const [lo, hi] = offDiagonalRange(matrix);
  const intensity = (v: number) => (hi - lo < 1e-6 ? 0.6 : (v - lo) / (hi - lo));
  const fill = (v: number) => `color-mix(in oklab, var(--accent) ${Math.round(12 + intensity(v) * 80)}%, var(--panel-2))`;

  return (
    <div>
      <div ref={ref} className="relative w-full" onMouseLeave={() => setHover(null)}>
        {width > 0 && (
          <svg width={size} height={size} role="img" aria-label="Paper similarity heatmap" className="mx-auto block">
            {papers.map((p, i) => (
              <g key={p.id}>
                <text
                  x={LABEL_W - 8}
                  y={LABEL_W + cell * i + cell / 2 + 4}
                  textAnchor="end"
                  fontSize={11}
                  fontWeight={600}
                  fill={hover && (hover.i === i || hover.j === i) ? "var(--ink)" : p.color}
                >
                  {i + 1}
                </text>
                <text
                  x={LABEL_W + cell * i + cell / 2}
                  y={LABEL_W - 10}
                  textAnchor="middle"
                  fontSize={11}
                  fontWeight={600}
                  fill={hover && (hover.i === i || hover.j === i) ? "var(--ink)" : p.color}
                >
                  {i + 1}
                </text>
              </g>
            ))}
            {matrix.map((row, i) =>
              row.map((v, j) => {
                const x = LABEL_W + cell * j;
                const y = LABEL_W + cell * i;
                const diagonal = i === j;
                const active = hover && (hover.i === i || hover.j === j);
                return (
                  <g key={`${i}-${j}`} onMouseEnter={() => setHover({ i, j })}>
                    <rect
                      x={x + 1.5}
                      y={y + 1.5}
                      width={cell - 3}
                      height={cell - 3}
                      rx={Math.min(8, cell / 5)}
                      fill={diagonal ? "var(--panel-2)" : fill(v)}
                      stroke={hover?.i === i && hover?.j === j ? "var(--ink)" : "none"}
                      opacity={hover && !active ? 0.55 : 1}
                    />
                    {cell >= 38 && (
                      <text
                        x={x + cell / 2}
                        y={y + cell / 2 + 4}
                        textAnchor="middle"
                        fontSize={10.5}
                        className="tabular"
                        fill={diagonal ? "var(--ink-3)" : intensity(v) > 0.6 ? "var(--accent-ink)" : "var(--ink-2)"}
                      >
                        {diagonal ? "—" : v.toFixed(2)}
                      </text>
                    )}
                  </g>
                );
              }),
            )}
          </svg>
        )}
        {hover && hover.i !== hover.j && (
          <ChartTooltip
            x={(width - size) / 2 + LABEL_W + cell * hover.j + cell / 2}
            y={LABEL_W + cell * hover.i}
            containerWidth={width}
          >
            <p className="font-medium text-ink tabular">cosine {matrix[hover.i][hover.j].toFixed(3)}</p>
            <p className="mt-1">
              {papers[hover.i]?.short} {papers[hover.i]?.title}
            </p>
            <p className="mt-0.5">
              {papers[hover.j]?.short} {papers[hover.j]?.title}
            </p>
          </ChartTooltip>
        )}
      </div>
      <div className="mt-3 flex items-center justify-center gap-2 text-[11px] tabular text-ink-3">
        <span>{lo.toFixed(2)}</span>
        <span
          className="h-1.5 w-28 rounded-full"
          style={{ background: `linear-gradient(90deg, ${fill(lo)}, ${fill(hi)})` }}
        />
        <span>{hi.toFixed(2)}</span>
        <span className="ml-1">cosine similarity</span>
      </div>
    </div>
  );
}

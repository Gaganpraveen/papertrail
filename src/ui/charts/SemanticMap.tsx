/**
 * Semantic map: every passage of every paper projected to 2-D (UMAP, or
 * classical MDS for tiny inputs), coloured by paper, with each paper's
 * centroid drawn larger. Nearby points are passages that say similar things.
 */
import { useMemo, useState } from "react";
import type { SurveyResult } from "@/types";
import { ChartEmpty, ChartTooltip, Legend, useSize } from "./common";

/** A paper as the survey charts label it: "[n]" plus a short title and colour. */
export interface SeriesPaper {
  id: string;
  short: string;
  title: string;
  color: string;
}

type MapPoint = SurveyResult["map"]["points"][number];

const HEIGHT = 380;
const PAD = 24;

export function SemanticMap({
  points,
  papers,
  describe,
  onSelect,
}: {
  points: MapPoint[];
  papers: SeriesPaper[];
  /** Text shown in the tooltip for a passage point. */
  describe?: (point: MapPoint) => string | undefined;
  onSelect?: (point: MapPoint) => void;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<{ point: MapPoint; x: number; y: number } | null>(null);
  const [focusPaper, setFocusPaper] = useState<string | null>(null);
  const byId = useMemo(() => new Map(papers.map((p) => [p.id, p])), [papers]);

  const scale = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const w = Math.max(1, width - PAD * 2);
    const h = HEIGHT - PAD * 2;
    // Equal units on both axes keep distances honest.
    const k = Math.min(w / (x1 - x0 || 1), h / (y1 - y0 || 1));
    const ox = PAD + (w - (x1 - x0) * k) / 2;
    const oy = PAD + (h - (y1 - y0) * k) / 2;
    return { x: (v: number) => ox + (v - x0) * k, y: (v: number) => oy + (y1 - v) * k };
  }, [points, width]);

  if (points.length === 0) return <ChartEmpty>No embeddings available for a semantic map.</ChartEmpty>;

  const passages = points.filter((p) => !p.isCentroid);
  const centroids = points.filter((p) => p.isCentroid);
  const dimmed = (p: MapPoint) => focusPaper !== null && p.paperId !== focusPaper;

  return (
    <div>
      <div ref={ref} className="relative w-full" style={{ height: HEIGHT }} onMouseLeave={() => setHover(null)}>
        {width > 0 && (
          <svg width={width} height={HEIGHT} role="img" aria-label="Semantic map of passages">
            <rect x={0.5} y={0.5} width={width - 1} height={HEIGHT - 1} rx={12} fill="none" stroke="var(--line)" />
            {passages.map((p, i) => (
              <circle
                key={p.chunkId ?? i}
                cx={scale.x(p.x)}
                cy={scale.y(p.y)}
                r={hover?.point === p ? 5.5 : 3.5}
                fill={byId.get(p.paperId)?.color ?? "var(--ink-3)"}
                opacity={dimmed(p) ? 0.08 : 0.6}
                className="cursor-pointer transition-opacity"
                onMouseEnter={() => setHover({ point: p, x: scale.x(p.x), y: scale.y(p.y) })}
                onClick={() => onSelect?.(p)}
              />
            ))}
            {centroids.map((p) => {
              const paper = byId.get(p.paperId);
              const cx = scale.x(p.x);
              const cy = scale.y(p.y);
              return (
                <g
                  key={`c-${p.paperId}`}
                  className="cursor-pointer"
                  opacity={dimmed(p) ? 0.25 : 1}
                  onMouseEnter={() => {
                    setFocusPaper(p.paperId);
                    setHover({ point: p, x: cx, y: cy });
                  }}
                  onMouseLeave={() => setFocusPaper(null)}
                >
                  <circle cx={cx} cy={cy} r={11} fill={paper?.color ?? "var(--ink-3)"} stroke="var(--panel)" strokeWidth={2.5} />
                  <text x={cx} y={cy + 3.5} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--panel)">
                    {paper?.short.replace(/[[\]]/g, "") ?? ""}
                  </text>
                </g>
              );
            })}
          </svg>
        )}
        {hover && (
          <ChartTooltip x={hover.x} y={hover.y} containerWidth={width}>
            <p className="font-medium text-ink">
              {byId.get(hover.point.paperId)?.short} {byId.get(hover.point.paperId)?.title}
            </p>
            <p className="mt-1 line-clamp-4">
              {hover.point.isCentroid ? "Paper centroid (mean of its passages)" : (describe?.(hover.point) ?? "Passage")}
            </p>
          </ChartTooltip>
        )}
      </div>
      <Legend className="mt-3" items={papers.map((p) => ({ label: `${p.short} ${p.title}`, color: p.color }))} />
    </div>
  );
}

/**
 * Rhetorical map: the whole paper drawn as one horizontal strip, one tick per
 * sentence, coloured by its argumentative zone (problem, method, result, ...).
 * Reading left to right shows how the argument unfolds; tick opacity encodes
 * the zoning classifier's confidence. Section names sit above the strip and
 * page numbers below it.
 */
import { useMemo, useState } from "react";
import type { MouseEvent } from "react";
import type { VisualData, Zone, ZonedSentence } from "@/types";
import { ZONES, cn, zoneColor, zoneLabel } from "@/ui/kit";
import { ChartEmpty, ChartTooltip, useSize } from "./common";

type MapPoint = VisualData["rhetoricalMap"][number];

const STRIP_TOP = 26;
const STRIP_HEIGHT = 46;
const PAD_X = 2;
const MIN_PAGE_LABEL_GAP = 30; // px between page labels before we thin them out
const MIN_SECTION_LABEL_GAP = 84;

interface Marker {
  position: number;
  label: string;
}

function pageMarkers(points: MapPoint[]): Marker[] {
  const markers: Marker[] = [];
  let lastPage = -1;
  for (const p of points) {
    if (p.page !== lastPage && p.page > 0) markers.push({ position: p.position, label: `p.${p.page}` });
    lastPage = p.page;
  }
  return markers;
}

function sectionMarkers(points: MapPoint[], byId: Map<string, ZonedSentence>): Marker[] {
  const markers: Marker[] = [];
  let lastSection = "";
  for (const p of points) {
    const s = byId.get(p.sentenceId);
    if (s && s.sectionId !== lastSection) markers.push({ position: p.position, label: s.sectionTitle });
    if (s) lastSection = s.sectionId;
  }
  return markers;
}

/** Keeps markers at least `gap` px apart so labels never collide. */
function thin(markers: Marker[], toX: (pos: number) => number, gap: number): Marker[] {
  const kept: Marker[] = [];
  let lastX = -Infinity;
  for (const m of markers) {
    const x = toX(m.position);
    if (x - lastX >= gap) {
      kept.push(m);
      lastX = x;
    }
  }
  return kept;
}

/** Index of the point whose position is closest to `pos` (points are sorted). */
function nearestIndex(points: MapPoint[], pos: number): number {
  let lo = 0;
  let hi = points.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid].position < pos) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(points[lo - 1].position - pos) < Math.abs(points[lo].position - pos)) return lo - 1;
  return lo;
}

export function RhetoricalMap({
  points,
  sentences,
  onSelect,
}: {
  points: MapPoint[];
  sentences?: ZonedSentence[];
  onSelect?: (sentence: ZonedSentence) => void;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const [focus, setFocus] = useState<Zone | null>(null);

  const sorted = useMemo(() => [...points].sort((a, b) => a.position - b.position), [points]);
  const byId = useMemo(() => new Map((sentences ?? []).map((s) => [s.id, s])), [sentences]);
  const presentZones = useMemo(() => {
    const seen = new Set(sorted.map((p) => p.zone));
    return ZONES.filter((z) => seen.has(z.zone));
  }, [sorted]);

  if (sorted.length === 0) return <ChartEmpty>No sentences were available to map.</ChartEmpty>;

  const height = STRIP_TOP + STRIP_HEIGHT + 22;
  const innerWidth = Math.max(0, width - PAD_X * 2);
  const toX = (pos: number) => PAD_X + pos * innerWidth;
  const tickWidth = Math.max(1, Math.min(8, innerWidth / sorted.length - 0.4));
  const pages = thin(pageMarkers(sorted), toX, MIN_PAGE_LABEL_GAP);
  const sections = thin(sectionMarkers(sorted, byId), toX, MIN_SECTION_LABEL_GAP);

  const hovered = hover !== null ? sorted[hover] : null;
  const hoveredSentence = hovered ? byId.get(hovered.sentenceId) : undefined;

  const locate = (event: MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const pos = (event.clientX - rect.left - PAD_X) / Math.max(1, innerWidth);
    return nearestIndex(sorted, Math.max(0, Math.min(1, pos)));
  };

  return (
    <div className="space-y-3">
      <div ref={ref} className="relative w-full" style={{ height }} onMouseLeave={() => setHover(null)}>
        {width > 0 && (
          <svg
            width={width}
            height={height}
            role="img"
            aria-labelledby="rhetorical-map-title"
            className={cn(onSelect && "cursor-pointer")}
            onMouseMove={(e) => setHover(locate(e))}
            onClick={(e) => {
              const s = byId.get(sorted[locate(e)].sentenceId);
              if (s && onSelect) onSelect(s);
            }}
          >
            <title id="rhetorical-map-title">
              {`Rhetorical map of ${sorted.length} sentences coloured by argumentative zone`}
            </title>
            {sections.map((m) => (
              <g key={`s${m.position}`}>
                <line x1={toX(m.position)} x2={toX(m.position)} y1={10} y2={STRIP_TOP} stroke="var(--line-strong)" />
                <text x={toX(m.position) + 4} y={18} fontSize={10.5} fill="var(--ink-3)">
                  {m.label.length > 14 ? `${m.label.slice(0, 13)}…` : m.label}
                </text>
              </g>
            ))}
            <rect x={PAD_X} y={STRIP_TOP} width={innerWidth} height={STRIP_HEIGHT} rx={6} fill="var(--panel-2)" />
            {sorted.map((p, i) => {
              const s = byId.get(p.sentenceId);
              const dimmed = focus !== null && p.zone !== focus;
              // Confidence-weighted opacity keeps uncertain labels visibly softer.
              const opacity = dimmed ? 0.08 : s ? 0.45 + 0.55 * Math.max(0, Math.min(1, s.confidence)) : 0.9;
              return (
                <rect
                  key={p.sentenceId || i}
                  x={toX(p.position) - tickWidth / 2}
                  y={STRIP_TOP + 4}
                  width={tickWidth}
                  height={STRIP_HEIGHT - 8}
                  rx={Math.min(1.5, tickWidth / 2)}
                  fill={zoneColor(p.zone)}
                  opacity={opacity}
                />
              );
            })}
            {hovered && (
              <rect
                x={toX(hovered.position) - Math.max(3, tickWidth) / 2 - 1.5}
                y={STRIP_TOP + 1}
                width={Math.max(3, tickWidth) + 3}
                height={STRIP_HEIGHT - 2}
                rx={2}
                fill="none"
                stroke="var(--ink)"
                strokeWidth={1.25}
              />
            )}
            {pages.map((m) => (
              <g key={`p${m.position}`}>
                <line
                  x1={toX(m.position)}
                  x2={toX(m.position)}
                  y1={STRIP_TOP + STRIP_HEIGHT}
                  y2={STRIP_TOP + STRIP_HEIGHT + 5}
                  stroke="var(--ink-3)"
                />
                <text x={toX(m.position)} y={height - 3} fontSize={10} fill="var(--ink-3)" textAnchor="middle" className="tabular">
                  {m.label}
                </text>
              </g>
            ))}
          </svg>
        )}
        {hovered && (
          <ChartTooltip x={toX(hovered.position)} y={STRIP_TOP} containerWidth={width}>
            <div className="flex items-center gap-1.5 text-[11px]">
              <span className="size-2 rounded-full" style={{ background: zoneColor(hovered.zone) }} />
              <span className="font-semibold text-ink">{zoneLabel(hovered.zone)}</span>
              {hoveredSentence && (
                <span className="text-ink-3 tabular">· {Math.round(hoveredSentence.confidence * 100)}% conf.</span>
              )}
              <span className="ml-auto text-ink-3 tabular">p.{hovered.page}</span>
            </div>
            {hoveredSentence && (
              <>
                <div className="mt-0.5 truncate text-[11px] text-ink-3">{hoveredSentence.sectionTitle}</div>
                <p className="mt-1.5 text-ink-2">
                  {hoveredSentence.text.length > 240 ? `${hoveredSentence.text.slice(0, 237)}…` : hoveredSentence.text}
                </p>
              </>
            )}
            {onSelect && <div className="mt-1.5 text-[10.5px] text-ink-3">Click to open the evidence</div>}
          </ChartTooltip>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {presentZones.map((z) => (
          <button
            key={z.zone}
            type="button"
            title={z.description}
            aria-pressed={focus === z.zone}
            onClick={() => setFocus((cur) => (cur === z.zone ? null : z.zone))}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] transition-colors",
              focus === z.zone ? "border-line-strong bg-panel-2 text-ink" : "border-transparent text-ink-3 hover:text-ink-2",
              focus !== null && focus !== z.zone && "opacity-50",
            )}
          >
            <span className="size-2 rounded-full" style={{ background: zoneColor(z.zone) }} />
            {z.label}
          </button>
        ))}
        {focus && (
          <button type="button" onClick={() => setFocus(null)} className="ml-1 text-[11px] text-accent hover:underline">
            Show all
          </button>
        )}
      </div>
    </div>
  );
}

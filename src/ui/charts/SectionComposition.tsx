/**
 * Section composition: one horizontal stacked bar where each segment is a
 * section, sized by word count and coloured by section type. Shows at a
 * glance whether a paper is experiment-heavy, method-heavy or mostly context.
 */
import { useId, useState } from "react";
import type { SectionKind, VisualData } from "@/types";
import { seriesColor } from "@/ui/kit";
import { ChartEmpty, ChartTooltip, useSize } from "./common";

type Segment = VisualData["sectionComposition"][number];

const KIND_STYLE: Record<SectionKind, { label: string; color: string }> = {
  abstract: { label: "Abstract", color: seriesColor(0) },
  introduction: { label: "Introduction", color: seriesColor(5) },
  related: { label: "Related work", color: seriesColor(9) },
  background: { label: "Background", color: seriesColor(8) },
  method: { label: "Method", color: seriesColor(4) },
  experiments: { label: "Experiments", color: seriesColor(1) },
  results: { label: "Results", color: seriesColor(6) },
  discussion: { label: "Discussion", color: seriesColor(2) },
  limitations: { label: "Limitations", color: seriesColor(7) },
  conclusion: { label: "Conclusion", color: seriesColor(3) },
  appendix: { label: "Appendix", color: "var(--ink-3)" },
  references: { label: "References", color: "var(--line-strong)" },
  acknowledgments: { label: "Acknowledgments", color: "var(--line-strong)" },
  other: { label: "Other", color: "var(--z-other)" },
};

const BAR_HEIGHT = 30;
const GAP = 1.5; // hairline gap between segments so adjacent sections stay distinct

export function SectionComposition({ sections }: { sections: Segment[] }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const clipId = `section-clip-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const segments = sections.filter((s) => s.words > 0);
  const total = segments.reduce((sum, s) => sum + s.words, 0);

  if (total === 0) return <ChartEmpty>Section structure is available when the full text is parsed.</ChartEmpty>;

  const totalsByKind = new Map<SectionKind, number>();
  for (const s of segments) totalsByKind.set(s.kind, (totalsByKind.get(s.kind) ?? 0) + s.words);
  const kinds = [...totalsByKind.entries()].sort((a, b) => b[1] - a[1]);

  const usable = Math.max(0, width - GAP * (segments.length - 1));
  let cursor = 0;
  const placed = segments.map((s) => {
    const w = (s.words / total) * usable;
    const seg = { ...s, x: cursor, w };
    cursor += w + GAP;
    return seg;
  });
  const active = hover !== null ? placed[hover] : null;

  return (
    <div className="space-y-4">
      <div ref={ref} className="relative w-full" style={{ height: BAR_HEIGHT }} onMouseLeave={() => setHover(null)}>
        {width > 0 && (
          <svg width={width} height={BAR_HEIGHT} role="img" aria-labelledby="section-comp-title">
            <title id="section-comp-title">{`Words per section, ${total.toLocaleString()} words in ${segments.length} sections`}</title>
            <defs>
              <clipPath id={clipId}>
                <rect width={width} height={BAR_HEIGHT} rx={8} />
              </clipPath>
            </defs>
            <g clipPath={`url(#${clipId})`}>
              {placed.map((s, i) => (
                <rect
                  key={`${s.title}-${i}`}
                  x={s.x}
                  y={0}
                  width={Math.max(0.5, s.w)}
                  height={BAR_HEIGHT}
                  fill={KIND_STYLE[s.kind].color}
                  opacity={hover === null || hover === i ? 0.9 : 0.35}
                  onMouseEnter={() => setHover(i)}
                />
              ))}
            </g>
          </svg>
        )}
        {active && (
          <ChartTooltip x={active.x + active.w / 2} y={0} containerWidth={width}>
            <div className="font-semibold text-ink">{active.title}</div>
            <div className="text-ink-3">{KIND_STYLE[active.kind].label}</div>
            <div className="mt-1 tabular">
              {active.words.toLocaleString()} words · {((active.words / total) * 100).toFixed(1)}% of the paper
            </div>
          </ChartTooltip>
        )}
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3">
        {kinds.map(([kind, words]) => (
          <div key={kind} className="flex min-w-0 items-center gap-2 text-xs">
            <span className="size-2 shrink-0 rounded-sm" style={{ background: KIND_STYLE[kind].color }} />
            <span className="truncate text-ink-2">{KIND_STYLE[kind].label}</span>
            <span className="ml-auto tabular text-ink-3">{Math.round((words / total) * 100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

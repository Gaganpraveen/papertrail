/**
 * Bibliographic coupling network: two papers are linked when they cite the
 * same works, with edge width proportional to the Jaccard overlap of their
 * reference lists. Layout is a force simulation run to rest (300 ticks)
 * before rendering, so the picture is static and deterministic.
 */
import { useMemo, useState } from "react";
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation } from "d3-force";
import type { SimulationLinkDatum, SimulationNodeDatum } from "d3-force";
import { ChartEmpty, ChartTooltip, useSize } from "./common";
import type { SeriesPaper } from "./SemanticMap";

const HEIGHT = 340;
const NODE_R = 15;

interface Coupling {
  source: string;
  target: string;
  shared: number;
  jaccard: number;
}

interface Node extends SimulationNodeDatum {
  id: string;
}

type Link = SimulationLinkDatum<Node> & { shared: number; jaccard: number };

function layout(papers: SeriesPaper[], links: Coupling[]): Map<string, { x: number; y: number }> {
  const nodes: Node[] = papers.map((p) => ({ id: p.id }));
  const simLinks: Link[] = links.map((l) => ({ ...l }));
  forceSimulation(nodes)
    .force(
      "link",
      // Strongly coupled papers sit closer together.
      forceLink<Node, Link>(simLinks)
        .id((d) => d.id)
        .distance((l) => 70 + 130 * (1 - l.jaccard)),
    )
    .force("charge", forceManyBody().strength(-420))
    .force("center", forceCenter(0, 0))
    .force("collide", forceCollide(NODE_R * 2.4))
    .stop()
    .tick(300);
  return new Map(nodes.map((n) => [n.id, { x: n.x ?? 0, y: n.y ?? 0 }]));
}

export function CouplingNetwork({ links, papers }: { links: Coupling[]; papers: SeriesPaper[] }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);
  const valid = useMemo(() => links.filter((l) => l.shared > 0), [links]);
  const positions = useMemo(() => layout(papers, valid), [papers, valid]);
  const byId = useMemo(() => new Map(papers.map((p) => [p.id, p])), [papers]);

  if (papers.length < 2) return <ChartEmpty>Needs at least two papers.</ChartEmpty>;

  // Fit the resting layout into the box.
  const pts = [...positions.values()];
  const [x0, x1] = [Math.min(...pts.map((p) => p.x)), Math.max(...pts.map((p) => p.x))];
  const [y0, y1] = [Math.min(...pts.map((p) => p.y)), Math.max(...pts.map((p) => p.y))];
  const pad = NODE_R + 40;
  const k = Math.min((width - pad * 2) / (x1 - x0 || 1), (HEIGHT - pad * 2) / (y1 - y0 || 1), 1.6);
  const sx = (v: number) => width / 2 + (v - (x0 + x1) / 2) * k;
  const sy = (v: number) => HEIGHT / 2 + (v - (y0 + y1) / 2) * k;
  const maxJ = Math.max(...valid.map((l) => l.jaccard), 0.01);
  const neighbours = new Set(
    valid.flatMap((l) => (l.source === hover ? [l.target] : l.target === hover ? [l.source] : [])),
  );
  const hovered = hover ? byId.get(hover) : undefined;
  const hoverLinks = valid.filter((l) => l.source === hover || l.target === hover);

  return (
    <div ref={ref} className="relative w-full" style={{ height: HEIGHT }}>
      {valid.length === 0 && (
        <p className="absolute top-3 left-0 w-full text-center text-xs text-ink-3">
          These papers share no references.
        </p>
      )}
      {width > 0 && (
        <svg width={width} height={HEIGHT} role="img" aria-label="Citation coupling network">
          {valid.map((l) => {
            const a = positions.get(l.source);
            const b = positions.get(l.target);
            if (!a || !b) return null;
            const active = hover === null || l.source === hover || l.target === hover;
            return (
              <g key={`${l.source}-${l.target}`}>
                <line
                  x1={sx(a.x)}
                  y1={sy(a.y)}
                  x2={sx(b.x)}
                  y2={sy(b.y)}
                  stroke="var(--accent)"
                  strokeOpacity={active ? 0.25 + 0.5 * (l.jaccard / maxJ) : 0.06}
                  strokeWidth={1 + 7 * (l.jaccard / maxJ)}
                  strokeLinecap="round"
                />
                {hover !== null && active && (
                  <text
                    x={(sx(a.x) + sx(b.x)) / 2}
                    y={(sy(a.y) + sy(b.y)) / 2 - 6}
                    textAnchor="middle"
                    fontSize={10}
                    fill="var(--ink-2)"
                    className="tabular"
                  >
                    {l.shared} shared
                  </text>
                )}
              </g>
            );
          })}
          {papers.map((p) => {
            const pos = positions.get(p.id);
            if (!pos) return null;
            const dim = hover !== null && hover !== p.id && !neighbours.has(p.id);
            return (
              <g
                key={p.id}
                transform={`translate(${sx(pos.x)},${sy(pos.y)})`}
                opacity={dim ? 0.3 : 1}
                className="cursor-default transition-opacity"
                onMouseEnter={() => setHover(p.id)}
                onMouseLeave={() => setHover(null)}
              >
                <circle r={NODE_R} fill={p.color} stroke="var(--panel)" strokeWidth={3} />
                <text y={4} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--panel)">
                  {p.short.replace(/[[\]]/g, "")}
                </text>
                <text y={NODE_R + 15} textAnchor="middle" fontSize={10.5} fill="var(--ink-2)">
                  {p.title.length > 26 ? `${p.title.slice(0, 25)}…` : p.title}
                </text>
              </g>
            );
          })}
        </svg>
      )}
      {hovered && positions.get(hovered.id) && (
        <ChartTooltip
          x={sx(positions.get(hovered.id)!.x)}
          y={sy(positions.get(hovered.id)!.y) - NODE_R}
          containerWidth={width}
        >
          <p className="font-medium text-ink">
            {hovered.short} {hovered.title}
          </p>
          {hoverLinks.length === 0 ? (
            <p className="mt-1">No references shared with the other papers.</p>
          ) : (
            <ul className="mt-1 space-y-0.5">
              {hoverLinks.map((l) => {
                const other = byId.get(l.source === hovered.id ? l.target : l.source);
                return (
                  <li key={`${l.source}-${l.target}`} className="tabular">
                    {other?.short}: {l.shared} shared · Jaccard {l.jaccard.toFixed(2)}
                  </li>
                );
              })}
            </ul>
          )}
        </ChartTooltip>
      )}
    </div>
  );
}

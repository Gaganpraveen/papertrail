/**
 * Key-phrase co-occurrence network. Nodes are key phrases (size = salience),
 * edges join phrases that appear in the same sentences (width = PMI weight).
 * Layout is a force simulation (d3-force: link springs, many-body repulsion,
 * collision) run for 300 ticks up front, so the result is a static, stable SVG
 * rather than a jittering animation. Hovering a node highlights its neighbours.
 */
import { useMemo, useState } from "react";
import { forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation } from "d3-force";
import type { SimulationLinkDatum, SimulationNodeDatum } from "d3-force";
import type { VisualData } from "@/types";
import { ChartEmpty, ChartTooltip, useSize } from "./common";

type Graph = VisualData["keyphraseGraph"];

interface SimNode extends SimulationNodeDatum {
  id: string;
  weight: number;
  r: number;
}
type SimLink = SimulationLinkDatum<SimNode> & { weight: number };

interface Layout {
  nodes: { id: string; weight: number; r: number; x: number; y: number }[];
  links: { source: string; target: string; weight: number }[];
}

const TICKS = 300;
const PAD_X = 64; // horizontal padding leaves room for labels beside edge nodes
const PAD_Y = 24;

function radiusFor(weight: number, maxWeight: number): number {
  return 5 + 11 * Math.sqrt(Math.max(0, weight) / (maxWeight || 1));
}

/** Runs the simulation to rest and fits the result into width × height. */
export function computeLayout(graph: Graph, width: number, height: number): Layout {
  const maxWeight = Math.max(...graph.nodes.map((n) => n.weight), 0);
  const maxLink = Math.max(...graph.links.map((l) => l.weight), 0) || 1;
  const nodes: SimNode[] = graph.nodes.map((n) => ({ id: n.id, weight: n.weight, r: radiusFor(n.weight, maxWeight) }));
  const ids = new Set(nodes.map((n) => n.id));
  const links: SimLink[] = graph.links
    .filter((l) => ids.has(l.source) && ids.has(l.target) && l.source !== l.target)
    .map((l) => ({ source: l.source, target: l.target, weight: l.weight }));

  const simulation = forceSimulation<SimNode>(nodes)
    .force(
      "link",
      forceLink<SimNode, SimLink>(links)
        .id((d) => d.id)
        .distance(70)
        .strength((l) => 0.15 + 0.6 * (l.weight / maxLink)),
    )
    .force("charge", forceManyBody<SimNode>().strength(-180))
    .force("center", forceCenter(0, 0))
    .force("collide", forceCollide<SimNode>().radius((d) => d.r + 14))
    .stop();
  simulation.tick(TICKS);

  // Fit the settled layout into the viewport, preserving aspect ratio.
  const xs = nodes.map((n) => n.x ?? 0);
  const ys = nodes.map((n) => n.y ?? 0);
  const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const scale = Math.min(
    (width - PAD_X * 2) / Math.max(1, maxX - minX),
    (height - PAD_Y * 2) / Math.max(1, maxY - minY),
    1.6,
  );
  const offsetX = width / 2 - ((minX + maxX) / 2) * scale;
  const offsetY = height / 2 - ((minY + maxY) / 2) * scale;

  return {
    nodes: nodes.map((n) => ({ id: n.id, weight: n.weight, r: n.r, x: (n.x ?? 0) * scale + offsetX, y: (n.y ?? 0) * scale + offsetY })),
    links: links.map((l) => ({
      source: typeof l.source === "object" ? l.source.id : String(l.source),
      target: typeof l.target === "object" ? l.target.id : String(l.target),
      weight: l.weight,
    })),
  };
}

export function KeyphraseNetwork({ graph, height = 340 }: { graph: Graph; height?: number }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);

  const layout = useMemo(
    () => (width > 0 && graph.nodes.length > 0 ? computeLayout(graph, width, height) : null),
    [graph, width, height],
  );
  const neighbours = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const l of layout?.links ?? []) {
      if (!map.has(l.source)) map.set(l.source, new Set());
      if (!map.has(l.target)) map.set(l.target, new Set());
      map.get(l.source)!.add(l.target);
      map.get(l.target)!.add(l.source);
    }
    return map;
  }, [layout]);

  if (graph.nodes.length === 0) return <ChartEmpty>No key phrases were extracted.</ChartEmpty>;

  const maxLink = Math.max(...(layout?.links.map((l) => l.weight) ?? [1]), 0) || 1;
  const byId = new Map(layout?.nodes.map((n) => [n.id, n]));
  const isLit = (id: string) => hover === null || id === hover || (neighbours.get(hover)?.has(id) ?? false);
  const hoveredNode = hover ? byId.get(hover) : undefined;
  const hoveredNeighbours = hover ? [...(neighbours.get(hover) ?? [])] : [];

  return (
    <div ref={ref} className="relative w-full" style={{ height }} onMouseLeave={() => setHover(null)}>
      {layout && (
        <svg width={width} height={height} role="img" aria-labelledby="kp-net-title">
          <title id="kp-net-title">{`Co-occurrence network of ${layout.nodes.length} key phrases`}</title>
          {layout.links.map((l) => {
            const a = byId.get(l.source);
            const b = byId.get(l.target);
            if (!a || !b) return null;
            const lit = hover !== null && (l.source === hover || l.target === hover);
            return (
              <line
                key={`${l.source}|${l.target}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={lit ? "var(--accent)" : "var(--line-strong)"}
                strokeWidth={0.75 + 2.25 * (l.weight / maxLink)}
                strokeOpacity={hover === null ? 0.9 : lit ? 1 : 0.25}
              />
            );
          })}
          {layout.nodes.map((n) => {
            const lit = isLit(n.id);
            const labelLeft = n.x > width - PAD_X - 20;
            return (
              <g
                key={n.id}
                onMouseEnter={() => setHover(n.id)}
                opacity={lit ? 1 : 0.3}
                className="cursor-default transition-opacity"
              >
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={n.r}
                  fill="var(--accent)"
                  fillOpacity={n.id === hover ? 0.95 : 0.55}
                  stroke="var(--accent)"
                  strokeWidth={1.25}
                />
                <text
                  x={labelLeft ? n.x - n.r - 4 : n.x + n.r + 4}
                  y={n.y + 3.5}
                  textAnchor={labelLeft ? "end" : "start"}
                  fontSize={11}
                  fontWeight={n.id === hover ? 600 : 450}
                  fill={n.id === hover ? "var(--ink)" : "var(--ink-2)"}
                  paintOrder="stroke"
                  stroke="var(--panel)"
                  strokeWidth={3}
                  strokeLinejoin="round"
                >
                  {n.id}
                </text>
              </g>
            );
          })}
        </svg>
      )}
      {hoveredNode && (
        <ChartTooltip x={hoveredNode.x} y={hoveredNode.y - hoveredNode.r} containerWidth={width}>
          <div className="font-semibold text-ink">{hoveredNode.id}</div>
          <div className="tabular text-ink-3">salience {hoveredNode.weight.toFixed(2)}</div>
          {hoveredNeighbours.length > 0 ? (
            <div className="mt-1">Co-occurs with {hoveredNeighbours.slice(0, 6).join(", ")}{hoveredNeighbours.length > 6 ? "…" : ""}</div>
          ) : (
            <div className="mt-1 text-ink-3">No strong co-occurrences</div>
          )}
        </ChartTooltip>
      )}
    </div>
  );
}

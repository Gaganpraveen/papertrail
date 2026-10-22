/**
 * Shared chart plumbing: a ResizeObserver size hook so every chart is a
 * responsive SVG, a tooltip that stays inside its card, and a compact legend.
 */
import { useLayoutEffect, useRef, useState } from "react";
import type { ReactNode, RefObject } from "react";
import { cn } from "@/ui/kit";

export interface Size {
  width: number;
  height: number;
}

/** Observes an element and returns its content-box size (0×0 until mounted). */
export function useSize<T extends HTMLElement>(): [RefObject<T | null>, Size] {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = (width: number, height: number) =>
      setSize((prev) =>
        Math.round(prev.width) === Math.round(width) && Math.round(prev.height) === Math.round(height)
          ? prev
          : { width, height },
      );
    const rect = el.getBoundingClientRect();
    update(rect.width, rect.height);
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (box) update(box.width, box.height);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return [ref, size];
}

/**
 * Tooltip positioned in the coordinate space of its nearest `relative`
 * ancestor. It flips to the left of the cursor in the right half of the
 * container so it never overflows the card.
 */
export function ChartTooltip({
  x,
  y,
  children,
  containerWidth,
}: {
  x: number;
  y: number;
  children: ReactNode;
  /** Optional; enables flipping near the right edge. */
  containerWidth?: number;
}) {
  const flip = containerWidth !== undefined && x > containerWidth * 0.6;
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-20 max-w-[280px] rounded-lg border border-line-strong bg-panel px-2.5 py-2 text-xs leading-relaxed text-ink-2 shadow-card animate-in"
      style={{
        left: x,
        top: y,
        transform: `translate(${flip ? "calc(-100% - 12px)" : "12px"}, calc(-100% - 8px))`,
      }}
    >
      {children}
    </div>
  );
}

export function Legend({
  items,
  className,
}: {
  items: { label: string; color: string }[];
  className?: string;
}) {
  return (
    <ul className={cn("flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[11px] text-ink-3", className)}>
      {items.map((item) => (
        <li key={item.label} className="inline-flex items-center gap-1.5">
          <span className="inline-block size-2 rounded-full" style={{ background: item.color }} />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

/** "Nice" tick values for a 0..max linear axis. */
export function niceTicks(max: number, count = 4): number[] {
  if (!(max > 0)) return [0];
  const rawStep = max / count;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const residual = rawStep / magnitude;
  const step = (residual > 5 ? 10 : residual > 2 ? 5 : residual > 1 ? 2 : 1) * magnitude;
  const ticks: number[] = [];
  for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(Number(v.toFixed(6)));
  if (ticks[ticks.length - 1] < max) ticks.push(Number((ticks[ticks.length - 1] + step).toFixed(6)));
  return ticks;
}

/** Placeholder shown inside a chart card when there is nothing to plot. */
export function ChartEmpty({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "dot-grid flex h-full min-h-[140px] items-center justify-center rounded-xl border border-dashed border-line px-6 text-center text-xs text-ink-3",
        className,
      )}
    >
      {children}
    </div>
  );
}

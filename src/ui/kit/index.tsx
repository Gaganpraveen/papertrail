/**
 * PaperTrail's design-system primitives. Views compose these instead of
 * hand-styling, which keeps spacing, radius and colour consistent.
 */
import { clsx } from "clsx";
import { useEffect, useRef, type ButtonHTMLAttributes, type HTMLAttributes, type ReactNode } from "react";
import type { Severity, Zone } from "@/types";

export const cn = clsx;

// ── Button ────────────────────────────────────────────────────────────────

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "lg";

const buttonVariants: Record<ButtonVariant, string> = {
  primary: "bg-accent text-accent-ink hover:brightness-110 shadow-sm",
  secondary: "bg-panel border border-line text-ink hover:bg-panel-2 hover:border-line-strong",
  ghost: "text-ink-2 hover:text-ink hover:bg-panel-2",
  danger: "text-bad hover:bg-bad-soft",
};

const buttonSizes: Record<ButtonSize, string> = {
  sm: "h-7 px-2.5 text-xs gap-1.5 rounded-md",
  md: "h-9 px-3.5 text-sm gap-2 rounded-lg",
  lg: "h-11 px-5 text-[15px] gap-2 rounded-xl",
};

export function Button({
  variant = "secondary",
  size = "md",
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: ButtonSize }) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center font-medium whitespace-nowrap transition-all",
        "disabled:opacity-50 disabled:pointer-events-none focus-visible:outline-2 focus-visible:outline-accent focus-visible:outline-offset-2",
        buttonVariants[variant],
        buttonSizes[size],
        className,
      )}
      {...props}
    />
  );
}

export function IconButton({
  label,
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      aria-label={label}
      title={label}
      className={cn(
        "inline-flex size-8 items-center justify-center rounded-lg text-ink-3 transition-colors",
        "hover:bg-panel-2 hover:text-ink disabled:opacity-40",
        className,
      )}
      {...props}
    />
  );
}

// ── Surfaces ──────────────────────────────────────────────────────────────

export function Card({
  className,
  children,
  ...props
}: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return (
    <div
      className={cn("rounded-2xl border border-line bg-panel shadow-card", className)}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  icon,
  actions,
  className,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start gap-3 px-5 pt-4 pb-3", className)}>
      {icon && <div className="mt-0.5 text-ink-3">{icon}</div>}
      <div className="min-w-0 flex-1">
        <h3 className="text-sm font-semibold tracking-tight text-ink">{title}</h3>
        {subtitle && <p className="mt-0.5 text-xs leading-relaxed text-ink-3">{subtitle}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-1">{actions}</div>}
    </div>
  );
}

export function CardBody({ className, children }: { className?: string; children?: ReactNode }) {
  return <div className={cn("px-5 pb-5", className)}>{children}</div>;
}

// ── Badges ────────────────────────────────────────────────────────────────

type Tone = "neutral" | "accent" | "good" | "warn" | "bad";

const tones: Record<Tone, string> = {
  neutral: "bg-panel-2 text-ink-2 border-line",
  accent: "bg-accent-soft text-accent border-transparent",
  good: "bg-good-soft text-good border-transparent",
  warn: "bg-warn-soft text-warn border-transparent",
  bad: "bg-bad-soft text-bad border-transparent",
};

export function Badge({
  tone = "neutral",
  className,
  children,
}: {
  tone?: Tone;
  className?: string;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium leading-none whitespace-nowrap",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export const severityTone: Record<Severity, Tone> = {
  major: "bad",
  moderate: "warn",
  minor: "neutral",
  strength: "good",
};

// ── Tabs & segmented control ─────────────────────────────────────────────

export function Tabs<T extends string>({
  value,
  onChange,
  items,
  className,
}: {
  value: T;
  onChange: (value: T) => void;
  /** shortLabel replaces label below `sm`, where the row has the least room. */
  items: { value: T; label: string; shortLabel?: string; icon?: ReactNode; count?: number }[];
  className?: string;
}) {
  const listRef = useRef<HTMLDivElement>(null);

  // The row scrolls sideways on narrow screens, so bring the active tab into
  // view when it changes from elsewhere (asking a question opens "Ask").
  useEffect(() => {
    const list = listRef.current;
    const tab = list?.querySelector<HTMLElement>('[aria-selected="true"]');
    if (!list || !tab) return;
    const box = list.getBoundingClientRect();
    const rect = tab.getBoundingClientRect();
    const delta = rect.left < box.left ? rect.left - box.left : rect.right > box.right ? rect.right - box.right : 0;
    if (delta !== 0) list.scrollBy({ left: delta, behavior: "smooth" });
  }, [value]);

  // Scrolls instead of overflowing the page on phones. The hairline is an
  // inset shadow rather than a border, so the active underline still sits on
  // it without poking out of the scroll box.
  return (
    <div
      ref={listRef}
      role="tablist"
      className={cn(
        "flex items-center gap-1 overflow-x-auto shadow-[inset_0_-1px_0_var(--line)] [scrollbar-width:none]",
        className,
      )}
    >
      {items.map((item) => {
        const active = item.value === value;
        return (
          <button
            key={item.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(item.value)}
            className={cn(
              "relative flex shrink-0 items-center gap-2 px-3 py-2.5 text-sm font-medium whitespace-nowrap transition-colors",
              // Inset so the scroll box does not clip the keyboard focus ring.
              "focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent",
              active ? "text-ink" : "text-ink-3 hover:text-ink-2",
            )}
          >
            {/* Text-only below `sm`, so all tabs fit on a phone without scrolling. */}
            {item.icon && <span className="hidden sm:contents">{item.icon}</span>}
            {item.shortLabel ? (
              <>
                <span className="sm:hidden">{item.shortLabel}</span>
                <span className="hidden sm:inline">{item.label}</span>
              </>
            ) : (
              item.label
            )}
            {item.count !== undefined && (
              <span className="rounded bg-panel-2 px-1 text-[10px] tabular text-ink-3">
                {item.count}
              </span>
            )}
            <span
              className={cn(
                "absolute inset-x-2 bottom-0 h-0.5 rounded-full transition-colors",
                active ? "bg-accent" : "bg-transparent",
              )}
            />
          </button>
        );
      })}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  onChange,
  items,
}: {
  value: T;
  onChange: (value: T) => void;
  items: { value: T; label: string }[];
}) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-panel-2 p-0.5">
      {items.map((item) => (
        <button
          key={item.value}
          onClick={() => onChange(item.value)}
          className={cn(
            "rounded-md px-2.5 py-1 text-xs font-medium transition-all",
            item.value === value ? "bg-panel text-ink shadow-sm" : "text-ink-3 hover:text-ink-2",
          )}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

// ── Feedback ──────────────────────────────────────────────────────────────

export function Progress({ value, className }: { value: number; className?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className={cn("h-1.5 overflow-hidden rounded-full bg-panel-2", className)}>
      <div
        className="h-full rounded-full bg-accent transition-[width] duration-300"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg className={cn("size-4 animate-spin text-accent", className)} viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeOpacity="0.2" strokeWidth="3" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-soft-pulse rounded-md bg-panel-2", className)} />;
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="rounded border border-line bg-panel-2 px-1.5 py-0.5 font-mono text-[10px] text-ink-3">
      {children}
    </kbd>
  );
}

export function EmptyState({
  icon,
  title,
  body,
  action,
}: {
  icon?: ReactNode;
  title: string;
  body?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 px-6 py-12 text-center">
      {icon && (
        <div className="flex size-11 items-center justify-center rounded-xl bg-panel-2 text-ink-3">
          {icon}
        </div>
      )}
      <div>
        <p className="text-sm font-semibold text-ink">{title}</p>
        {body && <p className="mx-auto mt-1 max-w-sm text-xs leading-relaxed text-ink-3">{body}</p>}
      </div>
      {action}
    </div>
  );
}

// ── Data display ─────────────────────────────────────────────────────────

export function Stat({
  label,
  value,
  hint,
  className,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <div className="text-[11px] font-medium uppercase tracking-wider text-ink-3">{label}</div>
      <div className="mt-1 text-xl font-semibold tabular tracking-tight text-ink">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-ink-3">{hint}</div>}
    </div>
  );
}

/** Circular 0..100 gauge used for the rigor score. */
export function ScoreRing({
  value,
  size = 64,
  stroke = 6,
  label,
}: {
  value: number;
  size?: number;
  stroke?: number;
  label?: ReactNode;
}) {
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, value)) / 100;
  const color = value >= 75 ? "var(--good)" : value >= 55 ? "var(--accent)" : value >= 40 ? "var(--warn)" : "var(--bad)";
  return (
    <div className="relative inline-flex shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--panel-2)" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={color}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - pct)}
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-base font-semibold tabular leading-none">{Math.round(value)}</span>
        {label && <span className="mt-0.5 text-[9px] uppercase tracking-wider text-ink-3">{label}</span>}
      </div>
    </div>
  );
}

// ── Zones (shared categorical palette) ───────────────────────────────────

export const ZONES: { zone: Zone; label: string; description: string }[] = [
  { zone: "background", label: "Background", description: "Context and prior knowledge" },
  { zone: "problem", label: "Problem", description: "The gap or question the paper addresses" },
  { zone: "contribution", label: "Contribution", description: "What the authors claim as new" },
  { zone: "method", label: "Method", description: "How the approach works" },
  { zone: "result", label: "Result", description: "Measured outcomes and findings" },
  { zone: "limitation", label: "Limitation", description: "Weaknesses the authors acknowledge" },
  { zone: "future", label: "Future work", description: "Open directions" },
  { zone: "other", label: "Other", description: "Unclassified" },
];

export const zoneColor = (zone: Zone) => `var(--z-${zone})`;
export const zoneLabel = (zone: Zone) => ZONES.find((z) => z.zone === zone)?.label ?? zone;
/** Categorical colour for papers / clusters (10 distinct hues). */
export const seriesColor = (i: number) => `var(--c-${((i % 10) + 10) % 10})`;

export function ZoneDot({ zone, className }: { zone: Zone; className?: string }) {
  return (
    <span
      className={cn("inline-block size-2 shrink-0 rounded-full", className)}
      style={{ background: zoneColor(zone) }}
    />
  );
}

export function SectionLabel({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("text-[11px] font-semibold uppercase tracking-wider text-ink-3", className)}>
      {children}
    </div>
  );
}

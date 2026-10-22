/**
 * Tiny "p.4" affordances attached to every claim. Clicking one opens the
 * evidence drawer with the verbatim quote and the highlighted PDF page.
 */
import type { Evidence } from "@/types";
import { useLibrary } from "@/store/library";
import { cn } from "@/ui/kit";

export function evidenceKey(e: Evidence): string {
  return `${e.page}|${e.quote.slice(0, 80)}`;
}

/** Drops repeated evidence (same page and opening words). */
export function uniqueEvidence(list: Evidence[]): Evidence[] {
  const seen = new Set<string>();
  return list.filter((e) => {
    const key = evidenceKey(e);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function EvidenceChip({ evidence, className }: { evidence: Evidence; className?: string }) {
  const openEvidence = useLibrary((s) => s.openEvidence);
  const label = evidence.page > 0 ? `p.${evidence.page}` : "abs";
  const preview = evidence.quote.length > 160 ? `${evidence.quote.slice(0, 157)}…` : evidence.quote;
  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        openEvidence(evidence);
      }}
      title={`${evidence.section ? `${evidence.section} — ` : ""}“${preview}”`}
      aria-label={`Show evidence on ${evidence.page > 0 ? `page ${evidence.page}` : "the abstract"}`}
      className={cn(
        "inline-flex h-[18px] items-center rounded border border-line bg-panel-2 px-1 align-middle font-mono text-[10px] leading-none text-ink-3",
        "transition-colors hover:border-accent hover:bg-accent-soft hover:text-accent",
        className,
      )}
    >
      {label}
    </button>
  );
}

export function EvidenceChips({ evidence, max = 3, className }: { evidence: Evidence[]; max?: number; className?: string }) {
  const unique = uniqueEvidence(evidence);
  if (unique.length === 0) return null;
  return (
    <span className={cn("ml-1.5 inline-flex flex-wrap items-center gap-1 align-middle", className)}>
      {unique.slice(0, max).map((e) => (
        <EvidenceChip key={evidenceKey(e)} evidence={e} />
      ))}
      {unique.length > max && <span className="text-[10px] text-ink-3">+{unique.length - max}</span>}
    </span>
  );
}

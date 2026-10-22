import { cn } from "@/ui/kit";

/** The PaperTrail mark: three pages fading into a trail behind the one in front. */
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" fill="none" aria-hidden className={cn("size-[22px] shrink-0 text-ink", className)}>
      <rect x="13" y="3" width="15" height="19" rx="3" fill="currentColor" opacity="0.22" />
      <rect x="8.5" y="6.5" width="15" height="19" rx="3" fill="currentColor" opacity="0.5" />
      <path
        d="M4 13a3 3 0 0 1 3-3h7.6L19 14.4V26a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V13Z"
        fill="var(--accent)"
      />
      <path d="M14.6 10v2.4a2 2 0 0 0 2 2H19" stroke="var(--bg)" strokeWidth="1.4" strokeLinejoin="round" />
    </svg>
  );
}

export function Brand({ onClick }: { onClick?: () => void }) {
  return (
    <button onClick={onClick} className="group flex items-center gap-2 rounded-lg text-left">
      <BrandMark />
      <span className="text-[15px] font-semibold tracking-[-0.02em] text-ink">PaperTrail</span>
    </button>
  );
}

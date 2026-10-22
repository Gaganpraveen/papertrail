import { cn } from "@/ui/kit";

/** The PaperTrail mark: a page with a dotted trail joining two waypoints. */
export function BrandMark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex size-7 shrink-0 items-center justify-center rounded-lg text-accent-ink",
        className,
      )}
      style={{ background: "linear-gradient(135deg, var(--accent), var(--c-4))" }}
      aria-hidden
    >
      <svg viewBox="0 0 32 32" fill="none" className="size-[22px]">
        <path
          d="M10.5 6.5h7.5l4.5 4.5v13.25c0 .69-.56 1.25-1.25 1.25H10.5c-.69 0-1.25-.56-1.25-1.25V7.75c0-.69.56-1.25 1.25-1.25Z"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinejoin="round"
        />
        <path
          d="M12.6 21.2c2.6 0 1.4-4.6 3.9-4.6s1.3-4.1 3.7-4.1"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeDasharray="0.1 2.6"
        />
        <circle cx="12.6" cy="21.2" r="1.6" fill="currentColor" />
        <circle cx="20.2" cy="12.5" r="1.6" fill="currentColor" />
      </svg>
    </span>
  );
}

export function Brand({ onClick }: { onClick?: () => void }) {
  return (
    <button onClick={onClick} className="group flex items-center gap-2.5 rounded-lg text-left">
      <BrandMark />
      <span className="text-[15px] font-semibold tracking-tight text-ink">PaperTrail</span>
    </button>
  );
}

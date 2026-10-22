import { CircleAlert, CircleCheck, Info, TriangleAlert, X } from "lucide-react";
import type { Toast } from "@/store/types";
import { cn } from "@/ui/kit";
import { store, useStore } from "@/ui/hooks/store";

const toneIcon: Record<Toast["tone"], typeof Info> = {
  info: Info,
  good: CircleCheck,
  warn: TriangleAlert,
  bad: CircleAlert,
};

const toneColor: Record<Toast["tone"], string> = {
  info: "text-accent",
  good: "text-good",
  warn: "text-warn",
  bad: "text-bad",
};

export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  if (toasts.length === 0) return null;
  return (
    <div
      aria-live="polite"
      className="pointer-events-none fixed right-4 bottom-14 z-50 flex w-[min(380px,calc(100vw-32px))] flex-col gap-2"
    >
      {toasts.map((t) => {
        const Icon = toneIcon[t.tone];
        return (
          <div
            key={t.id}
            role="status"
            className="animate-in pointer-events-auto flex items-start gap-2.5 rounded-xl border border-line-strong bg-panel px-3.5 py-3 text-sm shadow-card"
          >
            <Icon className={cn("mt-0.5 size-4 shrink-0", toneColor[t.tone])} />
            <p className="min-w-0 flex-1 leading-snug text-ink-2">{t.message}</p>
            <button
              onClick={() => store().dismissToast(t.id)}
              aria-label="Dismiss"
              className="-m-1 rounded p-1 text-ink-3 hover:text-ink"
            >
              <X className="size-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}

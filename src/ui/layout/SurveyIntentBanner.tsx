import { Layers, X } from "lucide-react";
import { Progress } from "@/ui/kit";
import { useStore } from "@/ui/hooks/store";
import { intentProgress, usePendingSurvey } from "@/ui/hooks/usePendingSurvey";

/** Shown while the sample survey waits for its papers to finish analysing. */
export function SurveyIntentBanner() {
  const arxivIds = usePendingSurvey((s) => s.arxivIds);
  const cancel = usePendingSurvey((s) => s.cancel);
  const papers = useStore((s) => s.papers);
  if (!arxivIds) return null;
  const { ready, failed, total } = intentProgress(papers, arxivIds);

  return (
    <div className="animate-in flex shrink-0 items-center gap-3 border-b border-line bg-accent-soft/60 px-4 py-2.5">
      <Layers className="size-4 shrink-0 text-accent" />
      <div className="min-w-0 flex-1">
        <p className="truncate text-xs font-medium text-ink">
          Preparing literature survey · {ready} of {total} papers analysed
          {failed > 0 && <span className="text-bad"> · {failed} failed</span>}
        </p>
        <Progress value={(ready + failed) / total} className="mt-1.5 h-1 max-w-sm" />
      </div>
      <span className="hidden text-[11px] text-ink-3 sm:block">The survey starts automatically.</span>
      <button onClick={cancel} aria-label="Cancel survey" className="rounded p-1 text-ink-3 hover:text-ink">
        <X className="size-3.5" />
      </button>
    </div>
  );
}

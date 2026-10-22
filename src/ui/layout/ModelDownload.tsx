/**
 * Plain-language progress for the one-time model download. Shown wherever a
 * wait depends on a model that is still loading, so the first visit explains
 * itself instead of looking stuck.
 */
import { Download } from "lucide-react";
import { ml } from "@/ml/client";
import { MODELS } from "@/ml/models";
import type { ModelKey } from "@/types";
import { Progress } from "@/ui/kit";
import { useStore } from "@/ui/hooks/store";

const FRIENDLY: Record<ModelKey, string> = {
  embedder: "search model",
  reranker: "ranking model",
  reader: "reading model",
};

/** Starts downloading models ahead of need. Failures surface later, on real use. */
export function warmModels(keys: ModelKey[]): void {
  if (typeof Worker === "undefined") return;
  for (const key of keys) void ml.warmup(key).catch(() => undefined);
}

export function ModelDownload({ keys }: { keys: ModelKey[] }) {
  const status = useStore((s) => s.ml);
  const loading = keys.filter((k) => status?.[k]?.state === "loading");
  if (loading.length === 0) return null;

  const totalMB = loading.reduce((sum, k) => sum + MODELS[k].sizeMB, 0);
  const done = loading.reduce((sum, k) => sum + MODELS[k].sizeMB * (status?.[k]?.progress ?? 0), 0);
  const progress = totalMB > 0 ? done / totalMB : 0;
  const names = loading.map((k) => FRIENDLY[k]).join(" and ");

  return (
    <div role="status" className="animate-in rounded-xl border border-accent/30 bg-accent-soft/60 px-4 py-3">
      <div className="flex items-center gap-2 text-xs font-medium text-ink">
        <Download className="size-3.5 text-accent" />
        <span className="flex-1">
          Downloading the {names} ({totalMB} MB), first visit only
        </span>
        <span className="tabular text-ink-3">{Math.round(progress * 100)}%</span>
      </div>
      <Progress value={progress} className="mt-2" />
      <p className="mt-2 text-[11px] leading-relaxed text-ink-3">
        The models run on your device and are saved in your browser, so the next paper starts in seconds.
      </p>
    </div>
  );
}

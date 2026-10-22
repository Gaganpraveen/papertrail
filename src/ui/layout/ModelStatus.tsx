/**
 * Footer popover listing the three on-device models, their size, purpose and
 * live load state. Models load lazily on first use; "Load" warms one up early.
 */
import { useEffect, useRef, useState } from "react";
import { Cpu } from "lucide-react";
import { MODELS } from "@/ml/models";
import { ml } from "@/ml/client";
import type { ModelKey, ModelStatus as ModelState } from "@/types";
import { Progress, cn } from "@/ui/kit";
import { useStore } from "@/ui/hooks/store";

const KEYS = Object.keys(MODELS) as ModelKey[];

const stateDot: Record<ModelState["state"], string> = {
  idle: "bg-line-strong",
  loading: "bg-accent animate-soft-pulse",
  ready: "bg-good",
  error: "bg-bad",
};

const stateText: Record<ModelState["state"], string> = {
  idle: "Not loaded",
  loading: "Loading",
  ready: "Ready",
  error: "Unavailable",
};

export function ModelStatus() {
  const status = useStore((s) => s.ml);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("pointerdown", onDown);
    return () => window.removeEventListener("pointerdown", onDown);
  }, [open]);

  const ready = KEYS.filter((k) => status[k]?.state === "ready").length;
  const totalMB = KEYS.reduce((sum, k) => sum + MODELS[k].sizeMB, 0);

  return (
    <div ref={ref} className="relative min-w-0 flex-1">
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs text-ink-2 hover:bg-panel-2"
      >
        <Cpu className="size-4 text-ink-3" />
        <span className="flex-1 truncate">On-device models</span>
        <span className="flex items-center gap-1">
          {KEYS.map((k) => (
            <span key={k} className={cn("size-1.5 rounded-full", stateDot[status[k]?.state ?? "idle"])} />
          ))}
        </span>
      </button>

      {open && (
        <div className="animate-in absolute bottom-full left-0 z-50 mb-2 w-[300px] rounded-xl border border-line-strong bg-panel p-3 shadow-card">
          <div className="mb-2 flex items-baseline justify-between">
            <p className="text-xs font-semibold text-ink">Models running in your browser</p>
            <p className="text-[11px] tabular text-ink-3">
              {ready}/{KEYS.length} ready · {totalMB} MB
            </p>
          </div>
          <ul className="space-y-2.5">
            {KEYS.map((key) => {
              const spec = MODELS[key];
              const s = status[key] ?? { state: "idle", progress: 0 };
              return (
                <li key={key} className="rounded-lg border border-line bg-panel-2/50 p-2.5">
                  <div className="flex items-center gap-2">
                    <span className={cn("size-2 rounded-full", stateDot[s.state])} />
                    <span className="flex-1 truncate text-xs font-medium text-ink">{spec.label}</span>
                    {s.state === "idle" || s.state === "error" ? (
                      <button
                        onClick={() => void ml.warmup(key).catch(() => undefined)}
                        className="rounded-md border border-line px-1.5 py-0.5 text-[10px] font-medium text-ink-2 hover:border-line-strong hover:text-ink"
                      >
                        {s.state === "error" ? "Retry" : "Load"}
                      </button>
                    ) : (
                      <span className="text-[10px] text-ink-3">{stateText[s.state]}</span>
                    )}
                  </div>
                  <p className="mt-1 text-[11px] leading-snug text-ink-3">{spec.purpose}</p>
                  <p className="mt-1 font-mono text-[10px] text-ink-3">
                    {spec.id} · {spec.sizeMB} MB · int8
                  </p>
                  {s.state === "loading" && <Progress value={s.progress} className="mt-2 h-1" />}
                  {s.state === "error" && s.error && (
                    <p className="mt-1 text-[11px] leading-snug text-bad">{s.error}</p>
                  )}
                </li>
              );
            })}
          </ul>
          <p className="mt-2.5 text-[11px] leading-snug text-ink-3">
            Inference runs in a Web Worker with ONNX Runtime (WebAssembly). If a model cannot load,
            PaperTrail falls back to lexical methods and says so.
          </p>
        </div>
      )}
    </div>
  );
}

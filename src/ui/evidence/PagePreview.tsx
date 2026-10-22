/**
 * Renders one PDF page from the locally cached file and highlights the quote
 * on it, so every claim can be checked against the original typesetting.
 * The drawer remounts it per piece of evidence, which resets the page.
 */
import { useEffect, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, FileText } from "lucide-react";
import { renderPage } from "@/parsing";
import { IconButton, Skeleton } from "@/ui/kit";
import { store } from "@/ui/hooks/store";

type LoadState = "loading" | "ready" | "missing" | "error";

export function PagePreview({
  paperId,
  page,
  pageCount,
  quote,
}: {
  paperId: string;
  page: number;
  pageCount?: number;
  quote: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [current, setCurrent] = useState(page);
  const [state, setState] = useState<LoadState>("loading");

  useEffect(() => {
    let cancelled = false;
    setState("loading");
    (async () => {
      const bytes = await store().getPdf(paperId);
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (!bytes || !canvas) {
        setState("missing");
        return;
      }
      // Only highlight on the page the quote was taken from.
      await renderPage(bytes, current, canvas, {
        scale: 1.4,
        highlight: current === page ? quote : undefined,
      });
      if (!cancelled) setState("ready");
    })().catch(() => {
      if (!cancelled) setState("error");
    });
    return () => {
      cancelled = true;
    };
  }, [paperId, current, page, quote]);

  if (state === "missing" || state === "error") {
    return (
      <div className="dot-grid flex flex-col items-center gap-2 rounded-xl border border-dashed border-line px-6 py-10 text-center">
        <FileText className="size-5 text-ink-3" />
        <p className="text-xs leading-relaxed text-ink-3">
          {state === "missing"
            ? "No PDF is cached for this paper (it was analysed from its abstract), so there is no page to show."
            : "This page could not be rendered."}
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[11px] font-medium tabular text-ink-3">
          Page {current}
          {pageCount ? ` of ${pageCount}` : ""}
          {current !== page && (
            <button onClick={() => setCurrent(page)} className="ml-2 text-accent hover:underline">
              back to p.{page}
            </button>
          )}
        </span>
        <span className="flex items-center gap-0.5">
          <IconButton label="Previous page" disabled={current <= 1} onClick={() => setCurrent(current - 1)}>
            <ChevronLeft className="size-4" />
          </IconButton>
          <IconButton
            label="Next page"
            disabled={pageCount !== undefined && current >= pageCount}
            onClick={() => setCurrent(current + 1)}
          >
            <ChevronRight className="size-4" />
          </IconButton>
        </span>
      </div>
      <div className="relative overflow-hidden rounded-xl border border-line bg-white">
        {state === "loading" && <Skeleton className="absolute inset-0 rounded-none" />}
        <canvas ref={canvasRef} className="block w-full" />
      </div>
    </div>
  );
}

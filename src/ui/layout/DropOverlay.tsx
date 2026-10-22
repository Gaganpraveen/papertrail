import { FileUp } from "lucide-react";

export function DropOverlay() {
  return (
    <div className="animate-in pointer-events-none fixed inset-0 z-[60] flex items-center justify-center bg-bg/70 p-6 backdrop-blur-sm">
      <div className="flex w-full max-w-md flex-col items-center gap-3 rounded-3xl border-2 border-dashed border-accent bg-panel/90 px-8 py-12 text-center shadow-card">
        <div className="flex size-12 items-center justify-center rounded-2xl bg-accent-soft text-accent">
          <FileUp className="size-6" />
        </div>
        <p className="text-lg font-semibold tracking-tight text-ink">Drop PDFs to analyse</p>
        <p className="text-sm leading-relaxed text-ink-3">
          Parsed locally with pdf.js. The file never leaves your device.
        </p>
      </div>
    </div>
  );
}

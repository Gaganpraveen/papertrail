/**
 * The single entry point: paste an arXiv id, DOI, PDF link or a list of them,
 * type a research topic, or upload PDFs. The detected input type is shown
 * live so the user knows what will happen before pressing Enter.
 */
import { useEffect, useRef, useState, type FormEvent } from "react";
import { ArrowRight, CornerDownLeft, FileUp, Search } from "lucide-react";
import { Kbd, cn } from "@/ui/kit";
import { store } from "@/ui/hooks/store";
import { FOCUS_COMMAND_EVENT } from "@/ui/hooks/useHotkeys";
import { isPdfFile } from "@/ui/hooks/useFileDrop";
import { previewInput } from "./inputKind";

const PLACEHOLDER = "Paste an arXiv ID, DOI or PDF link, or describe a topic…";

const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

export function CommandBar({ variant = "compact" }: { variant?: "hero" | "compact" }) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const preview = previewInput(value);
  const hero = variant === "hero";

  useEffect(() => {
    const focus = () => {
      inputRef.current?.focus();
      inputRef.current?.select();
    };
    window.addEventListener(FOCUS_COMMAND_EVENT, focus);
    return () => window.removeEventListener(FOCUS_COMMAND_EVENT, focus);
  }, []);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!preview || busy) return;
    setBusy(true);
    setValue("");
    try {
      const { submit } = store();
      // Sequential, so papers enter the analysis queue in the order given.
      for (const entry of preview.entries) await submit(entry);
    } finally {
      setBusy(false);
    }
  };

  const onFiles = (files: FileList | null) => {
    const pdfs = Array.from(files ?? []).filter(isPdfFile);
    if (pdfs.length > 0) void store().submitFiles(pdfs);
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <form
      onSubmit={onSubmit}
      className={cn(
        "group relative flex w-full items-center gap-2 border bg-panel transition-all",
        "border-line focus-within:border-accent/60 focus-within:shadow-[0_0_0_4px_var(--accent-soft)]",
        hero ? "h-14 rounded-2xl pr-2 pl-4 shadow-card" : "h-9 rounded-xl pr-1 pl-3",
      )}
    >
      <Search className={cn("shrink-0 text-ink-3", hero ? "size-5" : "size-4")} />
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={PLACEHOLDER}
        aria-label="Paper or topic"
        spellCheck={false}
        autoComplete="off"
        className={cn(
          "min-w-0 flex-1 bg-transparent text-ink outline-none placeholder:text-ink-3",
          hero ? "text-[15px]" : "text-sm",
        )}
      />
      {preview ? (
        <span
          className={cn(
            "hidden shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium sm:inline-flex",
            preview.kind === "topic" ? "bg-panel-2 text-ink-2" : "bg-accent-soft text-accent",
          )}
        >
          {preview.label}
        </span>
      ) : (
        !hero && (
          <span className="hidden shrink-0 sm:inline-flex">
            <Kbd>{isMac ? "⌘K" : "Ctrl K"}</Kbd>
          </span>
        )
      )}
      <input
        ref={fileRef}
        type="file"
        accept="application/pdf,.pdf"
        multiple
        hidden
        onChange={(e) => onFiles(e.target.files)}
      />
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        title="Upload PDFs (or drop them anywhere)"
        className={cn(
          "inline-flex shrink-0 items-center gap-1.5 rounded-lg text-ink-2 transition-colors hover:bg-panel-2 hover:text-ink",
          hero ? "h-10 px-2.5 text-sm sm:px-3" : "h-7 px-2 text-xs",
        )}
      >
        <FileUp className="size-4" />
        <span className={cn(hero ? "hidden sm:inline" : "hidden md:inline")}>Upload PDF</span>
      </button>
      {hero && (
        <button
          type="submit"
          disabled={!preview || busy}
          className="inline-flex h-10 shrink-0 items-center gap-1.5 rounded-xl bg-accent px-3 text-sm sm:px-4 font-medium text-accent-ink transition-all hover:brightness-110 disabled:opacity-40"
        >
          {preview?.kind === "topic" ? "Search" : "Analyze"}
          <ArrowRight className="size-4" />
        </button>
      )}
      {!hero && preview && (
        <button
          type="submit"
          className="inline-flex size-7 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-ink"
          aria-label="Submit"
        >
          <CornerDownLeft className="size-3.5" />
        </button>
      )}
    </form>
  );
}

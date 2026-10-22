/**
 * Right-hand drawer that answers "where does this come from?": the verbatim
 * quote, its section and page, and the rendered PDF page with the quote
 * highlighted. Opened from any evidence chip through store.openEvidence.
 */
import { useState } from "react";
import { Check, Copy, ExternalLink, Quote, X } from "lucide-react";
import { arxivAbsUrl, arxivPdfUrl } from "@/sources/arxiv";
import { Badge, IconButton, cn } from "@/ui/kit";
import { store, useStore } from "@/ui/hooks/store";
import { PagePreview } from "./PagePreview";

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <IconButton
      label={copied ? "Copied" : "Copy quote"}
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {copied ? <Check className="size-4 text-good" /> : <Copy className="size-4" />}
    </IconButton>
  );
}

export function EvidenceDrawer() {
  const evidence = useStore((s) => s.evidence);
  const record = useStore((s) => (s.evidence ? s.papers[s.evidence.paperId] : undefined));
  const open = evidence !== null;
  const meta = record?.meta;
  const pageCount = record?.doc?.pageCount;

  return (
    <aside
      aria-label="Evidence"
      aria-hidden={!open}
      className={cn(
        "fixed inset-y-0 right-0 z-40 flex w-full max-w-[480px] flex-col border-l border-line bg-panel shadow-card transition-transform duration-200 ease-out",
        open ? "translate-x-0" : "pointer-events-none translate-x-full",
      )}
    >
      {evidence && (
        <>
          <header className="flex items-start gap-3 border-b border-line px-5 py-4">
            <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent-soft text-accent">
              <Quote className="size-4" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">Evidence</p>
              <p className="mt-0.5 line-clamp-2 text-sm font-medium leading-snug text-ink">
                {meta?.title ?? evidence.paperId}
              </p>
            </div>
            <IconButton label="Close (Esc)" onClick={() => store().openEvidence(null)}>
              <X className="size-4" />
            </IconButton>
          </header>

          <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-5 py-5">
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge tone="accent">Page {evidence.page}</Badge>
              {evidence.section && <Badge>{evidence.section}</Badge>}
            </div>

            <figure className="group relative rounded-xl border border-line bg-panel-2/60 py-3.5 pr-10 pl-4">
              <div className="absolute inset-y-3 left-0 w-0.5 rounded-full bg-accent" />
              <blockquote className="text-[13px] leading-relaxed text-ink">“{evidence.quote}”</blockquote>
              <div className="absolute top-2 right-2">
                <CopyButton text={evidence.quote} />
              </div>
            </figure>

            {meta?.arxivId && (
              <div className="flex flex-wrap gap-2">
                <a
                  href={arxivAbsUrl(meta.arxivId, meta.version)}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1.5 text-xs font-medium text-ink-2 hover:border-line-strong hover:text-ink"
                >
                  Open on arXiv <ExternalLink className="size-3.5" />
                </a>
                <a
                  href={`${arxivPdfUrl(meta.arxivId, meta.version)}#page=${evidence.page}`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-lg border border-line px-2.5 py-1.5 text-xs font-medium text-ink-2 hover:border-line-strong hover:text-ink"
                >
                  PDF at page {evidence.page} <ExternalLink className="size-3.5" />
                </a>
              </div>
            )}

            <PagePreview
              key={`${evidence.paperId}:${evidence.page}:${evidence.quote}`}
              paperId={evidence.paperId}
              page={evidence.page}
              pageCount={pageCount}
              quote={evidence.quote}
            />
          </div>
        </>
      )}
    </aside>
  );
}

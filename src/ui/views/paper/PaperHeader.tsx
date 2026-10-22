/**
 * Paper header: bibliographic facts, provenance badges, the overall rigor
 * score and the export / survey actions.
 */
import { Braces, Check, Download, ExternalLink, FileText, ListPlus } from "lucide-react";
import type { ReactNode } from "react";
import { arxivAbsUrl, arxivPdfUrl } from "@/sources/arxiv";
import { briefingMarkdown, download } from "@/store/export";
import { useLibrary } from "@/store/library";
import { MAX_SURVEY_PAPERS, type PaperRecord } from "@/store/types";
import { Badge, Button, ScoreRing, Skeleton } from "@/ui/kit";

const MAX_AUTHORS = 4;

function slugify(text: string): string {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || "paper"
  );
}

function exportJson(record: PaperRecord): string {
  const { meta, mode, analysis, warnings, qa } = record;
  return JSON.stringify(
    {
      exportedBy: "PaperTrail",
      exportedAt: new Date().toISOString(),
      meta,
      mode,
      analysis,
      warnings,
      qa: qa.map(({ hits, ...answer }) => ({ ...answer, retrieved: hits.length })),
    },
    null,
    2,
  );
}

function ExternalAnchor({ href, icon, children }: { href: string; icon?: ReactNode; children: ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer noopener"
      className="inline-flex items-center gap-1 text-ink-3 transition-colors hover:text-accent"
    >
      {icon}
      {children}
    </a>
  );
}

function AuthorLine({ authors }: { authors: string[] }) {
  if (authors.length === 0) return <span className="text-ink-3">Unknown authors</span>;
  const shown = authors.slice(0, MAX_AUTHORS);
  const hidden = authors.length - shown.length;
  return (
    <span className="min-w-0" title={authors.join(", ")}>
      {shown.join(", ")}
      {hidden > 0 && <span className="text-ink-3"> +{hidden}</span>}
    </span>
  );
}

function PaperLinks({ record }: { record: PaperRecord }) {
  const { meta } = record;
  const links: ReactNode[] = [];
  if (meta.arxivId) {
    links.push(
      <ExternalAnchor key="abs" href={arxivAbsUrl(meta.arxivId, meta.version)} icon={<ExternalLink className="size-3.5" />}>
        arXiv {meta.arxivId}
      </ExternalAnchor>,
      <ExternalAnchor key="pdf" href={arxivPdfUrl(meta.arxivId, meta.version)} icon={<FileText className="size-3.5" />}>
        PDF
      </ExternalAnchor>,
    );
  } else if (meta.pdfUrl) {
    links.push(
      <ExternalAnchor key="pdf" href={meta.pdfUrl} icon={<FileText className="size-3.5" />}>
        PDF
      </ExternalAnchor>,
    );
  }
  if (meta.doi) {
    links.push(
      <ExternalAnchor key="doi" href={`https://doi.org/${meta.doi}`} icon={<ExternalLink className="size-3.5" />}>
        DOI {meta.doi}
      </ExternalAnchor>,
    );
  }
  if (!meta.arxivId && !meta.doi && meta.landingUrl) {
    links.push(
      <ExternalAnchor key="landing" href={meta.landingUrl} icon={<ExternalLink className="size-3.5" />}>
        Landing page
      </ExternalAnchor>,
    );
  }
  if (links.length === 0) return null;
  return <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">{links}</div>;
}

export function PaperHeader({ record }: { record: PaperRecord }) {
  const { meta, analysis } = record;
  const selected = useLibrary((s) => s.surveySelection.includes(record.id));
  const selectionFull = useLibrary((s) => s.surveySelection.length >= MAX_SURVEY_PAPERS);
  const toggleSurveySelection = useLibrary((s) => s.toggleSurveySelection);
  const ready = record.status === "ready" && !!analysis;
  // The mode is only settled once the analysis exists; a failed run never read the full text.
  const mode = record.status === "ready" ? (analysis?.mode ?? record.mode) : undefined;
  const slug = slugify(meta.title || meta.id);
  const year = meta.year ?? (meta.published ? Number.parseInt(meta.published.slice(0, 4), 10) : undefined);

  return (
    <header className="animate-in">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
        <div className="min-w-0 flex-1">
          <div className="mb-3 flex flex-wrap items-center gap-1.5">
            {mode === "full" && <Badge tone="good">Full text</Badge>}
            {mode === "abstract-only" && <Badge tone="warn">Abstract only</Badge>}
            {analysis && analysis.stats.pages > 0 && <Badge>{analysis.stats.pages} pages</Badge>}
            {analysis && analysis.stats.references > 0 && <Badge>{analysis.stats.references} references</Badge>}
            {meta.citationCount !== undefined && <Badge>{meta.citationCount.toLocaleString()} citations</Badge>}
            {meta.categories?.slice(0, 3).map((c) => (
              <Badge key={c} tone="accent">
                {c}
              </Badge>
            ))}
          </div>
          {meta.title ? (
            <h1 className="text-2xl leading-tight font-semibold tracking-tight text-balance text-ink">{meta.title}</h1>
          ) : (
            <Skeleton className="h-8 w-3/4" />
          )}
          <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-ink-2">
            <AuthorLine authors={meta.authors} />
            {year ? <span className="text-ink-3">· {year}</span> : null}
            {meta.venue && <span className="truncate text-ink-3">· {meta.venue}</span>}
          </div>
          <div className="mt-3">
            <PaperLinks record={record} />
          </div>
        </div>

        {analysis && (
          <div className="flex shrink-0 items-center gap-4 rounded-2xl border border-line bg-panel px-4 py-3">
            <ScoreRing value={analysis.rigor.overall} size={60} label="rigor" />
            <div>
              <div className="text-[11px] font-medium tracking-wider text-ink-3 uppercase">Rigor grade</div>
              <div className="text-2xl leading-tight font-semibold tracking-tight text-ink">{analysis.rigor.grade}</div>
              <div className="text-[11px] text-ink-3">weighted over {analysis.rigor.axes.length} axes</div>
            </div>
          </div>
        )}
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant={selected ? "primary" : "secondary"}
          disabled={!ready || (!selected && selectionFull)}
          onClick={() => toggleSurveySelection(record.id)}
          title={
            !ready
              ? "Available once the analysis finishes"
              : !selected && selectionFull
                ? `A survey holds at most ${MAX_SURVEY_PAPERS} papers`
                : undefined
          }
        >
          {selected ? <Check className="size-3.5" /> : <ListPlus className="size-3.5" />}
          {selected ? "In survey" : "Add to survey"}
        </Button>
        <Button size="sm" variant="ghost" disabled={!ready} onClick={() => download(`${slug}.md`, briefingMarkdown(record))}>
          <Download className="size-3.5" />
          Export Markdown
        </Button>
        <Button size="sm" variant="ghost" disabled={!ready} onClick={() => download(`${slug}.json`, exportJson(record))}>
          <Braces className="size-3.5" />
          Export JSON
        </Button>
        {analysis && (
          <span className="ml-auto text-[11px] text-ink-3 tabular">
            Analysis computed locally in {(analysis.computedMs / 1000).toFixed(1)} s
          </span>
        )}
      </div>
    </header>
  );
}

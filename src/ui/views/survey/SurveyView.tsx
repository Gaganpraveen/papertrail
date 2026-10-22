/**
 * Literature survey view: the generated review with live citations, then
 * the evidence behind it (semantic map, similarity, themes, comparison
 * matrix, timeline, coupling network, consensus and open gaps).
 */
import { useMemo, type ReactNode } from "react";
import {
  BookOpen,
  ChartNetwork,
  ChartScatter,
  CircleAlert,
  Download,
  GitCompareArrows,
  Grid3x3,
  Handshake,
  Layers,
  Lightbulb,
  Milestone,
  RotateCcw,
  Table,
  Trash2,
} from "lucide-react";
import { surveyMarkdown, download } from "@/store/export";
import type { PaperRecord, SurveyRecord } from "@/store/types";
import type { SurveyResult } from "@/types";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, Skeleton, Spinner, seriesColor } from "@/ui/kit";
import { CouplingNetwork } from "@/ui/charts/CouplingNetwork";
import { SemanticMap, type SeriesPaper } from "@/ui/charts/SemanticMap";
import { SimilarityHeatmap } from "@/ui/charts/SimilarityHeatmap";
import { Timeline } from "@/ui/charts/Timeline";
import { store, useStore } from "@/ui/hooks/store";
import { Markdown, refAnchor } from "./Markdown";
import { ComparisonMatrix, Consensus, Gaps, PaperTag, Themes } from "./sections";

const SECTIONS = [
  { id: "review", label: "Review" },
  { id: "map", label: "Semantic map" },
  { id: "themes", label: "Themes" },
  { id: "compare", label: "Comparison" },
  { id: "timeline", label: "Timeline" },
  { id: "coupling", label: "Coupling" },
  { id: "consensus", label: "Consensus" },
  { id: "gaps", label: "Gaps" },
];

function seriesFor(result: SurveyResult, papers: Record<string, PaperRecord>): SeriesPaper[] {
  return result.paperIds.map((id, i) => ({
    id,
    short: `[${i + 1}]`,
    title: papers[id]?.meta.title ?? id,
    color: seriesColor(i),
  }));
}

/** "prompting & reasoning vs. quantization & KV cache", from the clusters' top terms. */
function surveyTitle(result: SurveyResult): string {
  const themes = result.clusters.slice(0, 3).map((c) => c.terms.slice(0, 2).join(" & ")).filter(Boolean);
  if (themes.length === 0) return `Survey of ${result.paperIds.length} papers`;
  const title = themes.join(themes.length === 2 ? " vs. " : " · ");
  return title.charAt(0).toUpperCase() + title.slice(1);
}

function scrollToSection(id: string) {
  document.getElementById(`survey-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

/** Scrolls to a reference and flashes it, falling back to the paper chips. */
function jumpToReference(n: number) {
  const target = document.getElementById(refAnchor(n)) ?? document.getElementById(`survey-paper-${n}`);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.classList.add("bg-accent-soft");
  window.setTimeout(() => target.classList.remove("bg-accent-soft"), 1600);
}

function Section({
  id,
  title,
  subtitle,
  icon,
  children,
  actions,
}: {
  id: string;
  title: string;
  subtitle?: string;
  icon: ReactNode;
  children: ReactNode;
  actions?: ReactNode;
}) {
  // min-w-0: below `lg` the chart grids use an auto column, and a card sized by
  // its SVG would otherwise never let the charts shrink when the viewport does.
  return (
    <Card id={`survey-${id}`} className="scroll-mt-16 min-w-0">
      <CardHeader title={title} subtitle={subtitle} icon={icon} actions={actions} />
      <CardBody>{children}</CardBody>
    </Card>
  );
}

function RunningState({ survey }: { survey: SurveyRecord }) {
  const last = [...survey.trace].reverse().find((e) => e.type === "log" || e.type === "node:start");
  return (
    <div className="mx-auto max-w-5xl space-y-4 px-4 py-8 sm:px-8">
      <div className="flex items-center gap-3">
        <Spinner />
        <div>
          <p className="text-sm font-semibold text-ink">Synthesising a survey of {survey.paperIds.length} papers</p>
          <p className="text-xs text-ink-3">{last?.message ?? (last ? `Running ${last.node}…` : "Starting…")}</p>
        </div>
      </div>
      <Skeleton className="h-64 w-full rounded-2xl" />
      <div className="grid gap-4 md:grid-cols-2">
        <Skeleton className="h-72 rounded-2xl" />
        <Skeleton className="h-72 rounded-2xl" />
      </div>
    </div>
  );
}

export default function SurveyView({ id }: { id: string }) {
  const survey = useStore((s) => s.surveys[id]);
  const papers = useStore((s) => s.papers);
  const result = survey?.result;
  const series = useMemo(() => (result ? seriesFor(result, papers) : []), [result, papers]);
  const byId = useMemo(() => new Map(series.map((p) => [p.id, p])), [series]);

  if (!survey) {
    return (
      <EmptyState
        icon={<Layers className="size-5" />}
        title="Survey not found"
        body="It may have been deleted. Tick two or more analysed papers in the library to start a new one."
        action={<Button onClick={() => store().setView({ kind: "home" })}>Back home</Button>}
      />
    );
  }
  if (survey.status === "running") return <RunningState survey={survey} />;
  if (survey.status === "failed" || !result) {
    return (
      <div className="mx-auto max-w-xl px-4 py-12">
        <Card className="p-2">
          <EmptyState
            icon={<CircleAlert className="size-5 text-bad" />}
            title="The survey could not be built"
            body={survey.error ?? "Something went wrong while synthesising the papers."}
            action={
              <Button onClick={() => void store().runSurvey(survey.paperIds)}>
                <RotateCcw className="size-3.5" /> Try again
              </Button>
            }
          />
        </Card>
      </div>
    );
  }

  const describePoint = (point: SurveyResult["map"]["points"][number]) =>
    papers[point.paperId]?.chunks?.find((c) => c.id === point.chunkId)?.text;
  const openPoint = (point: SurveyResult["map"]["points"][number]) => {
    const chunk = papers[point.paperId]?.chunks?.find((c) => c.id === point.chunkId);
    if (!chunk) return;
    const quote = chunk.text.split(/\s+/).slice(0, 45).join(" ");
    store().openEvidence({ paperId: chunk.paperId, page: chunk.page, section: chunk.sectionTitle, quote, chunkId: chunk.id });
  };
  const exportMarkdown = () => {
    const date = new Date(survey.createdAt).toISOString().slice(0, 10);
    download(`papertrail-survey-${date}.md`, surveyMarkdown(survey, papers));
  };
  const remove = () => void store().removeSurvey(survey.id);

  return (
    <div className="mx-auto max-w-6xl px-4 pt-8 pb-16 sm:px-8">
      <header>
        <div className="flex flex-wrap items-start gap-4">
          <div className="min-w-0 flex-1">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">Literature survey</p>
            <h1 className="mt-1 text-2xl font-semibold tracking-tight text-ink">{surveyTitle(result)}</h1>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <Badge>{result.paperIds.length} papers</Badge>
              <Badge>{result.clusters.length} themes</Badge>
              <Badge tone="accent">{result.map.method === "umap" ? "UMAP projection" : "MDS projection"}</Badge>
              <span className="text-[11px] text-ink-3">
                {new Date(result.createdAt).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" onClick={exportMarkdown}>
              <Download className="size-3.5" /> Export Markdown
            </Button>
            <Button size="sm" variant="ghost" onClick={remove} aria-label="Delete survey">
              <Trash2 className="size-3.5" />
            </Button>
          </div>
        </div>

        <ul className="mt-5 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {series.map((p, i) => (
            <li key={p.id} id={`survey-paper-${i + 1}`} className="rounded-xl transition-colors duration-700">
              <button
                onClick={() => store().setView({ kind: "paper", id: p.id, tab: "briefing" })}
                className="flex w-full items-start gap-2 rounded-xl border border-line bg-panel px-3 py-2.5 text-left transition-colors hover:border-line-strong"
              >
                <PaperTag paper={p} className="mt-px shrink-0" />
                <span className="line-clamp-2 text-xs font-medium leading-snug text-ink">{p.title}</span>
              </button>
            </li>
          ))}
        </ul>
      </header>

      <nav className="sticky top-0 z-20 -mx-4 mt-6 mb-5 flex gap-1 overflow-x-auto border-b border-line bg-bg/85 px-4 py-2 backdrop-blur sm:-mx-8 sm:px-8">
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            onClick={() => scrollToSection(s.id)}
            className="shrink-0 rounded-md px-2.5 py-1 text-xs font-medium text-ink-3 transition-colors hover:bg-panel-2 hover:text-ink"
          >
            {s.label}
          </button>
        ))}
      </nav>

      <div className="space-y-5">
        <Section
          id="review"
          title="Literature review"
          subtitle="Written from the papers' own sentences; every claim is cited. Click a citation to jump to its reference."
          icon={<BookOpen className="size-4" />}
        >
          <div className="max-w-3xl">
            <Markdown
              source={result.narrative}
              onCite={jumpToReference}
              citeTitle={(n) => series[n - 1]?.title}
              citeColor={(n) => series[n - 1]?.color}
            />
          </div>
        </Section>

        <div className="grid gap-5 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <Section
            id="map"
            title="Semantic map"
            subtitle="Each dot is a passage; large dots are paper centroids. Click a passage to see it in the PDF."
            icon={<ChartScatter className="size-4" />}
          >
            <SemanticMap points={result.map.points} papers={series} describe={describePoint} onSelect={openPoint} />
          </Section>
          <Section
            id="similarity"
            title="Similarity"
            subtitle="How close each pair of papers is in embedding space."
            icon={<Grid3x3 className="size-4" />}
          >
            <SimilarityHeatmap matrix={result.similarity} papers={series} />
          </Section>
        </div>

        <Section
          id="themes"
          title="Themes"
          subtitle="Papers grouped by content; labels are the most distinctive terms of each group (class-based TF-IDF)."
          icon={<Layers className="size-4" />}
        >
          <Themes clusters={result.clusters} silhouette={result.silhouette} byId={byId} />
        </Section>

        <Section
          id="compare"
          title="Comparison matrix"
          subtitle="Extracted per paper from its analysis. Scroll sideways on small screens."
          icon={<Table className="size-4" />}
        >
          <ComparisonMatrix rows={result.comparison} byId={byId} />
        </Section>

        <div className="grid gap-5 lg:grid-cols-2">
          <Section id="timeline" title="Timeline" subtitle="Publication dates." icon={<Milestone className="size-4" />}>
            <Timeline items={result.timeline} papers={series} />
          </Section>
          <Section
            id="coupling"
            title="Citation coupling"
            subtitle="Edges join papers that cite the same works; width is the Jaccard overlap."
            icon={<ChartNetwork className="size-4" />}
          >
            <CouplingNetwork links={result.coupling} papers={series} />
          </Section>
        </div>

        <div className="grid gap-5 lg:grid-cols-2">
          <Section
            id="consensus"
            title="Consensus"
            subtitle="Findings that several papers state in similar terms."
            icon={<Handshake className="size-4" />}
          >
            <Consensus items={result.consensus} byId={byId} />
          </Section>
          <Section
            id="gaps"
            title="Research gaps"
            subtitle="Future work and limitations raised by one paper and not addressed by the others."
            icon={<Lightbulb className="size-4" />}
          >
            <Gaps items={result.gaps} byId={byId} />
          </Section>
        </div>

        <p className="flex items-center justify-center gap-2 pt-2 text-[11px] text-ink-3">
          <GitCompareArrows className="size-3.5" />
          Everything above is computed from the selected papers; nothing is generated from outside knowledge.
        </p>
      </div>
    </div>
  );
}

/**
 * Visual analysis tab: a grid of chart cards over the analysis output.
 */
import { BookOpen, ChartColumn, Gauge, Hash, History, Layers, Network, Radar, Waypoints } from "lucide-react";
import type { ReactNode } from "react";
import type { Evidence, PaperAnalysis, PaperMeta, ZonedSentence } from "@/types";
import { useLibrary } from "@/store/library";
import { Card, CardBody, CardHeader, Stat, cn } from "@/ui/kit";
import { CitationAges } from "@/ui/charts/CitationAges";
import { KeyphraseNetwork } from "@/ui/charts/KeyphraseNetwork";
import { MetricBars } from "@/ui/charts/MetricBars";
import { RhetoricalMap } from "@/ui/charts/RhetoricalMap";
import { RigorRadar } from "@/ui/charts/RigorRadar";
import { SectionComposition } from "@/ui/charts/SectionComposition";
import { ZoneDistribution } from "@/ui/charts/ZoneDistribution";

function ChartCard({
  icon,
  title,
  subtitle,
  className,
  children,
}: {
  icon: ReactNode;
  title: string;
  subtitle?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <Card className={cn("min-w-0", className)}>
      <CardHeader icon={icon} title={title} subtitle={subtitle} />
      <CardBody>{children}</CardBody>
    </Card>
  );
}

// Flesch Reading Ease bands (Flesch, 1948). Research papers typically land 10–40.
function fleschVerdict(score: number): { label: string; tone: string } {
  if (score >= 60) return { label: "Plain English, easy to follow", tone: "var(--good)" };
  if (score >= 50) return { label: "Fairly difficult", tone: "var(--accent)" };
  if (score >= 30) return { label: "Difficult: typical for a research paper", tone: "var(--warn)" };
  return { label: "Very difficult: dense, specialist prose", tone: "var(--bad)" };
}

// Around 20 words per sentence is the usual guidance for readable technical text.
function sentenceVerdict(avg: number): string {
  if (avg < 18) return "Short, punchy sentences.";
  if (avg <= 26) return "Typical academic sentence length.";
  return "Long sentences; expect to re-read.";
}

function jargonVerdict(density: number): string {
  if (density < 0.05) return "Light on field-specific terms.";
  if (density < 0.12) return "Moderate use of specialist vocabulary.";
  return "Heavy specialist vocabulary.";
}

function ReadabilityCard({ readability }: { readability: PaperAnalysis["visuals"]["readability"] }) {
  const flesch = Math.max(0, Math.min(100, readability.fleschReadingEase));
  const verdict = fleschVerdict(readability.fleschReadingEase);
  const bands = [
    { from: 0, to: 30, color: "var(--bad)" },
    { from: 30, to: 50, color: "var(--warn)" },
    { from: 50, to: 60, color: "var(--accent)" },
    { from: 60, to: 100, color: "var(--good)" },
  ];
  return (
    <div className="space-y-5">
      <div>
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-ink-3">Flesch reading ease</span>
          <span className="tabular text-xl font-semibold tracking-tight text-ink">{readability.fleschReadingEase.toFixed(0)}</span>
        </div>
        <svg className="mt-2 block h-3 w-full overflow-visible" role="img" aria-label={`Flesch reading ease ${flesch.toFixed(0)} of 100`}>
          {bands.map((b) => (
            <rect key={b.from} x={`${b.from}%`} width={`${b.to - b.from}%`} y={3} height={6} fill={b.color} opacity={0.35} />
          ))}
          <circle cx={`${flesch}%`} cy={6} r={5.5} fill="var(--panel)" stroke={verdict.tone} strokeWidth={2.5} />
        </svg>
        <div className="mt-1 flex justify-between text-[10px] text-ink-3 tabular">
          <span>0 · hard</span>
          <span>100 · easy</span>
        </div>
        <p className="mt-2 text-xs font-medium" style={{ color: verdict.tone }}>
          {verdict.label}
        </p>
      </div>
      <div className="grid grid-cols-3 gap-3 border-t border-line pt-4">
        <Stat label="Words / sentence" value={readability.avgSentenceLength.toFixed(1)} hint={sentenceVerdict(readability.avgSentenceLength)} />
        <Stat label="Reading time" value={`${Math.max(1, Math.round(readability.readingMinutes))} min`} hint="estimated at a steady reading pace" />
        <Stat label="Jargon density" value={`${(readability.jargonDensity * 100).toFixed(1)}%`} hint={jargonVerdict(readability.jargonDensity)} />
      </div>
    </div>
  );
}

function StatsStrip({ stats }: { stats: PaperAnalysis["stats"] }) {
  const items = [
    { label: "Pages", value: stats.pages },
    { label: "Words", value: stats.words.toLocaleString() },
    { label: "Sentences", value: stats.sentences.toLocaleString() },
    { label: "Passages", value: stats.chunks },
    { label: "Figures", value: stats.figures },
    { label: "Tables", value: stats.tables },
    { label: "References", value: stats.references },
  ];
  return (
    <Card className="grid grid-cols-2 gap-x-4 gap-y-4 px-5 py-4 sm:grid-cols-4 lg:grid-cols-7">
      {items.map((s) => (
        <Stat key={s.label} label={s.label} value={s.value} />
      ))}
    </Card>
  );
}

export function VisualsTab({
  analysis,
  meta,
  sentences,
}: {
  analysis: PaperAnalysis;
  meta: PaperMeta;
  sentences?: ZonedSentence[];
}) {
  const openEvidence = useLibrary((s) => s.openEvidence);
  const { visuals, rigor, briefing } = analysis;
  const showEvidence = (sentence: ZonedSentence) => {
    const evidence: Evidence = {
      paperId: meta.id,
      page: sentence.page,
      section: sentence.sectionTitle,
      quote: sentence.text,
      sentenceId: sentence.id,
    };
    openEvidence(evidence);
  };
  const abstractOnly = analysis.mode === "abstract-only";

  return (
    <div className="space-y-4 animate-in">
      <StatsStrip stats={analysis.stats} />
      <ChartCard
        icon={<Waypoints className="size-4" />}
        title="Rhetorical map"
        subtitle="Every sentence in reading order, coloured by its argumentative role. Hover to read it, click to see it on the page."
      >
        <RhetoricalMap points={visuals.rhetoricalMap} sentences={sentences} onSelect={showEvidence} />
      </ChartCard>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartCard
          icon={<Radar className="size-4" />}
          title="Rigor profile"
          subtitle={`Overall ${Math.round(rigor.overall)} / 100 (grade ${rigor.grade}). Hover an axis for the reasoning.`}
        >
          <RigorRadar rigor={rigor} />
        </ChartCard>
        <ChartCard icon={<ChartColumn className="size-4" />} title="Argument balance" subtitle="Sentences per argumentative zone.">
          <ZoneDistribution data={visuals.zoneDistribution} />
        </ChartCard>

        <ChartCard
          icon={<Layers className="size-4" />}
          title="Section composition"
          subtitle="Where the paper spends its words."
          className="lg:col-span-2"
        >
          <SectionComposition sections={visuals.sectionComposition} />
        </ChartCard>

        <ChartCard
          icon={<Network className="size-4" />}
          title="Key-phrase network"
          subtitle="Phrases linked when they co-occur in a sentence (PMI-weighted). Hover to see neighbours."
        >
          <KeyphraseNetwork graph={visuals.keyphraseGraph} />
        </ChartCard>
        <ChartCard icon={<History className="size-4" />} title="Citation ages" subtitle="Publication years of the references.">
          {abstractOnly ? (
            <p className="text-sm text-ink-3">The bibliography is only available when the full text is parsed.</p>
          ) : (
            <CitationAges years={visuals.citationYears} paperYear={meta.year} />
          )}
        </ChartCard>

        <ChartCard icon={<Hash className="size-4" />} title="Key numbers" subtitle="Quantities extracted from result sentences. Click one for its context.">
          <MetricBars numbers={briefing.keyNumbers} onSelect={openEvidence} />
        </ChartCard>
        <ChartCard icon={<BookOpen className="size-4" />} title="Readability" subtitle="How demanding the prose is to read.">
          <ReadabilityCard readability={visuals.readability} />
        </ChartCard>
      </div>
      {abstractOnly && (
        <p className="flex items-center gap-1.5 text-xs text-ink-3">
          <Gauge className="size-3.5" />
          Computed from the abstract only; structure, citation and network charts are limited.
        </p>
      )}
    </div>
  );
}

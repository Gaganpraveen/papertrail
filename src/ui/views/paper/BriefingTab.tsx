/**
 * Briefing tab: the paper in two minutes. Every statement is a sentence
 * selected from the paper (or a rule-based finding) and carries page-level
 * evidence chips.
 */
import { ArrowUpRight, Lightbulb, ScanSearch, Target, Workflow, FlaskConical, Quote } from "lucide-react";
import type { ReactNode } from "react";
import type { Briefing, BriefingItem, KeyNumber } from "@/types";
import { useLibrary } from "@/store/library";
import { Card, CardBody, CardHeader, SectionLabel, cn } from "@/ui/kit";
import { EvidenceChips } from "./EvidenceChips";

function ItemText({ item, className }: { item: BriefingItem; className?: string }) {
  return (
    <p className={cn("text-sm leading-relaxed text-ink-2", className)}>
      {item.text}
      <EvidenceChips evidence={item.evidence} />
    </p>
  );
}

function BulletList({ items, empty, marker }: { items: BriefingItem[]; empty: string; marker: string }) {
  if (items.length === 0) return <p className="text-sm text-ink-3">{empty}</p>;
  return (
    <ul className="space-y-2.5">
      {items.map((item, i) => (
        <li key={i} className="flex gap-2.5">
          <span className="mt-[9px] size-1.5 shrink-0 rounded-full" style={{ background: marker }} />
          <ItemText item={item} />
        </li>
      ))}
    </ul>
  );
}

function KeyNumberChip({ n }: { n: KeyNumber }) {
  const openEvidence = useLibrary((s) => s.openEvidence);
  return (
    <button
      type="button"
      onClick={() => openEvidence(n.evidence)}
      title={`“${n.evidence.quote}”`}
      className="group flex min-w-0 flex-col items-start rounded-xl border border-line bg-panel-2 px-3 py-2 text-left transition-colors hover:border-line-strong"
    >
      <span className="tabular text-base font-semibold tracking-tight text-ink">{n.value}</span>
      <span className="mt-0.5 line-clamp-1 max-w-[180px] text-[11px] text-ink-3 group-hover:text-ink-2">
        {n.label || "reported value"} · {n.evidence.page > 0 ? `p.${n.evidence.page}` : "abstract"}
      </span>
    </button>
  );
}

function Section({ icon, title, subtitle, children }: { icon: ReactNode; title: string; subtitle?: string; children: ReactNode }) {
  return (
    <Card>
      <CardHeader icon={icon} title={title} subtitle={subtitle} />
      <CardBody>{children}</CardBody>
    </Card>
  );
}

function Limitations({ limitations }: { limitations: Briefing["limitations"] }) {
  return (
    <Card>
      <CardHeader
        icon={<ScanSearch className="size-4" />}
        title="Limitations"
        subtitle="What the authors concede, and what PaperTrail's review rules flag on top of that."
      />
      <div className="grid gap-px overflow-hidden rounded-b-2xl border-t border-line bg-line md:grid-cols-2">
        <div className="bg-panel px-5 py-4">
          <SectionLabel className="mb-3 flex items-center gap-1.5">
            <span className="size-1.5 rounded-full" style={{ background: "var(--z-limitation)" }} />
            Stated by the authors
          </SectionLabel>
          <BulletList
            items={limitations.authorStated}
            marker="var(--z-limitation)"
            empty="The authors do not discuss limitations explicitly, which is itself worth noting."
          />
        </div>
        <div className="bg-panel-2 px-5 py-4">
          <SectionLabel className="mb-3 flex items-center gap-1.5 text-accent">
            <span className="size-1.5 rounded-full bg-accent" />
            Detected by PaperTrail
          </SectionLabel>
          <BulletList
            items={limitations.detected}
            marker="var(--accent)"
            empty="No additional weaknesses were detected by the review rules."
          />
        </div>
      </div>
    </Card>
  );
}

function Keyphrases({ phrases }: { phrases: Briefing["keyphrases"] }) {
  if (phrases.length === 0) return <p className="text-sm text-ink-3">No key phrases extracted.</p>;
  const max = Math.max(...phrases.map((p) => p.score)) || 1;
  return (
    <div className="flex flex-wrap gap-1.5">
      {phrases.map((p) => {
        const strength = p.score / max;
        return (
          <span
            key={p.phrase}
            title={`relevance ${p.score.toFixed(2)}`}
            className={cn(
              "rounded-lg border px-2 py-1 text-xs",
              strength > 0.66 ? "border-accent/40 bg-accent-soft text-accent" : "border-line bg-panel-2 text-ink-2",
            )}
          >
            {p.phrase}
          </span>
        );
      })}
    </div>
  );
}

export function BriefingTab({ briefing, onAsk }: { briefing: Briefing; onAsk: (question: string) => void }) {
  return (
    <div className="space-y-4 animate-in">
      <div className="relative overflow-hidden rounded-2xl border border-line bg-panel p-5 shadow-card">
        <div className="pointer-events-none absolute inset-y-0 left-0 w-0.5 bg-ink" />
        <SectionLabel className="mb-2 flex items-center gap-1.5">
          <Lightbulb className="size-3.5" />
          Why it matters
        </SectionLabel>
        <p className="text-[15px] leading-relaxed text-ink">
          {briefing.tldr.text}
          <EvidenceChips evidence={briefing.tldr.evidence} />
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          <Section icon={<Target className="size-4" />} title="Problem">
            <BulletList items={briefing.problem} marker="var(--z-problem)" empty="No explicit problem statement was found." />
          </Section>
          <Section icon={<Workflow className="size-4" />} title="Method">
            <BulletList items={briefing.method} marker="var(--z-method)" empty="No method description was found." />
          </Section>
          <Section icon={<FlaskConical className="size-4" />} title="Key results">
            {briefing.keyNumbers.length > 0 && (
              <div className="mb-4 grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-4">
                {briefing.keyNumbers.slice(0, 8).map((n, i) => (
                  <KeyNumberChip key={`${n.value}-${i}`} n={n} />
                ))}
              </div>
            )}
            <BulletList items={briefing.results} marker="var(--z-result)" empty="No result statements were found." />
          </Section>
          <Limitations limitations={briefing.limitations} />
        </div>

        <div className="space-y-4">
          <Section icon={<Quote className="size-4" />} title="Key phrases" subtitle="Ranked by relevance to the paper as a whole.">
            <Keyphrases phrases={briefing.keyphrases} />
          </Section>
          <Section
            icon={<ArrowUpRight className="size-4" />}
            title="Follow-up questions"
            subtitle="Generated from this paper's gaps. Click one to ask it."
          >
            {briefing.followUps.length === 0 ? (
              <p className="text-sm text-ink-3">No follow-up questions.</p>
            ) : (
              <ul className="-mx-2 space-y-0.5">
                {briefing.followUps.map((q) => (
                  <li key={q}>
                    <button
                      type="button"
                      onClick={() => onAsk(q)}
                      className="group flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left text-sm text-ink-2 transition-colors hover:bg-panel-2 hover:text-ink"
                    >
                      <ArrowUpRight className="mt-0.5 size-3.5 shrink-0 text-ink-3 group-hover:text-accent" />
                      {q}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Section>
        </div>
      </div>
    </div>
  );
}

/**
 * Critique tab: a structured peer review. Findings come from explainable
 * rules, each citing evidence (or explicitly stating what is absent), grouped
 * by severity; the rigor table shows exactly which signals drove each score.
 */
import { useState } from "react";
import { ChevronRight, CircleCheck, CircleQuestionMark, Gavel, ListChecks, OctagonAlert, TriangleAlert, Info } from "lucide-react";
import type { ReactNode } from "react";
import type { Critique, CritiqueFinding, RigorProfile, RigorSignal, Severity } from "@/types";
import { Badge, Card, CardBody, CardHeader, ScoreRing, cn, severityTone } from "@/ui/kit";
import { EvidenceChip, evidenceKey, uniqueEvidence } from "./EvidenceChips";

const SEVERITY_GROUPS: { severity: Severity; label: string; icon: ReactNode; blurb: string }[] = [
  { severity: "major", label: "Major concerns", icon: <OctagonAlert className="size-4 text-bad" />, blurb: "Could change the paper's conclusions." },
  { severity: "moderate", label: "Moderate concerns", icon: <TriangleAlert className="size-4 text-warn" />, blurb: "Weaken the evidence but not the core claim." },
  { severity: "minor", label: "Minor issues", icon: <Info className="size-4 text-ink-3" />, blurb: "Presentation or completeness." },
  { severity: "strength", label: "Strengths", icon: <CircleCheck className="size-4 text-good" />, blurb: "Practices that make the work more trustworthy." },
];

const CATEGORY_LABEL: Record<CritiqueFinding["category"], string> = {
  evaluation: "Evaluation",
  statistics: "Statistics",
  reproducibility: "Reproducibility",
  claims: "Claims",
  scope: "Scope",
  presentation: "Presentation",
  "related-work": "Related work",
};

function FindingRow({ finding }: { finding: CritiqueFinding }) {
  const [open, setOpen] = useState(false);
  const evidence = uniqueEvidence(finding.evidence);
  return (
    <li className="px-5 py-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={severityTone[finding.severity]}>{CATEGORY_LABEL[finding.category]}</Badge>
        <h4 className="text-sm font-semibold tracking-tight text-ink">{finding.title}</h4>
      </div>
      <p className="mt-1.5 text-sm leading-relaxed text-ink-2">{finding.rationale}</p>
      {evidence.length > 0 ? (
        <div className="mt-2">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            className="inline-flex items-center gap-1 text-xs font-medium text-ink-3 transition-colors hover:text-ink"
          >
            <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} />
            {evidence.length} evidence quote{evidence.length === 1 ? "" : "s"}
          </button>
          {open && (
            <ul className="mt-2 space-y-2 animate-in">
              {evidence.map((e) => (
                <li key={evidenceKey(e)} className="rounded-lg border-l-2 border-line-strong bg-panel-2 px-3 py-2 text-xs leading-relaxed text-ink-2">
                  “{e.quote}”
                  <span className="ml-1.5 text-ink-3">{e.section}</span>
                  <EvidenceChip evidence={e} className="ml-1.5" />
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <p className="mt-2 text-xs text-ink-3">Based on the absence of this signal anywhere in the parsed text.</p>
      )}
    </li>
  );
}

function VerdictCard({ critique, rigor }: { critique: Critique; rigor: RigorProfile }) {
  const counts = SEVERITY_GROUPS.map((g) => ({ ...g, n: critique.findings.filter((f) => f.severity === g.severity).length }));
  return (
    <Card className="p-5">
      <div className="flex flex-col gap-5 sm:flex-row sm:items-start">
        <ScoreRing value={rigor.overall} size={72} stroke={7} label={`grade ${rigor.grade}`} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5 text-[11px] font-semibold tracking-wider text-ink-3 uppercase">
            <Gavel className="size-3.5" />
            Verdict
          </div>
          <p className="mt-1.5 text-[15px] leading-relaxed text-ink">{critique.verdict}</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {counts.map((c) => (
              <Badge key={c.severity} tone={severityTone[c.severity]}>
                <span className="tabular">{c.n}</span> {c.label.split(" ")[0].toLowerCase()}
              </Badge>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}

function signalValue(signal: RigorSignal): string {
  if (typeof signal.value === "boolean") return signal.value ? "yes" : "no";
  return String(signal.value);
}

function scoreColor(score: number): string {
  return score >= 75 ? "var(--good)" : score >= 55 ? "var(--accent)" : score >= 40 ? "var(--warn)" : "var(--bad)";
}

function RigorTable({ rigor }: { rigor: RigorProfile }) {
  return (
    <Card>
      <CardHeader
        icon={<ListChecks className="size-4" />}
        title="Rigor breakdown"
        subtitle="Each axis is a saturating score over concrete signals; the overall score is the weighted sum."
      />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead>
            <tr className="border-y border-line text-[11px] tracking-wider text-ink-3 uppercase">
              <th className="px-5 py-2 font-medium">Axis</th>
              <th className="w-40 px-3 py-2 font-medium">Score</th>
              <th className="w-16 px-3 py-2 text-right font-medium">Weight</th>
              <th className="px-3 py-2 font-medium">Signals</th>
              <th className="px-5 py-2 font-medium">Why</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {rigor.axes.map((axis) => (
              <tr key={axis.key} className="align-top">
                <td className="px-5 py-3 font-medium whitespace-nowrap text-ink">{axis.label}</td>
                <td className="px-3 py-3">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-2">
                      <div className="h-full rounded-full" style={{ width: `${axis.score}%`, background: scoreColor(axis.score) }} />
                    </div>
                    <span className="w-7 text-right tabular text-xs text-ink-2">{Math.round(axis.score)}</span>
                  </div>
                </td>
                <td className="px-3 py-3 text-right tabular text-xs text-ink-3">{Math.round(axis.weight * 100)}%</td>
                <td className="px-3 py-3">
                  <ul className="space-y-1 text-xs">
                    {axis.signals.map((s) => (
                      <li key={s.label} className="flex flex-wrap items-center gap-x-1.5 text-ink-3">
                        {s.label}: <span className="tabular font-medium text-ink-2">{signalValue(s)}</span>
                        {s.evidence && <EvidenceChip evidence={s.evidence} />}
                      </li>
                    ))}
                  </ul>
                </td>
                <td className="px-5 py-3 text-xs leading-relaxed text-ink-2">{axis.explanation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

export function CritiqueTab({ critique, rigor }: { critique: Critique; rigor: RigorProfile }) {
  return (
    <div className="space-y-4 animate-in">
      <VerdictCard critique={critique} rigor={rigor} />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-4">
          {SEVERITY_GROUPS.map((group) => {
            const findings = critique.findings.filter((f) => f.severity === group.severity);
            if (findings.length === 0) return null;
            return (
              <Card key={group.severity}>
                <CardHeader
                  icon={group.icon}
                  title={
                    <span className="flex items-center gap-2">
                      {group.label}
                      <span className="tabular text-xs font-normal text-ink-3">{findings.length}</span>
                    </span>
                  }
                  subtitle={group.blurb}
                  className="border-b border-line"
                />
                <ul className="divide-y divide-line">
                  {findings.map((f) => (
                    <FindingRow key={f.id} finding={f} />
                  ))}
                </ul>
              </Card>
            );
          })}
          {critique.findings.length === 0 && (
            <Card className="p-5 text-sm text-ink-3">The review rules produced no findings for this paper.</Card>
          )}
        </div>
        <Card className="h-fit xl:sticky xl:top-4">
          <CardHeader
            icon={<CircleQuestionMark className="size-4" />}
            title="Questions for the authors"
            subtitle="What a careful reviewer would ask next."
          />
          <CardBody>
            {critique.reviewerQuestions.length === 0 ? (
              <p className="text-sm text-ink-3">No open questions.</p>
            ) : (
              <ol className="space-y-3">
                {critique.reviewerQuestions.map((q, i) => (
                  <li key={q} className="flex gap-2.5 text-sm leading-relaxed text-ink-2">
                    <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-md bg-panel-2 font-mono text-[10px] text-ink-3">
                      {i + 1}
                    </span>
                    {q}
                  </li>
                ))}
              </ol>
            )}
          </CardBody>
        </Card>
      </div>
      <RigorTable rigor={rigor} />
    </div>
  );
}


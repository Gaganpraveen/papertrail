/**
 * Single-paper view: header, then Briefing / Visual analysis / Critique / Ask.
 * The active tab lives in the store's view so it survives navigation and can
 * be deep-linked.
 */
import { useState } from "react";
import { ChartColumn, ChevronRight, FileQuestion, FileText, Gavel, Info, MessageSquareText, Sparkles } from "lucide-react";
import { useLibrary } from "@/store/library";
import type { PaperRecord, PaperTab } from "@/store/types";
import { Button, Card, EmptyState, Tabs } from "@/ui/kit";
import { AskTab } from "./AskTab";
import { BriefingTab } from "./BriefingTab";
import { CritiqueTab } from "./CritiqueTab";
import { PaperHeader } from "./PaperHeader";
import { FailedState, RunningState, UploadPdfButton } from "./PaperStatus";
import { VisualsTab } from "./VisualsTab";

function PipelineNotes({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) return null;
  return (
    <details className="group rounded-xl border border-line bg-panel px-4 py-2.5 text-xs text-ink-3">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 select-none hover:text-ink-2">
        <ChevronRight className="size-3.5 transition-transform group-open:rotate-90" />
        <Info className="size-3.5" />
        {warnings.length} note{warnings.length === 1 ? "" : "s"} from the pipeline
      </summary>
      <ul className="mt-2 ml-5 list-disc space-y-1 pl-4 text-ink-2">
        {warnings.map((w) => (
          <li key={w}>{w}</li>
        ))}
      </ul>
    </details>
  );
}

function AbstractOnlyNotice() {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-warn/30 bg-warn-soft px-4 py-3 sm:flex-row sm:items-center">
      <FileText className="size-4 shrink-0 text-warn" />
      <p className="flex-1 text-xs leading-relaxed text-ink-2">
        The full text could not be read, so this analysis is based on the abstract only. Upload the PDF for the
        complete review, charts and grounded Q&A.
      </p>
      <UploadPdfButton variant="secondary" />
    </div>
  );
}

function ReadyContent({
  record,
  tab,
  onTab,
  onAsk,
  pendingQuestion,
}: {
  record: PaperRecord;
  tab: PaperTab;
  onTab: (tab: PaperTab) => void;
  onAsk: (question: string) => void;
  pendingQuestion: string | null;
}) {
  const analysis = record.analysis;
  if (!analysis) {
    return <FailedState record={{ ...record, error: record.error ?? "The analysis finished without results." }} />;
  }
  const concerns = analysis.critique.findings.filter((f) => f.severity === "major" || f.severity === "moderate").length;

  return (
    <div className="space-y-5">
      {analysis.mode === "abstract-only" && <AbstractOnlyNotice />}
      <Tabs<PaperTab>
        value={tab}
        onChange={onTab}
        items={[
          { value: "briefing", label: "Briefing", icon: <Sparkles className="size-4" /> },
          { value: "visuals", label: "Visual analysis", icon: <ChartColumn className="size-4" /> },
          { value: "critique", label: "Critique", icon: <Gavel className="size-4" />, count: concerns },
          {
            value: "ask",
            label: "Ask",
            icon: <MessageSquareText className="size-4" />,
            count: record.qa.length || undefined,
          },
        ]}
        className="sticky top-0 z-20 -mx-1 bg-bg/90 px-1 backdrop-blur"
      />
      <div role="tabpanel">
        {tab === "briefing" && <BriefingTab briefing={analysis.briefing} onAsk={onAsk} />}
        {tab === "visuals" && <VisualsTab analysis={analysis} meta={record.meta} sentences={record.sentences} />}
        {tab === "critique" && <CritiqueTab critique={analysis.critique} rigor={analysis.rigor} />}
        {tab === "ask" && (
          <AskTab
            answers={record.qa}
            pending={record.qaPending}
            pendingQuestion={pendingQuestion}
            suggestions={analysis.briefing.followUps}
            onAsk={onAsk}
          />
        )}
      </div>
    </div>
  );
}

export default function PaperView({ id }: { id: string }) {
  const record = useLibrary((s) => s.papers[id]);
  const view = useLibrary((s) => s.view);
  const setView = useLibrary((s) => s.setView);
  const ask = useLibrary((s) => s.ask);
  const [pending, setPending] = useState<{ id: string; question: string } | null>(null);

  if (!record) {
    return (
      <div className="mx-auto w-full max-w-6xl px-4 py-16 sm:px-8">
        <Card>
          <EmptyState
            icon={<FileQuestion className="size-5" />}
            title="Paper not found"
            body="It may have been removed from your library."
            action={
              <Button size="sm" onClick={() => setView({ kind: "home" })}>
                Back to home
              </Button>
            }
          />
        </Card>
      </div>
    );
  }

  const tab: PaperTab = view.kind === "paper" && view.id === id ? view.tab : "briefing";
  const setTab = (next: PaperTab) => setView({ kind: "paper", id, tab: next });
  const askQuestion = (question: string) => {
    if (record.qaPending) return;
    setTab("ask");
    setPending({ id, question });
    void ask(id, question).finally(() =>
      setPending((cur) => (cur && cur.id === id && cur.question === question ? null : cur)),
    );
  };

  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 px-4 py-6 sm:px-8 sm:py-8">
      <PaperHeader record={record} />
      <PipelineNotes warnings={record.warnings} />
      {(record.status === "queued" || record.status === "running") && <RunningState record={record} />}
      {record.status === "failed" && <FailedState record={record} />}
      {record.status === "ready" && (
        <ReadyContent
          record={record}
          tab={tab}
          onTab={setTab}
          onAsk={askQuestion}
          pendingQuestion={pending && pending.id === id ? pending.question : null}
        />
      )}
    </div>
  );
}

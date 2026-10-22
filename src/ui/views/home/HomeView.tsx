/**
 * Landing view: what PaperTrail does, one input, and one-click samples.
 */
import {
  ArrowRight,
  BookOpen,
  FileSearch,
  Layers,
  Lock,
  MessageSquareQuote,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import type { ReactNode } from "react";
import { MODELS } from "@/ml/models";
import { PAPER_NODES } from "@/pipeline/paper";
import type { PaperRecord } from "@/store/types";
import { ScoreRing, cn } from "@/ui/kit";
import { store, useStore } from "@/ui/hooks/store";
import { submitForSurvey, usePendingSurvey } from "@/ui/hooks/usePendingSurvey";
import { CommandBar } from "@/ui/layout/CommandBar";
import { DOI_SAMPLE, PAPER_SAMPLES, SURVEY_SAMPLE, TOPIC_SAMPLE } from "./samples";

const MODEL_MB = Object.values(MODELS).reduce((sum, m) => sum + m.sizeMB, 0);

const CAPABILITIES: { icon: ReactNode; title: string; body: string; points: string[] }[] = [
  {
    icon: <BookOpen className="size-4" />,
    title: "Reviewer-grade briefing",
    body: "Why it matters, problem, method, results and limitations, each line linked to the page it came from.",
    points: ["LexRank + MMR summaries", "Argumentative zoning", "Key numbers extracted"],
  },
  {
    icon: <ShieldCheck className="size-4" />,
    title: "Critique & rigor score",
    body: "Explainable checks for baselines, ablations, statistics, reproducibility and overclaiming.",
    points: ["7-axis rigor profile", "Severity-ranked findings", "Reviewer questions"],
  },
  {
    icon: <MessageSquareQuote className="size-4" />,
    title: "Grounded Q&A",
    body: "Ask anything. Answers are spans copied from the paper, and it abstains when evidence is weak.",
    points: ["Hybrid BM25 + dense retrieval", "Cross-encoder rerank", "Extractive reader"],
  },
  {
    icon: <Layers className="size-4" />,
    title: "Literature survey",
    body: "Up to 10 papers: themes, semantic map, comparison matrix, consensus and open gaps.",
    points: ["Agglomerative clustering", "UMAP projection", "c-TF-IDF theme labels"],
  },
];

function Chip({
  onClick,
  children,
  className,
  title,
}: {
  onClick: () => void;
  children: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border border-line bg-panel px-3 py-1.5 text-xs text-ink-2 transition-all",
        "hover:-translate-y-px hover:border-line-strong hover:text-ink",
        className,
      )}
    >
      {children}
    </button>
  );
}

function Samples() {
  const pending = usePendingSurvey((s) => s.arxivIds);
  const submit = (input: string) => void store().submit(input);

  return (
    <div className="mt-5 flex flex-col items-center gap-2.5">
      <div className="flex flex-wrap items-center justify-center gap-2">
        <span className="text-[11px] font-medium uppercase tracking-wider text-ink-3">Try</span>
        {PAPER_SAMPLES.map((s) => (
          <Chip key={s.input} onClick={() => submit(s.input)} title={`${s.hint} · arXiv:${s.input}`}>
            <span className="font-medium">{s.label}</span>
            <span className="font-mono text-[10px] text-ink-3">{s.input}</span>
          </Chip>
        ))}
      </div>
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Chip onClick={() => submit(TOPIC_SAMPLE)}>
          <Search className="size-3.5 text-ink-3" />
          {TOPIC_SAMPLE}
        </Chip>
        <Chip onClick={() => submit(DOI_SAMPLE.input)} title={DOI_SAMPLE.input}>
          <span className="font-medium">{DOI_SAMPLE.label}</span>
          <span className="font-mono text-[10px] text-ink-3">{DOI_SAMPLE.input}</span>
        </Chip>
        <Chip
          onClick={() => void submitForSurvey(SURVEY_SAMPLE)}
          className="border-accent/40 bg-accent-soft text-accent hover:border-accent hover:text-accent"
          title={SURVEY_SAMPLE.map((id) => `arXiv:${id}`).join(", ")}
        >
          <Layers className="size-3.5" />
          {pending ? "Survey sample running…" : "Survey sample: 5 LLM papers"}
        </Chip>
      </div>
    </div>
  );
}

function RecentPapers() {
  const papers = useStore((s) => s.papers);
  const order = useStore((s) => s.order);
  const recent = order
    .map((id) => papers[id])
    .filter((r): r is PaperRecord => r?.status === "ready")
    .slice(0, 3);
  if (recent.length === 0) return null;

  return (
    <section className="mx-auto mt-14 max-w-5xl">
      <h2 className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-ink-3">Continue</h2>
      <div className="grid gap-3 md:grid-cols-3">
        {recent.map((r) => (
          <button
            key={r.id}
            onClick={() => store().setView({ kind: "paper", id: r.id, tab: "briefing" })}
            className="group flex items-start gap-3 rounded-2xl border border-line bg-panel p-4 text-left transition-all hover:-translate-y-0.5 hover:border-line-strong"
          >
            {r.analysis && <ScoreRing value={r.analysis.rigor.overall} size={44} stroke={4} />}
            <span className="min-w-0 flex-1">
              <span className="line-clamp-2 text-[13px] font-medium leading-snug text-ink">{r.meta.title}</span>
              <span className="mt-1 block truncate text-[11px] text-ink-3">
                {r.meta.authors[0] ?? "Unknown authors"}
                {r.meta.year ? ` · ${r.meta.year}` : ""}
              </span>
            </span>
            <ArrowRight className="mt-0.5 size-4 shrink-0 text-ink-3 transition-transform group-hover:translate-x-0.5" />
          </button>
        ))}
      </div>
    </section>
  );
}

function Pipeline() {
  const steps = PAPER_NODES.filter((n) => n.name !== "fallback");
  const fallback = PAPER_NODES.find((n) => n.name === "fallback");
  return (
    <section className="mx-auto mt-16 max-w-5xl">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-ink">How it works</h2>
          <p className="mt-1 text-sm text-ink-3">
            An explicit state graph runs each paper. Watch every step live in the trace dock.
          </p>
        </div>
        {fallback && (
          <p className="max-w-xs text-[11px] leading-relaxed text-ink-3">
            <span className="font-medium text-ink-2">{fallback.label}:</span> {fallback.description}.
          </p>
        )}
      </div>
      <ol className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-9">
        {steps.map((step, i) => (
          <li
            key={step.name}
            title={step.description}
            className="group relative rounded-xl border border-line bg-panel px-3 py-3 transition-colors hover:border-line-strong"
          >
            <span className="font-mono text-[10px] tabular text-ink-3">{String(i + 1).padStart(2, "0")}</span>
            <p className="mt-1 text-[13px] font-medium text-ink">{step.label}</p>
            <p className="mt-1 line-clamp-3 text-[11px] leading-snug text-ink-3">{step.description}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

export default function HomeView() {
  return (
    <div className="relative px-4 pt-14 pb-16 sm:px-8 sm:pt-20">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-[420px]"
        style={{
          background:
            "radial-gradient(60% 55% at 50% 0%, color-mix(in oklab, var(--accent) 16%, transparent), transparent 70%)",
        }}
      />
      <div aria-hidden className="dot-grid pointer-events-none absolute inset-x-0 top-0 h-[420px] opacity-40 [mask-image:linear-gradient(to_bottom,black,transparent)]" />

      <section className="relative mx-auto max-w-3xl text-center">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-panel/80 px-3 py-1 text-[11px] font-medium text-ink-2 backdrop-blur">
          <Sparkles className="size-3.5 text-accent" />
          No server, no API keys: {Object.keys(MODELS).length} small models run on your device
        </span>
        <h1 className="mt-5 text-4xl font-semibold tracking-[-0.035em] text-balance text-ink sm:text-5xl">
          Reviewer-grade paper analysis,{" "}
          <span
            className="bg-clip-text text-transparent"
            style={{ backgroundImage: "linear-gradient(100deg, var(--accent), var(--c-4) 60%, var(--c-3))" }}
          >
            entirely in your browser.
          </span>
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-[15px] leading-relaxed text-pretty text-ink-2">
          Paste an arXiv ID, DOI or PDF link, or search a topic. PaperTrail reads the full text, maps its
          argument, scores its rigor and answers questions with verbatim, page-linked evidence.
        </p>
        <div className="mt-8">
          <CommandBar variant="hero" />
        </div>
        <Samples />
      </section>

      <RecentPapers />

      <section className="mx-auto mt-14 grid max-w-5xl gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {CAPABILITIES.map((c) => (
          <article key={c.title} className="rounded-2xl border border-line bg-panel p-5 transition-colors hover:border-line-strong">
            <div className="flex size-8 items-center justify-center rounded-lg bg-accent-soft text-accent">{c.icon}</div>
            <h3 className="mt-4 text-sm font-semibold tracking-tight text-ink">{c.title}</h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-3">{c.body}</p>
            <ul className="mt-3 space-y-1">
              {c.points.map((p) => (
                <li key={p} className="flex items-center gap-2 text-[11px] text-ink-2">
                  <span className="size-1 rounded-full bg-accent" />
                  {p}
                </li>
              ))}
            </ul>
          </article>
        ))}
      </section>

      <Pipeline />

      <footer className="mx-auto mt-14 flex max-w-5xl flex-col items-center justify-between gap-3 border-t border-line pt-6 text-xs text-ink-3 sm:flex-row">
        <span className="inline-flex items-center gap-2">
          <Lock className="size-3.5" />
          Your papers never leave this device. PDFs and analyses are stored in your browser.
        </span>
        <span className="inline-flex items-center gap-2">
          <FileSearch className="size-3.5" />
          {MODEL_MB} MB of int8 ONNX models, cached after first use
        </span>
      </footer>
    </div>
  );
}

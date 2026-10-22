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
} from "lucide-react";
import { useEffect, useRef, type PointerEvent, type ReactNode } from "react";
import { MODELS } from "@/ml/models";
import { PAPER_NODES } from "@/pipeline/paper";
import type { PaperRecord } from "@/store/types";
import { ScoreRing, cn } from "@/ui/kit";
import { store, useStore } from "@/ui/hooks/store";
import { submitForSurvey, usePendingSurvey } from "@/ui/hooks/usePendingSurvey";
import { CommandBar } from "@/ui/layout/CommandBar";
import { DOI_SAMPLE, PAPER_SAMPLES, SURVEY_SAMPLE, TOPIC_SAMPLE } from "./samples";
import { Starfield } from "./Starfield";

const MODEL_MB = Object.values(MODELS).reduce((sum, m) => sum + m.sizeMB, 0);

const CAPABILITIES: { icon: ReactNode; title: string; body: string; points: string[] }[] = [
  {
    icon: <BookOpen className="size-4" />,
    title: "Reviewer-grade briefing",
    body: "Why it matters, problem, method, results and limitations, each line linked to the page it came from.",
    points: ["The most central sentences", "Every sentence tagged by role", "Key numbers pulled out"],
  },
  {
    icon: <ShieldCheck className="size-4" />,
    title: "Critique & rigor score",
    body: "Checks for baselines, ablations, statistics, reproducibility and overclaiming, each one explained.",
    points: ["Rigor score across 7 areas", "Findings ranked by severity", "Questions for the authors"],
  },
  {
    icon: <MessageSquareQuote className="size-4" />,
    title: "Grounded Q&A",
    body: "Ask anything. Answers are spans copied from the paper, and it abstains when evidence is weak.",
    points: ["Keyword and meaning search", "Best passages re-ranked", "Answer quoted from the text"],
  },
  {
    icon: <Layers className="size-4" />,
    title: "Literature survey",
    body: "Up to 10 papers: themes, semantic map, comparison matrix, consensus and open gaps.",
    points: ["Papers grouped into themes", "A map of how they relate", "Side-by-side comparison"],
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

const PLAIN_STEPS: Record<string, string> = {
  understand: "Works out what you pasted: arXiv, DOI, PDF link or file",
  resolve: "Finds the title, authors, abstract and the PDF",
  fetch: "Downloads the PDF and checks it is real",
  parse: "Rebuilds sections, references and captions",
  fallback: "No readable full text? It carries on from the abstract",
  segment: "Splits the text into sentences, keeping page numbers",
  zone: "Tags each sentence: problem, method, result, limitation",
  chunk: "Groups sentences into short passages",
  embed: "Indexes passages so you can search by meaning",
  analyze: "Builds the briefing, critique and charts",
};

/** Moves a soft spotlight under the pointer across a group of cards. */
function spotlight(e: PointerEvent<HTMLElement>) {
  const card = (e.target as HTMLElement).closest<HTMLElement>("[data-spot]");
  if (!card) return;
  const rect = card.getBoundingClientRect();
  card.style.setProperty("--mx", `${e.clientX - rect.left}px`);
  card.style.setProperty("--my", `${e.clientY - rect.top}px`);
}

/** Adds `is-visible` to `.reveal` children as they scroll into view. */
function useReveal() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const items = Array.from(ref.current?.querySelectorAll<HTMLElement>(".reveal") ?? []);
    if (typeof IntersectionObserver === "undefined") {
      items.forEach((el) => el.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      },
      { rootMargin: "0px 0px -8% 0px" },
    );
    items.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);
  return ref;
}

function Pipeline() {
  const steps = PAPER_NODES.filter((n) => n.name !== "fallback");
  const fallback = PAPER_NODES.find((n) => n.name === "fallback");
  return (
    <section className="reveal mx-auto mt-20 max-w-5xl">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-ink">How it works</h2>
          <p className="mt-1 text-sm text-ink-3">
            Every paper goes through the same steps. You can watch each one live in the trace panel.
          </p>
        </div>
        {fallback && (
          <p className="max-w-xs text-[11px] leading-relaxed text-ink-3">
            <span className="font-medium text-ink-2">{fallback.label}:</span> {PLAIN_STEPS.fallback}.
          </p>
        )}
      </div>
      <ol onPointerMove={spotlight} className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-9">
        {steps.map((step, i) => (
          <li
            key={step.name}
            data-spot
            style={{ animationDelay: `${i * 60}ms` }}
            className="spot group relative rounded-xl border border-line bg-panel px-3 py-3 shadow-card transition-all duration-300 hover:-translate-y-0.5 hover:border-accent/50"
          >
            <span className="inline-flex size-5 items-center justify-center rounded-md bg-accent-soft font-mono text-[10px] tabular text-accent">
              {i + 1}
            </span>
            <p className="mt-2 text-[13px] font-medium text-ink">{step.label}</p>
            <p className="mt-1 line-clamp-3 text-[11px] leading-snug text-ink-3">{PLAIN_STEPS[step.name] ?? step.description}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

export default function HomeView() {
  const ref = useReveal();
  return (
    <div ref={ref} className="relative px-4 pt-14 pb-16 sm:px-8 sm:pt-20">
      <Starfield className="pointer-events-none absolute inset-x-0 top-0 h-[620px] w-full [mask-image:linear-gradient(to_bottom,black_55%,transparent)]" />

      <section className="relative mx-auto max-w-3xl text-center">
        <span className="animate-in inline-flex items-center gap-2 rounded-full border border-line bg-panel/80 px-3 py-1 text-[11px] font-medium text-ink-2 shadow-card backdrop-blur">
          <span className="relative flex size-2">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-good opacity-60" />
            <span className="relative inline-flex size-2 rounded-full bg-good" />
          </span>
          No server, no API keys: {Object.keys(MODELS).length} small models run on your device
        </span>
        <h1 className="animate-in mt-5 text-4xl font-semibold tracking-[-0.035em] text-balance text-ink sm:text-[52px] sm:leading-[1.08]">
          Reviewer-grade paper analysis, <span className="text-accent">entirely in your browser.</span>
        </h1>
        <p className="animate-in mx-auto mt-4 max-w-2xl text-[15px] leading-relaxed text-pretty text-ink-2">
          Paste an arXiv ID, DOI or PDF link, or search a topic. PaperTrail reads the full text, maps its
          argument, scores its rigor and answers questions with verbatim, page-linked evidence.
        </p>
        <div className="animate-in mt-8">
          <CommandBar variant="hero" />
        </div>
        <Samples />
      </section>

      <RecentPapers />

      <section onPointerMove={spotlight} className="reveal mx-auto mt-16 grid max-w-5xl gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {CAPABILITIES.map((c) => (
          <article
            key={c.title}
            data-spot
            className="spot group rounded-2xl border border-line bg-panel p-5 shadow-card transition-all duration-300 hover:-translate-y-1 hover:border-accent/50"
          >
            <div className="flex size-9 items-center justify-center rounded-xl border border-accent/25 bg-accent-soft text-accent transition-transform duration-300 group-hover:scale-110">
              {c.icon}
            </div>
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

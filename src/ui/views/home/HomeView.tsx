/**
 * Landing view: what PaperTrail does, one input, and one-click samples.
 */
import { ArrowRight, BookOpen, Layers, Lock, MessageSquareQuote, Search, ShieldCheck } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";
import type { PaperRecord } from "@/store/types";
import { ScoreRing, cn } from "@/ui/kit";
import { store, useStore } from "@/ui/hooks/store";
import { submitForSurvey, usePendingSurvey } from "@/ui/hooks/usePendingSurvey";
import { CommandBar } from "@/ui/layout/CommandBar";
import { DOI_SAMPLE, PAPER_SAMPLES, SURVEY_SAMPLE, TOPIC_SAMPLE } from "./samples";
import { Starfield } from "./Starfield";

const CAPABILITIES: { icon: ReactNode; title: string; body: string }[] = [
  {
    icon: <BookOpen className="size-4" />,
    title: "Briefing",
    body: "The problem, method, results and limits in plain sentences, each linked to its page.",
  },
  {
    icon: <ShieldCheck className="size-4" />,
    title: "Critique",
    body: "What is strong, what is missing, and the questions a reviewer would ask.",
  },
  {
    icon: <MessageSquareQuote className="size-4" />,
    title: "Ask",
    body: "Answers quoted straight from the paper, so you can check every word.",
  },
  {
    icon: <Layers className="size-4" />,
    title: "Survey",
    body: "Compare up to 10 papers: shared themes, open gaps and a side-by-side table.",
  },
];

const STEPS: { title: string; body: string }[] = [
  { title: "Find the paper", body: "arXiv ID or link, DOI, PDF link, upload, or a topic search." },
  { title: "Read the full text", body: "Sections, figures, tables and references, straight from the PDF." },
  { title: "Analyse", body: "Briefing, critique and charts, every point traced to its source." },
  { title: "Ask and compare", body: "Question one paper, or survey several side by side." },
];

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
        "inline-flex items-center gap-1.5 rounded-full border border-line bg-panel/70 px-3 py-1.5 text-xs text-ink-2 backdrop-blur transition-colors",
        "hover:border-line-strong hover:bg-panel-2 hover:text-ink",
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
          className="border-line-strong text-ink hover:border-ink-3"
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
            className="group flex items-start gap-3 rounded-xl border border-line bg-panel p-4 text-left transition-colors hover:border-line-strong hover:bg-panel-2"
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

function HowItWorks() {
  return (
    <section className="reveal mx-auto mt-24 max-w-5xl">
      <h2 className="text-2xl font-semibold tracking-[-0.025em] text-ink">How it works</h2>
      <p className="mt-2 max-w-xl text-[15px] text-ink-3">
        Four steps, all on your device. You can watch each one happen in the trace panel.
      </p>
      <ol className="mt-8 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step, i) => (
          <li key={step.title} className="bg-bg p-5">
            <span className="font-mono text-[11px] tabular text-ink-3">{String(i + 1).padStart(2, "0")}</span>
            <p className="mt-3 text-sm font-medium text-ink">{step.title}</p>
            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-3">{step.body}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

export default function HomeView() {
  const ref = useReveal();
  return (
    <div ref={ref} className="relative px-4 pt-16 pb-16 sm:px-8 sm:pt-24">
      <Starfield className="pointer-events-none absolute inset-x-0 top-0 h-[640px] w-full [mask-image:linear-gradient(to_bottom,black_55%,transparent)]" />

      <section className="relative mx-auto max-w-4xl text-center">
        <span className="animate-in inline-flex items-center gap-2 rounded-full border border-line bg-panel/70 px-3 py-1 text-[12px] text-ink-2 backdrop-blur">
          <span className="size-1.5 rounded-full bg-good" />
          Runs entirely in your browser
        </span>
        <h1 className="animate-in mt-6 text-[38px] leading-[1.08] font-semibold tracking-[-0.04em] text-balance text-ink sm:text-[56px]">
          Understand any research paper.
          <span className="block text-ink-3">Every claim linked to its page.</span>
        </h1>
        <p className="animate-in mx-auto mt-5 max-w-xl text-[16px] leading-relaxed text-pretty text-ink-2">
          Paste an arXiv link, DOI or PDF, or search a topic. PaperTrail reads the whole paper and gives you a
          briefing, a critique and answers you can check.
        </p>
        <div className="animate-in mt-9">
          <CommandBar variant="hero" />
        </div>
        <Samples />
      </section>

      <RecentPapers />

      <section className="reveal relative mx-auto mt-24 grid max-w-5xl gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-2 lg:grid-cols-4">
        {CAPABILITIES.map((c) => (
          <article key={c.title} className="bg-bg p-6 transition-colors hover:bg-panel">
            <div className="text-ink-3">{c.icon}</div>
            <h3 className="mt-5 text-[15px] font-medium text-ink">{c.title}</h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-3">{c.body}</p>
          </article>
        ))}
      </section>

      <HowItWorks />

      <footer className="mx-auto mt-24 flex max-w-5xl flex-col items-center justify-between gap-3 border-t border-line pt-6 text-xs text-ink-3 sm:flex-row">
        <span className="inline-flex items-center gap-2">
          <Lock className="size-3.5" />
          Private by design: papers and analyses stay in your browser.
        </span>
        <span>No account, no server, no API keys.</span>
      </footer>
    </div>
  );
}

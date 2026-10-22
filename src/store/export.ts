/**
 * Markdown exports: a paper briefing and a literature survey, written so they
 * read well on their own (in a notes app, a PR, or a reading group doc).
 * Every claim keeps its page reference, so readers can check the source.
 */
import type { BriefingItem, CritiqueFinding, Evidence, PaperMeta, Severity } from "@/types";
import type { PaperRecord, SurveyRecord } from "./types";

const SEVERITY_ORDER: Severity[] = ["major", "moderate", "minor", "strength"];
const SEVERITY_LABEL: Record<Severity, string> = {
  major: "Major concerns",
  moderate: "Moderate concerns",
  minor: "Minor issues",
  strength: "Strengths",
};

/** Keeps table cells and inline text on one line and safe for Markdown. */
function inline(text: string): string {
  return text.replace(/\s+/g, " ").replace(/\|/g, "\\|").trim();
}

function pageRefs(evidence: Evidence[]): string {
  const pages = [...new Set(evidence.map((e) => e.page).filter((p) => p > 0))].sort((a, b) => a - b);
  if (pages.length === 0) return "";
  return ` _(p. ${pages.join(", ")})_`;
}

function bulletList(items: BriefingItem[], empty: string): string {
  if (items.length === 0) return `_${empty}_`;
  return items.map((item) => `- ${inline(item.text)}${pageRefs(item.evidence)}`).join("\n");
}

function quote(evidence: Evidence): string {
  const where = [evidence.section, evidence.page > 0 ? `p. ${evidence.page}` : ""].filter(Boolean).join(", ");
  return `  > ${inline(evidence.quote)}${where ? ` — ${where}` : ""}`;
}

export function paperLinks(meta: PaperMeta): string[] {
  const links: string[] = [];
  if (meta.arxivId) links.push(`[arXiv:${meta.arxivId}](https://arxiv.org/abs/${meta.arxivId})`);
  if (meta.doi && !/^10\.48550\//i.test(meta.doi)) links.push(`[doi:${meta.doi}](https://doi.org/${meta.doi})`);
  if (meta.pdfUrl && !meta.arxivId) links.push(`[PDF](${meta.pdfUrl})`);
  return links;
}

function authorLine(authors: string[]): string {
  if (authors.length === 0) return "Authors unknown";
  return authors.length > 8 ? `${authors.slice(0, 8).join(", ")} et al.` : authors.join(", ");
}

function findingsSection(findings: CritiqueFinding[]): string {
  const blocks: string[] = [];
  for (const severity of SEVERITY_ORDER) {
    const group = findings.filter((f) => f.severity === severity);
    if (group.length === 0) continue;
    const items = group.map((f) => {
      const lines = [`- **${inline(f.title)}** (${f.category}) — ${inline(f.rationale)}`];
      lines.push(...f.evidence.slice(0, 2).map(quote));
      return lines.join("\n");
    });
    blocks.push(`### ${SEVERITY_LABEL[severity]}\n\n${items.join("\n")}`);
  }
  return blocks.join("\n\n");
}

function header(record: PaperRecord): string {
  const { meta } = record;
  const facts = [authorLine(meta.authors), meta.year ? String(meta.year) : "", meta.venue ?? ""].filter(Boolean);
  const lines = [`# ${inline(meta.title)}`, "", facts.join(" · ")];
  const links = paperLinks(meta);
  if (links.length) lines.push("", links.join(" · "));
  if (record.mode === "abstract-only") {
    lines.push("", "> **Abstract-only analysis.** The full text could not be read, so findings rely on the abstract.");
  }
  return lines.join("\n");
}

/** Full briefing for one analysed paper. */
export function briefingMarkdown(record: PaperRecord): string {
  const parts = [header(record)];
  const analysis = record.analysis;
  if (!analysis) {
    parts.push("", record.error ? `_Analysis failed: ${inline(record.error)}_` : "_Analysis not available yet._");
    return parts.join("\n") + "\n";
  }
  const { briefing, critique, rigor, stats } = analysis;

  parts.push("", "## TL;DR", "", `${inline(briefing.tldr.text)}${pageRefs(briefing.tldr.evidence)}`);
  parts.push("", "## Problem", "", bulletList(briefing.problem, "No problem statement was identified."));
  parts.push("", "## Method", "", bulletList(briefing.method, "No method description was identified."));
  parts.push("", "## Results", "", bulletList(briefing.results, "No result statements were identified."));

  if (briefing.keyNumbers.length > 0) {
    parts.push("", "## Key numbers", "", "| Value | Measures | Page |", "| --- | --- | --- |");
    for (const k of briefing.keyNumbers) {
      parts.push(`| ${inline(k.value)} | ${inline(k.label)} | ${k.evidence.page > 0 ? k.evidence.page : "–"} |`);
    }
  }

  parts.push(
    "",
    "## Limitations",
    "",
    "### Stated by the authors",
    "",
    bulletList(briefing.limitations.authorStated, "The authors do not discuss limitations explicitly."),
    "",
    "### Detected by PaperTrail",
    "",
    bulletList(briefing.limitations.detected, "No additional limitations were detected."),
  );

  parts.push("", "## Critique", "", inline(critique.verdict), "", findingsSection(critique.findings));

  parts.push(
    "",
    "## Rigor profile",
    "",
    `Overall **${Math.round(rigor.overall)}/100**, grade **${rigor.grade}**.`,
    "",
    "| Axis | Score | Weight | Why |",
    "| --- | --- | --- | --- |",
    ...rigor.axes.map(
      (a) => `| ${a.label} | ${Math.round(a.score)} | ${Math.round(a.weight * 100)}% | ${inline(a.explanation)} |`,
    ),
  );

  if (critique.reviewerQuestions.length > 0) {
    parts.push("", "## Questions a reviewer would ask", "", ...critique.reviewerQuestions.map((q, i) => `${i + 1}. ${inline(q)}`));
  }
  if (briefing.followUps.length > 0) {
    parts.push("", "## Follow-up questions", "", ...briefing.followUps.map((q) => `- ${inline(q)}`));
  }
  if (briefing.keyphrases.length > 0) {
    parts.push("", "## Keyphrases", "", briefing.keyphrases.map((k) => `\`${k.phrase}\``).join(" "));
  }

  const answered = record.qa.filter((a) => a.status === "answered" && a.answer);
  if (answered.length > 0) {
    parts.push("", "## Questions & answers");
    for (const a of answered) {
      parts.push("", `**Q: ${inline(a.question)}**`, "", `${inline(a.answer!)}${pageRefs(a.evidence)}`);
      if (a.context && a.context !== a.answer) parts.push("", `> ${inline(a.context)}`);
    }
  }

  parts.push(
    "",
    "---",
    "",
    `_${stats.pages} pages · ${stats.words.toLocaleString("en-US")} words · ${stats.references} references · ` +
      `analysed in ${(analysis.computedMs / 1000).toFixed(1)} s by PaperTrail, entirely in the browser._`,
  );
  return parts.join("\n") + "\n";
}

/** Literature review plus the structured comparison behind it. */
export function surveyMarkdown(record: SurveyRecord, papers: Record<string, PaperRecord>): string {
  const result = record.result;
  const titles = record.paperIds.map((id) => papers[id]?.meta.title ?? id);
  if (!result) {
    return `# Literature survey\n\n${titles.map((t) => `- ${inline(t)}`).join("\n")}\n\n_${
      record.error ? `Survey failed: ${inline(record.error)}` : "Survey not available yet."
    }_\n`;
  }

  const parts = [result.narrative.trim()];
  const number = new Map(result.paperIds.map((id, i) => [id, i + 1]));
  const label = (id: string) => `[${number.get(id) ?? "?"}]`;

  if (result.comparison.length > 0) {
    parts.push(
      "",
      "## Appendix A: comparison matrix",
      "",
      "| Paper | Year | Problem | Approach | Datasets | Key result | Rigor | Code |",
      "| --- | --- | --- | --- | --- | --- | --- | --- |",
      ...result.comparison.map(
        (row) =>
          `| ${label(row.paperId)} ${inline(papers[row.paperId]?.meta.title ?? row.paperId)} | ${row.year ?? "–"} | ` +
          `${inline(row.problem)} | ${inline(row.approach.join(", "))} | ${inline(row.datasets.join(", ") || "–")} | ` +
          `${inline(row.keyResult)} | ${Math.round(row.rigor)} | ${row.code ? "yes" : "no"} |`,
      ),
    );
  }

  if (result.clusters.length > 0) {
    parts.push(
      "",
      "## Appendix B: themes",
      "",
      ...result.clusters.map((c) => `- **${inline(c.label)}**: ${c.paperIds.map(label).join(", ")}`),
      "",
      `_Themes from agglomerative clustering of paper embeddings (silhouette ${result.silhouette.toFixed(2)}), ` +
        `labelled with class-based TF-IDF; semantic map by ${result.map.method.toUpperCase()}._`,
    );
  }

  parts.push(
    "",
    "---",
    "",
    `_Survey of ${result.paperIds.length} papers generated by PaperTrail on ${new Date(result.createdAt)
      .toISOString()
      .slice(0, 10)}, entirely in the browser._`,
  );
  return parts.join("\n") + "\n";
}

/** Saves text as a file through a temporary object URL. */
export function download(filename: string, text: string): void {
  const type = filename.endsWith(".json")
    ? "application/json"
    : filename.endsWith(".md")
      ? "text/markdown"
      : "text/plain";
  const url = URL.createObjectURL(new Blob([text], { type: `${type};charset=utf-8` }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Revoke after the click has been handled, or some browsers cancel the download.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

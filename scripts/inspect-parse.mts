/**
 * Dev tool: runs the real parsing pipeline on PDFs and prints what it found.
 *
 *   node scripts/inspect-parse.mts [file.pdf | dir ...]        summary per paper (default: tests/fixtures/pdfs)
 *   node scripts/inspect-parse.mts --lines file.pdf 1,2        raw layout lines of pages 1-2
 *   node scripts/inspect-parse.mts --section file.pdf 3        full text of section #3
 *
 * Source modules are loaded through Vite's SSR module runner, so the `@/`
 * alias and extensionless imports resolve exactly as in the app.
 */
import { createServer, createServerModuleRunner } from "vite";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import type { ParsedDocument, TextLine } from "../src/types";

const DEFAULT_DIR = "tests/fixtures/pdfs";

interface ParsingModule {
  parsePdf(bytes: ArrayBuffer): Promise<ParsedDocument>;
  extractPdfLines(bytes: ArrayBuffer, opts?: { maxPages?: number }): Promise<{ lines: TextLine[] }>;
}

function readBytes(path: string): ArrayBuffer {
  const buf = readFileSync(path);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function pdfPaths(args: string[]): string[] {
  const targets = args.length ? args : [DEFAULT_DIR];
  return targets.flatMap((target) =>
    statSync(target).isDirectory()
      ? readdirSync(target)
          .filter((f) => f.endsWith(".pdf"))
          .sort()
          .map((f) => join(target, f))
      : [target],
  );
}

function printSummary(path: string, doc: ParsedDocument, ms: number): void {
  console.log(`\n━━ ${path}  (${doc.pagesParsed}/${doc.pageCount} pages, ${ms} ms)`);
  console.log(`title    : ${doc.title ?? "—"}`);
  console.log(`abstract : ${(doc.abstract ?? "—").slice(0, 200).replace(/\s+/g, " ")}`);
  console.log(`sections : ${doc.sections.length}`);
  for (const s of doc.sections) {
    const indent = "  ".repeat(s.level);
    const pages = s.pageStart === s.pageEnd ? `p${s.pageStart}` : `p${s.pageStart}-${s.pageEnd}`;
    console.log(`${indent}${s.number ? `${s.number} ` : ""}${s.title}  [${s.kind}, ${pages}, ${s.text.length} chars]`);
  }
  console.log(`references: ${doc.references.length}`);
  const step = Math.max(1, Math.floor(doc.references.length / 3));
  for (const r of [0, step, 2 * step].map((i) => doc.references[i]).filter(Boolean)) {
    console.log(`  - ${JSON.stringify({ ...r, raw: r.raw.slice(0, 90) })}`);
  }
  console.log(`captions : ${doc.captions.length} (${doc.captions.slice(0, 4).map((c) => c.label).join(", ")}…)`);
  console.log(`links    : ${doc.links.join("  ")}`);
  console.log(`quality  : ${doc.quality}`);
  console.log(`warnings : ${doc.warnings.join(" | ") || "—"}`);
}

const server = await createServer({
  root: process.cwd(),
  server: { middlewareMode: true, hmr: false, ws: false },
  appType: "custom",
  logLevel: "error",
});
const runner = createServerModuleRunner(server.environments.ssr, { hmr: false });
try {
  const parsing = (await runner.import("/src/parsing/index.ts")) as ParsingModule;
  const [mode, ...rest] = process.argv.slice(2);
  if (mode === "--lines") {
    const [file, pageList = "1"] = rest;
    const pages = pageList.split(",").map(Number);
    const { lines } = await parsing.extractPdfLines(readBytes(file), { maxPages: Math.max(...pages) });
    for (const l of lines.filter((line) => pages.includes(line.page))) {
      const style = `s${l.fontSize.toFixed(1)}${l.bold ? "B" : " "}`;
      console.log(`${l.page} c${l.column} y${l.y.toFixed(0).padStart(4)} x${l.x.toFixed(0).padStart(4)} ${style} | ${l.text}`);
    }
  } else if (mode === "--section") {
    const [file, index = "0"] = rest;
    const doc = await parsing.parsePdf(readBytes(file));
    const section = doc.sections[Number(index)];
    console.log(section ? `${section.number ?? ""} ${section.title}\n\n${section.text}` : "no such section");
  } else {
    for (const path of pdfPaths(mode ? [mode, ...rest] : [])) {
      const started = Date.now();
      const doc = await parsing.parsePdf(readBytes(path));
      printSummary(path, doc, Date.now() - started);
    }
  }
} finally {
  await runner.close();
  await server.close();
}

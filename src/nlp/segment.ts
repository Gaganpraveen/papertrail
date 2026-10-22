/**
 * Document segmentation: parsed sections -> ordered, located sentences.
 *
 * Every sentence keeps its section and an approximate page so evidence can
 * always point back to the PDF. Sections span page ranges, so each sentence
 * is located by searching a fingerprint of its opening characters (letters
 * and digits only, which survives hyphenation and spacing differences) in
 * the text of the section's pages. When the fingerprint is not found, the
 * page is interpolated from the sentence's position within the section.
 *
 * References and acknowledgments are skipped, and so are fragments that are
 * not prose: table rows and equations extracted as text (mostly digits or
 * symbols) would otherwise pollute summaries, zoning and retrieval.
 */
import type { PaperMeta, ParsedDocument, Section, SectionKind, Sentence } from "@/types";
import { splitSentences } from "./text";

const EXCLUDED_KINDS = new Set<SectionKind>(["references", "acknowledgments"]);
/** Characters of the sentence opening used to find its page. */
const PROBE_CHARS = 48;
/** Shorter fingerprints match by accident too often to be trusted. */
const MIN_PROBE_CHARS = 16;
/** Longer "sentences" are extraction artefacts (merged tables, broken layouts). */
const MAX_SENTENCE_CHARS = 1500;

/** Letters and digits only, lowercased: robust to hyphenation, ligatures and spacing. */
function fingerprint(text: string): string {
  return text
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "");
}

/** Prose has mostly letters; table rows and equations are dominated by digits and symbols. */
export function looksLikeProse(sentence: string): boolean {
  if (sentence.length > MAX_SENTENCE_CHARS) return false;
  const visible = sentence.replace(/\s+/g, "");
  if (visible.length === 0) return false;
  const letters = visible.match(/\p{L}/gu)?.length ?? 0;
  const digits = visible.match(/\p{N}/gu)?.length ?? 0;
  const wordCount = sentence.split(/\s+/).filter((w) => /\p{L}{2,}/u.test(w)).length;
  // Thresholds: an ordinary results sentence ("93.2% vs. 88.1% on GSM8K") stays well inside both.
  return letters / visible.length >= 0.55 && digits / visible.length <= 0.3 && wordCount >= 4;
}

class PageLocator {
  private readonly pageText = new Map<number, string>();

  constructor(pages: ParsedDocument["pages"]) {
    for (const page of pages) this.pageText.set(page.page, fingerprint(page.text));
  }

  /**
   * Page of `sentence` inside `section`. Pages from `hint` onwards are tried
   * first (sentences flow forward); captions, appended at the end of a
   * section, may sit on an earlier page, so those are tried next.
   */
  locate(sentence: string, section: Section, hint: number, fraction: number): number {
    const { pageStart, pageEnd } = section;
    if (pageStart >= pageEnd) return pageStart;
    const probe = fingerprint(sentence).slice(0, PROBE_CHARS);
    if (probe.length >= MIN_PROBE_CHARS) {
      const from = Math.min(Math.max(hint, pageStart), pageEnd);
      for (let p = from; p <= pageEnd; p++) if (this.pageText.get(p)?.includes(probe)) return p;
      for (let p = pageStart; p < from; p++) if (this.pageText.get(p)?.includes(probe)) return p;
    }
    const span = pageEnd - pageStart + 1;
    return Math.min(pageEnd, pageStart + Math.floor(fraction * span));
  }
}

function sectionSentences(section: Section, locator: PageLocator, firstIndex: number): Sentence[] {
  const texts = splitSentences(section.text).filter(looksLikeProse);
  const totalChars = texts.reduce((sum, t) => sum + t.length, 0);
  const sentences: Sentence[] = [];
  let charsBefore = 0;
  let hint = section.pageStart;
  for (const text of texts) {
    const fraction = totalChars > 0 ? charsBefore / totalChars : 0;
    const page = locator.locate(text, section, hint, fraction);
    hint = Math.max(hint, page);
    sentences.push({
      id: `${section.id}:${sentences.length}`,
      text,
      sectionId: section.id,
      sectionTitle: section.title,
      sectionKind: section.kind,
      page,
      index: firstIndex + sentences.length,
    });
    charsBefore += text.length;
  }
  return sentences;
}

/** Without detected sections, each page becomes a pseudo-section so nothing is lost. */
function pagesAsSections(doc: ParsedDocument): Section[] {
  return doc.pages
    .filter((p) => p.text.trim().length > 0)
    .map((p) => ({
      id: `p${p.page}`,
      title: `Page ${p.page}`,
      level: 1,
      kind: "other" as const,
      pageStart: p.page,
      pageEnd: p.page,
      text: p.text,
    }));
}

/** Sentences of every section except references and acknowledgments, in reading order. */
export function segmentDocument(doc: ParsedDocument): Sentence[] {
  const locator = new PageLocator(doc.pages);
  const sections = doc.sections.length > 0 ? doc.sections : pagesAsSections(doc);
  const out: Sentence[] = [];
  for (const section of sections) {
    if (EXCLUDED_KINDS.has(section.kind)) continue;
    out.push(...sectionSentences(section, locator, out.length));
  }
  return out;
}

/** Abstract-only mode: the metadata abstract as sentences of a single "abstract" section on page 1. */
export function segmentAbstract(meta: PaperMeta): Sentence[] {
  const abstract = meta.abstract?.trim();
  if (!abstract) return [];
  return splitSentences(abstract)
    .filter(looksLikeProse)
    .map((text, n) => ({
      id: `abstract:${n}`,
      text,
      sectionId: "abstract",
      sectionTitle: "Abstract",
      sectionKind: "abstract" as const,
      page: 1,
      index: n,
    }));
}

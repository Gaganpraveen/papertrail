/**
 * A deliberately tiny Markdown parser for the generated literature review.
 * It understands exactly what the narrative uses (headings, paragraphs,
 * bullet and numbered lists, pipe tables, block quotes, bold/italic/code,
 * links and [n] citations) and produces plain data, which the view renders as React
 * elements. No HTML is ever injected, so the output is safe by construction.
 */

export type Inline =
  | { type: "text"; text: string }
  | { type: "strong"; children: Inline[] }
  | { type: "em"; children: Inline[] }
  | { type: "code"; text: string }
  | { type: "link"; href: string; children: Inline[] }
  | { type: "cite"; refs: number[] };

export type Block =
  | { type: "heading"; level: number; text: string }
  | { type: "paragraph"; text: string }
  | { type: "list"; ordered: boolean; start: number; items: string[] }
  | { type: "quote"; text: string }
  | { type: "reference"; n: number; text: string }
  | { type: "table"; header: string[]; rows: string[][] }
  | { type: "rule" };

const HEADING = /^(#{1,4})\s+(.*)$/;
const BULLET = /^\s*[-*+]\s+(.*)$/;
const ORDERED = /^\s*(\d+)[.)]\s+(.*)$/;
const REFERENCE = /^\[(\d+)\]\s+(.*)$/;
const RULE = /^(-{3,}|\*{3,})\s*$/;
const TABLE_ROW = /^\|.*\|$/;
const TABLE_DIVIDER = /^\|?(\s*:?-{2,}:?\s*\|)+\s*:?-*:?\s*\|?$/;

function tableCells(row: string): string[] {
  return row
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

export function parseMarkdown(source: string): Block[] {
  const lines = source.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let paragraph: string[] = [];

  const flush = () => {
    if (paragraph.length) blocks.push({ type: "paragraph", text: paragraph.join(" ").trim() });
    paragraph = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    let m: RegExpMatchArray | null;

    if (!trimmed) {
      flush();
    } else if ((m = trimmed.match(HEADING))) {
      flush();
      blocks.push({ type: "heading", level: m[1].length, text: m[2].trim() });
    } else if (RULE.test(trimmed)) {
      flush();
      blocks.push({ type: "rule" });
    } else if ((m = trimmed.match(REFERENCE))) {
      flush();
      blocks.push({ type: "reference", n: Number(m[1]), text: m[2] });
    } else if (TABLE_ROW.test(trimmed)) {
      flush();
      const rows: string[][] = [];
      for (; i < lines.length && TABLE_ROW.test(lines[i].trim()); i++) {
        if (!TABLE_DIVIDER.test(lines[i].trim())) rows.push(tableCells(lines[i]));
      }
      i--;
      blocks.push({ type: "table", header: rows[0] ?? [], rows: rows.slice(1) });
    } else if (trimmed.startsWith(">")) {
      flush();
      blocks.push({ type: "quote", text: trimmed.replace(/^>\s?/, "") });
    } else if ((m = line.match(BULLET)) || (m = line.match(ORDERED))) {
      flush();
      const ordered = ORDERED.test(line);
      const start = ordered ? Number(line.match(ORDERED)![1]) : 1;
      const items: string[] = [];
      // Consume consecutive items of the same list type; indented lines continue an item.
      for (; i < lines.length; i++) {
        const item = lines[i].match(ordered ? ORDERED : BULLET);
        if (item) items.push(ordered ? item[2] : item[1]);
        else if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && items.length) {
          items[items.length - 1] += ` ${lines[i].trim()}`;
        } else break;
      }
      i--;
      blocks.push({ type: "list", ordered, start, items });
    } else {
      paragraph.push(trimmed);
    }
  }
  flush();
  return blocks;
}

/** "[1, 3]" / "[2–4]" -> [1, 3] / [2, 3, 4]. Returns null for non-citations. */
export function parseCitation(body: string): number[] | null {
  if (!/^\d+(\s*[,;–-]\s*\d+)*$/.test(body.trim())) return null;
  const refs: number[] = [];
  for (const part of body.split(/[,;]/)) {
    const range = part.split(/[–-]/).map((s) => Number(s.trim()));
    if (range.length === 2 && range[1] >= range[0] && range[1] - range[0] < 20) {
      for (let n = range[0]; n <= range[1]; n++) refs.push(n);
    } else refs.push(...range);
  }
  return refs.filter((n) => Number.isInteger(n) && n > 0);
}

const INLINE = /(\*\*([^*]+)\*\*)|(`([^`]+)`)|(\[([^\]]+)\]\((https?:\/\/[^)\s]+)\))|(\[([\d,;\s–-]+)\])|(\*([^*\s][^*]*)\*)|(_([^_\s][^_]*)_)/g;

export function parseInline(text: string): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const m of text.matchAll(INLINE)) {
    const at = m.index ?? 0;
    if (at > last) out.push({ type: "text", text: text.slice(last, at) });
    if (m[1]) out.push({ type: "strong", children: parseInline(m[2]) });
    else if (m[3]) out.push({ type: "code", text: m[4] });
    else if (m[5]) out.push({ type: "link", href: m[7], children: parseInline(m[6]) });
    else if (m[8]) {
      const refs = parseCitation(m[9]);
      out.push(refs ? { type: "cite", refs } : { type: "text", text: m[8] });
    } else if (m[10]) out.push({ type: "em", children: parseInline(m[11]) });
    else if (m[12]) out.push({ type: "em", children: parseInline(m[13]) });
    last = at + m[0].length;
  }
  if (last < text.length) out.push({ type: "text", text: text.slice(last) });
  return out;
}

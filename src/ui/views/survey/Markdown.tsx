import { Fragment, type ReactNode } from "react";
import { cn, safeHref } from "@/ui/kit";
import { parseInline, parseMarkdown, type Block, type Inline } from "./markdown";

export const refAnchor = (n: number) => `survey-ref-${n}`;

interface CiteProps {
  onCite: (n: number) => void;
  citeTitle: (n: number) => string | undefined;
  citeColor: (n: number) => string | undefined;
}

function renderInline(nodes: Inline[], cite: CiteProps): ReactNode[] {
  return nodes.map((node, i) => {
    switch (node.type) {
      case "text":
        return <Fragment key={i}>{node.text}</Fragment>;
      case "strong":
        return (
          <strong key={i} className="font-semibold text-ink">
            {renderInline(node.children, cite)}
          </strong>
        );
      case "em":
        return <em key={i}>{renderInline(node.children, cite)}</em>;
      case "code":
        return (
          <code key={i} className="rounded bg-panel-2 px-1 py-0.5 font-mono text-[0.85em]">
            {node.text}
          </code>
        );
      case "link":
        return (
          <a key={i} href={safeHref(node.href)} target="_blank" rel="noreferrer noopener" className="text-accent hover:underline">
            {renderInline(node.children, cite)}
          </a>
        );
      case "cite":
        return (
          <span key={i} className="whitespace-nowrap">
            {node.refs.map((n) => (
              <button
                key={n}
                onClick={() => cite.onCite(n)}
                title={cite.citeTitle(n)}
                className="mx-px inline-flex h-[18px] min-w-[18px] -translate-y-px items-center justify-center rounded-md px-1 align-middle text-[10.5px] font-semibold tabular transition-colors hover:brightness-110"
                style={{
                  color: cite.citeColor(n) ?? "var(--accent)",
                  background: `color-mix(in oklab, ${cite.citeColor(n) ?? "var(--accent)"} 16%, transparent)`,
                }}
              >
                {n}
              </button>
            ))}
          </span>
        );
    }
  });
}

function renderBlock(block: Block, key: number, cite: CiteProps, inReferences: boolean): ReactNode {
  const inline = (text: string) => renderInline(parseInline(text), cite);
  switch (block.type) {
    case "heading": {
      const cls = cn(
        "tracking-tight text-ink",
        block.level <= 1 ? "mt-2 text-lg font-semibold" : block.level === 2 ? "text-base font-semibold" : "text-sm font-semibold",
      );
      return block.level <= 2 ? (
        <h2 key={key} className={cls}>
          {inline(block.text)}
        </h2>
      ) : (
        <h3 key={key} className={cls}>
          {inline(block.text)}
        </h3>
      );
    }
    case "paragraph":
      return <p key={key}>{inline(block.text)}</p>;
    case "quote":
      return (
        <blockquote key={key} className="border-l-2 border-line-strong pl-3 text-ink-3 italic">
          {inline(block.text)}
        </blockquote>
      );
    case "rule":
      return <hr key={key} className="my-4 border-line" />;
    case "reference":
      // Prose can open with a citation ("[2] stands on its own here…"). Only the
      // References section holds citation targets; elsewhere it stays prose with
      // a live chip, so anchor ids stay unique and chips jump to the bibliography.
      if (!inReferences) return <p key={key}>{inline(`[${block.n}] ${block.text}`)}</p>;
      return (
        <p
          key={key}
          id={refAnchor(block.n)}
          className="scroll-mt-24 rounded-lg px-2 py-1 text-[13px] text-ink-2 transition-colors duration-700"
        >
          <span className="mr-2 font-semibold tabular" style={{ color: cite.citeColor(block.n) }}>
            [{block.n}]
          </span>
          {inline(block.text)}
        </p>
      );
    case "table":
      return (
        <div key={key} className="my-4 overflow-x-auto rounded-xl border border-line">
          <table className="w-full border-collapse text-[12.5px]">
            <thead className="bg-panel-2/60">
              <tr>
                {block.header.map((cell, i) => (
                  <th key={i} className="px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wider whitespace-nowrap text-ink-3">
                    {inline(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, r) => (
                <tr key={r} className="border-t border-line align-top">
                  {row.map((cell, c) => (
                    <td key={c} className="px-3 py-2 leading-relaxed text-ink-2">
                      {inline(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "list": {
      const items = block.items.map((item, i) => {
        // A numbered list under "References" doubles as the citation target.
        const anchor = block.ordered && inReferences ? refAnchor(block.start + i) : undefined;
        return (
          <li key={i} id={anchor} className="scroll-mt-24 rounded-md transition-colors duration-700">
            {inline(item)}
          </li>
        );
      });
      return block.ordered ? (
        <ol key={key} start={block.start} className="list-decimal pl-5">
          {items}
        </ol>
      ) : (
        <ul key={key}>{items}</ul>
      );
    }
  }
}

/** Renders the survey narrative; [n] citations become chips that jump to reference n. */
export function Markdown({ source, ...cite }: { source: string } & CiteProps) {
  const blocks = parseMarkdown(source);
  let inReferences = false;
  return (
    <div className="prose-pt text-[14px] text-ink-2">
      {blocks.map((block, i) => {
        if (block.type === "heading") inReferences = /^(references|bibliography)\b/i.test(block.text);
        return renderBlock(block, i, cite, inReferences);
      })}
    </div>
  );
}

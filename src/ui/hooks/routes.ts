/**
 * Hash routes for the single-page app (GitHub Pages serves one static file,
 * so hash routing is the only routing that survives a reload).
 *
 *   #/                       home
 *   #/search                 current topic search
 *   #/paper/<id>/<tab>       a paper in the library
 *   #/survey/<id>            a saved survey
 *   #/analyze/<input>        deep link: analyse an arXiv id, DOI or URL on load
 */
import type { PaperTab, View } from "@/store/types";

const PAPER_TABS: PaperTab[] = ["briefing", "visuals", "critique", "ask"];

export type Route = View | { kind: "analyze"; input: string };

export function viewToHash(view: View): string {
  switch (view.kind) {
    case "home":
      return "#/";
    case "search":
      return "#/search";
    case "paper":
      return `#/paper/${encodeURIComponent(view.id)}/${view.tab}`;
    case "survey":
      return `#/survey/${encodeURIComponent(view.id)}`;
  }
}

export function hashToRoute(hash: string): Route | null {
  const [kind, ...rest] = hash.replace(/^#\/?/, "").split("/");
  const arg = rest[0] ? safeDecode(rest[0]) : "";
  switch (kind) {
    case "":
      return { kind: "home" };
    case "search":
      return { kind: "search" };
    case "paper": {
      const tab = PAPER_TABS.find((t) => t === rest[1]) ?? "briefing";
      return arg ? { kind: "paper", id: arg, tab } : null;
    }
    case "survey":
      return arg ? { kind: "survey", id: arg } : null;
    case "analyze": {
      const input = safeDecode(rest.join("/")).trim();
      return input ? { kind: "analyze", input } : null;
    }
    default:
      return null;
  }
}

function safeDecode(text: string): string {
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}

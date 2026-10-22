/**
 * Bibliographic coupling (Kessler, 1963): two papers are related when they
 * cite the same work. Coupling strength is the Jaccard overlap of their
 * reference sets, |A ∩ B| / |A ∪ B|.
 *
 * The hard part is deciding that two differently formatted references are the
 * same work. Each reference yields several keys (arXiv id without version,
 * DOI, normalised title); a union-find over keys links a reference cited by
 * arXiv id in one paper and by title in another. Direct citations (paper A's
 * bibliography contains paper B itself) are detected with the same keys.
 */
import type { PaperMeta, Reference, SurveyResult } from "@/types";

export type CouplingEdge = SurveyResult["coupling"][number];

const ARXIV_DOI_RE = /^10\.48550\/arxiv\.(.+)$/i;
/** Titles shorter than this collide too easily ("Attention", "Introduction"). */
const MIN_TITLE_WORDS = 3;
const MIN_TITLE_CHARS = 15;

export function normalizeArxivId(id: string): string {
  return id.trim().toLowerCase().replace(/^arxiv:/, "").replace(/v\d+$/, "");
}

export function normalizeTitle(title: string): string {
  return title
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function identityKeys(ids: { arxivId?: string; doi?: string; title?: string }): string[] {
  const keys: string[] = [];
  const doi = ids.doi?.trim().toLowerCase();
  const arxivFromDoi = doi ? ARXIV_DOI_RE.exec(doi)?.[1] : undefined;
  const arxiv = ids.arxivId ?? arxivFromDoi;
  if (arxiv) keys.push(`arxiv:${normalizeArxivId(arxiv)}`);
  if (doi && !arxivFromDoi) keys.push(`doi:${doi}`);
  if (ids.title) {
    const title = normalizeTitle(ids.title);
    if (title.split(" ").length >= MIN_TITLE_WORDS && title.length >= MIN_TITLE_CHARS) keys.push(`title:${title}`);
  }
  return keys;
}

export function referenceKeys(ref: Reference): string[] {
  return identityKeys(ref);
}

export function paperKeys(meta: PaperMeta): string[] {
  return identityKeys(meta);
}

export interface CouplingPaper {
  id: string;
  meta: PaperMeta;
  references: Reference[];
}

/** Maps every reference key to a canonical work id, merging keys that co-occur on one reference. */
function canonicalWorks(papers: CouplingPaper[]): (key: string) => string {
  const parent = new Map<string, string>();
  const find = (k: string): string => {
    const p = parent.get(k) ?? k;
    if (p === k) return k;
    const root = find(p);
    parent.set(k, root);
    return root;
  };
  const union = (a: string, b: string) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(rb, ra);
  };
  for (const paper of papers) {
    for (const ref of paper.references) {
      const keys = referenceKeys(ref);
      for (const key of keys) if (!parent.has(key)) parent.set(key, key);
      for (let i = 1; i < keys.length; i++) union(keys[0], keys[i]);
    }
  }
  return find;
}

/** The set of distinct works each paper cites. */
export function citedWorks(papers: CouplingPaper[]): Set<string>[] {
  const work = canonicalWorks(papers);
  return papers.map((paper) => {
    const works = new Set<string>();
    for (const ref of paper.references) {
      const keys = referenceKeys(ref);
      if (keys.length > 0) works.add(work(keys[0]));
    }
    return works;
  });
}

/** Pairs of papers that share at least one reference, strongest first. */
export function bibliographicCoupling(papers: CouplingPaper[]): CouplingEdge[] {
  const works = citedWorks(papers);
  const edges: CouplingEdge[] = [];
  for (let i = 0; i < papers.length; i++) {
    for (let j = i + 1; j < papers.length; j++) {
      let shared = 0;
      for (const w of works[i]) if (works[j].has(w)) shared++;
      if (shared === 0) continue;
      const union = works[i].size + works[j].size - shared;
      edges.push({ source: papers[i].id, target: papers[j].id, shared, jaccard: shared / union });
    }
  }
  return edges.sort((a, b) => b.jaccard - a.jaccard || b.shared - a.shared);
}

export interface DirectCitation {
  citing: string;
  cited: string;
}

/** Paper A cites paper B when one of A's references carries one of B's identity keys. */
export function directCitations(papers: CouplingPaper[]): DirectCitation[] {
  const citations: DirectCitation[] = [];
  for (const citing of papers) {
    const keys = new Set(citing.references.flatMap(referenceKeys));
    for (const cited of papers) {
      if (cited.id === citing.id) continue;
      if (paperKeys(cited.meta).some((k) => keys.has(k))) citations.push({ citing: citing.id, cited: cited.id });
    }
  }
  return citations;
}

/**
 * OpenAlex adapter: maps an OpenAlex "work" to PaperMeta.
 *
 * OpenAlex is our widest-coverage, CORS-friendly index and the source of
 * citation counts. For legal reasons it ships abstracts as an inverted index
 * ({ word: [positions] }), which we turn back into text by placing every word
 * at its positions and reading them in order.
 *
 * Its arXiv records are merged from several upstreams and are sometimes
 * wrong (a stray DOI, a placeholder date), so callers prefer DataCite for
 * arXiv metadata and use OpenAlex mainly to enrich (citations, venue, OA PDF).
 */
import type { PaperMeta } from "@/types";
import { asArray, asNumber, asRecords, asString, isRecord } from "./http";
import {
  arxivIdFromDoi,
  arxivIdFromUrl,
  cleanMetaText,
  encodeDoiPath,
  normalizeAbstract,
  normalizeDoi,
  normalizePersonName,
  type ArxivRef,
} from "./input";

export const OPENALEX_API = "https://api.openalex.org";
/** OpenAlex source id of the arXiv repository. */
export const OPENALEX_ARXIV_SOURCE = "S4306400194";
/** Root fields we read; asking only for these keeps search responses ~10x smaller. */
// prettier-ignore
export const OPENALEX_SELECT = [
  "id", "doi", "display_name", "title", "publication_year", "publication_date", "ids",
  "authorships", "abstract_inverted_index", "cited_by_count", "primary_location",
  "best_oa_location", "locations", "type",
].join(",");

/** Rebuilds abstract text from OpenAlex's { word: [positions] } index. */
export function reconstructAbstract(
  inverted: Record<string, number[]> | null | undefined,
): string | undefined {
  if (!inverted || typeof inverted !== "object") return undefined;
  const placed: [number, string][] = [];
  for (const [word, positions] of Object.entries(inverted)) {
    if (!Array.isArray(positions)) continue;
    for (const position of positions) {
      if (Number.isInteger(position) && position >= 0) placed.push([position, word]);
    }
  }
  if (placed.length === 0) return undefined;
  // Sorting pairs (rather than indexing a dense array) stays safe with sparse or huge positions.
  placed.sort((a, b) => a[0] - b[0]);
  const text = placed
    .map(([, word]) => word)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
  return text || undefined;
}

function isArxivSource(source: unknown): boolean {
  if (!isRecord(source)) return false;
  const id = asString(source.id) ?? "";
  const name = asString(source.display_name) ?? "";
  return id.endsWith(OPENALEX_ARXIV_SOURCE) || /^arxiv\b/i.test(name);
}

/** Looks for an arXiv ID in locations (landing/PDF URLs, per-location DOIs) and the work's own ids. */
function findArxivRef(work: Record<string, unknown>): ArxivRef | null {
  const locations = asRecords(work.locations);
  // arXiv-hosted locations first: their URLs are the most reliable evidence.
  const ordered = [...locations.filter((l) => isArxivSource(l.source)), ...locations];
  for (const location of ordered) {
    for (const url of [asString(location.landing_page_url), asString(location.pdf_url)]) {
      const ref = url ? arxivIdFromUrl(url) : null;
      if (ref) return { id: ref.id };
    }
    const locationDoi = normalizeDoi(asString(location.doi) ?? "");
    const ref = locationDoi ? arxivIdFromDoi(locationDoi) : null;
    if (ref) return ref;
  }
  const ids = isRecord(work.ids) ? work.ids : {};
  for (const candidate of [asString(work.doi), asString(ids.doi)]) {
    const doi = candidate ? normalizeDoi(candidate) : null;
    const ref = doi ? arxivIdFromDoi(doi) : null;
    if (ref) return ref;
  }
  return null;
}

/** Earliest possible date of a new-style arXiv ID ("2510.00231" -> "2025-10"). */
function arxivMonth(arxivId: string): string | undefined {
  const match = /^(\d{2})(\d{2})\./.exec(arxivId);
  return match ? `20${match[1]}-${match[2]}` : undefined;
}

/**
 * OpenAlex occasionally dates an arXiv paper before it existed (a placeholder
 * "YYYY-01-01" or a merged record). A paper cannot predate its arXiv ID, so
 * such dates are replaced by the month encoded in the ID.
 */
function sanitizeDate(published: string | undefined, arxivId: string | undefined): string | undefined {
  const floor = arxivId ? arxivMonth(arxivId) : undefined;
  if (!floor) return published;
  if (!published || published.slice(0, 7) < floor) return floor;
  return published;
}

function authorNames(work: Record<string, unknown>): string[] {
  return asRecords(work.authorships)
    .map((authorship) => {
      const author = isRecord(authorship.author) ? authorship.author : {};
      return asString(author.display_name) ?? asString(authorship.raw_author_name) ?? "";
    })
    .map(normalizePersonName)
    .filter((name) => name.length > 0);
}

function venueOf(work: Record<string, unknown>): string | undefined {
  const primary = isRecord(work.primary_location) ? work.primary_location : {};
  if (isArxivSource(primary.source)) return "arXiv";
  const source = isRecord(primary.source) ? primary.source : {};
  return asString(source.display_name);
}

function openAccessPdf(work: Record<string, unknown>): string | undefined {
  const best = isRecord(work.best_oa_location) ? work.best_oa_location : {};
  const fromBest = asString(best.pdf_url);
  if (fromBest) return fromBest;
  for (const location of asRecords(work.locations)) {
    const pdf = asString(location.pdf_url);
    if (pdf && location.is_oa !== false) return pdf;
  }
  return undefined;
}

function invertedIndexOf(value: unknown): Record<string, number[]> | null {
  if (!isRecord(value)) return null;
  const index: Record<string, number[]> = {};
  for (const [word, positions] of Object.entries(value)) {
    index[word] = asArray(positions).filter((p): p is number => typeof p === "number");
  }
  return index;
}

/** Maps one OpenAlex work (single-work response or a search result item). */
export function mapOpenAlexWork(json: unknown): PaperMeta | null {
  if (!isRecord(json)) return null;
  const title = cleanMetaText(asString(json.display_name) ?? asString(json.title));
  if (!title) return null;

  const ids = isRecord(json.ids) ? json.ids : {};
  const doi = normalizeDoi(asString(json.doi) ?? asString(ids.doi) ?? "") ?? undefined;
  const arxiv = findArxivRef(json);
  const openAlexId = asString(json.id)?.replace(/^https?:\/\/openalex\.org\//i, "");
  const id = arxiv ? `arxiv:${arxiv.id}` : doi ? `doi:${doi}` : openAlexId ? `openalex:${openAlexId}` : null;
  if (!id) return null;

  const reportedYear = asNumber(json.publication_year);
  const reportedDate = asString(json.publication_date) ?? (reportedYear ? String(reportedYear) : undefined);
  const published = sanitizeDate(reportedDate, arxiv?.id);
  const year = published ? Number(published.slice(0, 4)) : reportedYear;
  const authors = authorNames(json);
  const abstract = normalizeAbstract(
    reconstructAbstract(invertedIndexOf(json.abstract_inverted_index)),
    authors,
  );
  const primary = isRecord(json.primary_location) ? json.primary_location : {};

  const meta: PaperMeta = {
    id,
    title,
    authors,
    source: "openalex",
  };
  if (abstract) meta.abstract = abstract;
  if (published) meta.published = published;
  if (year) meta.year = year;
  if (arxiv) meta.arxivId = arxiv.id;
  if (doi) meta.doi = doi;
  const venue = venueOf(json);
  if (venue) meta.venue = venue;
  const citations = asNumber(json.cited_by_count);
  if (citations !== undefined) meta.citationCount = citations;
  const pdfUrl = openAccessPdf(json);
  if (pdfUrl) meta.pdfUrl = pdfUrl;
  const landingUrl = asString(primary.landing_page_url) ?? (doi ? `https://doi.org/${doi}` : undefined);
  if (landingUrl) meta.landingUrl = landingUrl;
  return meta;
}

// prettier-ignore
const FILLABLE_FIELDS = [
  "abstract", "published", "year", "arxivId", "version", "doi", "categories", "pdfUrl", "landingUrl",
] as const;

function fillGap<K extends keyof PaperMeta>(target: PaperMeta, source: PaperMeta, key: K): void {
  if (target[key] === undefined && source[key] !== undefined) target[key] = source[key];
}

/**
 * Fills the gaps of `primary` from `secondary` (both describing the same paper).
 * Values already present in `primary` win, with two deliberate exceptions:
 * a peer-reviewed venue beats the generic "arXiv", and a known citation count
 * beats an unknown one.
 */
export function mergeMeta(primary: PaperMeta, secondary: PaperMeta | null | undefined): PaperMeta {
  if (!secondary) return primary;
  const merged: PaperMeta = { ...primary };
  for (const key of FILLABLE_FIELDS) fillGap(merged, secondary, key);
  if (merged.authors.length === 0) merged.authors = secondary.authors;
  if ((!merged.venue || merged.venue === "arXiv") && secondary.venue && secondary.venue !== "arXiv") {
    merged.venue = secondary.venue;
  } else if (!merged.venue && secondary.venue) {
    merged.venue = secondary.venue;
  }
  if (merged.citationCount === undefined && secondary.citationCount !== undefined) {
    merged.citationCount = secondary.citationCount;
  }
  return merged;
}

/** GET /works/doi:<doi> — the canonical single-work lookup. */
export function openAlexWorkUrl(doi: string): string {
  return `${OPENALEX_API}/works/doi:${encodeDoiPath(doi)}`;
}

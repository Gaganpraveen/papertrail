/**
 * arXiv adapter.
 *
 * arXiv's own export API speaks Atom XML and is built for batch harvesting,
 * not for a web page calling it on every keystroke. Every arXiv paper is also
 * registered at DataCite under the DOI 10.48550/arXiv.<id>, and DataCite's
 * JSON API is CORS-enabled and carries what we need: title, authors,
 * abstract, arXiv categories, per-version submission dates and the author
 * comment (often "Accepted at ICML 2024"). OpenAlex is the fallback when
 * DataCite fails or has not indexed a brand-new paper yet.
 */
import type { PaperMeta } from "@/types";
import {
  asNumber,
  asRecords,
  asString,
  fetchJson,
  isAbortError,
  isRecord,
  mostInformativeError,
  SourceError,
} from "./http";
import {
  arxivIdFromDoi,
  arxivIdFromUrl,
  cleanMetaText,
  encodeDoiPath,
  normalizeAbstract,
  normalizeDoi,
  normalizePersonName,
  parseArxivId,
  type ArxivRef,
} from "./input";
import { mapOpenAlexWork, openAlexWorkUrl } from "./openalex";

export const DATACITE_API = "https://api.datacite.org";

function normalizeVersion(version: string | undefined): string | undefined {
  if (!version) return undefined;
  const match = /^v?(\d+)$/i.exec(version.trim());
  return match ? `v${match[1]}` : undefined;
}

export function arxivPdfUrl(id: string, version?: string): string {
  return `https://arxiv.org/pdf/${id}${normalizeVersion(version) ?? ""}`;
}

export function arxivAbsUrl(id: string, version?: string): string {
  return `https://arxiv.org/abs/${id}${normalizeVersion(version) ?? ""}`;
}

/** The DataCite DOI arXiv mints for every paper (lower-cased, like all our DOIs). */
export function arxivDoi(id: string): string {
  return `10.48550/arxiv.${id}`;
}

export function dataCiteDoiUrl(doi: string): string {
  return `${DATACITE_API}/dois/${encodeDoiPath(doi)}`;
}

// ───────────────────────────── DataCite mapping ─────────────────────────────

/** Accepts a whole /dois/{doi} response, its "data" object, a list item, or bare attributes. */
function dataCiteAttributes(json: unknown): Record<string, unknown> | null {
  if (!isRecord(json)) return null;
  if (isRecord(json.data)) return dataCiteAttributes(json.data);
  if (isRecord(json.attributes)) return json.attributes;
  return "titles" in json ? json : null;
}

function findArxivRef(attrs: Record<string, unknown>): ArxivRef | null {
  for (const item of asRecords(attrs.identifiers)) {
    if (asString(item.identifierType)?.toLowerCase() === "arxiv") {
      const ref = parseArxivId(asString(item.identifier) ?? "");
      if (ref) return ref;
    }
  }
  for (const item of asRecords(attrs.alternateIdentifiers)) {
    if (asString(item.alternateIdentifierType)?.toLowerCase() === "arxiv") {
      const ref = parseArxivId(asString(item.alternateIdentifier) ?? "");
      if (ref) return ref;
    }
  }
  const doi = normalizeDoi(asString(attrs.doi) ?? "");
  const fromDoi = doi ? arxivIdFromDoi(doi) : null;
  if (fromDoi) return fromDoi;
  const url = asString(attrs.url);
  return url ? arxivIdFromUrl(url) : null;
}

function creatorNames(attrs: Record<string, unknown>): string[] {
  return asRecords(attrs.creators)
    .map((creator) => {
      const given = asString(creator.givenName);
      const family = asString(creator.familyName);
      if (given && family) return cleanMetaText(`${given} ${family}`);
      return normalizePersonName(asString(creator.name) ?? "");
    })
    .filter((name) => name.length > 0);
}

function descriptionOfType(attrs: Record<string, unknown>, type: string): string | undefined {
  const match = asRecords(attrs.descriptions).find(
    (d) => asString(d.descriptionType)?.toLowerCase() === type,
  );
  return match ? asString(match.description) : undefined;
}

/**
 * First public appearance: the earliest "Submitted" date (that is v1),
 * otherwise the coarser "Available"/"Issued" dates, otherwise the year.
 */
function firstPublished(attrs: Record<string, unknown>): string | undefined {
  const dates = asRecords(attrs.dates)
    .map((d) => ({ type: asString(d.dateType)?.toLowerCase(), date: asString(d.date) }))
    .filter(
      (d): d is { type: string | undefined; date: string } => d.date !== undefined && /^\d{4}/.test(d.date),
    );
  for (const type of ["submitted", "available", "issued"]) {
    const ofType = dates
      .filter((d) => d.type === type)
      .map((d) => d.date.slice(0, 10))
      .sort();
    if (ofType.length > 0) return ofType[0];
  }
  const year = asNumber(attrs.publicationYear);
  return year ? String(year) : undefined;
}

/** "Computation and Language (cs.CL)" -> "cs.CL"; non-arXiv schemes (e.g. OECD FOS) are dropped. */
function arxivCategories(attrs: Record<string, unknown>): string[] {
  const categories: string[] = [];
  for (const subject of asRecords(attrs.subjects)) {
    if (asString(subject.subjectScheme)?.toLowerCase() !== "arxiv") continue;
    const label = asString(subject.subject);
    if (!label) continue;
    const code = /\(([a-z-]+(?:\.[A-Za-z-]+)?)\)\s*$/.exec(label)?.[1] ?? label;
    if (!categories.includes(code)) categories.push(code);
  }
  return categories;
}

/** Well-known ML/systems venues, as written in arXiv comments ("ICML2024", "NeurIPS'23"). */
// prettier-ignore
const VENUE_NAMES = [
  "NeurIPS", "NIPS", "ICML", "ICLR", "ACL", "EMNLP", "NAACL", "EACL", "AACL", "COLING", "COLM", "TACL",
  "CVPR", "ICCV", "ECCV", "WACV", "AAAI", "IJCAI", "KDD", "SIGIR", "WWW", "WSDM", "CIKM", "RecSys", "UAI",
  "AISTATS", "CoRL", "ICRA", "IROS", "RSS", "MLSys", "OSDI", "SOSP", "NSDI", "ISCA", "MICRO", "ASPLOS",
  "SIGMOD", "VLDB", "ICSE", "INTERSPEECH", "ICASSP", "MICCAI", "JMLR", "TMLR", "TPAMI",
];
const VENUE_ACRONYM = new RegExp(`\\b(${VENUE_NAMES.join("|")})\\s*'?(\\d{4}|\\d{2})?\\b`);
const VENUE_WORDS = /\b(conference|journal|proceedings|workshop|transactions|symposium|findings|letters)\b/i;
/** "<accepted|published|...> ... <at|in|to|by|for> [the] VENUE" up to a sentence break or a format note. */
const ACCEPTANCE_NOTE = new RegExp(
  "\\b(?:accepted|published|to appear|appears|appeared|presented)\\b[^.;]*?" +
    "\\b(?:at|in|to|by|for)\\s+(?:the\\s+)?(.+?)" +
    "(?:\\s*[.;(]|,\\s*(?:forthcoming|camera|oral|spotlight|poster|main|long|short)\\b|$)",
  "i",
);
const MAX_VENUE_LENGTH = 80;

/**
 * Best-effort venue from the arXiv author comment ("Accepted to Findings of
 * EMNLP 2026, forthcoming" -> "Findings of EMNLP 2026", "ICML2024" ->
 * "ICML 2024"). Returns undefined for comments such as "8 pages, 6 figures".
 */
export function venueFromComment(comment: string | undefined): string | undefined {
  if (!comment) return undefined;
  const text = cleanMetaText(comment);
  const note = ACCEPTANCE_NOTE.exec(text)?.[1]?.trim();
  if (note && note.length <= MAX_VENUE_LENGTH && (VENUE_ACRONYM.test(note) || VENUE_WORDS.test(note))) {
    return note;
  }
  const acronym = VENUE_ACRONYM.exec(text);
  if (!acronym) return undefined;
  const [, name, year] = acronym;
  if (!year) return name;
  return `${name} ${year.length === 2 ? `20${year}` : year}`;
}

/** Maps a DataCite DOI record to PaperMeta; null when it has no usable title. */
export function mapDataCite(json: unknown): PaperMeta | null {
  const attrs = dataCiteAttributes(json);
  if (!attrs) return null;
  const titleRecord = asRecords(attrs.titles)[0];
  const title = cleanMetaText(titleRecord ? asString(titleRecord.title) : undefined);
  if (!title) return null;

  const arxiv = findArxivRef(attrs);
  const doi = normalizeDoi(asString(attrs.doi) ?? "") ?? undefined;
  const id = arxiv ? `arxiv:${arxiv.id}` : doi ? `doi:${doi}` : null;
  if (!id) return null;

  const meta: PaperMeta = { id, title, authors: creatorNames(attrs), source: "datacite" };
  const abstract = normalizeAbstract(descriptionOfType(attrs, "abstract"), meta.authors);
  if (abstract) meta.abstract = abstract;
  const published = firstPublished(attrs);
  if (published) {
    meta.published = published;
    meta.year = Number(published.slice(0, 4));
  }
  if (doi) meta.doi = doi;
  const categories = arxivCategories(attrs);
  if (categories.length > 0) meta.categories = categories;

  if (arxiv) {
    meta.arxivId = arxiv.id;
    // DataCite's "version" is the latest arXiv version number ("2" -> "v2").
    const latest = normalizeVersion(String(asNumber(attrs.version) ?? ""));
    if (latest) meta.version = latest;
    meta.venue = venueFromComment(descriptionOfType(attrs, "other")) ?? "arXiv";
    meta.pdfUrl = arxivPdfUrl(arxiv.id);
    meta.landingUrl = arxivAbsUrl(arxiv.id);
  } else {
    const publisher = asString(attrs.publisher);
    if (publisher) meta.venue = cleanMetaText(publisher);
    const url = asString(attrs.url);
    if (url) meta.landingUrl = url;
  }
  return meta;
}

// ───────────────────────────────── Fetching ─────────────────────────────────

/**
 * Points a record at arXiv: stable "arxiv:<id>" key and arXiv PDF. A
 * publisher landing page is kept (it is the version of record); otherwise the
 * abstract page becomes the landing URL.
 */
export function withArxivLinks(meta: PaperMeta, arxivId: string, version?: string): PaperMeta {
  const requested = normalizeVersion(version);
  const keepLanding = meta.landingUrl !== undefined && !/arxiv\.org|10\.48550/i.test(meta.landingUrl);
  const linked: PaperMeta = {
    ...meta,
    id: `arxiv:${arxivId}`,
    arxivId,
    pdfUrl: arxivPdfUrl(arxivId, requested),
    landingUrl: keepLanding ? meta.landingUrl : arxivAbsUrl(arxivId, requested),
  };
  const shownVersion = requested ?? meta.version;
  if (shownVersion) linked.version = shownVersion;
  return linked;
}

/**
 * Metadata for one arXiv paper: DataCite first, OpenAlex as fallback.
 * The returned id is "arxiv:<id>"; `version` pins the PDF link to that version.
 */
export async function fetchArxivMeta(id: string, version?: string, signal?: AbortSignal): Promise<PaperMeta> {
  const ref = parseArxivId(id);
  if (!ref) throw new SourceError(`"${id}" is not a valid arXiv identifier`, "invalid");
  const requestedVersion = normalizeVersion(version) ?? ref.version;
  const doi = arxivDoi(ref.id);
  const errors: unknown[] = [];

  const lookups: { url: string; map: (json: unknown) => PaperMeta | null }[] = [
    { url: dataCiteDoiUrl(doi), map: mapDataCite },
    { url: openAlexWorkUrl(doi), map: mapOpenAlexWork },
  ];
  for (const { url, map } of lookups) {
    try {
      const meta = map(await fetchJson(url, { signal }));
      if (meta) return withArxivLinks(meta, ref.id, requestedVersion);
      errors.push(new SourceError("The metadata record had no title", "invalid", url));
    } catch (error) {
      if (isAbortError(error)) throw error;
      errors.push(error);
    }
  }
  throw mostInformativeError(errors, new SourceError(`No arXiv paper found for ${ref.id}`, "not-found"));
}

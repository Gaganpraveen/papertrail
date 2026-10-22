/**
 * DOI resolution by consensus of three free, CORS-enabled indexes.
 *
 *  - Crossref is the registry of record for publisher DOIs: authoritative
 *    title, authors, venue and date, but often no abstract and paywalled PDFs.
 *  - OpenAlex fills the abstract, contributes citation counts and open-access
 *    PDF locations (including an arXiv copy when it knows one).
 *  - Semantic Scholar is asked only for its arXiv cross-reference
 *    (externalIds.ArXiv); it rate-limits aggressively, so it gets no retries
 *    and its failures are ignored.
 *
 * When an arXiv copy exists we analyse that PDF: it is free, stable and
 * served from one predictable URL.
 */
import type { PaperMeta } from "@/types";
import { fetchArxivMeta, mapDataCite, withArxivLinks } from "./arxiv";
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
  cleanMetaText,
  encodeDoiPath,
  normalizeAbstract,
  normalizeDoi,
  parseArxivId,
} from "./input";
import { mapOpenAlexWork, mergeMeta, openAlexWorkUrl } from "./openalex";
import { dataCiteSearchUrl, dataCiteTerms } from "./search";

export const CROSSREF_API = "https://api.crossref.org";
export const SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1";

// ───────────────────────────── Crossref mapping ─────────────────────────────

function crossrefMessage(json: unknown): Record<string, unknown> | null {
  if (!isRecord(json)) return null;
  if (isRecord(json.message)) return json.message;
  return "DOI" in json || "title" in json ? json : null;
}

/**
 * JATS abstracts embed a heading and citation superscripts ("effort<sup>1–4</sup>").
 * Citation markers are removed; a superscript glued to a digit is an exponent
 * ("10<sup>6</sup>" -> "10^6") and is kept.
 */
export function stripJats(abstract: string): string {
  const withoutHeading = abstract.replace(/<(?:jats:)?title\b[^>]*>[\s\S]*?<\/(?:jats:)?title>/gi, " ");
  const withoutCitations = withoutHeading.replace(
    /(\d?)(\s*)<(?:jats:)?sup\b[^>]*>([\s\S]*?)<\/(?:jats:)?sup>/gi,
    (match, digitBefore: string, space: string, body: string) => {
      const isCitation = /^[\s\d,–—-]+$/.test(body);
      if (digitBefore && !space) return `${digitBefore}^${body.trim()}`;
      return isCitation ? digitBefore : match;
    },
  );
  return cleanMetaText(withoutCitations);
}

/** Crossref dates are { "date-parts": [[2021, 7, 15]] } with month/day optional. */
function crossrefDate(value: unknown): string | undefined {
  if (!isRecord(value)) return undefined;
  const parts = Array.isArray(value["date-parts"]) ? value["date-parts"][0] : undefined;
  if (!Array.isArray(parts)) return undefined;
  const [year, month, day] = parts.map((p) => asNumber(p));
  if (!year) return undefined;
  const pad = (n: number) => String(n).padStart(2, "0");
  if (month && day) return `${year}-${pad(month)}-${pad(day)}`;
  if (month) return `${year}-${pad(month)}`;
  return String(year);
}

function firstString(value: unknown): string | undefined {
  return Array.isArray(value) ? asString(value[0]) : asString(value);
}

function crossrefTitle(message: Record<string, unknown>): string {
  const title = cleanMetaText(firstString(message.title));
  const subtitle = cleanMetaText(firstString(message.subtitle));
  if (!title) return "";
  return subtitle && !title.toLowerCase().includes(subtitle.toLowerCase()) ? `${title}: ${subtitle}` : title;
}

function crossrefAuthors(message: Record<string, unknown>): string[] {
  return asRecords(message.author)
    .map((author) => {
      const given = asString(author.given);
      const family = asString(author.family);
      return cleanMetaText(given && family ? `${given} ${family}` : (family ?? asString(author.name)));
    })
    .filter((name) => name.length > 0);
}

/** Crossref "relation" sometimes links the published article to its arXiv preprint. */
function crossrefArxivId(message: Record<string, unknown>): string | undefined {
  const relation = isRecord(message.relation) ? message.relation : {};
  for (const item of asRecords(relation["has-preprint"])) {
    const id = asString(item.id) ?? "";
    const type = asString(item["id-type"])?.toLowerCase();
    const ref = type === "arxiv" ? parseArxivId(id) : arxivIdFromDoi(normalizeDoi(id) ?? "");
    if (ref) return ref.id;
  }
  return undefined;
}

function crossrefPdf(message: Record<string, unknown>): string | undefined {
  const pdf = asRecords(message.link).find((link) => asString(link["content-type"]) === "application/pdf");
  return pdf ? asString(pdf.URL) : undefined;
}

/** Maps a Crossref /works/{doi} response (or its "message"). */
export function mapCrossref(json: unknown): PaperMeta | null {
  const message = crossrefMessage(json);
  if (!message) return null;
  const title = crossrefTitle(message);
  const doi = normalizeDoi(asString(message.DOI) ?? "");
  if (!title || !doi) return null;

  const meta: PaperMeta = {
    id: `doi:${doi}`,
    title,
    authors: crossrefAuthors(message),
    doi,
    source: "crossref",
  };
  const rawAbstract = asString(message.abstract);
  const abstract = rawAbstract ? normalizeAbstract(stripJats(rawAbstract), meta.authors) : undefined;
  if (abstract) meta.abstract = abstract;
  const published = ["published", "issued", "published-online", "published-print", "created"]
    .map((key) => crossrefDate(message[key]))
    .find((date) => date !== undefined);
  if (published) {
    meta.published = published;
    meta.year = Number(published.slice(0, 4));
  }
  const event = isRecord(message.event) ? message.event : {};
  const venue = cleanMetaText(firstString(message["container-title"]) ?? asString(event.name));
  if (venue) meta.venue = venue;
  const citations = asNumber(message["is-referenced-by-count"]);
  if (citations !== undefined) meta.citationCount = citations;
  const arxivId = crossrefArxivId(message);
  if (arxivId) meta.arxivId = arxivId;
  const pdfUrl = crossrefPdf(message);
  if (pdfUrl) meta.pdfUrl = pdfUrl;
  const resource =
    isRecord(message.resource) && isRecord(message.resource.primary) ? message.resource.primary : {};
  const landingUrl = asString(resource.URL) ?? asString(message.URL);
  if (landingUrl) meta.landingUrl = landingUrl;
  return meta;
}

// ───────────────────────────── Semantic Scholar ─────────────────────────────

interface ScholarHints {
  arxivId?: string;
  citationCount?: number;
  pdfUrl?: string;
}

function mapSemanticScholar(json: unknown): ScholarHints {
  if (!isRecord(json)) return {};
  const hints: ScholarHints = {};
  const external = isRecord(json.externalIds) ? json.externalIds : {};
  const arxiv = parseArxivId(asString(external.ArXiv) ?? "");
  if (arxiv) hints.arxivId = arxiv.id;
  const citations = asNumber(json.citationCount);
  if (citations !== undefined) hints.citationCount = citations;
  const pdf = isRecord(json.openAccessPdf) ? asString(json.openAccessPdf.url) : undefined;
  // Its "openAccessPdf" is frequently a landing page (e.g. a doi.org link), so only real PDF paths count.
  if (pdf && /\.pdf($|[?#])/i.test(pdf)) hints.pdfUrl = pdf;
  return hints;
}

// ──────────────────────────────── Resolution ────────────────────────────────

function settledValue<T>(result: PromiseSettledResult<T>): T | null {
  return result.status === "fulfilled" ? result.value : null;
}

function settledError<T>(result: PromiseSettledResult<T>): unknown {
  return result.status === "rejected" ? result.reason : undefined;
}

/** Abstract (and arXiv categories) from DataCite when the publisher record has none. */
async function arxivDetails(arxivId: string, signal?: AbortSignal): Promise<PaperMeta | null> {
  try {
    return await fetchArxivMeta(arxivId, undefined, signal);
  } catch (error) {
    if (isAbortError(error)) throw error;
    return null;
  }
}

const titleKey = (title: string) => title.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

/**
 * Finds the arXiv preprint of a published paper by its title. Many venues
 * (ACL, NeurIPS, ICML) publish papers whose authors also posted them on
 * arXiv; registries rarely link the two, but the titles almost always match
 * exactly once case and punctuation are ignored. Only an exact match counts,
 * so a similar-sounding paper is never analysed by mistake.
 */
export async function findArxivByTitle(title: string, signal?: AbortSignal): Promise<string | undefined> {
  const terms = dataCiteTerms(title);
  if (!terms) return undefined;
  try {
    const json = await fetchJson(dataCiteSearchUrl(terms, 10), { signal, retries: 0 });
    const target = titleKey(title);
    const items = isRecord(json) ? asRecords(json.data) : [];
    return items.map(mapDataCite).find((meta) => meta?.arxivId && titleKey(meta.title) === target)?.arxivId;
  } catch (error) {
    if (isAbortError(error)) throw error;
    return undefined;
  }
}

/**
 * Resolves any DOI to PaperMeta. arXiv DOIs go straight to the arXiv adapter;
 * other DOIs are looked up in Crossref, OpenAlex and Semantic Scholar in
 * parallel and merged (Crossref facts first, OpenAlex for gaps and citations).
 */
export async function resolveDoi(doi: string, signal?: AbortSignal): Promise<PaperMeta> {
  const normalized = normalizeDoi(doi);
  if (!normalized) throw new SourceError(`"${doi}" is not a valid DOI`, "invalid");
  const arxivRef = arxivIdFromDoi(normalized);
  if (arxivRef) return fetchArxivMeta(arxivRef.id, arxivRef.version, signal);

  const path = encodeDoiPath(normalized);
  const [crossrefResult, openAlexResult, scholarResult] = await Promise.allSettled([
    fetchJson(`${CROSSREF_API}/works/${path}`, { signal }).then(mapCrossref),
    fetchJson(openAlexWorkUrl(normalized), { signal }).then(mapOpenAlexWork),
    fetchJson(`${SEMANTIC_SCHOLAR_API}/paper/DOI:${path}?fields=externalIds,citationCount,openAccessPdf`, {
      signal,
      retries: 0,
    }).then(mapSemanticScholar),
  ]);
  if (signal?.aborted) throw signal.reason ?? new DOMException("The operation was aborted.", "AbortError");

  const crossref = settledValue(crossrefResult);
  const openAlex = settledValue(openAlexResult);
  const scholar = settledValue(scholarResult) ?? {};
  const primary = crossref ?? openAlex;
  const arxivId =
    crossref?.arxivId ??
    openAlex?.arxivId ??
    scholar.arxivId ??
    (primary?.title ? await findArxivByTitle(primary.title, signal) : undefined);

  if (!primary) {
    // Neither registry knows the DOI, but Semantic Scholar may still map it to an arXiv preprint.
    if (arxivId) return { ...(await fetchArxivMeta(arxivId, undefined, signal)), doi: normalized };
    throw mostInformativeError(
      [settledError(crossrefResult), settledError(openAlexResult)],
      new SourceError(`No metadata found for DOI ${normalized}`, "not-found"),
    );
  }

  let meta = mergeMeta(primary, openAlex);
  meta.doi = normalized;
  // One citation source across the app keeps "impact" comparable between papers.
  const citations = openAlex?.citationCount ?? scholar.citationCount ?? crossref?.citationCount;
  if (citations !== undefined) meta.citationCount = citations;

  if (arxivId) {
    if (!meta.abstract) meta = mergeMeta(meta, await arxivDetails(arxivId, signal));
    return withArxivLinks(meta, arxivId);
  }

  const pdfUrl = openAlex?.pdfUrl ?? scholar.pdfUrl ?? crossref?.pdfUrl;
  if (pdfUrl) meta.pdfUrl = pdfUrl;
  meta.id = `doi:${normalized}`;
  return meta;
}

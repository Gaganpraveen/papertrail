export function arxivId(input) {
  const q = input.trim();
  const match = q.match(
    /^(?:https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\/)?((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?\/\d{7})(?:v\d+)?)(?:\.pdf)?\/?$/i,
  );
  return match?.[1] || null;
}
export async function searchPapers(query, signal) {
  const response = await fetch(
    `https://api.openalex.org/works?search=${encodeURIComponent(query)}&per_page=6&select=id,doi,display_name,publication_year,authorships,primary_location,open_access,cited_by_count`,
    { signal },
  );
  if (!response.ok)
    throw Error(
      response.status === 429
        ? "The public search service is rate-limited. Try again later or upload a PDF."
        : "Paper search is unavailable. Upload a PDF to continue.",
    );
  const body = await response.json();
  return (body.results || []).map((p) => ({
    title: p.display_name,
    year: p.publication_year,
    authors: p.authorships
      ?.slice(0, 3)
      .map((a) => a.author.display_name)
      .join(", "),
    url: p.doi || p.primary_location?.landing_page_url || p.id,
    pdf: p.primary_location?.pdf_url || p.open_access?.oa_url,
    citations: p.cited_by_count,
  }));
}
export function safeUrl(value) {
  try {
    const u = new URL(value);
    return u.protocol === "https:" ? u.href : null;
  } catch {
    return null;
  }
}

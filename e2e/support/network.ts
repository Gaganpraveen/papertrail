/**
 * Deterministic network for end-to-end tests: arXiv PDFs and scholarly APIs
 * are served from recorded fixtures so the suite is fast, offline and stable.
 * Everything else outside the app's own origin is blocked, which also proves
 * the models and runtime are self-hosted.
 */
import type { Page, Route } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const FIXTURES = join(import.meta.dirname, "..", "..", "tests", "fixtures");
const EXTRA_PDF_DIR = process.env.PAPERTRAIL_SAMPLES;

const cors = { "access-control-allow-origin": "*" };

function json(route: Route, file: string | null, status = 200) {
  if (!file || !existsSync(file)) return route.fulfill({ status: 404, headers: cors, body: "{}" });
  return route.fulfill({
    status,
    headers: { ...cors, "content-type": "application/json" },
    body: readFileSync(file),
  });
}

function pdfPath(id: string): string | null {
  const candidates = [join(FIXTURES, "pdfs", `${id}.pdf`)];
  if (EXTRA_PDF_DIR) candidates.push(join(EXTRA_PDF_DIR, `${id}.pdf`));
  return candidates.find((p) => existsSync(p)) ?? null;
}

export interface NetworkLog {
  blocked: string[];
  served: string[];
}

export async function useFixtureNetwork(page: Page): Promise<NetworkLog> {
  const log: NetworkLog = { blocked: [], served: [] };

  // Playwright runs the most recently registered matching route first, so the
  // catch-all goes in first and the specific fixtures below override it.
  // Anything else off-origin (Semantic Scholar, Hugging Face, CDNs) is blocked.
  await page.route(/^https?:\/\/(?!127\.0\.0\.1|localhost)/, (route) => {
    log.blocked.push(route.request().url());
    return route.abort("blockedbyclient");
  });

  await page.route("https://arxiv.org/pdf/**", (route) => {
    const id = new URL(route.request().url()).pathname.replace(/^\/pdf\//, "").replace(/v\d+$/, "").replace(/\.pdf$/, "");
    const file = pdfPath(id);
    log.served.push(route.request().url());
    if (!file) return route.fulfill({ status: 404, headers: cors, body: "not found" });
    return route.fulfill({ status: 200, headers: { ...cors, "content-type": "application/pdf" }, body: readFileSync(file) });
  });

  await page.route("https://api.datacite.org/**", (route) => {
    const url = new URL(route.request().url());
    log.served.push(url.href);
    const single = decodeURIComponent(url.pathname).match(/\/dois\/10\.48550\/arxiv\.(.+)$/i);
    if (single) return json(route, join(FIXTURES, "api", "datacite", `${single[1]}.json`));
    const query = (url.searchParams.get("query") ?? "").toLowerCase();
    const file = query.includes("cache")
      ? "datacite-search-kv-cache.json"
      : query.includes("bert")
        ? "datacite-search-bert.json"
        : "datacite-search-empty.json";
    return json(route, join(FIXTURES, "api", file));
  });

  await page.route("https://api.openalex.org/**", (route) => {
    const url = new URL(route.request().url());
    log.served.push(url.href);
    const filter = (url.searchParams.get("filter") ?? "").toLowerCase();
    if (url.pathname === "/works" && filter.includes("cache")) {
      return json(route, join(FIXTURES, "api", "openalex-search-kv-cache.json"));
    }
    if (decodeURIComponent(url.pathname).toLowerCase() === "/works/doi:10.18653/v1/n19-1423") {
      return json(route, join(FIXTURES, "api", "openalex-doi-bert.json"));
    }
    if (url.pathname === "/works") {
      return route.fulfill({ status: 200, headers: { ...cors, "content-type": "application/json" }, body: '{"meta":{"count":0},"results":[]}' });
    }
    return json(route, null);
  });

  await page.route("https://api.crossref.org/**", (route) => {
    const url = new URL(route.request().url());
    log.served.push(url.href);
    const bert = decodeURIComponent(url.pathname).toLowerCase() === "/works/10.18653/v1/n19-1423";
    return json(route, bert ? join(FIXTURES, "api", "crossref-bert.json") : null);
  });

  return log;
}

/**
 * Smoke tests against the deployed site with the real network: arXiv PDFs,
 * DataCite, OpenAlex and Crossref. They run after each deployment
 * (see .github/workflows/deploy.yml) and are skipped otherwise.
 */
import { expect, test } from "@playwright/test";
import { submit, waitForBriefing, watchErrors } from "./support/app";

test.skip(!process.env.PAPERTRAIL_LIVE_URL, "set PAPERTRAIL_LIVE_URL to run against a deployed site");
test.describe.configure({ retries: 1 });

test("analyses an arXiv paper with the real network", async ({ page }) => {
  test.setTimeout(300_000);
  const errors = watchErrors(page);
  await page.goto("./");
  await submit(page, "2402.02750");
  await waitForBriefing(page, 240_000);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("KIVI");
  await expect(page.getByText("Full text", { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test("resolves a publisher DOI", async ({ page }) => {
  test.setTimeout(300_000);
  await page.goto("./");
  await submit(page, "10.18653/v1/N19-1423");
  await waitForBriefing(page, 240_000);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("BERT");
});

test("searches a topic", async ({ page }) => {
  await page.goto("./");
  await submit(page, "retrieval augmented generation");
  await expect(page.getByRole("button", { name: "Analyze", exact: true }).first()).toBeVisible({ timeout: 60_000 });
});

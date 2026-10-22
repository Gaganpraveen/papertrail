import { expect, test } from "@playwright/test";
import { fileURLToPath } from "node:url";
import { useFixtureNetwork } from "./support/network";
import { shot, submit, waitForBriefing, watchErrors } from "./support/app";

const pdf = (id: string) => fileURLToPath(new URL(`../tests/fixtures/pdfs/${id}.pdf`, import.meta.url));

test.beforeEach(async ({ page }) => {
  await useFixtureNetwork(page);
  await page.goto("./");
});

test("finds papers for a topic and analyses a picked candidate", async ({ page }) => {
  const errors = watchErrors(page);
  await submit(page, "kv cache compression");
  const analyze = page.getByRole("button", { name: "Analyze", exact: true });
  await expect(analyze.first()).toBeVisible({ timeout: 60_000 });
  expect(await analyze.count()).toBeGreaterThan(3);
  await shot(page, "search");

  await analyze.first().click();
  // No PDF fixture for these papers, so the graph falls back to the abstract.
  await waitForBriefing(page);
  await expect(page.getByText(/based on the abstract only/)).toBeVisible();
  expect(errors).toEqual([]);
});

test("resolves a publisher DOI to its arXiv preprint", async ({ page }) => {
  await submit(page, "https://doi.org/10.18653/v1/N19-1423");
  await waitForBriefing(page);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("BERT");
  await expect(page.getByText("arXiv 1810.04805")).toBeVisible();
});

test("analyses an uploaded PDF", async ({ page }) => {
  const errors = watchErrors(page);
  const chooser = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: /Upload PDF/ }).first().click();
  await (await chooser).setFiles(pdf("2201.11903"));
  await waitForBriefing(page);
  await expect(page.getByRole("heading", { level: 1 })).toContainText(/Chain[- ]of[- ]Thought/i);
  await expect(page.getByText("Full text", { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test("explains when an arXiv ID does not exist", async ({ page }) => {
  await submit(page, "2501.99999");
  await expect(page.getByRole("main").getByText(/No arXiv paper found for 2501\.99999/)).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("button", { name: /Retry/ })).toBeVisible();
});

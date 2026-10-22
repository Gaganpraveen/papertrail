import { expect, test } from "@playwright/test";
import { useFixtureNetwork } from "./support/network";
import { shot, watchErrors } from "./support/app";

test("builds a literature survey from the five-paper sample", async ({ page }) => {
  test.setTimeout(900_000);
  const errors = watchErrors(page);
  await useFixtureNetwork(page);
  await page.goto("./");

  await page.getByRole("button", { name: /Survey sample/ }).click();
  const title = page.getByRole("heading", { level: 1 });
  await expect(page.getByRole("heading", { name: "Comparison matrix" })).toBeVisible({ timeout: 840_000 });
  await expect(title).toBeVisible();

  for (const section of ["Literature review", "Semantic map", "Similarity", "Themes", "Timeline", "Citation coupling"]) {
    await expect(page.getByRole("heading", { name: section, exact: true }).first()).toBeVisible();
  }
  // Every claim in the review carries at least one [n] citation.
  await expect(page.getByText(/\[\d+\]/).first()).toBeVisible();
  await shot(page, "survey");
  await shot(page, "survey-full", true);
  expect(errors).toEqual([]);
});

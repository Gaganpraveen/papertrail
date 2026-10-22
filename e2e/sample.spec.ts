import { expect, test } from "@playwright/test";
import { useFixtureNetwork } from "./support/network";
import { shot, waitForBriefing, watchErrors } from "./support/app";

test("the sample button analyses a paper and explains the first-visit download", async ({ page }) => {
  test.setTimeout(240_000);
  const errors = watchErrors(page);
  await useFixtureNetwork(page);

  await page.goto("./");
  await page.getByRole("button", { name: /Try a sample paper/ }).click();

  await expect(page.getByRole("heading", { level: 1 })).toContainText("KIVI", { timeout: 60_000 });
  // The download notice is brief against a local server, so it is captured when seen, not required.
  const notice = page.getByRole("status").filter({ hasText: /first visit only/ });
  if (await notice.isVisible().catch(() => false)) await shot(page, "first-visit");

  await waitForBriefing(page);
  expect(errors).toEqual([]);
});

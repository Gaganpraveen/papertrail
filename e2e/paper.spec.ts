import { expect, test } from "@playwright/test";
import { useFixtureNetwork } from "./support/network";
import { openTab, shot, submit, waitForBriefing, watchErrors } from "./support/app";

test("analyses an arXiv paper and answers questions with evidence", async ({ page }) => {
  test.setTimeout(360_000);
  const errors = watchErrors(page);
  const network = await useFixtureNetwork(page);

  await page.goto("./");
  await expect(page.getByRole("textbox", { name: "Paper or topic" })).toBeVisible();
  await shot(page, "home");

  await test.step("resolves an arXiv URL and builds the briefing", async () => {
    await submit(page, "https://arxiv.org/abs/2402.02750v2");
    await waitForBriefing(page);
    await expect(page.getByRole("heading", { level: 1 })).toContainText("KIVI");
    await expect(page.getByText("Full text", { exact: true })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Problem", exact: true })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Key phrases" })).toBeVisible();
    await shot(page, "hero");
  });

  await test.step("draws the visual analysis", async () => {
    await openTab(page, /Visual analysis/);
    await expect(page.getByRole("heading", { name: "Rhetorical map" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Rigor profile" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Key-phrase network" })).toBeVisible();
    await expect(page.getByRole("list", { name: "Key numbers reported in the paper" })).toBeVisible();
    await shot(page, "visuals");
  });

  await test.step("lists critique findings with a rigor breakdown", async () => {
    await openTab(page, /Critique/);
    await expect(page.getByRole("heading", { name: "Rigor breakdown" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Questions for the authors" })).toBeVisible();
    await shot(page, "critique");
  });

  await test.step("answers from the paper and abstains when it cannot", async () => {
    await openTab(page, /^Ask/);
    const box = page.getByRole("textbox", { name: "Ask a question about this paper" });
    await box.fill("How much does KIVI reduce peak memory usage?");
    await box.press("Enter");
    await expect(page.getByText("Answer from the paper").first()).toBeVisible({ timeout: 90_000 });
    await expect(page.getByText(/2\.6\s*×/).first()).toBeVisible();

    await box.fill("What is the boiling point of water on Mars?");
    await box.press("Enter");
    await expect(page.getByText("Not found in the paper")).toBeVisible({ timeout: 90_000 });
    await shot(page, "ask");
  });

  await test.step("opens the cited page in the evidence drawer", async () => {
    await page.getByRole("button", { name: /Show evidence on page/ }).first().click();
    const drawer = page.getByRole("complementary", { name: "Evidence" });
    await expect(drawer).toBeVisible();
    await expect(drawer.locator("canvas").first()).toBeVisible({ timeout: 30_000 });
    await shot(page, "evidence");
    await page.keyboard.press("Escape");
    await expect(drawer).toBeHidden();
  });

  await test.step("keeps the analysis and answers after a reload", async () => {
    await page.reload();
    await expect(page.getByRole("heading", { level: 1 })).toContainText("KIVI");
    await openTab(page, /^Ask/);
    await expect(page.getByText("Answer from the paper").first()).toBeVisible();
  });

  expect(network.blocked, "every request stays on fixtures or the app's own origin").toEqual([]);
  expect(errors).toEqual([]);
});

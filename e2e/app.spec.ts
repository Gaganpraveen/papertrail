import { expect, test } from "@playwright/test";
import { useFixtureNetwork } from "./support/network";
import { shot, watchErrors } from "./support/app";

test.beforeEach(async ({ page }) => {
  await useFixtureNetwork(page);
});

test("home offers one input, samples and the keyboard shortcut", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("./");
  const box = page.getByRole("textbox", { name: "Paper or topic" });
  await expect(box).toBeVisible();
  await expect(page.getByRole("button", { name: /Survey sample/ })).toBeVisible();
  await expect(page.getByText("How it works")).toBeVisible();

  await page.locator("body").click({ position: { x: 5, y: 450 } });
  await page.keyboard.press("Control+k");
  await expect(box).toBeFocused();
  expect(errors).toEqual([]);
});

test("switches between dark and light themes and remembers the choice", async ({ page }) => {
  await page.goto("./");
  const html = page.locator("html");
  const before = await html.getAttribute("data-theme");
  await page.getByRole("button", { name: /Switch to (light|dark) theme/ }).click();
  const after = await html.getAttribute("data-theme");
  expect(after).not.toBe(before);
  await shot(page, `home-${after}`);
  await page.reload();
  await expect(html).toHaveAttribute("data-theme", after!);
});

test.describe("on a phone", () => {
  test.use({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });

  test("fits the screen and opens the library menu", async ({ page }) => {
    await page.goto("./");
    await expect(page.getByRole("textbox", { name: "Paper or topic" }).first()).toBeVisible();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(0);
    await page.getByRole("button", { name: "Open menu" }).click();
    await expect(page.getByRole("button", { name: "Close menu" })).toBeVisible();
    await shot(page, "mobile-menu");
  });
});

test("ships a strict content security policy", async ({ page }) => {
  await page.goto("./");
  const policy = await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute("content");
  expect(policy).toContain("script-src 'self'");
  expect(policy).toContain("object-src 'none'");
  expect(policy).toContain("frame-src 'none'");
});

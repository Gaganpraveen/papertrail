/**
 * Small helpers shared by the end-to-end specs: submit an input, wait for an
 * analysis to finish, collect console errors and (optionally) save screenshots.
 */
import { expect, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

/** Set PAPERTRAIL_SHOTS=<dir> to save screenshots of each step for review. */
const SHOTS = process.env.PAPERTRAIL_SHOTS;

export async function shot(page: Page, name: string, fullPage = false): Promise<void> {
  if (!SHOTS) return;
  mkdirSync(SHOTS, { recursive: true });
  // Let entry animations finish and keep notifications out of the picture.
  await page.addStyleTag({ content: '[aria-live="polite"] { display: none !important; }' });
  await page.waitForTimeout(600);
  await page.screenshot({ path: join(SHOTS, `${name}.png`), fullPage });
}

/** Records uncaught exceptions and console errors so a spec can assert there were none. */
export function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() !== "error") return;
    const text = message.text();
    // Blocked off-origin requests and fixture 404s are expected; the specs check those via the network log.
    if (/Failed to load resource|ERR_BLOCKED_BY_CLIENT|status of 404/.test(text)) return;
    errors.push(`console: ${text}`);
  });
  return errors;
}

export async function submit(page: Page, input: string): Promise<void> {
  const box = page.getByRole("textbox", { name: "Paper or topic" });
  await box.fill(input);
  await box.press("Enter");
}

/** Waits until the paper view shows a finished briefing. */
export async function waitForBriefing(page: Page, timeout = 180_000): Promise<void> {
  await expect(page.getByText(/why it matters/i).first()).toBeVisible({ timeout });
}

export async function openTab(page: Page, name: RegExp): Promise<void> {
  await page.getByRole("tab", { name }).click();
}

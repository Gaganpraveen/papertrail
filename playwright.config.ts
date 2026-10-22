import { defineConfig, devices } from "@playwright/test";
import { existsSync } from "node:fs";

// Use a preinstalled Chromium when one is available (offline sandboxes),
// otherwise Playwright's own download (CI).
const localChromium = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome";

// PAPERTRAIL_LIVE_URL points the suite at a deployed site (see e2e/live.spec.ts);
// otherwise it runs against a local preview of the production build.
const liveUrl = process.env.PAPERTRAIL_LIVE_URL;

export default defineConfig({
  testDir: "e2e",
  timeout: 240_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: liveUrl ?? "http://127.0.0.1:4317/papertrail/",
    viewport: { width: 1440, height: 900 },
    colorScheme: "dark",
    trace: "retain-on-failure",
    launchOptions: existsSync(localChromium) ? { executablePath: localChromium } : {},
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: liveUrl
    ? undefined
    : {
        command: "npx vite preview --port 4317 --strictPort --host 127.0.0.1",
        url: "http://127.0.0.1:4317/papertrail/",
        reuseExistingServer: false,
        timeout: 60_000,
      },
});

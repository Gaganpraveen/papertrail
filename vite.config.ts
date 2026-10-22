import { createHash } from "node:crypto";
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

/**
 * Adds a Content Security Policy to the built page. Scripts may only come
 * from the site itself (plus the hashed inline theme script), nothing can be
 * framed or posted elsewhere, and network requests must use HTTPS. Requests
 * stay open to any HTTPS host because users can paste a PDF link from
 * anywhere. Build only: the dev server injects its own inline scripts.
 */
function contentSecurityPolicy(): Plugin {
  return {
    name: "papertrail-csp",
    apply: "build",
    transformIndexHtml: {
      order: "post",
      handler(html) {
        const hashes = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)].map(
          ([, body]) => `'sha256-${createHash("sha256").update(body).digest("base64")}'`,
        );
        const policy = [
          "default-src 'self'",
          `script-src 'self' 'wasm-unsafe-eval' ${hashes.join(" ")}`.trim(),
          "style-src 'self' 'unsafe-inline'",
          "img-src 'self' data: blob:",
          "font-src 'self' data:",
          "connect-src 'self' https: blob: data:",
          "worker-src 'self' blob:",
          "object-src 'none'",
          "frame-src 'none'",
          "base-uri 'self'",
          "form-action 'self'",
        ].join("; ");
        return html.replace(
          "<head>",
          `<head>\n    <meta http-equiv="Content-Security-Policy" content="${policy}" />`,
        );
      },
    },
  };
}

// The app is served from https://<user>.github.io/papertrail/, and dev/preview
// use the same base so paths behave identically everywhere.
export default defineConfig({
  base: "/papertrail/",
  plugins: [react(), tailwindcss(), contentSecurityPolicy()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  worker: { format: "es" },
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500,
  },
});

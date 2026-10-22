import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@/styles/index.css";
import App from "@/ui/App";

/**
 * pdf.js 6 relies on the ES2026 "upsert" methods (Map.prototype.getOrInsert
 * and getOrInsertComputed) when rendering pages. Browsers released before
 * them get this spec-equivalent shim; pdf.js itself is loaded lazily, so
 * installing it here runs before any PDF is opened.
 */
function installUpsertShim(): void {
  type Upsert = {
    has(key: unknown): boolean;
    get(key: unknown): unknown;
    set(key: unknown, value: unknown): unknown;
  };
  for (const proto of [Map.prototype, WeakMap.prototype] as object[]) {
    if (!("getOrInsert" in proto)) {
      Object.defineProperty(proto, "getOrInsert", {
        configurable: true,
        writable: true,
        value(this: Upsert, key: unknown, value: unknown) {
          if (!this.has(key)) this.set(key, value);
          return this.get(key);
        },
      });
    }
    if (!("getOrInsertComputed" in proto)) {
      Object.defineProperty(proto, "getOrInsertComputed", {
        configurable: true,
        writable: true,
        value(this: Upsert, key: unknown, compute: (key: unknown) => unknown) {
          if (!this.has(key)) this.set(key, compute(key));
          return this.get(key);
        },
      });
    }
  }
}

installUpsertShim();

const root = document.getElementById("root");
if (!root) throw new Error("Missing #root element");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

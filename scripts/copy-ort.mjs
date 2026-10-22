// Copies the ONNX Runtime WebAssembly binaries next to the app so that
// inference never depends on a third-party CDN at runtime.
import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { join } from "node:path";

const candidates = [
  "node_modules/@huggingface/transformers/node_modules/onnxruntime-web/dist",
  "node_modules/onnxruntime-web/dist",
];
const src = candidates.find((dir) => existsSync(dir));
if (!src) {
  console.warn("[copy-ort] onnxruntime-web not installed yet, skipping");
  process.exit(0);
}
const dest = join("public", "ort");
mkdirSync(dest, { recursive: true });
const files = readdirSync(src).filter((f) => /^ort-wasm.*\.(wasm|mjs)$/.test(f));
for (const file of files) cpSync(join(src, file), join(dest, file));
console.log(`[copy-ort] copied ${files.length} runtime files to public/ort`);

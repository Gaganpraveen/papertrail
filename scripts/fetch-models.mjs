// Downloads the three ONNX models PaperTrail runs in the browser into
// public/models so the deployed site serves them from its own origin.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const MODELS = [
  "Xenova/all-MiniLM-L6-v2", // sentence embeddings (semantic search, zoning, clustering)
  "Xenova/ms-marco-MiniLM-L-6-v2", // cross-encoder reranker
  "Xenova/distilbert-base-cased-distilled-squad", // extractive question answering
];
const FILES = [
  "config.json",
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "onnx/model_quantized.onnx",
];

const root = join("public", "models");
for (const model of MODELS) {
  for (const file of FILES) {
    const target = join(root, model, file);
    if (existsSync(target)) continue;
    const url = `https://huggingface.co/${model}/resolve/main/${file}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} while downloading ${url}`);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, Buffer.from(await res.arrayBuffer()));
    console.log(`fetched ${model}/${file}`);
  }
}
console.log("models ready in public/models");

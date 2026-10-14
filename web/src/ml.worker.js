import { pipeline, env } from "@huggingface/transformers";
import {
  makeChunks,
  prompts,
  selectBriefing,
  answerEvidence,
  measurements,
} from "./analysis.js";
env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1;
let extractor,
  activeDocument = null;
const MODEL = "Xenova/all-MiniLM-L6-v2";
const REVISION = "751bff37182d3f1213fa05d7196b954e230abad9";
function progress(id, stage, detail, ratio) {
  self.postMessage({ id, type: "progress", stage, detail, ratio });
}
async function model(id) {
  if (!extractor)
    extractor = await pipeline("feature-extraction", MODEL, {
      revision: REVISION,
      dtype: "q8",
      device: "wasm",
      progress_callback: (p) => {
        if (p.status === "progress")
          progress(
            id,
            "Loading semantic model",
            `${p.file} · ${Math.round(p.progress)}%`,
            p.progress / 100,
          );
      },
    });
  return extractor;
}
async function embed(texts, id) {
  const pipe = await model(id);
  const rows = [];
  for (let i = 0; i < texts.length; i += 8) {
    const tensor = await pipe(texts.slice(i, i + 8), {
      pooling: "mean",
      normalize: true,
    });
    rows.push(...tensor.tolist());
    progress(
      id,
      "Indexing passages",
      `${Math.min(i + 8, texts.length)} of ${texts.length} passages`,
      Math.min(1, (i + 8) / texts.length),
    );
  }
  return rows;
}
self.onmessage = async ({ data }) => {
  const { id, action } = data;
  try {
    if (action === "analyze") {
      activeDocument = null;
      const chunks = makeChunks(data.pages, data.documentId);
      if (chunks.length < 2)
        throw Error(
          "Not enough readable text. Choose a text-based research PDF; scanned PDFs need OCR.",
        );
      if (chunks.length > 1600)
        throw Error(
          "This document has too much text for the browser edition. Choose a shorter paper.",
        );
      const vectors = await embed(
        chunks.map((c) => c.text),
        id,
      );
      progress(
        id,
        "Preparing briefing",
        "Selecting representative passages by section",
        0.9,
      );
      const pv = await embed(Object.values(prompts), id);
      const brief = selectBriefing(
        chunks,
        vectors,
        Object.fromEntries(Object.keys(prompts).map((k, i) => [k, pv[i]])),
      );
      activeDocument = { id: data.documentId, chunks, vectors };
      self.postMessage({
        id,
        type: "result",
        result: {
          chunks,
          vectors,
          brief,
          measurements: measurements(chunks),
          model: MODEL,
          revision: REVISION,
        },
      });
    } else if (action === "ask") {
      if (activeDocument?.id !== data.documentId)
        activeDocument = {
          id: data.documentId,
          chunks: data.chunks,
          vectors: data.vectors,
        };
      progress(
        id,
        "Retrieving evidence",
        "Matching your question against this document",
      );
      const [qv] = await embed([data.question], id);
      self.postMessage({
        id,
        type: "result",
        result: answerEvidence(
          activeDocument.chunks,
          activeDocument.vectors,
          data.question,
          qv,
        ),
      });
    } else if (action === "dispose") {
      await extractor?.dispose();
      extractor = null;
      activeDocument = null;
    }
  } catch (e) {
    self.postMessage({
      id,
      type: "error",
      error:
        e.message ||
        "Browser model could not run. Check your connection and retry.",
    });
  }
};

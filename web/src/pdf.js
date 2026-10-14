import { readingLines } from "./layout.js";
import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
export async function openPdf(bytes, signal) {
  signal?.throwIfAborted();
  const task = pdfjs.getDocument({
    data: new Uint8Array(bytes),
    isEvalSupported: false,
    stopAtErrors: true,
  });
  let encrypted = false;
  const abort = () => {
    task.destroy();
  };
  signal?.addEventListener("abort", abort, { once: true });
  task.onPassword = () => {
    encrypted = true;
    task.destroy();
  };
  try {
    const result = await task.promise;
    if (encrypted) throw Error("Encrypted PDF");
    return result;
  } catch {
    await task.destroy();
    throw Error(
      "This PDF is damaged, encrypted or unsupported. Choose an unencrypted, text-readable PDF.",
    );
  } finally {
    signal?.removeEventListener("abort", abort);
  }
}
export async function parsePdf(pdf, onProgress, signal) {
  if (pdf.numPages > 100) throw Error("Choose a PDF with 100 pages or fewer.");
  const pages = [];
  let characters = 0;
  for (let n = 1; n <= pdf.numPages; n++) {
    signal.throwIfAborted();
    const page = await pdf.getPage(n);
    const content = await page.getTextContent();
    const lines = readingLines(
      content.items,
      page.getViewport({ scale: 1 }).width,
    );
    characters += lines.join(" ").length;
    pages.push({ number: n, lines, characters: lines.join(" ").length });
    onProgress(n, pdf.numPages);
    page.cleanup();
  }
  if (
    characters < 500 ||
    pages.filter((p) => p.characters >= 80).length / pdf.numPages < 0.5
  )
    throw Error(
      "This PDF has little readable text. Scanned and image-only PDFs require OCR, which is not supported.",
    );
  let title = "";
  try {
    const { info } = await pdf.getMetadata();
    title = info.Title || "";
  } catch {}
  if (!title || /^untitled|^microsoft|\.docx?$/i.test(title)) title = "";
  return { pages, title };
}

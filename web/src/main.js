import "./style.css";
import { openPdf, parsePdf } from "./pdf.js";
import { loadDocuments, saveDocument, removeDocument } from "./storage.js";
import { searchPapers, arxivId, safeUrl } from "./search.js";
import { validateComparison } from "./analysis.js";

const root = document.querySelector("#app");
const escape = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const paths = {
  paper: "M6 3h8l4 4v14H6z M14 3v5h5 M9 12h6 M9 16h6",
  plus: "M12 5v14 M5 12h14",
  search: "M21 21l-5-5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  arrow: "M5 12h14 M13 6l6 6-6 6",
  upload: "M12 16V3 M6 9l6-6 6 6 M4 16v5h16v-5",
  book: "M3 4h7l2 2 2-2h7v15h-7l-2 2-2-2H3z M12 6v15",
  chart: "M4 3v17h17 M8 16v-5 M13 16V6 M18 16V9",
  chat: "M4 4h16v12H9l-5 4z M8 8h8 M8 12h5",
  check: "M5 12l4 4L19 6",
  lock: "M6 10h12v11H6z M8 10V6a4 4 0 0 1 8 0v4",
  close: "M5 5l14 14 M19 5L5 19",
  download: "M12 3v12 M6 9l6 6 6-6 M4 17v4h16v-4",
  chevron: "M9 5l7 7-7 7",
  trash: "M3 6h18 M9 6V3h6v3 M6 6l1 15h10l1-15 M10 10v7 M14 10v7",
  globe:
    "M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0 M3 12h18 M12 3c5 5 5 13 0 18-5-5-5-13 0-18",
  flow: "M4 3h6v5H4z M14 16h6v5h-6z M7 8v9h7 M7 12h10v4",
  clock: "M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0 M12 6v6l4 2",
};
const icon = (name, cls = "") =>
  `<svg class="icon ${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[name] || paths.paper}"/></svg>`;
const uid = () => crypto.randomUUID();
let docs = [],
  current = null,
  tab = "briefing",
  notice = "",
  error = "",
  busy = null,
  worker = null,
  pdf = null,
  renderTask = null,
  readerPage = 1,
  evidence = null,
  searchResults = null,
  searchQuery = "",
  searchBusy = false,
  modal = false,
  selectedMetrics = new Set();
const operations = new Map();
function ensureWorker() {
  if (worker) return worker;
  worker = new Worker(new URL("./ml.worker.js", import.meta.url), {
    type: "module",
  });
  worker.onmessage = ({ data }) => {
    const op = operations.get(data.id);
    if (!op) return;
    if (data.type === "progress") {
      if (busy) {
        busy.stage = data.stage;
        busy.detail = data.detail;
        busy.ratio = data.ratio;
        updateProgress();
      }
    } else {
      operations.delete(data.id);
      data.type === "error"
        ? op.reject(Error(data.error))
        : op.resolve(data.result);
    }
  };
  worker.onerror = () => {
    for (const op of operations.values())
      op.reject(
        Error(
          "The browser model stopped unexpectedly. Retry, or use a browser with more available memory.",
        ),
      );
    operations.clear();
    worker?.terminate();
    worker = null;
  };
  return worker;
}
function runWorker(action, data) {
  const id = uid();
  return new Promise((resolve, reject) => {
    operations.set(id, { resolve, reject });
    ensureWorker().postMessage({ id, action, ...data });
  });
}
function stopWork(
  message = "Processing cancelled. You can choose another file or retry.",
) {
  busy?.controller.abort();
  worker?.terminate();
  worker = null;
  for (const op of operations.values()) op.reject(Error(message));
  operations.clear();
  if (busy?.timer) clearTimeout(busy.timer);
  busy = null;
  error = message;
  render();
}
function begin(stage) {
  error = "";
  notice = "";
  const op = {
    stage,
    detail: "",
    start: performance.now(),
    ratio: 0,
    controller: new AbortController(),
  };
  op.timer = setTimeout(
    () =>
      stopWork(
        "Processing reached the 3-minute limit and was stopped. Retry with a shorter PDF or a faster connection.",
      ),
    180000,
  );
  busy = op;
  render();
  return op;
}
function finish(op) {
  clearTimeout(op.timer);
  if (busy === op) busy = null;
}
function updateProgress() {
  const el = document.querySelector("#progress");
  if (!el || !busy) return;
  const elapsed = ((performance.now() - busy.start) / 1000).toFixed(0);
  el.innerHTML = `<div class="progress-copy"><span class="spinner"></span><div><strong>${escape(busy.stage)}</strong><span>${escape(busy.detail)}</span></div><span class="elapsed">${elapsed}s</span><button class="icon-button" id="cancel" aria-label="Cancel processing">${icon("close")}</button></div><div class="track"><i style="width:${Math.max(3, (busy.ratio || 0.1) * 100)}%"></i></div>`;
  el.querySelector("#cancel").onclick = () => stopWork();
}
setInterval(() => {
  if (busy) updateProgress();
}, 1000);

function shell(content) {
  return `<div class="shell"><aside class="rail"><button class="brand" data-home aria-label="PaperTrail home">${icon("paper")}<span>papertrail<span class="brand-dot">.</span></span></button><button class="new-paper" data-new ${busy ? "disabled" : ""}>${icon("plus")} New paper</button><div class="rail-label">YOUR WORKSPACE <span>${docs.length}</span></div><nav class="document-list" aria-label="Your papers">${docs.length ? docs.map((d) => `<button class="document ${current?.id === d.id ? "selected" : ""}" data-doc="${d.id}" ${busy ? "disabled" : ""}>${icon("paper")}<span>${escape(d.title)}<small>${d.pages.length} pages · ${new Date(d.created).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</small></span></button>`).join("") : '<p class="library-empty">Your papers will appear here.<br>Saved only in this browser.</p>'}</nav><div class="rail-bottom"><button class="plain" data-about>${icon("flow")} How it works</button><div class="privacy">${icon("lock")}<span>Private by design<small>PDFs stay on this device</small></span></div></div></aside><div class="surface"><header class="topbar"><div class="breadcrumb">Workspace ${current ? `${icon("chevron")} <span>${escape(current.filename)}</span>` : ""}</div><span class="local-badge"><i></i> Browser edition</span></header><main>${content}</main><footer><span>Read closely. Follow the evidence.</span><button class="plain" data-about>Methods & limitations ${icon("arrow")}</button></footer></div></div>${modal ? about() : ""}`;
}
function status() {
  return `${error ? `<div role="alert" class="message error">${icon("close")}<span>${escape(error)}</span><button class="plain" data-dismiss>Dismiss</button></div>` : ""}${notice ? `<div role="status" class="message">${icon("check")}<span>${escape(notice)}</span></div>` : ""}${busy ? '<div id="progress" role="status" aria-live="polite"></div>' : ""}`;
}
function home() {
  return `<div class="home"><div class="eyebrow"><span></span> THE RESEARCH READING ROOM</div><h1>From paper to<br><em>perspective.</em></h1><p class="lede">Find the argument. Examine the results.<br>Ask better questions, with the evidence beside you.</p>${status()}<div class="entry-grid"><section id="dropzone" class="dropzone" aria-label="PDF drop zone"><div class="upload-symbol">${icon("upload")}</div><h2>Bring your own paper</h2><p>Drop a research PDF here to start reading.</p><button class="button primary" data-new ${busy ? "disabled" : ""}>Choose PDF ${icon("arrow")}</button><small>Text-based PDF · up to 30 MB · 100 pages</small><button class="plain sample-button" id="sample" ${busy ? "disabled" : ""}>Try a sample paper ↗</button><div class="upload-privacy">${icon("lock")} Processed on your device, never uploaded.</div></section><section class="search-entry"><span class="eyebrow">OR START WITH A SEARCH</span><h2>Follow your curiosity.</h2><p>Find a paper by title or topic, or open an exact arXiv ID.</p><form id="searchForm"><label class="sr-only" for="searchInput">Paper title, topic or arXiv ID</label><div class="searchbox">${icon("search")}<input id="searchInput" placeholder="e.g. efficient attention mechanisms" value="${escape(searchQuery)}" required maxlength="250"><button aria-label="Search papers" ${searchBusy ? "disabled" : ""}>${icon("arrow")}</button></div></form><div class="suggestions"><button data-query="retrieval augmented generation">Retrieval augmented generation</button><button data-query="1706.03762">1706.03762</button></div><small>Search sends your query to OpenAlex. Some publishers require downloading the PDF before importing it.</small></section></div>${searchPanel()}<div class="steps"><article><span>01 / READ</span><h3>A briefing with receipts</h3><p>Key passages organised into the problem, method, results and limitations.</p></article><article><span>02 / QUESTION</span><h3>Keep the source in sight</h3><p>Find relevant evidence and inspect the original page in one click.</p></article><article><span>03 / EXAMINE</span><h3>Look beyond the headline</h3><p>Explore reported measurements, context and related research.</p></article></div><div class="edition-note">${icon("flow")} <p><strong>Small models. Transparent methods.</strong> This edition uses neural retrieval and extractive summaries, not a generative chatbot. The first analysis downloads a small embedding model; later runs reuse its browser cache.</p></div></div>`;
}
function searchPanel() {
  if (searchBusy)
    return '<div class="search-results"><span class="spinner"></span> Searching the research index…</div>';
  if (!searchResults) return "";
  return `<section class="search-results"><div class="section-top"><h3>${searchResults.length ? "Choose a paper" : "No matching papers found"}</h3><span>Metadata search · not a full-text review</span></div>${searchResults.map((p, i) => `<article><div><span class="eyebrow">${escape(p.year || "Date unavailable")}</span><h3>${escape(p.title)}</h3><p>${escape(p.authors || "Authors unavailable")}</p></div><div class="result-actions">${safeUrl(p.url) ? `<a class="button secondary" href="${escape(safeUrl(p.url))}" target="_blank" rel="noopener noreferrer">Open source ↗</a>` : ""}${safeUrl(p.pdf) ? `<button class="button primary" data-import="${i}" ${busy ? "disabled" : ""}>Import PDF</button>` : ""}</div></article>`).join("")}</section>`;
}
function cite(c, label = "View evidence") {
  return `<button class="citation" data-cite="${escape(c.chunkId)}" data-quote="${escape(c.text || c.quote)}">${icon("paper")} p. ${c.page} <span>${label}</span> ↗</button>`;
}
function briefing() {
  return `<div class="reading-intro"><span class="eyebrow">THE PAPER, IN PERSPECTIVE</span><h2>Start with the essentials.</h2><p>Selected source passages, organised for a first reading. Wording is preserved; this is an extractive briefing.</p></div>${Object.entries(
    current.brief,
  )
    .map(
      ([section, claims], i) =>
        `<section class="brief-section" id="section-${section}"><div class="section-number">0${i + 1}</div><div><h3>${section === "Overview" ? "At a glance" : section === "Problem" ? "The question" : section === "Method" ? "The approach" : section === "Results" ? "What the paper reports" : "Limitations & open questions"}</h3>${claims.length ? claims.map((c) => `<div class="claim"><p>${escape(c.text)}</p>${cite(c)}</div>`).join("") : '<p class="muted">No suitable source passage was identified for this section. Inspect the PDF for information the retriever may have missed.</p>'}</div></section>`,
    )
    .join(
      "",
    )}<div class="next-action"><div><strong>There’s more in the details.</strong><p>Ask a focused question and follow its evidence.</p></div><button class="button secondary" data-tab="chat">Ask this paper ${icon("arrow")}</button></div>`;
}
function chat() {
  return `<div class="reading-intro"><span class="eyebrow">FOLLOW THE EVIDENCE</span><h2>Ask the paper.</h2><p>Questions search only this document. Responses show matching source passages, not generated conclusions.</p></div><div class="conversation">${(current.exchanges || []).map((x) => `<article class="exchange"><div class="question"><span>YOU</span><p>${escape(x.question)}</p></div><div class="answer"><span class="answer-status">${x.answer.status === "evidence_found" ? "RETRIEVED EVIDENCE" : "INSUFFICIENT EVIDENCE"} <small>${x.seconds}s</small></span><p class="muted">${escape(x.answer.explanation)}</p>${x.answer.evidence.map((c) => `<blockquote>${escape(c.text)}</blockquote>${cite(c)}`).join("")}</div></article>`).join("") || `<div class="chat-empty">${icon("chat")}<h3>A good question is a good start.</h3><p>Ask about a method, result or limitation.<br>Check every quotation in its original context.</p></div>`}</div><div class="question-suggestions">${["What method does this paper propose?", "What results does the paper report?", "What limitations are discussed?"].map((q) => `<button data-question="${escape(q)}">${escape(q)}</button>`).join("")}</div><form id="askForm" class="ask-form"><label class="sr-only" for="question">Ask about this document</label><textarea id="question" rows="2" maxlength="700" placeholder="What would you like to understand?" required ${busy ? "disabled" : ""}></textarea><button class="button primary" aria-label="Ask question" ${busy ? "disabled" : ""}>${icon("arrow")}</button></form><small class="muted">Retrieval can miss evidence. A missing answer is not proof that the paper does not contain one.</small>`;
}
function findings() {
  const ms = current.measurements;
  const selected = ms.filter((m) => selectedMetrics.has(m.id));
  const issue = validateComparison(selected);
  return `<div class="reading-intro"><span class="eyebrow">EXAMINE THE CLAIMS</span><h2>Numbers need context.</h2><p>Measurements extracted from prose. They are reported values, not independently reproduced results.</p></div><div class="chart-card"><div class="section-top"><div><h3>Document map</h3><p class="muted">Extracted text by page · not a quality score</p></div><span>${current.pages.length} pages</span></div><div class="page-chart" role="img" aria-label="Bar chart showing readable text characters on each PDF page">${current.pages.map((p) => `<button data-page="${p.number}" style="--height:${Math.max(2, (p.characters / Math.max(...current.pages.map((x) => x.characters))) * 100)}%" title="Page ${p.number}: ${p.characters} text characters" aria-label="Open page ${p.number}, ${p.characters} extracted characters"><i></i><span>${p.number}</span></button>`).join("")}</div></div><section class="measurements"><div class="section-top"><h3>Reported measurements <span class="count">${ms.length}</span></h3></div><p class="muted">Select 2–6 values with the same unit to inspect side by side. Check the datasets, methods and evaluation conditions in each quotation first.</p>${selected.length ? `<div class="comparison"><strong>${issue ? "Comparison needs attention" : "Selected values · source context required"}</strong>${issue ? `<p>${escape(issue)}</p>` : `<p>Equal units do not imply equal experimental conditions. This chart is a reading aid, not a ranking.</p>${selected.map((m, i) => `<div class="metric-bar"><span>${i + 1}. Page ${m.page}</span><div><i style="width:${Math.max(1, (m.value / Math.max(...selected.map((x) => x.value), 1)) * 100)}%"></i></div><strong>${m.value} ${escape(m.unit)}</strong></div>`).join("")}`}</div>` : ""}${ms.length ? ms.map((m) => `<article class="measurement"><label><input type="checkbox" data-metric="${m.id}" ${selectedMetrics.has(m.id) ? "checked" : ""}><span class="metric-value">${escape(m.label)}</span></label><div><p>${escape(m.quote)}</p>${cite(m)}</div></article>`).join("") : '<div class="empty-note">No supported numerical measurements were found in the extracted prose. Tables and charts are not reconstructed automatically.</div>'}</section>`;
}
function related() {
  return `<div class="reading-intro"><span class="eyebrow">WIDEN THE READING LIST</span><h2>Put it in context.</h2><p>Search the scholarly index for related work. Metadata and citation counts help discovery; they do not verify a paper’s claims.</p></div><form id="searchForm"><label for="searchInput">Related topic or paper title</label><div class="searchbox">${icon("search")}<input id="searchInput" value="${escape(searchQuery || current.title)}" required maxlength="250"><button aria-label="Search papers">${icon("arrow")}</button></div></form><small class="muted">Only this query is sent to OpenAlex. Your PDF and its text are not sent.</small>${searchPanel()}<div class="empty-note">To compare another paper’s findings, download and import its PDF. PaperTrail does not invent benchmark values from abstracts or assume different experiments are comparable.</div>`;
}
function reader() {
  return `<aside class="reader-panel"><div class="reader-top"><span>${icon("book")} SOURCE DOCUMENT</span><div><button id="previousPage" class="icon-button" aria-label="Previous page" ${readerPage <= 1 ? "disabled" : ""}>‹</button><span>${readerPage} / ${current.pages.length}</span><button id="nextPage" class="icon-button" aria-label="Next page" ${readerPage >= current.pages.length ? "disabled" : ""}>›</button></div></div><div class="canvas-wrap"><canvas id="pdfCanvas" aria-label="Original PDF page ${readerPage}"></canvas><p id="renderError" class="muted"></p></div>${evidence ? `<div class="evidence-panel"><span class="eyebrow">SELECTED EVIDENCE · PAGE ${evidence.page}</span><blockquote>${escape(evidence.text)}</blockquote><details><summary>Full retrieved passage</summary><p>${escape(evidence.context)}</p></details></div>` : `<div class="reader-hint">${icon("paper")} Click a page citation to inspect its source here.</div>`}<details class="page-text"><summary>Read extracted page text</summary><p>${escape(current.pages[readerPage - 1]?.lines.join(" "))}</p></details></aside>`;
}
function workspace() {
  return `<div class="workspace"><div class="paper-heading"><div><span class="eyebrow">RESEARCH BRIEFING</span><h1>${escape(current.title)}</h1><p>${current.metadataTitle ? "PDF title metadata" : "Filename used as title"} · ${current.pages.length} pages · ${current.chunks.length} passages <span>· ${current.seconds}s analysis</span></p><p class="source-meta">${current.source ? `Source: <a href="${escape(safeUrl(current.source.url))}" target="_blank" rel="noopener noreferrer">${escape(current.source.title)}</a> · ${escape(current.source.authors || "Authors unavailable")} · ${escape(current.source.year || "Date unavailable")} ${current.source.license ? "· " + escape(current.source.license) : ""}` : "Uploaded PDF · authors and publication date not inferred"} · Document ${current.id.slice(0, 12)}</p></div><div class="paper-actions"><button class="button secondary" id="rebuild" title="Run a new analysis of this saved PDF; replaces its previous analysis and questions" ${busy ? "disabled" : ""}>Rebuild analysis</button><button class="icon-button" id="export" aria-label="Download analysis report" title="Download report">${icon("download")}</button><button class="icon-button" id="delete" aria-label="Delete this paper from this browser" title="Delete local paper" ${busy ? "disabled" : ""}>${icon("trash")}</button></div></div>${status()}<nav class="tabs" aria-label="Paper views">${[
    ["briefing", "book", "Briefing"],
    ["findings", "chart", "Findings"],
    ["chat", "chat", "Ask the paper"],
    ["related", "globe", "Related work"],
  ]
    .map(
      ([id, ico, label]) =>
        `<button data-tab="${id}" class="${tab === id ? "active" : ""}" aria-current="${tab === id ? "page" : "false"}">${icon(ico)}${label}</button>`,
    )
    .join(
      "",
    )}</nav><div class="workspace-grid"><div class="reading-pane">${tab === "briefing" ? briefing() : tab === "findings" ? findings() : tab === "chat" ? chat() : related()}</div>${reader()}</div></div>`;
}
function about() {
  return `<div class="modal-backdrop"><section class="modal" role="dialog" aria-modal="true" aria-labelledby="aboutTitle"><button class="icon-button modal-close" data-close aria-label="Close methods">${icon("close")}</button><span class="eyebrow">A LOOK UNDER THE HOOD</span><h2 id="aboutTitle">A trail you can explain.</h2><p>This browser edition runs the analysis on your device. It does not call a paid LLM or train a model on your document.</p><div class="logic-map">${[
    ["PDF.js", "Extract text + page numbers"],
    ["MiniLM embeddings", "Encode passage meaning"],
    ["Hybrid retrieval", "BM25 + cosine similarity"],
    ["Rank fusion", "Combine retrieval rankings"],
    ["Evidence selection", "Choose original quotations"],
    ["Inspect + question", "Open the original PDF page"],
  ]
    .map(
      ([a, b], i) =>
        `<div><span>0${i + 1}</span><strong>${a}</strong><small>${b}</small></div>`,
    )
    .join(
      "",
    )}</div><h3>What the output means</h3><p>Briefings are selected source sentences. Answers are relevant passages, not generated explanations. Citation checks establish provenance, not scientific correctness. False matches and missed evidence remain possible.</p><h3>Privacy & external services</h3><p>PDFs, indexes and chat history are stored in this browser’s IndexedDB. Delete a paper to remove its local record. Model files download from Hugging Face and are cached by your browser. Searches send your query to OpenAlex; source links contact their publishers. Interface fonts are served by Google Fonts. No analytics or API keys.</p><h3>Boundaries</h3><p>English, text-readable PDFs work best. Scans, encryption, dense tables and complex layouts may fail or lose information. Charts show extracted values, not verified benchmarks. The existing Python edition remains available for local generative RAG.</p><button class="button primary" data-close>Back to reading ${icon("arrow")}</button></section></div>`;
}
function render() {
  root.innerHTML = shell(current ? workspace() : home());
  bind();
  if (busy) updateProgress();
  if (current) drawPage();
  if (modal) document.querySelector("[data-close]")?.focus();
}
async function loadSample() {
  if (busy) return;
  const op = begin("Opening sample PDF");
  try {
    const response = await fetch(
      `${import.meta.env.BASE_URL}samples/active-retrieval.pdf`,
      { signal: op.controller.signal },
    );
    if (!response.ok)
      throw Error(
        "The sample could not be downloaded. Please choose a PDF instead.",
      );
    const bytes = await response.arrayBuffer();
    op.controller.signal.throwIfAborted();
    finish(op);
    await processFile(
      new File([bytes], "Active Retrieval Augmented Generation.pdf", {
        type: "application/pdf",
      }),
      true,
      {
        title: "Active Retrieval Augmented Generation",
        authors: "Jiang et al.",
        year: 2023,
        url: "https://aclanthology.org/2023.emnlp-main.495/",
        license: "CC BY 4.0",
      },
    );
  } catch (e) {
    if (busy === op) error = e.message;
    finish(op);
    render();
  }
}
function bind() {
  document.querySelector("#sample")?.addEventListener("click", loadSample);
  document.querySelectorAll("[data-new]").forEach(
    (b) =>
      (b.onclick = () => {
        if (!busy) document.querySelector("#pdfFile").click();
      }),
  );
  document.querySelector("#pdfFile").onchange = (e) => {
    if (e.target.files[0]) processFile(e.target.files[0]);
  };
  document.querySelector("[data-home]").onclick = () => {
    if (busy) return;
    current = null;
    error = "";
    notice = "";
    render();
  };
  document
    .querySelectorAll("[data-doc]")
    .forEach((b) => (b.onclick = () => selectDocument(b.dataset.doc)));
  document.querySelectorAll("[data-tab]").forEach(
    (b) =>
      (b.onclick = () => {
        tab = b.dataset.tab;
        render();
      }),
  );
  document.querySelectorAll("[data-about]").forEach(
    (b) =>
      (b.onclick = () => {
        modal = true;
        render();
      }),
  );
  document.querySelectorAll("[data-close]").forEach(
    (b) =>
      (b.onclick = () => {
        modal = false;
        render();
      }),
  );
  document.querySelector("[data-dismiss]")?.addEventListener("click", () => {
    error = "";
    render();
  });
  const drop = document.querySelector("#dropzone");
  if (drop) {
    for (const ev of ["dragenter", "dragover"])
      drop.addEventListener(ev, (e) => {
        e.preventDefault();
        drop.classList.add("dragging");
      });
    for (const ev of ["dragleave", "drop"])
      drop.addEventListener(ev, (e) => {
        e.preventDefault();
        drop.classList.remove("dragging");
      });
    drop.addEventListener("drop", (e) => {
      if (busy) return;
      if (e.dataTransfer.files.length !== 1) {
        error = "Drop one PDF at a time.";
        render();
        return;
      }
      processFile(e.dataTransfer.files[0]);
    });
  }
  document.querySelector("#searchForm")?.addEventListener("submit", (e) => {
    e.preventDefault();
    search(document.querySelector("#searchInput").value);
  });
  document
    .querySelectorAll("[data-query]")
    .forEach((b) => (b.onclick = () => search(b.dataset.query)));
  document
    .querySelectorAll("[data-import]")
    .forEach(
      (b) =>
        (b.onclick = () => importPdf(searchResults[Number(b.dataset.import)])),
    );
  document.querySelectorAll("[data-cite]").forEach(
    (b) =>
      (b.onclick = () => {
        const chunk = current.chunks.find((c) => c.id === b.dataset.cite);
        if (!chunk) return;
        readerPage = chunk.page;
        evidence = {
          page: chunk.page,
          text: b.dataset.quote,
          context: chunk.text,
        };
        render();
        document
          .querySelector(".reader-panel")
          ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }),
  );
  document.querySelectorAll("[data-page]").forEach(
    (b) =>
      (b.onclick = () => {
        readerPage = Number(b.dataset.page);
        evidence = null;
        render();
      }),
  );
  document.querySelector("#previousPage")?.addEventListener("click", () => {
    readerPage--;
    evidence = null;
    render();
  });
  document.querySelector("#nextPage")?.addEventListener("click", () => {
    readerPage++;
    evidence = null;
    render();
  });
  document.querySelectorAll("[data-metric]").forEach(
    (b) =>
      (b.onchange = () => {
        b.checked
          ? selectedMetrics.add(b.dataset.metric)
          : selectedMetrics.delete(b.dataset.metric);
        render();
      }),
  );
  document.querySelectorAll("[data-question]").forEach(
    (b) =>
      (b.onclick = () => {
        document.querySelector("#question").value = b.dataset.question;
        document.querySelector("#question").focus();
      }),
  );
  document.querySelector("#askForm")?.addEventListener("submit", (e) => {
    e.preventDefault();
    ask(document.querySelector("#question").value);
  });
  document.querySelector("#export")?.addEventListener("click", exportReport);
  document.querySelector("#rebuild")?.addEventListener("click", () => {
    const d = current;
    processFile(
      new File([d.bytes], d.filename, { type: "application/pdf" }),
      true,
      d.source,
    );
  });
  document.querySelector("#delete")?.addEventListener("click", async () => {
    if (
      !confirm(
        "Remove this paper and its questions from this browser? This cannot be undone.",
      )
    )
      return;
    const id = current.id;
    await removeDocument(id);
    docs = docs.filter((d) => d.id !== id);
    current = null;
    evidence = null;
    await pdf?.loadingTask.destroy();
    pdf = null;
    notice = "Paper, extracted text and questions removed from this browser.";
    render();
  });
}
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && modal) {
    modal = false;
    render();
  }
});
async function processFile(file, force = false, source = null) {
  if (busy) return;
  if (
    !file.name.toLowerCase().endsWith(".pdf") ||
    file.size === 0 ||
    file.size > 30 * 1024 * 1024
  ) {
    error = "Choose a non-empty PDF up to 30 MB.";
    render();
    return;
  }
  current = null;
  selectedMetrics.clear();
  evidence = null;
  const op = begin("Opening PDF");
  let operationPdf = null;
  try {
    const bytes = await file.arrayBuffer();
    op.controller.signal.throwIfAborted();
    if (new TextDecoder().decode(bytes.slice(0, 5)) !== "%PDF-")
      throw Error("This file is not a valid PDF. Choose another file.");
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    const id = Array.from(new Uint8Array(digest), (b) =>
      b.toString(16).padStart(2, "0"),
    ).join("");
    const existing = docs.find((d) => d.id === id && d.analysisVersion === 3);
    if (existing && !force) {
      finish(op);
      await selectDocument(id);
      notice =
        "Reopened this document’s saved browser analysis. No new model inference was run.";
      render();
      return;
    }
    op.controller.signal.throwIfAborted();
    await pdf?.loadingTask.destroy();
    operationPdf = await openPdf(bytes.slice(0), op.controller.signal);
    op.controller.signal.throwIfAborted();
    pdf = operationPdf;
    op.controller.signal.addEventListener(
      "abort",
      () => operationPdf?.loadingTask.destroy(),
      { once: true },
    );
    op.stage = "Reading PDF";
    op.detail = file.name;
    updateProgress();
    const { pages, title } = await parsePdf(
      pdf,
      (n, total) => {
        op.detail = `${file.name} · page ${n} of ${total}`;
        op.ratio = n / total;
        updateProgress();
      },
      op.controller.signal,
    );
    const result = await runWorker("analyze", { pages, documentId: id });
    op.controller.signal.throwIfAborted();
    const doc = {
      id,
      analysisVersion: 3,
      source,
      filename: file.name,
      title: title || file.name.replace(/\.pdf$/i, "").replace(/[_-]+/g, " "),
      metadataTitle: !!title,
      pages,
      bytes,
      ...result,
      exchanges: [],
      created: Date.now(),
      seconds: ((performance.now() - op.start) / 1000).toFixed(3),
    };
    try {
      await saveDocument(doc);
    } catch {
      notice =
        "Analysis is ready, but browser storage is full or unavailable. This paper will not survive a refresh.";
    }
    docs = docs.filter((d) => d.id !== doc.id);
    docs.unshift(doc);
    current = doc;
    readerPage = 1;
    tab = "briefing";
  } catch (e) {
    if (busy === op) error = e.message;
    await operationPdf?.loadingTask.destroy();
    if (pdf === operationPdf) pdf = null;
  } finally {
    finish(op);
    render();
  }
}
async function selectDocument(id) {
  if (busy) return;
  const d = docs.find((x) => x.id === id);
  if (!d) return;
  await pdf?.loadingTask.destroy();
  pdf = null;
  current = d;
  tab = "briefing";
  readerPage = 1;
  evidence = null;
  selectedMetrics.clear();
  error = "";
  notice = "";
  try {
    pdf = await openPdf(d.bytes.slice(0));
  } catch (e) {
    error = e.message;
  }
  render();
}
let drawVersion = 0;
async function drawPage() {
  const version = ++drawVersion;
  if (!pdf) return;
  renderTask?.cancel();
  const canvas = document.querySelector("#pdfCanvas");
  if (!canvas) return;
  try {
    const page = await pdf.getPage(readerPage);
    if (version !== drawVersion) return;
    const width = canvas.parentElement.clientWidth - 24;
    const viewport = page.getViewport({
      scale:
        Math.max(1, width / page.getViewport({ scale: 1 }).width) *
        Math.min(2, devicePixelRatio),
    });
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    canvas.style.width = "100%";
    renderTask = page.render({
      canvasContext: canvas.getContext("2d"),
      viewport,
    });
    await renderTask.promise;
  } catch (e) {
    if (e.name !== "RenderingCancelledException" && version === drawVersion) {
      const el = document.querySelector("#renderError");
      if (el)
        el.textContent =
          "Page rendering is unavailable. The extracted page text is available below.";
    }
  }
}
async function ask(question) {
  if (!current || busy || !question.trim()) return;
  const doc = current;
  const op = begin("Retrieving evidence");
  try {
    const answer = await runWorker("ask", {
      documentId: doc.id,
      chunks: doc.chunks,
      vectors: doc.vectors,
      question: question.trim(),
    });
    op.controller.signal.throwIfAborted();
    doc.exchanges.push({
      question: question.trim(),
      answer,
      seconds: ((performance.now() - op.start) / 1000).toFixed(3),
    });
    await saveDocument(doc);
  } catch (e) {
    if (busy === op) error = e.message;
  } finally {
    finish(op);
    render();
    document
      .querySelector("#askForm")
      ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}
let searchVersion = 0;
async function search(query) {
  const version = ++searchVersion;
  searchQuery = query.trim();
  error = "";
  notice = "";
  const id = arxivId(searchQuery);
  if (id) {
    searchBusy = false;
    searchResults = [
      {
        title: `arXiv ${id}`,
        authors: "Exact requested identifier; metadata not inferred.",
        url: `https://arxiv.org/abs/${id}`,
        pdf: `https://arxiv.org/pdf/${id}`,
      },
    ];
    render();
    return;
  }
  searchBusy = true;
  render();
  try {
    const found = await searchPapers(searchQuery, AbortSignal.timeout(20000));
    if (version === searchVersion) searchResults = found;
  } catch (e) {
    if (version !== searchVersion) return;
    error =
      e.name === "TimeoutError"
        ? "Paper search timed out. Retry or upload a PDF."
        : e.message;
    searchResults = null;
  } finally {
    if (version === searchVersion) {
      searchBusy = false;
      render();
    }
  }
}
async function importPdf(paper) {
  if (busy) return;
  const url = safeUrl(paper.pdf);
  if (!url) return;
  const op = begin("Downloading selected PDF");
  try {
    const response = await fetch(url, { signal: op.controller.signal });
    if (!response.ok) throw Error("Publisher download failed.");
    const declared = Number(response.headers.get("content-length"));
    if (declared > 30 * 1024 * 1024)
      throw Error("The remote PDF exceeds 30 MB.");
    const reader = response.body.getReader();
    let size = 0;
    const chunks = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.length;
      if (size > 30 * 1024 * 1024) {
        await reader.cancel();
        throw Error("The remote PDF exceeds 30 MB.");
      }
      chunks.push(value);
    }
    finish(op);
    await processFile(
      new File(chunks, [...paper.title].slice(0, 160).join("") + ".pdf", {
        type: "application/pdf",
      }),
      false,
      paper,
    );
  } catch (e) {
    if (busy === op)
      error = `The publisher could not provide this PDF to the browser. Open the source, download the requested PDF, then choose it here. No other paper was substituted. (${e.name === "TypeError" ? "Cross-origin download blocked" : e.message})`;
    finish(op);
    render();
  }
}
function exportReport() {
  const doc = current;
  const text = [
    `# ${doc.title}`,
    "",
    "Extractive browser analysis. Quotations are selected source text, not independent verification.",
    `Document SHA-256: ${doc.id}`,
    `Model: ${doc.model}`,
    `Pages: ${doc.pages.length}`,
    "",
    ...Object.entries(doc.brief).flatMap(([s, cs]) => [
      `## ${s}`,
      ...cs.flatMap((c) => [
        `> ${c.text}`,
        `PDF page ${c.page} · passage ${c.chunkId}`,
        "",
      ]),
    ]),
    "## Questions",
    ...doc.exchanges.flatMap((x) => [
      `### ${x.question}`,
      x.answer.explanation,
      ...x.answer.evidence.map((e) => `> ${e.text}\n\nPDF page ${e.page}`),
      "",
    ]),
  ].join("\n");
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "papertrail-analysis.md";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
render();
loadDocuments()
  .then((saved) => {
    docs = saved.sort((a, b) => b.created - a.created);
    render();
  })
  .catch(() => {
    notice =
      "Browser storage is unavailable. You can still analyze a PDF for this visit.";
    render();
  });
window.addEventListener("pagehide", () => {
  worker?.terminate();
  pdf?.loadingTask.destroy();
});

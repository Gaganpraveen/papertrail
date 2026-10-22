# Design decisions and tradeoffs

This page explains the main choices I made while building PaperTrail, what each
one cost, and what I would change with more time.

## 1. Everything runs in the browser

**Decision.** PaperTrail is a static site. Parsing, embeddings, retrieval, question
answering, critique and the literature survey all run in the visitor's browser.
There is no server, no LLM, and no API key.

**Why.**
- **Free and easy to run.** The brief asks for something anyone can run for free.
  Here the only setup is opening a URL.
- **Private.** Papers never leave the visitor's machine.
- **Every number can be explained.** A generative model can't reliably say why it
  produced a sentence. Each output here comes from an algorithm I can walk through.

**Cost.**
- **No abstractive summaries.** The briefing is extractive: it selects and arranges
  the paper's own sentences instead of writing new prose.
- **Small models.** The models are small: about 110 MB in total, quantised to 8 bits.
- **First visit is slower.** It downloads those models once; the browser caches them
  after that.

## 2. An explicit state graph instead of a single prompt chain

**Decision.** I wrote a small graph runtime (`src/agent/graph.ts`) with:
- typed shared state;
- nodes that return partial updates;
- conditional edges and per-node retries;
- trace events for every step.

The paper pipeline, the topic search and the QA loop are each a graph built on it.

**Why.**
- **Failure handling is part of the design.** These cases are edges in the graph,
  not `try/catch` blocks scattered through the code:
  - a PDF that can't be downloaded;
  - a scanned PDF with no text layer;
  - a topic search that returns nothing.

  A model that fails to load is handled inside the node that needed it: the run
  continues on a weaker path (keyword retrieval, cue-phrase zoning) and records
  a warning.
- **The graph can be shown.** The trace dock draws it live.

**Tradeoff vs. LangGraph.** LangGraph is Python, and this app runs in a browser.
The runtime is about 200 lines, follows the same ideas (nodes, edges, conditional
routing, checkpoints via `onState`), and has no dependencies.

## 3. Hybrid retrieval with a reranker

**Decision.** Chunks follow section boundaries, about 170 words each, with one
sentence of overlap. They are searched two ways:
- with BM25 (keywords);
- with MiniLM embeddings (meaning).

The two rankings are fused with Reciprocal Rank Fusion, which keeps the top 20.
A cross-encoder then re-scores the top 8, and the reader looks for an answer in
the top 3.

**Why.**
- BM25 catches exact terms: dataset names, acronyms, numbers.
- Dense vectors catch paraphrases.
- RRF combines the two without calibrating their very different score scales.
- The cross-encoder reads the question and the passage together, which is much more
  accurate than comparing vectors. It is slower, so it only sees the top 8.

## 4. Extractive QA, with abstention

**Decision.** Answers are spans copied out of the paper by a SQuAD-trained reader
(DistilBERT). If the combined confidence (reranker relevance × reader score ×
sentence support) is below a threshold, PaperTrail says the paper doesn't answer
the question.

**Why.**
- The brief's biggest grounding requirement is to not hallucinate.
- An extractive reader *cannot* invent a fact: every answer is a quote with a page
  number.
- I set the threshold from measured scores, not by guessing. On a separate labelled
  set of 51 questions across three papers ([evaluation](evaluation.md)), correct
  reader answers scored 0.13–0.91 (median 0.66) but unanswerable questions scored
  0.02–0.70 (median 0.21), so 0.2 is a trade-off rather than a clean gap: it
  answers 62% of answerable questions correctly but still answers 7 of the 12
  unanswerable ones.

**Cost.** The reader can only return a span. "Why" and "compare" questions get the
best supporting sentence instead of a synthesised answer.

## 5. A reviewer built from transparent rules

**Decision.** The critique and the Rigor Profile come from signals extracted from
the text:
- datasets, baselines and ablations;
- variance, seeds and significance tests;
- code links, compute, stated limitations;
- hedging vs. overclaiming language;
- how recent the references are.

Every finding cites the sentence it is based on, or says what was searched for and
not found.

**Why.**
- Reviewers can check it.
- It is consistent across papers, which makes the survey's comparison matrix fair.

**Cost.** Some things can't be measured from lexical evidence. A rule can see that a
paper reports no variance; it can't judge whether the idea is novel. The UI presents
the score as a rigor profile, not a quality verdict.

## 6. Where the data comes from

The arXiv API doesn't send CORS headers, so a browser can't call it. arXiv PDFs and
these four APIs do allow browser requests:

| Source | Used for | Notes |
| --- | --- | --- |
| DataCite | arXiv metadata (arXiv DOIs are registered there) and title search | |
| OpenAlex | Full-text search and citation counts | Also the fallback for arXiv metadata |
| Crossref | Publisher DOIs | |
| Semantic Scholar | arXiv ID, citations and PDF link for publisher DOIs | Rate-limits hard, so no retries and failures are ignored |

I checked every endpoint's headers before choosing it. When a PDF can't be fetched,
the graph falls back to an abstract-only analysis and offers the upload path.

## What I would do with more time

- **OCR for scanned PDFs**, with tesseract.js in a worker.
- **Read arXiv's HTML (LaTeXML) version when it exists.** Its section structure is
  cleaner than PDF layout heuristics.
- **A labelled evaluation set** of 100+ questions across fields, to tune the
  abstention threshold per question type.
- **An optional small on-device generative model** (WebGPU) to rephrase the
  extractive briefing, kept strictly grounded by the same verifier.
- **Add an HNSW index over the IndexedDB library**, so surveys can go beyond
  10 papers to hundreds.

# PaperTrail interview guide

Use this as a study aid. Explain the design in your own words and point to real code or saved output. Do not present hypothetical evaluation results or this draft's wording as your personal experience.

## A short project explanation

PaperTrail turns a research topic or arXiv ID into a briefing for one selected paper, then answers follow-up questions from retrieved passages. Every accepted factual claim carries source quotations and page references. The CLI and local browser interface drive the same explicit state graph and persist progress, so failures can resume from the saved step. It exports Markdown, JSON and a portable HTML evidence reader.

It is a local, single-user research reading assistant. `papertrail web` starts live interaction on `127.0.0.1`; computation stays on that computer. Exported HTML and GitHub Pages contain saved results. It does not fine-tune a model, perform an exhaustive literature review, or guarantee scientific correctness.

## Explain the pipeline

1. **Understand:** normalize an arXiv ID/URL or extract bounded literal keywords from a topic query.
2. **Retrieve:** request candidate metadata from the official arXiv API. Recent/latest intent affects the API sort.
3. **Select:** rank bounded candidates using abstract embeddings and title-weighted BM25; select one paper and pin its revision. The alternatives stay inspectable.
4. **Fetch:** download the PDF using an allowed source URL, enforce a size limit and compute its SHA-256 checksum.
5. **Parse:** extract page-aware text with section labels and a conservative column heuristic. Chunks are 180 words with 30-word overlap and do not cross pages.
6. **Index:** embed passage text using BGE-small-en-v1.5 through FastEmbed on CPU. Persist vectors and passage payloads in Qdrant local mode.
7. **Brief:** retrieve section coverage, ask local Qwen3.5 4B through Ollama for structured claims, resolve evidence aliases in Python, check provenance, and run a second model support review. Rejected paraphrases can become visibly labeled source excerpts; rejected limitations are removed.
8. **Validate:** check the resolved briefing against the allowed retrieved source passages, then advance to ready.
9. **Ready / QA:** retrieve evidence for a question, generate and validate an answer, save the exchange, and return to ready. The CLI also supports inspection and export.

Python controls these transitions. The LLM cannot choose arbitrary tools, execute code, replace paper metadata or choose download destinations. The explicit `EDGES` map in `src/papertrail/graph.py` is the executable graph.

## Evidence: the detail that matters most

The model receives sentence-like spans from retrieved chunks with short aliases such as `E1`. It selects aliases and writes claim text. Python resolves the aliases and copies quotations from its source registry; the model does not author the saved quotation text.

Deterministic checks ensure that passage IDs are known, quotations occur in normalized source text and numerical values in a claim occur in its attached evidence. A second LLM pass checks whether each claim follows from its quotations. It can reject a related quotation that does not support the statement. Schema and provenance failures receive one repair attempt, then fail closed.

Briefings have a conservative fallback for semantic-review rejections: Python substitutes up to two attached source quotations, visibly labeled as source wording requiring inspection. Rejected limitations are removed instead of being reframed as limitations of the current paper. The fallback keeps source text inspectable but sacrifices synthesis, and it does not prove relevance. QA has no excerpt fallback; still-rejected answers become explicit evidence-validation abstentions.

These controls reduce particular failure modes. They do **not** prove semantic entailment or scientific correctness. The same local model can miss an error during review. Number matching does not validate units, signs, comparisons or experimental fairness. A correct quotation can be interpreted incorrectly.

## Likely interview questions

### Why use a graph instead of one prompt or a large framework?

Named steps make failures and recovery inspectable. The saved next-node field identifies where to continue. A short sequential workflow does not require a framework scheduler. A framework could become useful for parallel branches, human approval steps or more complex execution.

### What makes this an agentic workflow?

It coordinates source access, retrieval, generation, validation and persistence through explicit state. Its autonomy is bounded: the graph owns tool use, and the LLM returns structured content inside prescribed stages. Be clear that it is not an open-ended autonomous tool loop.

### Why use both embeddings and BM25?

Dense embeddings can retrieve paraphrases; BM25 helps preserve exact technical terms and names. Within-paper retrieval combines rankings using reciprocal-rank fusion, summing `1 / (60 + rank)` from each list. It avoids adding raw lexical scores to cosine similarity scores. Whether the combination helps must be established with evaluation.

Candidate-paper ranking is separate: the current implementation uses a weighted blend of abstract/title semantic similarity and normalized title-weighted BM25. Do not claim that every ranking stage uses RRF.

### Why do you need Qdrant and SQLite?

Qdrant retrieves vector representations of source passages and stores their payloads. SQLite stores application state, selected paper metadata, graph progress and QA history. The original PDF and parsed text are separate artifacts. Each store has a different responsibility.

### What happens if the process stops halfway through?

The graph saves the current node before execution and advances it after successful completion. A failed or interrupted node is re-entered on resume. Earlier successful stages remain complete. Stable vector point IDs make index upserts idempotent. This supports retry-safe progress; it is not a claim that every external operation executes exactly once.

### How is RAG different from fine-tuning?

RAG supplies relevant document text when a question is asked. It does not change model weights. Fine-tuning changes weights using training examples. This application uses an existing local model and source retrieval; it has no model-training pipeline.

### Does a citation guarantee a correct answer?

No. Provenance confirms where the quoted text came from. Entailment asks whether the complete claim follows from that text. Python checks provenance and a second model screens support, but both retrieval and interpretation can still fail. Users can inspect the source quotation and page.

### Why use a second model review if it is fallible?

It screens an error that substring and numeric checks cannot: a real but unrelated quotation. The tradeoff is additional latency and another fallible judgment. It is one layer to evaluate, not a truth oracle or an independent human review.

### How do follow-up questions work?

The session persists question/answer history. A simple referential rule can append the previous question to the retrieval query, and recent questions provide context to generation. Earlier generated answers are never source evidence. This is limited conversational support rather than a general coreference resolver.

### What if the paper has no explicit limitations section?

The system may still retrieve relevant limitations from other sections, but it should not invent them. If selected evidence has none, the briefing says no explicit limitations were identified in that evidence. That is not proof that the paper has no limitations.

### What can go wrong with PDFs?

Columns, equations, tables, figure labels and unusual layouts can extract poorly. The parser records warnings and refuses unreadable or oversized inputs. OCR, figure interpretation and reliable table reconstruction are outside scope. Page citations help inspection, but they cannot recover information that extraction lost.

### How would you evaluate it?

Separate retrieval quality from answer support. Build held-out questions with labeled relevant passages, compare dense, BM25 and hybrid retrieval, then judge whether accepted claims are supported and whether abstentions are appropriate. Include unanswerable questions and difficult layouts. Publish actual outputs and limitations; do not generalize from a small demonstration set.

### Why not build a hosted web application?

The application includes a live local browser interface with digest and question forms, background jobs, saved sessions and an embedded evidence reader. It binds only to `127.0.0.1`, checks Host/Origin, bounds JSON bodies and serializes jobs through the existing application lock. A public multi-user service would add authentication, resource controls and concurrency requirements. GitHub Pages displays saved reports only.

### How do you defend against malicious source content?

Source passages are treated as untrusted data in prompts. Python restricts source destinations and keeps workflow control outside the model. The LLM cannot execute instructions as tools. This reduces impact, but prompt instructions alone do not guarantee that generated prose ignores every malicious passage.

### What would you improve next?

Start with a larger, held-out evaluation set and classify failures by search, parsing, retrieval, generation and support review. Use the evidence to prioritize layout extraction, query rewriting, reranking or model changes. Concurrency-safe storage and job handling would be needed before serving multiple users.

## Practical demonstration

Follow the README installation steps first, including the configured local model. Start Ollama if needed, then run:

```bash
papertrail doctor --warmup
papertrail digest "1706.03762"
papertrail sessions
```

For the local browser interface, run `papertrail web --port 8765`, open `http://127.0.0.1:8765`, and keep the server and Ollama running. See [the web demo guide](web-demo.md) for request boundaries and recovery details.

Copy the real completed session ID. Replace `SESSION` and `CHUNK` below with actual IDs:

```bash
papertrail show SESSION
papertrail ask SESSION "What problem does this paper address?"
papertrail ask SESSION "What is the main method?"
papertrail ask SESSION "What experiments support the method?"
papertrail inspect SESSION CHUNK
papertrail export SESSION --output exports/demo
```

These are rehearsal commands, not claims about observed output. Present the actual result, including an abstention. The first model download and source fetch require network access. Generation speed depends on local hardware.

During the walkthrough, show one accepted claim, inspect its quotation and visit the original PDF page. Reopen the session to explain persistence. If an interrupted pipeline needs recovery, use `papertrail resume SESSION` after fixing the cause. If QA fails on a ready session, retry `ask`. Keep a clearly labeled export of a real saved run available.

## Code reading route

| Module | Question it answers |
| --- | --- |
| `schema.py` | What is the shape of state, claims and evidence? |
| `graph.py` | What happens next, and what is saved after each stage? |
| `sources.py` | How are arXiv candidates searched and downloaded? |
| `parsing.py` | How do source pages become traceable chunks? |
| `retrieval.py` | How are papers ranked and passages retrieved? |
| `llm.py` | How are drafts constrained, resolved and reviewed? |
| `grounding.py` | Which provenance checks are deterministic? |
| `storage.py` | How do checkpoints and single-writer locking work? |
| `cli.py` / `rendering.py` | How can a user operate and inspect the result? |

## Official background references

- [arXiv API user manual](https://info.arxiv.org/help/api/user-manual.html)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Qdrant Python client and local mode](https://github.com/qdrant/qdrant-client#local-mode)
- [PaperTrail repository](https://github.com/Gaganpraveen/papertrail)

These references explain upstream interfaces. The local source is authoritative for this application's current behavior.

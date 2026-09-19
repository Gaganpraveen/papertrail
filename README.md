# PaperTrail

PaperTrail turns an arXiv paper into a structured briefing and answers follow-up questions with inspectable quotations and PDF page references. It accepts an arXiv ID, URL, or research topic, selects one paper per session, and preserves its source, index, execution history, and answers.

**Live application:** after setup, run `papertrail web` and open **[http://127.0.0.1:8765](http://127.0.0.1:8765)**. This runs new paper processing and local model inference.

**[GitHub Pages](https://gaganpraveen.github.io/papertrail/) is a static saved report, not a live application.** Public inference is not deployed. The current server is deliberately limited to a local single-user session; it must not be exposed through a public tunnel or used to expose raw Ollama.

[Architecture](docs/architecture.md) · [Browser guide](docs/web-demo.md) · [Release measurements](docs/release-rescue.md) · [Evaluation](docs/evaluation.md) · [BERT report](examples/rescue-bert/report.html) · [Topic report](examples/rescue-topic/report.html)

## Setup and run

Requirements: Python **3.12** for the documented setup, [Ollama](https://ollama.com/download), and sufficient memory for `qwen3.5:4b`. This model was tested on an Apple M5 with 16 GB RAM; its download is approximately 3.4 GB. The package declares Python 3.11–3.13 support. Model downloads and arXiv retrieval require internet access; inference and storage run locally without a paid text-generation provider.

```sh
git clone --branch codex/release-rescue https://github.com/Gaganpraveen/papertrail.git
cd papertrail
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
ollama pull qwen3.5:4b
papertrail doctor --warmup
papertrail web
```

Ollama must be running; `ollama serve` starts it in another terminal if necessary. On Windows, activate with `.venv\Scripts\Activate.ps1` and use the installed Python 3.12 launcher. Initial setup also downloads the CPU embedding model. The tested dependency snapshot and platform scope are documented in [reproducibility notes](docs/reproducibility.md).

Open the local application, enter `https://arxiv.org/abs/1810.04805`, and select **Create briefing**. The selected title and revision appear above the briefing. Follow-up questions appear in the same report with expandable source quotations. Keep Ollama and the application process running.

The default Ollama address is `http://127.0.0.1:11434`. The original demonstration Mac uses a separate service on port 11435:

```sh
PAPERTRAIL_OLLAMA_URL=http://127.0.0.1:11435 .venv/bin/papertrail web
```

## CLI example

```sh
papertrail digest 'https://arxiv.org/abs/1810.04805'
papertrail sessions
papertrail show SESSION
papertrail ask SESSION 'What does BERT stand for?'
papertrail ask SESSION 'What was the total training cost in US dollars?'
papertrail export SESSION --output exports/bert
papertrail resume SESSION
```

`SESSION` is the identifier printed by `digest` or `sessions`. A topic example is `papertrail digest 'research about electron'`. Topic availability depends on arXiv; a successful lookup does not guarantee successful parsing or generation. `papertrail chat SESSION` opens interactive QA, and `papertrail inspect SESSION CHUNK` displays a retrieved source passage.

## Architecture and state

```text
understand → retrieve → select → fetch → parse → index → brief → validate → ready
                                                                           │
                                    retrieve evidence ← follow-up question ┘
                                            ↓
                                validated answer / abstention → saved history
```

The graph in [`graph.py`](src/papertrail/graph.py) uses typed Pydantic state and SQLite checkpoints. arXiv metadata determines the paper and revision; the model cannot choose arbitrary download URLs. PDF parsing produces page-aware passages with section labels. Local BGE embeddings in Qdrant and BM25 retrieval are combined through reciprocal-rank fusion.

A briefing covers the summary, problem, method, results, limitations or a missing-evidence note, and suggested questions. Generation is followed by deterministic provenance checks and a separate support-review pass through the same local model. QA uses the saved paper and index. Previous generated answers are not source evidence.

The browser starts one isolated worker per operation. It displays the actual stage, stage and total elapsed times, completed timings, and the operation deadline. On timeout, the parent stops the worker and releases its resources; incomplete model output is not accepted. Graph checkpoints avoid repeating completed downloads, parsing, and indexing during resume. Compatible cached PDFs and indexes are reused across sessions.

## Grounding and tradeoffs

The model selects evidence sentence IDs; Python resolves them to source quotations. Validation checks passage identity, quotation provenance, and numerical values, including numbers attached to units. Unicode normalization handles PDF ligatures consistently. Parser version 4 infers two-column reading order and records warnings where layout requires inspection. If no Abstract heading is detected, the first two opening-page passages are included in briefing evidence. New digests use the parser-versioned index; existing saved parsed artifacts are not silently rewritten. BM25 handles question boilerplate and simple plural/hyphen variants without requiring new embeddings.

Rejected QA abstains. A rejected briefing paraphrase can instead become exact cited source wording, visibly labeled **“Source wording (automatic review requested inspection)”** with a warning. This is an inspectable fallback, not a successful semantic-review verdict. Rejected limitations are removed; absent limitations evidence is stated explicitly. Schema or evidence failures receive one bounded repair attempt.

An explicit Python graph makes transitions and checkpoint recovery inspectable without an additional orchestration framework. SQLite and local Qdrant keep setup small; one operation at a time avoids conflicting writers. Local Ollama avoids per-token provider charges but requires model downloads, memory, and compute. Dense retrieval handles paraphrases while lexical retrieval preserves exact terms; neither guarantees that all relevant evidence is retrieved.

The support reviewer shares the generator's potential errors. Numeric agreement does not establish units, comparisons, or scientific validity. Dense numeric tables and displaced mathematical spans are conservatively withheld from generation, which can omit useful evidence. PDF layout heuristics can mishandle figures, tables, equations, or columns. OCR, arbitrary PDF upload, publisher-wide search, cross-paper synthesis, and multi-user hosting are not implemented. PDFs are limited to 30 MB and 100 pages. Model input uses a 32,768-token context with a conservative 28,000-byte message budget; oversize input fails explicitly.

## Recorded QA and browser verification

A fresh BERT run resolved the URL above to `1810.04805v2`, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, session `7016d3da0645`. The browser operation took about **31 s**, using cached metadata/PDF and a compatible index. Generation and review were new. One result paraphrase became labeled source wording with a warning. The [report](examples/rescue-bert/report.html) and [session JSON](examples/rescue-bert/session.json) preserve these results.

| Question | Actual result | Evidence and timing |
| --- | --- | --- |
| What does BERT stand for? | “BERT stands for Bidirectional Encoder Representations from Transformers.” | Quotation verified in the browser and original [PDF page 1](https://arxiv.org/pdf/1810.04805v2#page=1), chunk `e1796cabe1b618c0`; backend 4.579 s, browser 5.3 s |
| What was the total training cost in US dollars? | “I could not find enough evidence in the retrieved passages to answer this question.” | `insufficient_evidence`, no factual claims; backend 2.472 s, browser 3.2 s |
| What was the total energy resolution of the setup? (physics topic) | “The setup has a total energy resolution of < 22meV (limited by the bandwidth of the 6eV pulses).” | [PDF page 3](https://arxiv.org/pdf/1401.3078v2#page=3), chunk `4455b47adb1a80bf`, verified in the browser and original PDF; backend 4.050 s, browser 4.8 s |

The final fresh topic run, `research about electron`, selected arXiv `1401.3078v2`, *Ultrafast Electron Dynamics in the Topological Insulator Bi2Se3 Studied by Time-Resolved Photoemission Spectroscopy*, session `d757bedd4a1c`. It completed in **36 s** in the browser using cached source files and a compatible index. The summary is plain-English synthesis; one result still uses visibly labeled source wording with a warning. The research-budget question abstained in **0.800 s** in the backend and **1.5 s** in the browser. The [topic report](examples/rescue-topic/report.html) preserves the final results. Earlier session `51fc27237fea` failed after 44 s on attached-unit validation, then resumed in 28 s with a weak summary; that history is retained separately rather than presented as the final run.

A real five-second deadline check stopped an in-flight operation at **5.1 s**. Ollama logged cancellation and release of the inference slot; a subsequent question on an existing paper completed in **3.7 s**. This verifies resource recovery, not fresh-paper correctness. [Release measurements](docs/release-rescue.md) distinguish fresh generation, cache reuse, failures, and nested stage timings. They are observations on one machine, not controlled latency benchmarks.

## Configuration and recovery

Settings are read from the process environment; `.env` files are not loaded automatically. [`.env.example`](.env.example) lists optional overrides. Saved sessions retain their model configuration.

| Setting | Default | Purpose |
| --- | --- | --- |
| `PAPERTRAIL_MODEL` | `qwen3.5:4b` | Local Ollama model |
| `PAPERTRAIL_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `PAPERTRAIL_DATA_DIR` | `.papertrail` | Source files, sessions, embeddings, and indexes |
| `PAPERTRAIL_MODEL_TIMEOUT` | `240` | Model HTTP request timeout, seconds |
| `PAPERTRAIL_OPERATION_TIMEOUT` | `240` | Total browser operation deadline, including retrieval, review, and export |

**Resume saved session** continues a failed pipeline checkpoint. **Retry last operation** retries failed QA or resumes a recoverable pipeline. **Refresh** reconnects to active work without creating a duplicate job. Invalid input must be corrected in a new session. A server restart preserves disk checkpoints but loses its in-memory job registry; interrupted QA must be retried.

Transient source failures and HTTP 429/5xx receive at most three attempts. HTTP 401/403 are not repeatedly retried. Some valid arXiv topics return empty HTTP 406 responses; one equivalent canonical query preserves the terms and sort intent, and a successful response is cached. Persistent failures remain explicit. No canned paper replaces a failed lookup. Missing models require starting Ollama and pulling the saved session's model. Failed grounding checks remain failures or abstentions rather than successful results.

## Tests and evaluation

The complete local regression passed **175 tests in 20.99 s** before the final opening-page evidence correction. After that correction and artifact-path redaction, all **12 affected graph/evaluation tests passed in 0.44 s**, including the new regression. Ruff lint and formatting, dependency checks, and wheel/CLI verification passed. The final full GitHub CI result is reported separately; these local checks do not establish arXiv availability or model correctness.

```sh
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
python -m pip check
papertrail --help
```

Tests cover source validation, parsing, persistent retrieval, checkpoint recovery, evidence validation, abstention, context/repair limits, HTTP boundaries, and timeout cleanup. External services and model behavior are represented by explicit test doubles in offline tests. GitHub Actions runs the automated checks on Python 3.11 and 3.12.

The [retrieval-only regression](examples/evaluation-rescue.json) found labeled support in the top five for **10/10** existing Attention questions in dense, BM25, and hybrid modes; hybrid mean reciprocal rank was **1.0**. The model-answer evaluation was not rerun for the rescue changes. The [earlier answer evaluation](examples/evaluation-release.json) recorded 8/10 answerable questions answered and 3/3 unsupported controls abstained, with two false abstentions. This is a developer-constructed, single-paper smoke evaluation, not a blind or held-out benchmark. [Evaluation notes](docs/evaluation.md) and [failure history](examples/rescue-verification.json) retain the protocol and observed problems.

## Repository structure

```text
src/papertrail/       graph, sources, parsing, retrieval, grounding, storage, CLI, web
tests/              offline regression and integration tests
evals/              source-pinned question labels
scripts/            evaluation and reproducible demo tooling
examples/           saved reports, session records, timings, and evaluation results
docs/               architecture, evaluation, run instructions, static report
.github/workflows/  automated CI
pyproject.toml      package metadata, dependencies, and CLI entry point
requirements-tested.txt  platform-scoped dependency snapshot
.env.example        optional environment settings
```

Published historical records replace the local workspace prefix with `<project>`; answers, timings, and source hashes are unchanged.

Runtime data lives under `.papertrail/` and is excluded from Git. Model weights, downloaded paper PDFs, local databases, caches, and secrets are not part of the source repository. Application code uses the [MIT License](LICENSE); model weights and source papers retain their respective licenses.

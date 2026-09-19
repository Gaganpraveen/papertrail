# PaperTrail

An arXiv reading assistant with persistent sessions, structured briefings, and answers linked to inspectable source evidence. Enter a research topic, arXiv ID, or arXiv paper URL; PaperTrail selects one paper, parses its PDF, and saves the path from each accepted claim back to its quoted passage and page.

[Repository](https://github.com/Gaganpraveen/papertrail) · [Architecture](docs/architecture.md) · [Evaluation](docs/evaluation.md) · [Interview guide PDF](output/pdf/papertrail-guide.pdf) · [Reflection draft](docs/reflection-script.md)

[Verified paper run](examples/release-attention/report.html) · [Verified topic run](examples/release-topic/report.html) · [Earlier CLI recording](examples/demo.cast) · [Release verification](docs/release-verification.md). These saved artifacts preserve real results and inspectable citations. Live processing runs locally through the CLI or browser interface. **Public publishing remains blocked by GitHub authentication**; the [planned Pages demo](https://gaganpraveen.github.io/papertrail/) is not a verified deployment. Pages will serve saved results, not a live model backend.

## What it does

- Searches the official arXiv API or resolves a supplied ID, preserves candidate metadata, and selects one revision per session.
- Extracts page-aware PDF passages with section labels, including abstract and references, and records extraction warnings.
- Combines local BGE embeddings in Qdrant with BM25 using reciprocal-rank fusion.
- Produces a briefing with a summary, problem, method, results, mandatory limitations or a missing-evidence note, and suggested questions.
- Answers follow-up questions with quoted evidence, or explicitly abstains when support is insufficient or validation fails.
- Saves graph checkpoints, model configuration, source checksums, and conversation history; exports Markdown, JSON, and a portable HTML evidence reader.

No paid API key, hosted database, or cloud inference account is required. Model downloads and access to arXiv require internet access. Inference and the vector database run on your computer; speed and memory use depend on the selected model and hardware.

## Setup

Use Python **3.12** for the documented setup. The package declares Python 3.11–3.13 support. Install [Ollama](https://ollama.com/download) and ensure its local service is running; if needed, run `ollama serve` in another terminal.

```sh
git clone https://github.com/Gaganpraveen/papertrail.git
cd papertrail
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` and use your Python 3.12 launcher in place of `python3.12`.

The default local model is **`qwen3.5:4b`**, tested locally on an Apple M5 with 16 GB RAM. Its download is about 3.4 GB. The setup commands explicitly select this model; see [reproducibility notes](docs/reproducibility.md) for the tested dependency snapshot and model-pinning limits.

```sh
ollama pull qwen3.5:4b
export PAPERTRAIL_MODEL=qwen3.5:4b
papertrail doctor --warmup
papertrail digest '1706.03762'
```

For PowerShell, set the model with `$env:PAPERTRAIL_MODEL = 'qwen3.5:4b'`. The first `doctor --warmup` also downloads the CPU embedding model. Allow several minutes for initial downloads and local generation; no hardware-independent latency is promised.

The final command requests *Attention Is All You Need*. A base ID resolves through official metadata to a revision, which is then retained in the session. You can instead supply a versioned ID or a plain arXiv `/abs/` or `/pdf/` URL.

## Local browser interface

After setup, start the local interface:

```sh
papertrail web
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) in your browser. Keep the process and Ollama running for live processing and questions. This interface runs on your computer and is separate from the static Pages export. See the [local web demo guide](docs/web-demo.md) for the walkthrough.

## Use the saved session

After a successful digest, copy its printed session ID. Replace `SESSION` below with that ID and `CHUNK` with a passage ID from an answer or export.

```sh
papertrail sessions
papertrail show SESSION
papertrail ask SESSION 'How many GPUs were used, and how long was the big model trained?'
papertrail ask SESSION 'What was the total training cost in US dollars?'
papertrail chat SESSION
papertrail inspect SESSION CHUNK
papertrail export SESSION --output exports/attention
```

The saved demo answered the hardware and base-training questions, abstained on dollar cost, and falsely abstained on label smoothing after its support checker rejected a valid interpretation of noisy extracted text. These outcomes remain visible; the demo does not hide the failure. Type `/exit` to leave interactive chat. Questions and accepted answers or abstentions are saved between CLI invocations. `show`, `inspect`, `sessions`, and export reading do not require a running language model.

`digest` automatically writes an export under `.papertrail/runs/SESSION/export/`. Open `report.html` in a browser to expand quotations, follow PDF page links, and inspect the execution trail. The `export` command writes `report.html`, `briefing.md`, and `session.json` to the chosen directory. The HTML contains no live chat backend. Exported quotations remain readable offline; links to arXiv require internet access.

Topic search uses the same pipeline:

```sh
papertrail digest 'recent work on KV-cache compression'
```

**Observed source limitation:** during validation, several syntactically valid topic queries received empty HTTP 406 responses from the arXiv API, while a documented example query and the base ID `1706.03762` returned metadata. The underlying cause has not been confirmed. Topic search remains implemented; a blocked topic request is reported as a source error. Retry the saved session later or use a known paper ID. A successful ID lookup does not imply that the complete briefing has succeeded.

## Architecture

```text
understand → retrieve → select → fetch → parse → index → brief → validate → ready
                                                                           │
                                    retrieve evidence ← follow-up question ┘
                                            ↓
                                validated answer / abstention → saved history
```

The transition map in [`graph.py`](src/papertrail/graph.py) is executable Python state, with Pydantic records and SQLite checkpoints after successful nodes. The LLM writes schema-constrained content; it cannot select arbitrary tools, modify source metadata, or choose download URLs. Failed or interrupted pipeline nodes can be re-entered without repeating already completed nodes.

| Component | Responsibility |
| --- | --- |
| `sources.py` | Input normalization, official Atom metadata, bounded PDF download, metadata caching |
| `parsing.py` | PDF text extraction, section detection, page-aware chunks and provenance |
| `retrieval.py` | FastEmbed BGE-small on CPU, persistent Qdrant vectors, BM25 and rank fusion |
| `llm.py` / `grounding.py` | Ollama JSON generation, evidence resolution, validation and support review |
| `graph.py` / `storage.py` | Explicit transitions, SQLite state, durable artifacts and operation locking |
| `cli.py` / `web.py` / `rendering.py` | CLI, loopback HTTP jobs and browser interface, saved Markdown/JSON/HTML |

The default data directory is `.papertrail/`: SQLite stores sessions, `vectors/` stores Qdrant collections, `models/` caches embeddings, and `runs/SESSION/` retains the source PDF, parsed passages, and exports. Each passage carries its paper ID, page, section, offsets, and stable chunk ID. One operation at a time holds the data-directory lock. See the [architecture and state documentation](docs/architecture.md) for the full graph and recovery boundaries.

The shared [`RunState`](src/papertrail/schema.py) contains the query and intent, candidate and selected paper metadata, model names, PDF checksum, page/chunk counts, vector collection reference, briefing and evidence IDs, QA exchanges, warnings, timings, status, and next node. PDF text and vectors are separate artifacts referenced by this durable state.

### Repository structure

```text
papertrail/
├── src/papertrail/          # graph, schemas, sources, parsing, retrieval, generation,
│                           # grounding, storage, CLI, local web, and rendering
├── tests/                  # offline unit/integration checks with explicit test doubles
├── evals/                  # source-pinned question labels
├── scripts/                # evaluation, real demo recording/replay, PDF guide builder
├── examples/attention/     # saved briefing.md, report.html, and session.json
├── examples/               # terminal recording/transcript and evaluation reports
├── docs/                   # architecture, tradeoffs, evaluation, guides, static index
├── output/pdf/             # explanatory and interview guide
├── .github/workflows/      # CI configuration
├── pyproject.toml          # install metadata, dependencies, CLI entry point
├── requirements-tested.txt # platform-scoped tested dependency snapshot
└── .env.example            # documented optional environment settings
```

## Evidence checks and limits

The model selects sentence IDs before writing claims. Python resolves those IDs and copies quotations from the retrieved source; it checks quotation provenance and that numeric values in each claim occur in its evidence. A second pass through the same local model reviews whether the attached quotations support the claim. Invalid generation receives one bounded repair attempt with the rejected draft and validation reason.

For briefings only, a paraphrase rejected by semantic review can be replaced with exact cited source wording, visibly labeled **“Source wording (automatic review requested inspection)”** and accompanied by a warning. This conservative fallback preserves something the reader can inspect when generation or review is unreliable. The replacement still undergoes deterministic provenance checks; it is not a successful semantic-review verdict. Rejected limitations are removed instead of receiving this fallback; when none remain, the briefing states that explicit limitations were not identified in the selected evidence. Unrepaired schema or evidence failures prevent acceptance. QA retains stricter behavior: a rejected draft produces an explicit evidence-validation abstention.

Dense numeric excerpts that may be flattened tables are conservatively withheld from model generation; the original passages remain in the local index. This heuristic can also omit useful prose. Ordinary retrieval excludes references unless the question requests citations or references. Previous questions can supply conversational context; previous generated answers are never source evidence.

These checks **do not prove correctness**. The support reviewer shares the generator's potential errors and can also reject valid paraphrases. An exact source excerpt can omit context or be placed under an unsuitable briefing heading; inspect fallback wording in the paper before interpreting it. Matching numbers does not verify units or comparisons. Retrieval can miss evidence, so abstention means insufficient retrieved support, not proof that the paper lacks an answer. PDF section and column detection are heuristic; equations, complex tables, figures, scans, and non-English material can extract poorly. OCR and cross-paper synthesis are not implemented.

PDFs are capped at 30 MB and 100 pages; insufficient readable text fails explicitly. Model requests use a 32,768-token context with a conservative 28,000-byte message budget, including repair history. This bound assumes the Qwen byte-level tokenizer; an alternative model must support the same context and tokenization assumptions. Oversized input is rejected instead of silently clipped by the application.

## Configuration and recovery

Global options precede the command:

```sh
papertrail --data-dir ./my-reading --model qwen3.5:4b digest '1706.03762'
papertrail --data-dir ./my-reading resume SESSION
papertrail --debug resume SESSION
```

| Setting | Default | Purpose |
| --- | --- | --- |
| `PAPERTRAIL_MODEL` | `qwen3.5:4b` | Ollama model for new sessions |
| `PAPERTRAIL_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `PAPERTRAIL_DATA_DIR` | `.papertrail` | Sessions, source files, embeddings and vector storage |
| `PAPERTRAIL_MODEL_TIMEOUT` | `240` | Per-request generation timeout in seconds |

Settings come from the process environment. [`.env.example`](.env.example) documents the variables; the application does not automatically load an `.env` file. Saved sessions retain their model and embedding configuration. Start a new digest to change models for a paper.

| Failure | Recovery |
| --- | --- |
| arXiv unavailable, timeout, or HTTP 406 | Keep the printed session ID and run `resume SESSION` later; a known-ID digest is an alternative to topic search |
| Ollama unavailable or model missing | Start Ollama, pull the required model, run `doctor`, then resume the pipeline or retry `ask` |
| Generation exceeds the timeout | Increase `PAPERTRAIL_MODEL_TIMEOUT`, then retry; changing models requires a new digest |
| Briefing uses labeled source wording | Inspect the warning, quotation, and original page before interpreting the excerpt |
| Briefing schema or evidence validation still fails | Inspect the failure and retry `resume`; repeated rejection is a limitation, not a verified result |
| QA lacks support or fails evidence validation | Read the abstention, inspect retrieved passages, or ask a more specific question |
| Another operation holds the lock | Wait for it to finish or exit an open `chat`; keep the same data directory for existing sessions |
| PDF unreadable, oversized, or saved passages/index missing | Read the actionable error; use a suitable paper or create a new digest to rebuild missing artifacts |

## Verification and assessment material

The frozen release passes **130 automated tests**, including the local HTTP adapter. Ruff lint and formatting, dependency checks, and an isolated wheel-install smoke test also pass. [Release verification](docs/release-verification.md) records the real browser scenarios and remaining publication steps. Run the automated checks:

```sh
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
papertrail --help
```

Offline tests cover source validation and caching, parsing, persistent retrieval, checkpoint recovery, evidence validation, safe abstention, generation repair/context limits, rendering, and evaluation helpers. They use explicit test doubles and do not establish live model quality or arXiv availability. CI runs the offline checks on Python 3.11 and 3.12.

The [evaluation protocol](docs/evaluation.md) documents a developer-constructed, single-paper smoke evaluation: ten answerable questions and three unanswerable controls. It compares the application's dense, BM25, and hybrid retrieval and separates optional answer checks from retrieval metrics. The [final release evaluation](examples/evaluation-release.json) recorded 11/13 expected-status matches: 8/10 answerable questions answered, all 3 unanswerable controls abstained, and two false abstentions. All three retrieval modes found labeled support in their top five results for 10/10 questions. The [earlier report](examples/evaluation.json) remains available, with provenance and qualitative review in the evaluation notes. The sample is neither blind nor held out, and status agreement does not establish answer correctness.

Additional materials:

- [Design decisions and tradeoffs](docs/design-decisions.md)
- [Interview guide PDF](output/pdf/papertrail-guide.pdf) and [Markdown guide](docs/interview-guide.md)
- [Reflection recording draft](docs/reflection-script.md), to adapt to the presenter's actual contribution and record personally
- [Demo recorder](scripts/record_demo.py), which records real CLI subprocess output and stops on failure

Application code is available under the [MIT License](LICENSE). Model weights and source papers retain their respective licenses. Downloaded PDFs, model weights, local databases, and environment secrets are excluded from the repository.

## Design decisions and tradeoffs

The core submission is an explicit Python graph with typed shared state and SQLite checkpoints. This makes transitions and recovery easy to inspect without introducing an orchestration framework for a short sequential pipeline. The QA loop uses the same saved paper and index across process restarts. One paper is selected per session; cross-paper comparisons and conflicting claims require a broader evidence design.

Local Ollama, CPU embeddings, and Qdrant avoid paid API credentials and cloud database setup. They require model downloads and sufficient local memory, and generation speed depends on hardware. Dense retrieval handles paraphrases while BM25 preserves exact terms; reciprocal-rank fusion combines their rankings. The small evaluation compares these choices without claiming general superiority.

Evidence IDs, copied quotations, and numeric checks make provenance inspectable. The same-model support reviewer adds another check but can repeat the generator's errors or reject valid answers. Briefing-only source-excerpt fallback is visible and needs human inspection; QA instead abstains on rejected support. PDF layout heuristics, unresolved source contradictions, and missed retrieval evidence remain material limitations.

The CLI fulfills the assessment's interaction requirement. The local web adapter and static export are optional presentation aids; public inference, authentication, and production deployment are not part of the system. With more time, the priority would be a held-out evaluation across papers and layouts, human review of claim support and abstention, and improvements guided by whether errors originate in parsing, retrieval, or generation. The full [design notes](docs/design-decisions.md) expand these choices.

## Recorded demonstration

Session `4ca1290ef789` processed 15 pages into 55 passages. Its execution trail retains the initial generation failure and successful checkpoint recovery. The terminal recording shows the saved briefing followed by three fresh QA calls; the final HTML also includes a fourth question submitted through the live browser interface. Replay with `python scripts/replay_demo.py examples/demo.cast` (idle gaps are capped at three seconds by default). Generate your own recording with `python scripts/record_demo.py`.

A source inconsistency is visible in this paper: its abstract reports 41.8 English–French BLEU, while the results prose reports 41.0. The briefing cites the abstract value; the attached source passages preserve both. PaperTrail does not automatically reconcile contradictory source claims.

### Actual input, briefing, and three saved QA exchanges

**Input:** `papertrail digest '1706.03762'`. The resulting saved session is `4ca1290ef789`, using `qwen3.5:4b`, for *Attention Is All You Need*, arXiv `1706.03762v7`, published 2017-06-12. The [complete briefing](examples/attention/briefing.md) includes its authors, problem, method, results, explicit limitations note, and suggested questions.

**Saved summary:** “The authors propose the Transformer, a new network architecture based solely on attention mechanisms that dispenses with recurrence and convolutions entirely. Experiments on machine translation tasks demonstrate that these models are superior in quality while being more parallelizable and requiring significantly less time to train.” [Source: PDF page 1, Abstract](https://arxiv.org/pdf/1706.03762v7#page=1), chunk `c8b5eec8c07826e6`.

The following answers are copied from the [saved session JSON](examples/attention/session.json), not newly generated for this README:

1. **Question:** “How many GPUs were used, and how long was the big model trained?” **Answer:** “8 NVIDIA P100 GPUs were used to train the big model.” “The big model was trained for 3.5 days.” [Source: PDF page 7, Hardware and Schedule](https://arxiv.org/pdf/1706.03762v7#page=7), chunk `abf6f22b8f92a712`; corroborating [page 8](https://arxiv.org/pdf/1706.03762v7#page=8), chunk `b37abe7f7725f407`.
2. **Question:** “How many training steps and hours did the base Transformer models use?” **Answer:** “The base models were trained for a total of 100,000 steps or 12 hours.” “For the base models, each training step took about 0.4 seconds.” [Source: PDF page 7, Hardware and Schedule](https://arxiv.org/pdf/1706.03762v7#page=7), chunk `abf6f22b8f92a712`.
3. **Question:** “What was the total training cost in US dollars?” **Answer:** “I could not find enough evidence in the retrieved passages to answer this question.” Status: `insufficient_evidence`. This abstention makes no factual claim requiring a paper citation; its retrieval IDs and response are preserved in the [saved exchange](examples/attention/session.json).

The saved label-smoothing question also produced an evidence-validation abstention despite relevant source material; it remains visible in the full transcript. The required personal reflection video is a separate deliverable: [the script](docs/reflection-script.md) is preparation material and does not replace a recording of at most four minutes.

## Final browser verification

The release was exercised through the browser, including a fresh versioned URL, natural-language search, answerable QA, abstention, and invalid-input handling. [Current attention-paper run](examples/release-attention/report.html) contains the hardware answer, dollar-cost abstention, and a supported answer of 4,000 warmup steps. [Natural-language topic run](examples/release-topic/report.html) records `research about electron`, actual arXiv candidate selection, a downloaded 15-page paper, and cited QA. These are saved exports; use `papertrail web` for live local interaction.

On HTTP 406, a simple alphanumeric topic receives one equivalent canonical arXiv API query preserving its terms and sort intent. This enabled the recorded topic run; some other queries still fail upstream. The generation context retains short factual sentences and conservatively withholds displaced formula spans and dense numeric tables. A lexical uncertainty check prevents dropping words such as “may” or “appear”; these heuristics can omit useful evidence or cause false abstentions and do not prove semantic correctness.

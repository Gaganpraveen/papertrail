# Initial release verification (historical)

This record preserves the initial 2026-09-19 release checks and their original metrics. [Release rescue verification](release-rescue.md) contains the later fixes, fresh workflows, and current regression results. [PR #1](https://github.com/Gaganpraveen/papertrail/pull/1) was merged into `main` at `d92aea7`; the application is no longer awaiting that merge.

## Functional coverage at initial release

| Requirement | Local evidence | Release status / limitation |
| --- | --- | --- |
| Topic or arXiv ID/URL input | `sources.py`: intent parsing and official Atom API; ranking in `retrieval.py` | Final browser URL session `0d03c77529e0` and topic session `620d2cdae9d1` reached `ready`. Some multi-term topics can still receive HTTP 406. |
| Explicit nodes, edges, shared state | `graph.py` executable `EDGES`; `schema.py` `RunState`; [state diagram](architecture.md) | Present. QA uses saved state and separate retrieval/answer transitions. |
| Candidate metadata and selection | Official metadata includes title, authors, abstract, categories, dates, PDF link; candidate list retained | Browser topic run retrieved eight official candidates and selected `hep-th/0505137v1`; one paper per session. |
| PDF sections, abstract, references | `parsing.py`; saved run has 15 pages and 55 passages | Heuristic extraction; scanned PDFs and unsupported sizes fail explicitly. Complex tables and equations remain limited. |
| Chunking, embeddings, vector DB | Page-aware chunks, FastEmbed, persistent local Qdrant, BM25/rank fusion | Implemented; retrieval evaluation and offline coverage exist. |
| Required briefing fields | Final [attention Markdown](../examples/release-attention/briefing.md), [attention JSON](../examples/release-attention/session.json), and [topic Markdown](../examples/release-topic/briefing.md) | Present, including explicit limitations handling. Visibly labeled source-wording fallback remains possible and is not a semantic guarantee. |
| Grounded free-form follow-up QA and abstention | Final attention hardware, warmup, and dollar-cost exchanges; final topic QA, all with saved evidence | Hardware answer independently checked against original PDF pages 7/8; warmup answer is 4000 in the export; cost question abstained. DSR citations verified in browser. Historical false abstention remains disclosed. |
| State persistence and realistic recovery | SQLite checkpoints and saved QA history; historical failure/recovery trail | Final invalid-input session `1bf62c413c3f` failed at `understand` and correctly instructed the user to start a new briefing with a corrected ID/topic, without the misleading resume hint. |
| Python, free/local tools, official arXiv API, no paid keys | `pyproject.toml`, `config.py`, source/LLM adapters; README setup | Present. Ollama and embedding weights require downloads and sufficient local resources. |
| Published source and setup | Tracked code, README, configuration, and saved reports are published | Repository access was corrected and all 63 tracked files were published. [PR #1](https://github.com/Gaganpraveen/papertrail/pull/1) was subsequently merged into `main` at `d92aea7`. Source ZIP built with `git archive` and verified against the complete tracked-file manifest; archive integrity passes and runtime files, credentials, model weights, and downloaded PDFs are excluded. |
| README graph/state description, setup, input→briefing→2–3 QA, tradeoffs | README now includes shared-state description, actual repository tree, saved summary and three real QA exchanges, concise tradeoffs | Documentation published with the source. |

## Real example provenance

The example input was `1706.03762`; official metadata resolved it to `1706.03762v7`. Session `4ca1290ef789` is saved with status `ready`, model `qwen3.5:4b`, 15 parsed pages, 55 passages, and four QA exchanges in [`examples/attention/session.json`](../examples/attention/session.json).

Those historical exchanges include big-model hardware/duration, base-model steps/hours, and an abstention on training cost in dollars. The accepted hardware and schedule answers cite PDF page 7, chunk `abf6f22b8f92a712`, with page 8 corroboration for the big model. The label-smoothing false abstention remains in the complete historical example rather than being removed from the report. Final release exports are separately identified below.

The saved briefing also exposes an unresolved source inconsistency: its abstract-based result uses 41.8 English–French BLEU, while a cited body passage uses 41.0. Exact citation checks do not reconcile that inconsistency. The saved artifacts retain the source-excerpt fallback, false abstention, and inconsistent source values.

## Fresh browser release checks

- **Final paper URL:** `https://arxiv.org/abs/1706.03762v7` created session `0d03c77529e0` after the grounding fixes. It reached `ready` with 15 pages and 55 passages. Its [saved JSON](../examples/release-attention/session.json) confirms the input, completed state, and three QA exchanges. This supersedes earlier URL session `272b19dbec52` as the final release example.
- **Final hardware QA:** “How many GPUs were used, and how long was the big model trained?” received “The big model was trained for 3.5 days on 8 P100 GPUs.” It also reported “The big models were trained for 300,000 steps (3.5 days).” The attached quotations were checked directly in the original PDF with `x_tolerance=2` on [page 7](https://arxiv.org/pdf/1706.03762v7#page=7), chunk `abf6f22b8f92a712`, and [page 8](https://arxiv.org/pdf/1706.03762v7#page=8), chunk `b37abe7f7725f407`.
- **Final abstention:** “What was the total training cost in US dollars?” returned `insufficient_evidence` with no claims in 2.471 seconds (approximately 2.5 seconds).
- **Final warmup QA:** “How many warmup steps were used for the learning rate?” returned “We used warmup_steps = 4000.” The saved answer cites the exact same text from chunk `a3d64752b5cce068`; this value was checked directly in the final export.
- **Natural-language topic:** `research about electron` created session `620d2cdae9d1`. The official API returned eight candidates; ranking selected `hep-th/0505137v1`. The PDF produced 15 pages and 44 chunks, and all pipeline nodes completed in approximately 60 seconds.
- **Topic QA:** “Which theoretical framework does this paper use to discuss the electron mass?” received an answer identifying DSR. The browser opened the source evidence on [PDF page 1, Abstract](https://arxiv.org/pdf/hep-th/0505137v1#page=1), chunk `ae766c886b044ee6`, and [page 2](https://arxiv.org/pdf/hep-th/0505137v1#page=2), chunk `1a09b8d685f1ab28`; the exact supporting text was verified in the browser.
- **Final invalid-input recheck:** `1706.invalid` created session `1bf62c413c3f` and failed at `understand`. The browser now instructs the user to start a new briefing with a corrected ID or topic; it no longer suggests resuming the invalid input. This supersedes the earlier failure session `56d47023f067` and verifies the corrected message.

A canonical-equivalent request fallback was added for the observed arXiv HTTP 406 behavior. The successful electron topic run demonstrates one genuine topic workflow; it does not establish that every topic now succeeds. Other multi-term queries can still return HTTP 406.

Final exports have been written and inspected: [attention report](../examples/release-attention/report.html), [attention Markdown](../examples/release-attention/briefing.md), [attention JSON](../examples/release-attention/session.json), [topic report](../examples/release-topic/report.html), [topic Markdown](../examples/release-topic/briefing.md), and [topic JSON](../examples/release-topic/session.json). The source-wording fallback and PDF/math extraction limitations remain applicable. A successful pipeline is not an independent correctness judgment on every generated claim.

## Verification ledger

| Check | Evidence at audit start | Historical result |
| --- | --- | --- |
| Offline tests | Previously reported **97 passing tests** | **130 passed in 10.94 seconds** before removal of an unused source attribute; **28 source tests passed again** after that cleanup. No full-suite rerun after the cleanup is claimed. The 97-test result remains historical. |
| Ruff lint and formatting | Earlier checks exist in project work | **Passed:** Ruff lint and formatting checks on frozen release code. |
| Dependency consistency | Earlier environment snapshot | **Passed:** `pip check`. |
| Static type checking | No configured type checker | Not run; no static type-check pass is claimed. |
| GitHub Actions | Not yet published at audit start | [Run 35457051935](https://github.com/Gaganpraveen/papertrail/actions/runs/35457051935) passed on published application commit `c9f2eda`: Python 3.11.16, 130 tests in 12.35 seconds; Python 3.12.14, 130 tests in 12.24 seconds. Both also passed Ruff lint/format and CLI smoke checks. |
| Wheel packaging and isolated install | Earlier package check | Wheel built and installed in an isolated temporary environment. CLI help imported from the wheel and included `web` and `digest`; passed. |
| Browser known-ID workflow | Historical ready session and browser QA artifact | Final versioned-URL session `0d03c77529e0` reached `ready`, 15 pages/55 passages, after grounding fixes. |
| Browser topic workflow | API HTTP 406 observed during earlier validation | `research about electron` completed as `620d2cdae9d1`, eight candidates, 15 pages/44 chunks. Some other queries still receive 406. |
| Browser cited QA and abstention | Historical saved answers | Final hardware answer and 300,000-step detail checked against original PDF; warmup answer confirmed as 4000; dollar-cost question abstained. Topic DSR evidence verified on pages 1/2. |
| Browser failure handling | Code and offline tests cover errors | Final invalid-input session `1bf62c413c3f` verified the corrected start-a-new-briefing message, with no misleading resume hint. |
| Saved artifacts and links | Local report, Markdown, JSON, terminal cast/transcript, evaluation reports present | Final `release-topic` and `release-attention` HTML/Markdown/JSON exports exist and were inspected. Relative document links resolve. |
| Final 13-case model evaluation | Earlier evaluation reports remain available | **Completed:** [release report](../examples/evaluation-release.json), 11/13 expected-status matches (84.6%), 8/10 answerables answered, 3/3 controls abstained, zero execution errors and two false abstentions. All three retrieval modes found labeled support in the top five for 10/10 questions. This is a same-paper developer smoke evaluation, not held-out accuracy. |
| Public GitHub contents | Initial README reported at audit start | **Published:** `codex/release-ready`, [PR #1](https://github.com/Gaganpraveen/papertrail/pull/1), subsequently merged into `main` at `d92aea7`. Application commit `c9f2eda` has the exact verified local tree `05b9c9d5ea565e97e3b5a52d084203ef9a7f158c`. The earlier HTTP 403 was resolved by granting the connector access to this repository. Prior local history is preserved on `codex/local-release-snapshot`. |
| GitHub Pages | `docs/index.html` exists locally; intended URL is [gaganpraveen.github.io/papertrail](https://gaganpraveen.github.io/papertrail/) | **Deployed and browser-verified:** [saved demo](https://gaganpraveen.github.io/papertrail/) from `codex/release-ready` `/docs`. [Pages build 35457127406](https://github.com/Gaganpraveen/papertrail/actions/runs/35457127406) succeeded. The deployed report shows session `0d03c77529e0`, all three QA exchanges, and an expanded hardware quotation linking to PDF page 8. This is a saved report; new paper processing and QA run locally. |

This document distinguishes historical evidence from final release checks. A successful pipeline does not establish every claim’s correctness; the evaluation notes retain the false abstentions and limits of same-model support review.

# Release rescue verification

This record distinguishes the observed failures, changes, and checks for the release rescue from the earlier saved demonstration. The prior published release commit `d229442` remains on remote `codex/release-ready`; `codex/pre-rescue-release` is an additional local backup branch. The rescue work is on `codex/release-rescue`.

## Actual initial failure

The reported input was:

> Concierge Agent: A Local Hybrid Vision-Language Framework For Privacy Preserving Autonomous Computer Use

Reproducing this exact input failed at **arXiv metadata retrieval in 0.281 seconds**. Parsing, indexing, model loading, generation, and evidence review did not run. The evidence does not support blaming the model for this failure.

Additional bounded official API probes returned empty HTTP 406 responses for `concierge` (0.344 s), `transformer` (0.257 s), the phrase `machine learning` (0.256 s), and `quantum` (0.322 s). The responses contained `via: 1.1 varnish, 1.1 varnish`, `x-cache: MISS, MISS`, and `cache-control: private, no-store`. A live `all:electron` request succeeded in 0.274 s with a valid Atom feed and Google Frontend/Varnish headers. These observations suggest an upstream/cache-dependent failure; the exact source-side cause is not established.

The exact Concierge title is present on its [publisher's page](https://pspac.info/index.php/dlbh/article/view/395), DOI `10.46121/pspc.54.2.50`. No arXiv ID was verified for that title. PaperTrail's current intake is arXiv-only, so success cannot be promised for this publisher-hosted paper. No unrelated paper or hard-coded title mapping was substituted. The HTTP 406 failure remains visible rather than being marked fixed.

## Measured successful baseline

A fresh processing run of *Attention Is All You Need*, session `15d7a2fe9c9a`, took **58.660 seconds** before the rescue changes. It reused cached metadata; this is not a cold-source measurement.

| Stage | Seconds | Interpretation |
| --- | ---: | --- |
| Metadata retrieval | 0.000 | Cached response |
| PDF fetch | 0.258 | Measured fetch stage |
| PDF parsing | 0.781 | Measured parser stage |
| Indexing | 0.016 | Measured index stage |
| Briefing and evidence review | 57.585 | Included model calls below |
| Entire processing run | 58.660 | Includes small unlisted orchestration stages |

The first draft's Ollama-reported duration was 21.595 s: model loading 3.068 s, prompt evaluation 3.754 s, and token generation 14.746 s. Evidence-review requests reported 8.436 s, 24.015 s, and 3.416 s. These model subtimings are nested within the briefing stage and must not be added again to the total. Server-reported durations and application wall-clock timings have different boundaries.

This baseline shows that model generation and evidence review dominate a successful run, while the reported title failed earlier at the source. Both cases previously looked like undifferentiated waiting in the browser.

## Focused implementation changes

- The browser reports the actual graph/model stage, stage elapsed time, completed-stage timings, and total time against its deadline.
- A spawned worker isolates each browser operation. The parent enforces `PAPERTRAIL_OPERATION_TIMEOUT` (240 seconds by default), terminates timed-out work with a bounded kill fallback, and releases the operation slot after cleanup. Source retrieval, PDF work, inference, review, and export share that deadline.
- Model loading is visible separately from generation and evidence review. Ollama timing metadata is retained with completed model events. Stream interruption does not produce an accepted partial answer.
- Browser resume continues from durable graph checkpoints. Failed QA can be retried against the saved paper and index. Refresh/page reload reconnects to active work instead of submitting a duplicate job.
- Transient source retries remain bounded; access-denied responses are not repeatedly retried. A successful equivalent canonical topic request is cached under the original parameters to avoid repeating a known rejected request on subsequent runs. All query terms and sort intent are preserved.
- Existing evidence provenance, numeric checks, bounded repair, semantic support review, and abstention remain enabled. A failed check is not silently accepted to improve latency.
- Numeric checks recognize attached units such as `22meV` and decimal values without weakening quotation or semantic review. When no Abstract heading is detected, briefing evidence includes the first two opening-page passages. This corrected an omitted opening abstract in the physics topic run.
- Parser version 4 infers two-column reading order for documents such as BERT instead of interleaving adjacent columns; the report warns that this inference is heuristic. Unicode NFKC normalization handles PDF ligatures consistently during quote checks. Evidence still has to match its source passage.
- BM25 ignores question boilerplate and normalizes simple plural and hyphen variants. This generic lexical correction addresses observed missed evidence without paper-specific mappings or rebuilding embeddings. New digests use a parser-versioned vector collection; existing sessions retain their saved parsed artifacts instead of being silently rewritten.

## Real timeout and resource-recovery check

A dedicated local browser server on `127.0.0.1:8766` was started with `PAPERTRAIL_OPERATION_TIMEOUT=5`. The browser resumed the saved BERT briefing checkpoint. Its operation stopped at **5.1 seconds**, rather than continuing to hold the application busy.

The actual Ollama log confirms cancellation of task `2770`, release of its inference slot, and then `all slots are idle`. Immediately afterward, a new browser question against existing Attention session `0d03c77529e0` completed in **3.7 seconds**, with Ollama reporting **2.664 seconds** of inference. The follow-up used the existing paper and index; it was not a fresh-paper acceptance run. The local diagnostic capture is `.cache/rescue-timeout-evidence.json` (an ignored runtime artifact).

This check demonstrates cancellation of a real in-flight model task, resource release, and acceptance of later work. It does not establish that every paper completes within the default deadline or that a fresh briefing is correct.

## Fresh acceptance runs

These were actual new browser operations, not replayed saved demonstrations. Source metadata/PDF and compatible indexes were reused where noted; a new session does not imply a cold-cache run. The final exact-paper and topic runs are recorded below; complete CI/publication state is separate from the local measurements.

| Workflow | Session/input | Observed result | Timings |
| --- | --- | --- | --- |
| Unfamiliar paper URL and briefing | `https://arxiv.org/abs/1810.04805`, session `7016d3da0645` | Resolved to `1810.04805v2`, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. Fresh briefing completed after parser/quote corrections; one result paraphrase used visibly labeled source wording with a warning. | Browser 31 s |
| Supported QA and source inspection | BERT: “What does BERT stand for?” | Answered “Bidirectional Encoder Representations from Transformers.” The exact attached quotation was inspected in the browser and the original PDF, page 1, chunk `e1796cabe1b618c0`. | Backend 4.579 s; browser 5.3 s |
| Unsupported QA | BERT: “What was the total training cost in US dollars?” | `insufficient_evidence`, no factual claims; no cost was invented. | Backend 2.472 s; browser 3.2 s |
| Realistic topic, selected paper, and briefing | `research about electron`, final session `d757bedd4a1c`, selected arXiv `1401.3078v2` | Selected *Ultrafast Electron Dynamics in the Topological Insulator Bi2Se3 Studied by Time-Resolved Photoemission Spectroscopy*. Fresh summary is coherent plain-English synthesis; one result retains labeled source-wording fallback and a warning. | Browser 36 s |
| Topic supported QA | “What was the total energy resolution of the setup?” | Answered less than 22 meV, citing the exact source passage on PDF page 3, chunk `4455b47adb1a80bf`; verified in the browser and original PDF. | Backend 4.050 s; browser 4.8 s |
| Topic unsupported QA | “What was the total research budget in US dollars?” | `insufficient_evidence`, no factual claims. | Backend 0.800 s; browser 1.5 s |
| Failure recovery and deadline cleanup | BERT checkpoint on dedicated port 8766; subsequent question on existing session `0d03c77529e0` | Real deadline cancellation, Ollama slot release/idle, and a successful subsequent question verified; this is resource recovery only. | Deadline 5 s; observed stop 5.1 s; next QA 3.7 s |

The [BERT HTML report](../examples/rescue-bert/report.html), [briefing](../examples/rescue-bert/briefing.md), and [session record](../examples/rescue-bert/session.json) preserve the actual fresh results. The one source-wording fallback is not a successful semantic-review verdict: inspect its quotation and warning. The application retained that distinction rather than hiding the rejected paraphrase.

### BERT stage timings

| Stage | Seconds | Cache/measurement boundary |
| --- | ---: | --- |
| Metadata retrieval | 0.000 | Cached official metadata |
| PDF fetch | 0.002 | Cached PDF |
| PDF parsing | 0.840 | Actual parser pass |
| Indexing | 0.028 | Compatible existing vector index reused |
| Model loading | 0.001 | Model already warm |
| Briefing generation | 18.715 | Application wall-clock model call |
| Evidence review, three requests | 3.759 + 3.968 + 3.065 | Application wall-clock model calls |
| Entire briefing stage | 29.618 | Includes loading, generation, reviews, and orchestration above |
| Browser operation | About 31 | Includes remaining graph work and browser polling |

The model substage times are nested within the entire briefing stage and must not be added twice.

### Final topic stage timings

The [topic report](../examples/rescue-topic/report.html) and [session record](../examples/rescue-topic/session.json) preserve session `d757bedd4a1c`. Official metadata and the PDF were cached, while briefing generation and QA were fresh. The 10-page paper produced 57 passages.

| Stage | Seconds |
| --- | ---: |
| Metadata retrieval, cached | 0.000 |
| Candidate selection | 0.601 |
| PDF fetch, cached | 0.007 |
| PDF parsing | 0.636 |
| Index reuse | 0.030 |
| Warm model load | 0.001 |
| Briefing generation | 21.830 |
| Evidence review, three calls | 4.539 + 4.491 + 3.120 |
| Entire briefing stage, including model substages | 34.033 |
| Browser operation | About 36 |

The summary uses the opening-page abstract evidence that was missing from the earlier selection. One result still uses exact source wording after support review; its visible warning remains a reading-quality limitation.

### Failures retained and corrected

The first BERT session, `0cd5dd7a02fe`, correctly selected the paper but failed quotation construction. Recovery completed in 44 s; browser inspection then exposed cross-column PDF extraction. A later diagnostic session, `a38837a872b4`, recorded two false abstentions before the lexical retrieval correction. These intermediate failures and retries remain in [the diagnostic record](../examples/rescue-verification.json). They are not counted as successful acceptance runs. The final BERT session above was generated after the parser and lexical corrections.

Earlier physics session `51fc27237fea` failed after 44 s on attached-unit validation (`22meV`). After that correction, resume completed in 28 s but produced a weak summary from incomplete evidence. Including opening-page passages when an Abstract heading is absent corrected that evidence-selection gap; the separate fresh session `d757bedd4a1c` above is the final topic result. The failed and weak intermediate results remain in the diagnostic history.

### Same-paper latency observation

After the changes, a new Attention session `e830d31be1c4` completed in **31.266 s** of graph time, approximately **32 s** in the browser. Evidence review took **10.803 s** of application wall-clock time. The earlier same-paper baseline took 58.660 s.

This is an observation, not a controlled speedup benchmark. The later model was warm, whereas the baseline included 3.068 s of model loading, and generated claims/review work differed. The implementation keeps grounding checks and exposes their cost; it does not promise a fixed improvement for other papers or hardware.

## Regression and publication status

The complete local suite passed **175 tests in 20.99 s** before the final opening-page evidence correction. After that correction and artifact-path redaction, **12 affected graph/evaluation tests passed in 0.44 s**, including the new regression. This distinction preserves the actual verification order; the full final CI result must be read from its completed run.

The [retrieval-only regression](../examples/evaluation-rescue.json) was rerun against the existing Attention question set after parser and lexical changes. Dense, BM25, and hybrid retrieval each found labeled support in the top five for **10/10** answerable questions; hybrid mean reciprocal rank was **1.0**. The model answer evaluation was **not rerun**. This developer-constructed, single-paper result does not establish correctness of generated answers or generalization to other papers.

- Local automated verification: 175-test full suite, followed by 12 affected tests after the last targeted correction, as recorded above. Final full GitHub CI status is separate.
- Ruff lint and formatting, dependency consistency, wheel build/install, and CLI verification: **passed locally**.
- Release branch: [`codex/release-rescue`](https://github.com/Gaganpraveen/papertrail/tree/codex/release-rescue). This measurement record does not assert remote publication or CI completion before verification.
- Public live application: **not deployed**. Suitable authenticated hosting compute was unavailable. No paid text-generation provider was introduced, raw Ollama was not exposed, and private shared session storage was not published.
- Live local entry: after [setup](../README.md#setup), run `papertrail web` and open [http://127.0.0.1:8765](http://127.0.0.1:8765). The original demonstration Mac uses `PAPERTRAIL_OLLAMA_URL=http://127.0.0.1:11435 .venv/bin/papertrail web`.
- [GitHub Pages](https://gaganpraveen.github.io/papertrail/) remains a **static saved report**, not a live deployment.

## Remaining boundaries

arXiv topic availability is not guaranteed. No arbitrary PDF upload, publisher-wide search, OCR, model migration, or new frontend framework was added. The browser remains a local single-user application. Grounding review can falsely reject a valid answer or accept an unsupported interpretation; quoted evidence and original pages require inspection. The earlier single-paper evaluation cannot establish broad research accuracy.

Published historical artifacts replace the local workspace prefix with `<project>`. Source hashes, answer content, and timings are unchanged.

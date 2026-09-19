# Evaluation scope and reproduction

This is a small **developer-constructed smoke evaluation**, not a blind or held-out benchmark and not a statistical estimate of research-assistant quality. The ten answerable questions were written after inspecting the actual parsed text of *Attention Is All You Need*, arXiv `1706.03762v7`. Three additional questions request information absent from that source. No retrieval scores were consulted when constructing these initial labels.

`evals/attention-questions.json` contains the questions, expected answer/abstention status, exact supporting snippets, and one-based PDF page numbers. The source PDF SHA-256 is pinned in the file. Gold snippets preserve extraction quirks, including the misplaced `model` subscript in the parsing passage. They are a small, incomplete set of relevant passages, not exhaustive relevance judgments. A semantically valid alternative passage can therefore score as a miss. The German translation result accepts either the abstract or the results-section statement. The dataset excludes the English-to-French score because the parsed abstract/table and body report different values.

From the repository root, after creating a session for this PDF and installing the project:

```sh
python scripts/evaluate.py --session 0d03c77529e0 --validate-only
python scripts/evaluate.py --session 0d03c77529e0 --output .cache/evaluation-retrieval.json
```

Use a session ID from the local `papertrail sessions` output when reproducing on another machine. `--data-dir` defaults to `.papertrail`. The script reads the saved SQLite session in read-only mode, verifies its PDF checksum against the gold labels, and verifies that every gold snippet occurs on its declared page before opening the index. `--validate-only` stops there and performs no retrieval or model calls. The saved session must already have its vector collection; this script does not rebuild a missing index. It acquires the same operation lock as the CLI, so wait for the demo or another command to finish before running it.

The default run compares the application's `dense`, `bm25`, and `hybrid` modes with the same verbatim question, non-reference chunks, overlap suppression, and `k=5`. These are the actual application retrieval modes rather than independent reimplementations. Dense ranking uses cosine similarity in the saved Qdrant collection; BM25 uses the application's lexical scores; hybrid uses reciprocal rank fusion. The reported `score` is the application's rank-fusion score even in a single mode; raw `dense_score` and `lexical_score` are recorded separately when available.

For the **ten answerable questions only**:

- **Support-hit@5** is the fraction with at least one returned passage containing at least one normalized gold snippet. Normalization uses the application's Unicode, whitespace, line-break hyphenation, and case normalization. Gold pages establish annotation provenance; retrieval matching is based on passage text.
- **MRR@5** averages `1 / first_support_rank`; a question with no matching passage in the first five has reciprocal rank zero.

The three unanswerable controls have null retrieval metrics and are excluded from both denominators. Retrieval always returns plausible candidates; finding a passage for an unanswerable question is not itself evidence of a failure or successful abstention.

The JSON report preserves the top-five passage IDs, ranks, pages, scores, matching-gold indices, and per-question timings. Full retrieved passage text stays in the local parsed artifact; use the recorded IDs to inspect it. The report also records session/model names, PDF and parsed-artifact checksums, dataset checksum, package versions, and protocol. Timings are a single sequential pass and include lazy loading on the first query, so they are not a fair latency benchmark. The tiny, single-paper sample does not establish that hybrid retrieval is better in general, even if its measured score is higher here. Failed retrieval runs do not produce a completed result report; existing output files should not be mistaken for a new run.

Optional local answer checks use hybrid top-five passages:

```sh
python scripts/evaluate.py --session 0d03c77529e0 --answers --output .cache/evaluation-with-answers.json
```

This calls `Ollama.answer` directly with empty prior-question history and the saved session's model. It does not append exchanges or alter the demo session. This deliberately differs from conversational `Agent.ask`, which can expand referential questions and normally retrieves seven passages. The model availability/digest, generated responses, calls/tokens, and answer timings are saved. The application can perform multiple generation/review calls per question, so this option is slower than retrieval alone.

Generation conservatively withholds dense numeric excerpts and selected displaced formula lines because flattening PDF layout can lose column or mathematical relationships. The original text remains in the parsed artifact and retrieval index. These heuristics are not a table or equation parser and can also withhold useful evidence. Short factual sentences, including scalar settings, remain eligible for citation. Retrieval metrics are computed on full retrieved passage text, before this generation-context filtering, so a support hit does not guarantee that every part of that passage is available as a model citation.

**Status agreement is not answer correctness.** It tests only whether the model returns `answered` for the answerable questions and `insufficient_evidence` for the controls. Execution errors count as disagreements and are listed separately. A grounding rejection uses the application's safe abstention behavior and is separately flagged, so a rejection is distinguishable from a deliberate abstention. Provenance checks are reported separately: known retrieved chunk IDs, quotes matching those chunks, and numeric values appearing in cited evidence. Abstentions have no claims to check. These deterministic checks do not establish semantic entailment, and the application's same-model support review is not an independent judge. A human must review the actual answer and quoted support to assess correctness, completeness, and calibrated uncertainty.

The offline unit tests validate scoring, source-label checks, unanswerable denominator handling, and answer-error reporting; synthetic test passages are never presented as live retrieval results.

## Current verification record

[Release rescue verification](release-rescue.md) records the later parser/retrieval corrections, fresh BERT and topic workflows, timeout recovery, and current regression results. The model-answer results below belong to the earlier implementation and have not been relabeled as a new benchmark.

## Initial release smoke run (historical)

The initial [release evaluation report](../examples/evaluation-release.json) was generated on 2026-09-19 at **16:11:19 UTC** with `--answers`, ready session `0d03c77529e0`, `BAAI/bge-small-en-v1.5` embeddings, and local `qwen3.5:4b` (`Q4_K_M`). It uses the release implementation with short-fact handling, formula/qualification guards, and focused QA instructions. The report records the full Ollama model digest and input checksums. The commands above write new reproductions under `.cache` to preserve the committed report history.

| Retrieval mode | Support-hit@5, 10 answerable questions | MRR@5 |
| --- | --- | --- |
| Dense | 10/10 (100%) | 1.0000 |
| BM25 | 10/10 (100%) | 0.8833 |
| Hybrid | 10/10 (100%) | 1.0000 |

The hybrid answer checks completed all **13 status cases**, with **11/13 status agreement (84.6%)** and **zero execution errors**. Eight of the ten answerable questions received answers; all three unanswerable controls received abstentions. Both answerable abstentions followed validation rejection. This is **not 84.6% answer accuracy**: the metric checks status only, and exact gold retrieval did not guarantee a useful answer.

Inspection of all 13 saved cases against their attached quotations and gold evidence found:

- `q01`, `q03`, `q04`, `q05`, `q06`, `q07`, `q08`, and `q09` contain answers supported by their attached evidence. No material unsupported accepted claim was identified in this inspection. This qualitative assessment is not an independent accuracy score.
- The prior table-column attribution error in `q08` and mathematical misreading in `q04` are absent. `q03` now preserves the source's **many appear** qualification. `q07` correctly gives the linear warmup, 4000-step duration, and inverse-square-root decay; the short duration sentence is available as evidence.
- `q02-decoder-masking` and `q10-parsing-small-data` still falsely abstain despite rank-one gold support. The saved rejection reasons concern an incomplete masking quotation and an unnecessary semi-supervised parsing claim. A failed extra claim can prevent a supported core answer from being returned. The report preserves the errors but not every rejected draft, limiting diagnosis of the generation/review interaction.
- `q07` and `q08` repeat supported information. `q09` includes a truncated extra quotation alongside a complete quotation that independently supports its beam-size and length-penalty answer. These remain presentation and evidence-selection weaknesses.
- All three unanswerable controls abstain, consistent with the developer labels.

All eight returned answers passed the recorded deterministic provenance checks. Neither those checks nor this qualitative inspection establish reliability outside these cases. The evaluation's 13 questions are separate from the CLI recording and subsequent browser QA exchanges, and the harness does not append them to session history.

## Earlier development runs retained unchanged

| Report, all on 2026-09-19 | Session | Status agreement | Material observations |
| --- | --- | --- | --- |
| [Before numeric guard, 15:37:02 UTC](../examples/evaluation-before-table-guard.json) | `4ca1290ef789` | 10/13 (76.9%) | Accepted `q08` Deep-Att table-column attribution error; qualification loss. |
| [After numeric guard, 15:41:56 UTC](../examples/evaluation.json) | `4ca1290ef789` | 10/13 (76.9%) | Table error absent; accepted `q04` flattened-math misreading; `q07` false abstention. |
| [Initial release, 16:11:19 UTC](../examples/evaluation-release.json) | `0d03c77529e0` | 11/13 (84.6%) | Earlier accepted errors absent in this run; `q02` and `q10` false abstentions remain. |

In the first run, the accepted `q08` answer grouped Deep-Att among English-to-German baselines although that model's table entries are English-to-French. In the second run, an extra `q04` claim called a positional encoding a linear function of `pos+k`, misreading the source relationship between `PE(pos+k)` and `PE(pos)`. Both passed the same-model review and deterministic quotation checks. These reports remain available because they demonstrate failures that aggregate status metrics missed.

Those observations informed the general guards and prompt changes. Every run reused this same developer-constructed set; the final 11/13 result is an in-development smoke check, **not a held-out demonstration of improved accuracy**. The guards do not establish general table/equation understanding, and may exclude useful evidence. Retrieval scores are unchanged across these runs; generation and validation behavior changed.

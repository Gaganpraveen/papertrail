# PDF upload verification

These checks used fresh browser operations in an isolated application data directory. Existing sessions were preserved. Downloaded model weights were reused, but the directory initially contained no source-paper cache or vector index. The measurements below describe this machine and these inputs, not a general latency guarantee.

## Uploaded research paper

The input was Leo Breiman's *Random Forests* ([official Berkeley PDF](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)), a readable, non-arXiv PDF with 33 pages. The application identified it as an uploaded document; it did not invent an arXiv ID or bibliographic metadata. The source file was retained locally for evidence inspection.

| Browser operation | Observed result | Measured time |
| --- | --- | --- |
| Upload and create briefing | Completed with newly parsed passages, a new index, and local generation/support review | 42 s in browser |
| “How does the forest combine the generated trees for classification?” | Answered with a quotation verified against the original PDF's page 2 | 12.366 s backend; 13 s in browser |
| “What was the total research budget in US dollars?” | Abstained with `insufficient_evidence`; no unsupported factual answer displayed | 3.295 s backend; 4 s in browser |
| Upload a malformed PDF | Rejected during parsing with a damaged/encrypted-file explanation; no model inference or briefing | 0.7 s in browser |

For the first upload, parsing took 0.709 s, indexing 5.230 s, and model loading 3.340 s. Briefing generation took 20.512 s; the three evidence-review batches took 4.119 s, 3.737 s, and 3.349 s. The supported answer required a repair after an initial support rejection; that failure was retained in the session trace rather than hidden.

## arXiv regression and failure recovery

The previously failing exact input, `https://arxiv.org/abs/1901.00003v3`, was processed as *Learning Spatial Common Sense with Geometry-Aware Recurrent Networks*, retaining revision **v3**. The API returned HTTP 406; the application recovered metadata from the official arXiv abstract page, verified its explicit revision marker, and reported this fallback. Download URLs were constructed from the verified ID.

The first run used a fresh source download and index and completed its briefing in **56 s** in the browser. Metadata retrieval took 6.063 s, PDF download 9.884 s, parsing 0.945 s, and indexing 5.087 s. The briefing phase took 33.703 s, including generation and evidence review.

That run exposed two false abstentions: “What does GRNN stand for?” took 2.585 s; “What are the networks trained to predict?” took 15.687 s and failed support validation. Both questions have source support, so these are recorded as failures, not successful abstention tests. The unsupported research-budget question correctly abstained in 3.538 s.

Parser version 5 corrects the observed column-reading problem. A subsequent new session reused the source metadata/PDF but performed new parsing, indexing, briefing generation, and support review. It is not a cold-source measurement.

The corrected arXiv briefing completed in **42 s** with source metadata/PDF reuse and fresh parsing/indexing/generation. The prediction question now answered in **8.769 s backend / 9.6 s browser**, with original PDF pages 2 and 5 checked visually. The research-budget question abstained in **2.062 s / 2.9 s**. The GRNN acronym question still falsely abstained in **2.334 s / 3.0 s**; that limitation remains.

## Fresh topic: a retained failure

`research about quantum` selected the relevant, previously unused `2003.11810v4`, *Searching for Coherent States: From Origins to Quantum Gravity*. Identity was checked against the original PDF. Metadata retrieval took 3.026 s, PDF download 3.485 s, parsing 0.919 s, and new indexing 6.131 s. No source/index cache was reused on this initial run.

The initial briefing **failed after 56 s**: a draft included an uncited numeric value, and its bounded repair removed an uncertainty qualifier. Validation now reports all failing briefing fields together without changing any check or adding attempts. Browser **Resume saved session** reused the completed stages, and page refresh reconnected to the active job. The resumed briefing **still failed after 37 s** on uncertainty qualifiers. No briefing or topic QA was accepted; topic end-to-end success is **not established** by this test.

The corrected parser was also run over the uploaded Random Forests source: all 82 passages, IDs and warnings were unchanged. Its existing successful checks remain valid.

After switching from the arXiv paper back to the uploaded Random Forests document, the question “Does this paper train GRNNs to predict novel camera views?” correctly abstained in **4.7 s browser time**. The prior paper’s answer was not reused. The live picker, filename/stage display, quotations and page links were checked; native operating-system drag-and-drop was not separately automated. The in-app browser did not render the original PDF preview, although the local document route and source bytes were verified in HTTP tests.

## Regression checks and boundaries

- The complete automated suite passed **224 tests** after the upload, parser and repair-diagnostic changes. These tests include explicit test doubles and do not establish general model accuracy.
- Source tests cover missing or ambiguous HTML metadata, paper/revision mismatches, untrusted PDF URLs, injected source text, and access-denied responses. HTTP 401/403 do not trigger the HTML fallback.
- PDF upload provides a path for a locally available paper when external retrieval is unavailable. It does not make arXiv topic searches universally reliable; the fresh `diffusion` API probe still returned HTTP 406.
- Scanned/image-only, damaged, encrypted, oversized, or otherwise unsupported PDFs can fail explicitly. Complex layouts, equations, tables, and imperfect retrieval can still cause omissions or false abstentions.
- Briefing source-wording fallbacks remain visibly labeled. Inspect the quotations before relying on the interpretation.
- Raw verification PDFs, session JSON, screenshots, and application data are local verification material and are not included in this public document.

Use the [README setup and live application instructions](../README.md) to process a new paper. Saved reports are evidence records, not a public live inference service.

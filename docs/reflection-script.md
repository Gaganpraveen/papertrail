# Reflection recording draft

This is a first-person draft to adapt, not a statement of Gagan's personal experience. Replace or remove anything that does not reflect your real contribution. Describe AI/tool assistance accurately if required by the assessment. Do not invent debugging stories, user research, test results or measured performance. Record the reflection yourself.

Aim for approximately 3 minutes 40 seconds to 3 minutes 50 seconds, leaving buffer under a four-minute limit. Rehearse with a timer and shorten as needed. The timing plan concerns the recording, not application latency.

| Target | Focus | Visual |
| --- | --- | --- |
| 0:00-0:30 | Problem and scope | Repository and saved briefing |
| 0:30-1:10 | Graph and persistence | State diagram and session |
| 1:10-2:00 | Retrieval and evidence checks | Claim, quotation and PDF page |
| 2:00-2:45 | Follow-up and durable history | Actual QA result |
| 2:45-3:30 | Tradeoff, limitation, improvement | Relevant code or documented limitation |
| 3:30-3:50 | Personal contribution and learning | Camera or concise closing slide |

## Spoken draft

Hello, I'm Gagan P, an AI and Machine Learning graduate. This is PaperTrail, a local assistant for reading research papers. A user can enter a research topic or arXiv ID, get a briefing for one selected paper, and ask follow-up questions. The key idea is to keep the path from an answer back to its source visible.

The workflow is an explicit Python state graph. It understands the input, retrieves and selects a paper, downloads the PDF, parses it into page-aware passages, builds an index, and creates a validated briefing. SQLite saves the current state and the next node. If a stage fails, a resumed run re-enters that stage instead of restarting the entire pipeline. Qdrant stores the passage vectors separately.

For retrieval, dense embeddings help find related meaning, while BM25 helps with exact scientific terms. Reciprocal-rank fusion combines their rankings. The default embedding model runs on CPU through FastEmbed, and Qwen3.5 4B runs locally through Ollama.

The evidence design is the part I would highlight. The model sees source spans with short IDs, selects those IDs and writes a claim. Python copies quotation text from the source registry and checks IDs, source matches and numerical provenance. A second model pass reviews support. Rejected briefing prose can become clearly labeled source excerpts for inspection; rejected limitations are removed. QA instead abstains on rejected support. Neither review nor copying source text guarantees correctness.

Here is a saved briefing. I can open a quotation and follow its page reference to the original paper. I can also ask a follow-up using the same session. The answer is saved with its evidence and can abstain. The local browser and CLI share the same agent. The browser submits background jobs so it stays responsive. The exported HTML and public demo display saved results.

One tradeoff is the local setup. It avoids paid API credentials and keeps the application self-contained after model downloads, but speed and quality depend on the machine and model. Another limitation is PDF extraction: equations, figures and complex tables are not reliably understood. A relevant passage can also be missed during retrieval.

My next improvement would be a larger held-out evaluation set. I would measure whether retrieval finds the required evidence, compare dense, BM25 and hybrid retrieval, and review claim support and abstention separately. That would help identify whether the next change should target parsing, retrieval or generation.

Before recording, replace this closing paragraph with two truthful sentences: what you personally implemented or understood most deeply, and one concrete thing you learned. If you used AI assistance, describe your own review, testing and decisions accurately. End with the main lesson you can defend from this project.

## Adaptation prompts

- Which file or function can you explain without reading from notes?
- What real failure did you observe, and how did the code or workflow handle it?
- Which choice did you make or verify personally, and what alternative did you consider?
- What outcome can you demonstrate from a real saved run?
- What help or tools did you use, and what did you independently check?

Do not read these prompts or the replacement instruction aloud. Prepare the final closing in your own words, then time the complete recording. If a live question is too slow for the recording, show a clearly labeled saved result and explain how it was produced.

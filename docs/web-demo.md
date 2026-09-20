# Live local application

The live application is **[http://127.0.0.1:8765](http://127.0.0.1:8765)** after starting the server. The public GitHub Pages page is a static saved report: it cannot accept a new paper or run inference. All live model computation runs on the machine running PaperTrail and Ollama.

Follow the [README setup](../README.md#setup-and-run), start Ollama, then run:

```sh
papertrail doctor --warmup
papertrail web --port 8765
```

Keep that terminal and Ollama running. The default Ollama endpoint is `http://127.0.0.1:11434`. The original demonstration Mac uses port 11435:

```sh
PAPERTRAIL_OLLAMA_URL=http://127.0.0.1:11435 .venv/bin/papertrail web
```

## One walkthrough

1. Open the live local application. Drag a text-readable PDF onto the upload area or select **Choose PDF**. Alternatively, enter `1810.04805` and choose **Create briefing**. IDs and plain arXiv URLs resolve to a specific paper; a topic search selects one candidate and shows its title.
2. Read the actual processing stage and elapsed time. Source retrieval, PDF download/parsing, indexing, model loading, briefing generation, and evidence review are reported separately. Completed timings remain visible while the next stage runs.
3. Verify the selected title before interpreting the briefing. Ask a question about the paper, then expand its quotations and inspect the original PDF page.
4. Ask an unsupported question, such as the total training cost in US dollars. An abstention is a valid result when retrieved evidence does not support an answer.
5. Use **Open full report** to inspect the saved evidence reader. This saved report is separate from the live controls.

PaperTrail accepts PDF uploads, arXiv IDs, arXiv URLs, and research topics. Uploads do not use arXiv. The filename is shown as a display title; unverified bibliographic metadata stays unknown. PDFs must contain readable text and fit the 30 MiB / 100-page limits. Encrypted, corrupt and image-only files are rejected. It does not search every publisher. Some valid topic requests receive HTTP 406 from arXiv; the application reports that failure and does not replace the requested paper with a canned demonstration.

## Progress, deadlines, and recovery

Each accepted browser upload, digest, resume, or question has a **240-second total deadline**, configured by `PAPERTRAIL_OPERATION_TIMEOUT`. This includes worker startup, source access, parsing, indexing, model loading, generation, evidence review, and export. It is separate from the model HTTP timeout. The browser shows stage elapsed time and total elapsed time against this deadline.

Operations run in an isolated spawned worker. On deadline or server shutdown, the parent terminates that worker and, if needed, kills it after a bounded wait. This closes its connections and releases its process-held storage lock; later requests can run. Incomplete model streams are never accepted as answers. The Ollama service remains available for subsequent operations; stopping an operation does not shut down the user's Ollama server.

- **Resume saved session** continues a failed or interrupted pipeline from the last successful graph checkpoint. Completed download, parsing, and indexing nodes are retained. A failed node itself may need to run again.
- **Retry last operation** resumes a recoverable pipeline or resends the failed question. QA uses the saved paper and index; it does not rebuild a briefing.
- **Refresh**, or reloading the page, reconnects to the current active job. Polling also makes two bounded reconnect attempts after connection errors. Neither action creates duplicate work.
- An invalid input must be corrected before creating a new session. Repeating the invalid input cannot repair it.
- If the server process restarts, its in-memory job registry is lost, while graph checkpoints remain on disk. Select the saved session and resume it. An interrupted question must be asked again.

Exact-ID requests rejected by the API with HTTP 406 can use the official arXiv abstract page. Its paper ID, explicit revision marker and citation metadata must match before a trusted PDF URL is constructed. This fallback does not replace a requested revision or bypass HTTP 401/403 access denials. Official source requests retry transient network failures and HTTP 429/5xx at most three times. HTTP 401/403 are not retried. An HTTP 406 topic request gets at most one equivalent canonical query preserving its terms and sort intent; a successful response is cached. Generation/schema repair is bounded to one retry, and the operation deadline bounds the entire browser job. Grounding checks remain enabled on retries.

## Local API and execution

| Method and path | Behavior |
| --- | --- |
| `GET /api/sessions` | List recent persisted sessions, checkpoint/recovery information, and existing reports |
| `GET /api/active` | Return the active job so the browser can reconnect |
| `POST /api/upload` | Receive bounded PDF bytes with `application/pdf` and a URL-encoded `X-Filename`; start the shared parsing pipeline |
| `GET /document/<session ID>/paper.pdf` | Open that uploaded session’s local PDF for page inspection |
| `POST /api/digest` | Start a job with JSON `{"query":"1810.04805"}` |
| `POST /api/resume` | Resume a saved pipeline with JSON `{"session":"<12-character ID>"}` |
| `POST /api/ask` | Ask a saved paper with JSON `{"session":"<12-character ID>","question":"What is the method?"}` |
| `GET /api/jobs/<job ID>` | Read status, stage, elapsed times, deadline, completed timings, and recovery details |
| `GET /report/<session ID>` | Read that session's existing generated HTML export |

Accepted work returns HTTP 202 promptly. Only one web job runs at a time; additional submissions receive HTTP 409. The data-directory lock also prevents conflicting CLI operations. Existing sessions restore their saved model configuration. The registry retains at most 100 jobs; graph checkpoints, reports, and QA history persist on disk.

## Request and hosting boundaries

The server binds only to `127.0.0.1`; it refuses wildcard, LAN, and IPv6 binds. Requests must have the expected local Host header. POST requests require the matching HTTP Origin and a single bounded Content-Length. Normal operations require JSON and valid text fields; uploads require PDF bytes, a bounded display filename, and a 30-second transfer limit. Cross-origin requests are rejected. Reports are selected by strict session ID and checked against the data directory; arbitrary files are not exposed.

Browser content uses no external scripts, fonts, or assets. A content security policy, escaped report content, and a sandboxed report frame constrain the frontend. This is a local single-user interface without authentication. Do not tunnel or reverse-proxy it onto a public address or expose raw Ollama. A hosted multi-user service needs authenticated access, session isolation, and resource limits.

No public live deployment was completed for this release rescue. The available Hugging Face account did not provide the required compute eligibility and repository-write access. A static Pages report does not satisfy live hosting. See [release rescue verification](release-rescue.md) for measured results and current limitations.

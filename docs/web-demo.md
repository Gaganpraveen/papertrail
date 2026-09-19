# Local live web demo

PaperTrail has a browser interface backed by the same Python agent, retrieval, validation and checkpoint storage as the CLI. All model computation runs on the machine running the server. The public GitHub Pages site remains a saved-report demonstration.

After following the README setup and starting Ollama:

```bash
papertrail doctor --warmup
papertrail web --port 8765
```

Open `http://127.0.0.1:8765` in your browser. Keep that terminal and Ollama running.

1. Enter an arXiv ID, arXiv URL or topic, then choose **Create briefing**. A topic search selects one paper.
2. The status area shows progress while a background job performs the local work. Initial downloads and generation may take several minutes.
3. Select a completed session, ask a question, and inspect its answer and source quotations in the report.
4. Use **Open full report** to inspect the saved evidence reader in a separate tab. **Refresh** reloads saved sessions from disk.

The interface makes genuine calls to the existing agent. It does not fabricate answers or return hard-coded demonstrations. An answer can abstain, and a run can fail validation. A failed job reports its error. Saved pipeline sessions can be recovered through `papertrail resume SESSION` after fixing the cause, then refreshed in the browser.

## Local API and execution

| Method and path | Behavior |
| --- | --- |
| `GET /api/sessions` | List recent persisted sessions and whether an HTML export exists |
| `POST /api/digest` | Start a job with JSON `{"query":"1706.03762"}` |
| `POST /api/ask` | Start a job with JSON `{"session":"<12-character ID>","question":"What is the method?"}` |
| `GET /api/jobs/<job ID>` | Read a job's running, succeeded or failed state |
| `GET /report/<session ID>` | Read that session's existing generated HTML export |

Accepted work returns HTTP 202 promptly. The browser polls the job endpoint while the model runs. Only one web job runs at a time; additional submissions receive HTTP 409. The existing data-directory lock also prevents conflicting CLI operations. Asking about an existing session restores that session's saved model configuration.

The job registry is in memory and bounded to the latest 100 jobs. Graph checkpoints, reports and QA history persist on disk. If the web process stops, job polling does not survive; use the saved session to recover pipeline work from the CLI. An interrupted QA request should be retried. Starting the server again does not automatically resume jobs.

## Request boundaries

The server binds only to `127.0.0.1`; it refuses wildcard, LAN and IPv6 binds. Requests must have the expected local Host header. POST requests require the matching HTTP Origin, JSON content type, a single bounded Content-Length and valid text fields. Cross-origin requests are rejected and no permissive CORS header is returned. Reports are selected by strict session ID and checked against the data directory; the server does not expose arbitrary files.

Browser content uses no external scripts, fonts or assets. A content security policy, escaped report content, and a sandboxed report frame constrain the frontend. This is a local single-user interface without authentication. Do not tunnel or reverse-proxy it onto a public address. A hosted multi-user service would need a separate authentication, resource and concurrency design.

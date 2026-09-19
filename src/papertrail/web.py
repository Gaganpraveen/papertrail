"""Loopback-only web adapter. Model work stays in the existing checkpointed Agent."""

import hashlib
import json
import re
import threading
from base64 import b64encode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

from papertrail.errors import PaperTrailError
from papertrail.graph import Agent, settings_for_session
from papertrail.rendering import export_run
from papertrail.storage import Store

SESSION = re.compile(r"[a-f0-9]{12}")
MAX_BODY = 8192
SCRIPT = r"""
const $ = id => document.getElementById(id);
let selected = '', busy = false, ready = false;
async function api(path, data) {
  const response = await fetch(path, data ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)} : {});
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'Request failed');
  return result;
}
function status(message, error=false) { $('status').textContent=message; $('status').classList.toggle('error',error); }
function controls(locked) { busy=locked; $('digestButton').disabled=locked; $('askButton').disabled=locked || !ready; }
function choose(session) {
  selected=session.id; ready=session.status==='ready'; $('selected').textContent=session.title || session.query;
  $('sessionId').textContent=session.id+' / '+session.model;
  $('askButton').disabled=busy || session.status!=='ready';
  $('reader').hidden=!session.report; $('empty').hidden=!!session.report;
  $('reportLink').hidden=!session.report;
  if(session.report) { $('reader').src='/report/'+session.id+'?v='+Date.now(); $('reportLink').href='/report/'+session.id; }
  else $('empty').textContent=session.status==='ready' ? 'No exported report yet. Ask a question to create one.' : 'Session '+session.status+'. If interrupted, resume it from the CLI: papertrail resume '+session.id;
}
async function sessions(preferred) {
  const result=await api('/api/sessions'); $('sessions').replaceChildren();
  result.sessions.forEach(session=>{
    const button=document.createElement('button'); button.className='session';
    const title=document.createElement('strong'); title.textContent=session.title || session.query;
    const detail=document.createElement('small'); detail.textContent=session.status+' / '+session.id;
    button.append(title,detail); button.onclick=()=>choose(session); $('sessions').append(button);
  });
  if(!result.sessions.length) $('sessions').textContent='Your saved papers will appear here.';
  const target=result.sessions.find(s=>s.id===(preferred || selected)) || (!selected && result.sessions[0]);
  if(target) choose(target);
}
async function submit(path, data) {
  controls(true); status('Starting a local job...');
  try {
    let job=await api(path,data);
    while(job.status==='running') {
      status((job.stage || 'Working')+' - running on this computer. You can keep reading.');
      await new Promise(resolve=>setTimeout(resolve,1200));
      job=await api('/api/jobs/'+job.id);
    }
    if(job.status==='failed') throw Error(job.error);
    await sessions(job.session); status('Saved. Open the evidence below to inspect this result.');
    if(path==='/api/ask') $('question').value='';
  } catch(error) { status(error.message,true); await sessions().catch(()=>{}); }
  finally { controls(false); }
}
$('digest').onsubmit=event=>{event.preventDefault(); submit('/api/digest',{query:$('query').value});};
$('ask').onsubmit=event=>{event.preventDefault(); submit('/api/ask',{session:selected,question:$('question').value});};
$('refresh').onclick=()=>sessions().catch(error=>status(error.message,true));
sessions().catch(error=>status(error.message,true));
"""
HTML = (
    """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PaperTrail - local research desk</title><style>
:root{--ink:#183b3a;--muted:#637571;--paper:#faf9f5;--line:#dce3dc;--accent:#176459;--wash:#eff3ed;--gold:#b88e4c}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}button,input{font:inherit}button,a{touch-action:manipulation}button{cursor:pointer}button:disabled{opacity:.5;cursor:wait}button:focus-visible,a:focus-visible,input:focus-visible{outline:3px solid var(--gold);outline-offset:3px}a{color:var(--accent)}.layout{display:grid;grid-template-columns:270px 1fr;min-height:100vh}aside{padding:30px 22px;border-right:1px solid var(--line);background:var(--wash)}.brand{font-size:27px;font-weight:750;letter-spacing:-1px}.brand span{color:var(--gold)}.eyebrow{text-transform:uppercase;font-size:10px;letter-spacing:1.8px;color:var(--muted)}.sidehead{display:flex;justify-content:space-between;align-items:center;margin:45px 0 12px}.textbutton{border:0;background:transparent;color:var(--accent);padding:5px}.session{display:block;text-align:left;width:100%;padding:13px 0;border:0;border-top:1px solid var(--line);background:transparent;color:var(--ink)}.session strong{display:block;font-size:13px;font-weight:550;line-height:1.4}.session small{display:block;margin-top:6px;color:var(--muted);font-size:11px}.local{margin-top:35px;font-size:12px;color:var(--muted)}main{min-width:0;padding:35px 42px}.top{display:flex;justify-content:space-between;gap:16px}.badge{font-size:10px;letter-spacing:1px;border:1px solid var(--line);border-radius:20px;padding:4px 12px;white-space:nowrap}h1{font:normal clamp(32px,4vw,48px)/1.15 Georgia,serif;letter-spacing:-1.2px;margin:25px 0 12px}h2{font:normal 25px/1.3 Georgia,serif;margin:0 0 14px}p{margin:10px 0}.intro{color:var(--muted);max-width:670px}section{margin:30px 0}.card{border:1px solid var(--line);background:white;border-radius:10px;padding:22px 25px}.row{display:flex;gap:10px;align-items:stretch}input{width:100%;min-width:0;border:1px solid var(--line);background:var(--paper);color:var(--ink);padding:12px 14px;border-radius:6px}label{display:block;font-size:12px;font-weight:650;margin-bottom:8px}.primary{border:0;border-radius:6px;background:var(--accent);color:white;padding:12px 20px;white-space:nowrap}.hint{font-size:12px;color:var(--muted)}#status{padding:13px 17px;background:var(--wash);border-left:3px solid var(--accent);font-size:13px;white-space:pre-line;overflow-wrap:anywhere}.error{border-color:#9e533b!important;color:#8c3d28}.paperhead{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}#selected{font:normal 23px/1.35 Georgia,serif;margin-bottom:5px}#sessionId{font-size:11px;color:var(--muted)}#reportLink{font-size:12px;white-space:nowrap}.reader{width:100%;height:760px;border:1px solid var(--line);border-radius:8px;background:white}#empty{padding:35px 20px;background:var(--wash);text-align:center;color:var(--muted);font-size:13px}.foot{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:18px}[hidden]{display:none!important}@media(max-width:850px){.layout{grid-template-columns:1fr}aside{padding:20px 22px;border-right:0;border-bottom:1px solid var(--line)}.sidehead{margin-top:20px}#sessions{max-height:180px;overflow:auto}.local{margin-top:15px}main{padding:25px 20px}.card{padding:20px}.row{flex-direction:column}.paperhead,.top{flex-wrap:wrap}.reader{height:650px}}
</style></head><body><div class="layout"><aside><div class="brand">papertrail<span>.</span></div><div class="eyebrow">Research with receipts</div><div class="sidehead"><b class="eyebrow">Saved sessions</b><button id="refresh" class="textbutton" type="button">Refresh</button></div><div id="sessions">Loading sessions...</div><p class="local"><b>Local compute</b><br>This interface runs on your computer. Keep the server and Ollama running. The public Pages demo contains saved reports only.</p></aside><main><div class="top"><div class="eyebrow">Your research desk</div><span class="badge">LIVE / LOCALHOST ONLY</span></div><h1>A paper. A question. A trail.</h1><p class="intro">Find one research paper, read its briefing, and ask questions with inspectable source quotations.</p><section class="card"><h2>Start with a paper</h2><form id="digest"><label for="query">Research topic, arXiv ID or arXiv URL</label><div class="row"><input id="query" name="query" required maxlength="1000" placeholder="e.g. 1706.03762 or efficient attention"><button class="primary" id="digestButton" type="submit">Create briefing</button></div></form><p class="hint">A topic search selects one paper. Downloading and local generation may take several minutes.</p></section><p id="status" role="status" aria-live="polite">Ready when you are. Select a saved paper or create a briefing.</p><section class="card"><div class="paperhead"><div><div class="eyebrow">Selected paper</div><div id="selected">No paper selected</div><div id="sessionId"></div></div><a id="reportLink" hidden target="_blank" rel="noopener">Open full report</a></div><form id="ask"><label for="question">Ask about this paper</label><div class="row"><input id="question" name="question" required maxlength="1000" placeholder="What evidence supports the main result?"><button class="primary" id="askButton" type="submit" disabled>Ask question</button></div></form><p class="hint">Answers may abstain when retrieved evidence is insufficient. Every exchange is saved.</p></section><div id="empty">Create or select a completed session to view its evidence.</div><iframe id="reader" class="reader" title="Saved briefing and source evidence" sandbox="allow-popups allow-popups-to-escape-sandbox" hidden></iframe><p class="foot">Source quotations establish provenance. Model review adds a fallible support check; neither guarantees scientific correctness.</p></main></div><script>"""
    + SCRIPT
    + "</script></body></html>"
)


class LocalApp:
    def __init__(self, settings, agent_factory=Agent):
        self.settings, self.factory = settings, agent_factory
        self.store = Store(settings.data_dir)
        self.active, self.lock = threading.Lock(), threading.Lock()
        self.jobs = {}

    def update(self, identity, **fields):
        with self.lock:
            self.jobs[identity].update(fields)

    def submit(self, action, payload):
        if not self.active.acquire(blocking=False):
            return None
        identity = uuid4().hex
        with self.lock:
            if len(self.jobs) >= 100:
                self.jobs.pop(next(iter(self.jobs)))
            self.jobs[identity] = dict(
                id=identity, status="running", stage="Starting", session=payload.get("session")
            )
            job = dict(self.jobs[identity])
        threading.Thread(target=self.work, args=(identity, action, payload), daemon=True).start()
        return job

    def work(self, identity, action, payload):
        def observe(event):
            fields = {"stage": f"{event.node}: {event.status}"}
            if SESSION.fullmatch(event.detail):
                fields["session"] = event.detail
            self.update(identity, **fields)

        try:
            with self.store.exclusive():
                settings = (
                    settings_for_session(self.settings, payload["session"])
                    if action == "ask"
                    else self.settings
                )
                agent = self.factory(settings, observer=observe)
                try:
                    self.update(
                        identity,
                        stage="Retrieving and answering" if action == "ask" else "Finding paper",
                    )
                    state = (
                        agent.ask(payload["session"], payload["question"])
                        if action == "ask"
                        else agent.new(payload["query"])
                    )
                    export_run(
                        state, agent.parsed(state).chunks, self.store.run_dir(state.id) / "export"
                    )
                finally:
                    agent.close()
            self.update(
                identity, status="succeeded", session=state.id, report=f"/report/{state.id}"
            )
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, PaperTrailError)
                else "The local job failed unexpectedly. Inspect the saved session with the CLI."
            )
            self.update(identity, status="failed", error=message)
        finally:
            self.active.release()


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, settings, host="127.0.0.1", port=8765, agent_factory=Agent):
        if host != "127.0.0.1":
            raise ValueError("PaperTrail's web interface binds only to 127.0.0.1.")
        self.app = LocalApp(settings, agent_factory)
        super().__init__((host, port), Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "PaperTrail"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_args):
        pass

    def send(self, status, data, content_type="application/json"):
        body = json.dumps(data).encode() if content_type == "application/json" else data
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        digest = b64encode(hashlib.sha256(SCRIPT.encode()).digest()).decode()
        self.send_header(
            "Content-Security-Policy",
            f"default-src 'none'; script-src 'sha256-{digest}'; style-src 'unsafe-inline'; connect-src 'self'; frame-src 'self'; frame-ancestors 'self'; base-uri 'none'; form-action 'self'",
        )
        self.end_headers()
        self.wfile.write(body)

    def trusted(self):
        hosts, origins = self.headers.get_all("Host", []), self.headers.get_all("Origin", [])
        port = self.server.server_port
        valid_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if len(hosts) != 1 or hosts[0] not in valid_hosts or len(origins) > 1:
            return False
        if origins and origins[0] != "http://" + hosts[0]:
            return False
        return self.command != "POST" or bool(origins)

    def do_GET(self):
        if not self.trusted():
            return self.send(403, {"error": "Only this local origin can access PaperTrail."})
        app, path = self.server.app, self.path.split("?", 1)[0]
        if path == "/":
            return self.send(200, HTML.encode(), "text/html")
        if path == "/api/sessions":
            rows = [
                dict(
                    id=s.id,
                    title=s.paper.title if s.paper else None,
                    query=s.query,
                    status=s.status,
                    model=s.model,
                    report=bool((report := self.report_path(s.id)) and report.is_file()),
                )
                for s in app.store.recent()
            ]
            return self.send(200, {"sessions": rows})
        if re.fullmatch(r"/api/jobs/[a-f0-9]{32}", path):
            with app.lock:
                job = dict(app.jobs.get(path.rsplit("/", 1)[1], {}))
            return (
                self.send(200, job)
                if job
                else self.send(
                    404, {"error": "Job not found. Check saved sessions if the server restarted."}
                )
            )
        if re.fullmatch(r"/report/[a-f0-9]{12}", path):
            report = self.report_path(path.rsplit("/", 1)[1])
            if report and report.is_file() and report.stat().st_size <= 10_000_000:
                return self.send(200, report.read_bytes(), "text/html")
        self.send(404, {"error": "Not found."})

    def report_path(self, session):
        root = self.server.app.settings.data_dir.resolve()
        path = root / "runs" / session / "export" / "report.html"
        return path if SESSION.fullmatch(session) and path.resolve() == path else None

    def do_POST(self):
        if not self.trusted():
            return self.send(403, {"error": "Only this local origin can start jobs."})
        if self.path not in {"/api/digest", "/api/ask"}:
            return self.send(404, {"error": "Not found."})
        if self.headers.get_content_type() != "application/json":
            return self.send(415, {"error": "Use application/json."})
        lengths = self.headers.get_all("Content-Length", [])
        if (
            self.headers.get("Transfer-Encoding")
            or len(lengths) != 1
            or not re.fullmatch(r"[0-9]{1,5}", lengths[0])
        ):
            return self.send(400, {"error": "A single Content-Length is required."})
        if not 0 < int(lengths[0]) <= MAX_BODY:
            return self.send(413, {"error": "Request body must contain 1-8192 bytes."})
        try:
            payload = json.loads(self.rfile.read(int(lengths[0])))
            keys = {"query"} if self.path == "/api/digest" else {"session", "question"}
            if not isinstance(payload, dict) or set(payload) != keys:
                raise ValueError
            if any(
                not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in payload.values()
            ):
                raise ValueError
            payload = {k: v.strip() for k, v in payload.items()}
            if "session" in payload and not SESSION.fullmatch(payload["session"]):
                raise ValueError
        except (ValueError, UnicodeError, TimeoutError, RecursionError):
            return self.send(
                400,
                {
                    "error": "Provide valid JSON with non-empty text fields (maximum 1000 characters) and a valid session ID."
                },
            )
        job = self.server.app.submit(self.path.rsplit("/", 1)[1], payload)
        self.send(202, job) if job else self.send(
            409, {"error": "Another local job is running. Wait for it to finish."}
        )


def make_server(settings, host="127.0.0.1", port=8765, agent_factory=Agent):
    return LocalServer(settings, host, port, agent_factory)


def serve(settings, port=8765):
    with make_server(settings, port=port) as server:
        print(f"PaperTrail local compute: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

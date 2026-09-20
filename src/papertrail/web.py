"""Loopback-only web adapter. Model work stays in the existing checkpointed Agent."""

import hashlib
import json
import logging
import multiprocessing
import re
import threading
import time
from base64 import b64encode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote
from uuid import uuid4

from papertrail.errors import PaperTrailError
from papertrail.graph import Agent
from papertrail.schema import Event, Node
from papertrail.storage import Store
from papertrail.worker import process_operation, run_operation

SESSION = re.compile(r"[a-f0-9]{12}")
MAX_BODY = 8192
UPLOAD_READ_SECONDS = 30
SCRIPT = r"""
const $ = id => document.getElementById(id);
let selected = '', busy = false, ready = false, resumable = false, lastRetry = null, watchVersion = 0;
const seconds = value => Math.max(0, value || 0).toFixed((value || 0)<10 ? 1 : 0)+'s';
const stageNames = {initialize:'Opening local index', understand:'Understanding input', retrieve:'Retrieving arXiv metadata', select:'Selecting the matching paper', fetch:'Downloading PDF', parse:'Reading PDF', index:'Indexing passages', brief:'Preparing briefing evidence', validate:'Validating citations', 'model.load':'Loading local model', 'brief.generate':'Writing briefing', 'brief.review':'Checking briefing evidence', 'qa.retrieve':'Finding answer evidence', 'qa.generate':'Answering question', 'qa.review':'Checking answer evidence', export:'Saving report'};
const stageName = node => stageNames[node] || node;
async function api(path, data) {
  const response = await fetch(path, {...(data ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)} : {}), signal:AbortSignal.timeout(15000)});
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'Request failed');
  return result;
}
function status(message, error=false) { $('status').textContent=message; $('status').classList.toggle('error',error); }
function controls(locked) {
  busy=locked; $('digestButton').disabled=locked; $('askButton').disabled=locked || !ready;
  $('resumeButton').disabled=locked || !resumable; $('retryButton').disabled=locked;
  $('choosePdf').disabled=locked; $('pdfFile').disabled=locked;
  document.querySelectorAll('.session').forEach(button=>button.disabled=locked);
}
function pendingPaper(query) {
  selected=''; ready=false; resumable=false; $('question').value='';
  $('selected').textContent='Selecting paper…'; $('sessionId').textContent='Requested: '+query;
  $('reader').hidden=true; $('reader').removeAttribute('src');
  $('reportLink').hidden=true; $('reportLink').removeAttribute('href'); $('resumeButton').hidden=true;
  $('empty').hidden=false; $('empty').textContent='Processing your input. The selected paper and its briefing will appear here.';
}
function choose(session) {
  if(selected!==session.id) $('question').value='';
  selected=session.id; ready=session.status==='ready'; resumable=session.can_resume;
  $('selected').textContent=session.title || session.query; $('sessionId').textContent=session.id+' / '+session.model;
  $('resumeButton').hidden=!resumable; controls(busy);
  $('reader').hidden=!session.report; $('empty').hidden=!!session.report; $('reportLink').hidden=!session.report;
  if(session.report) { $('reader').src='/report/'+session.id+'?v='+Date.now(); $('reportLink').href='/report/'+session.id; }
  else $('empty').textContent=session.error || (resumable ? 'Saved at '+session.checkpoint+'. Resume this session to continue from its last checkpoint.' : session.status==='ready' ? 'No exported report yet. Ask a question to create one.' : 'Enter a corrected ID or topic to start a new briefing.');
}
async function sessions(preferred) {
  const result=await api('/api/sessions'); $('sessions').replaceChildren();
  result.sessions.forEach(session=>{
    const button=document.createElement('button'); button.className='session';
    const title=document.createElement('strong'); title.textContent=session.title || session.query;
    const detail=document.createElement('small'); detail.textContent=session.status+' / '+session.id;
    button.append(title,detail); button.disabled=busy; button.onclick=()=>{if(!busy) choose(session);}; $('sessions').append(button);
  });
  if(!result.sessions.length) $('sessions').textContent='Your saved papers will appear here.';
  const target=result.sessions.find(s=>s.id===(preferred || selected)) || (!selected && result.sessions[0]);
  if(target) choose(target);
}
function progress(job) {
  const timing=(job.timings || []).filter(event=>event.status==='completed').map(event=>stageName(event.node)+' '+seconds(event.seconds)).join(' · ');
  status((job.filename ? job.filename+'\n' : '')+stageName(job.stage)+' — '+seconds(job.stage_elapsed_seconds)+' in this stage; '+seconds(job.elapsed_seconds)+' total / '+seconds(job.deadline_seconds)+' limit.'+(job.detail ? '\n'+job.detail : '')+(timing ? '\nCompleted: '+timing : ''));
}
async function watch(job) {
  const version=++watchVersion;
  controls(true); $('retryButton').hidden=true;
  try {
    let reconnects=0;
    while(job.status==='running' && version===watchVersion) {
      progress(job); await new Promise(resolve=>setTimeout(resolve,1000));
      try { job=await api('/api/jobs/'+job.id); reconnects=0; }
      catch(error) { if(++reconnects>2) throw error; status('Connection interrupted; reconnecting to the current job...',true); }
    }
    if(version!==watchVersion) return;
    await sessions(job.session);
    if(job.status==='failed') {
      lastRetry=job.retry; $('retryButton').hidden=!lastRetry;
      status(job.error+'\nElapsed: '+seconds(job.elapsed_seconds)+'.',true);
    } else {
      lastRetry=null; status('Saved in '+seconds(job.elapsed_seconds)+'. Open the evidence below to inspect this result.'+(job.cleanup_warning ? '\n'+job.cleanup_warning : ''),!!job.cleanup_warning);
      if(job.action==='ask') $('question').value='';
    }
  } catch(error) { if(version===watchVersion) status(error.message+' Refresh to reconnect; the server still enforces its operation deadline.',true); }
  finally { if(version===watchVersion) controls(false); }
}
async function submit(path, data) {
  if(path==='/api/digest') pendingPaper(data.query);
  controls(true); status('Starting an operation...'); $('retryButton').hidden=true;
  try { await watch(await api(path,data)); }
  catch(error) { status(error.message,true); controls(false); }
}
async function uploadPdf(file) {
  if(busy || !file) return;
  const maxBytes=Number($('uploadDrop').dataset.maxBytes);
  if(file.size===0 || file.size>maxBytes) { status('Choose a non-empty PDF no larger than '+Math.round(maxBytes/1024/1024)+' MB.',true); return; }
  if(!file.name.toLowerCase().endsWith('.pdf')) { status('Choose a PDF file. Other document formats are not supported.',true); return; }
  pendingPaper(file.name); controls(true); $('retryButton').hidden=true;
  $('uploadName').textContent=file.name; status('Uploading '+file.name+'…');
  try {
    const response=await fetch('/api/upload',{method:'POST',headers:{'Content-Type':'application/pdf','X-Filename':encodeURIComponent(file.name)},body:file,signal:AbortSignal.timeout(35000)});
    const result=await response.json();
    if(!response.ok) throw Error(result.error || 'Upload failed. Choose the file again to retry.');
    await watch(result);
  } catch(error) { status(error.message+' Choose the file again to retry, or refresh to reconnect to an accepted job.',true); controls(false); }
  finally { $('pdfFile').value=''; }
}
$('choosePdf').onclick=()=>{if(!busy) $('pdfFile').click();};
$('pdfFile').onchange=()=>uploadPdf($('pdfFile').files[0]);
for(const name of ['dragenter','dragover']) $('uploadDrop').addEventListener(name,event=>{event.preventDefault(); if(!busy) $('uploadDrop').classList.add('dragging');});
for(const name of ['dragleave','drop']) $('uploadDrop').addEventListener(name,event=>{event.preventDefault(); $('uploadDrop').classList.remove('dragging');});
$('uploadDrop').addEventListener('drop',event=>{if(event.dataTransfer.files.length!==1) {status('Drop one PDF at a time.',true); return;} uploadPdf(event.dataTransfer.files[0]);});
window.addEventListener('dragover',event=>event.preventDefault());
window.addEventListener('drop',event=>event.preventDefault());
$('digest').onsubmit=event=>{event.preventDefault(); submit('/api/digest',{query:$('query').value});};
$('ask').onsubmit=event=>{event.preventDefault(); submit('/api/ask',{session:selected,question:$('question').value});};
$('resumeButton').onclick=()=>submit('/api/resume',{session:selected});
$('retryButton').onclick=()=>{ if(lastRetry) submit(lastRetry.path,lastRetry.payload); };
async function reconnect() {
  try { await sessions(); const result=await api('/api/active'); if(result.job) { if(['upload','digest'].includes(result.job.action)) pendingPaper(result.job.filename || 'Current paper'); await watch(result.job); } }
  catch(error) { status(error.message,true); }
}
$('refresh').onclick=reconnect;
reconnect();
"""
HTML = (
    """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PaperTrail - local research desk</title><style>
:root{--ink:#183b3a;--muted:#637571;--paper:#faf9f5;--line:#dce3dc;--accent:#176459;--wash:#eff3ed;--gold:#b88e4c}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}button,input{font:inherit}button,a{touch-action:manipulation}button{cursor:pointer}button:disabled{opacity:.5;cursor:wait}button:focus-visible,a:focus-visible,input:focus-visible{outline:3px solid var(--gold);outline-offset:3px}a{color:var(--accent)}.layout{display:grid;grid-template-columns:270px 1fr;min-height:100vh}aside{padding:30px 22px;border-right:1px solid var(--line);background:var(--wash)}.brand{font-size:27px;font-weight:750;letter-spacing:-1px}.brand span{color:var(--gold)}.eyebrow{text-transform:uppercase;font-size:10px;letter-spacing:1.8px;color:var(--muted)}.sidehead{display:flex;justify-content:space-between;align-items:center;margin:45px 0 12px}.textbutton{border:0;background:transparent;color:var(--accent);padding:5px}.session{display:block;text-align:left;width:100%;padding:13px 0;border:0;border-top:1px solid var(--line);background:transparent;color:var(--ink)}.session strong{display:block;font-size:13px;font-weight:550;line-height:1.4}.session small{display:block;margin-top:6px;color:var(--muted);font-size:11px}.local{margin-top:35px;font-size:12px;color:var(--muted)}main{min-width:0;padding:35px 42px}.top{display:flex;justify-content:space-between;gap:16px}.badge{font-size:10px;letter-spacing:1px;border:1px solid var(--line);border-radius:20px;padding:4px 12px;white-space:nowrap}h1{font:normal clamp(32px,4vw,48px)/1.15 Georgia,serif;letter-spacing:-1.2px;margin:25px 0 12px}h2{font:normal 25px/1.3 Georgia,serif;margin:0 0 14px}p{margin:10px 0}.intro{color:var(--muted);max-width:670px}section{margin:30px 0}.card{border:1px solid var(--line);background:white;border-radius:10px;padding:22px 25px}.row{display:flex;gap:10px;align-items:stretch}input{width:100%;min-width:0;border:1px solid var(--line);background:var(--paper);color:var(--ink);padding:12px 14px;border-radius:6px}label{display:block;font-size:12px;font-weight:650;margin-bottom:8px}.primary{border:0;border-radius:6px;background:var(--accent);color:white;padding:12px 20px;white-space:nowrap}.upload{border:1px dashed var(--muted);border-radius:8px;padding:18px;margin-bottom:20px;background:var(--paper)}.upload.dragging{border-color:var(--accent);background:var(--wash)}#uploadName{overflow-wrap:anywhere}.hint{font-size:12px;color:var(--muted)}#status{padding:13px 17px;background:var(--wash);border-left:3px solid var(--accent);font-size:13px;white-space:pre-line;overflow-wrap:anywhere}.error{border-color:#9e533b!important;color:#8c3d28}.paperhead{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}#selected{font:normal 23px/1.35 Georgia,serif;margin-bottom:5px}#sessionId{font-size:11px;color:var(--muted)}#reportLink{font-size:12px;white-space:nowrap}.reader{width:100%;height:760px;border:1px solid var(--line);border-radius:8px;background:white}#empty{padding:35px 20px;background:var(--wash);text-align:center;color:var(--muted);font-size:13px}.foot{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:18px}[hidden]{display:none!important}@media(max-width:850px){.layout{grid-template-columns:1fr}aside{padding:20px 22px;border-right:0;border-bottom:1px solid var(--line)}.sidehead{margin-top:20px}#sessions{max-height:180px;overflow:auto}.local{margin-top:15px}main{padding:25px 20px}.card{padding:20px}.row{flex-direction:column}.paperhead,.top{flex-wrap:wrap}.reader{height:650px}}
</style></head><body><div class="layout"><aside><div class="brand">papertrail<span>.</span></div><div class="eyebrow">Research with receipts</div><div class="sidehead"><b class="eyebrow">Saved sessions</b><button id="refresh" class="textbutton" type="button">Refresh</button></div><div id="sessions">Loading sessions...</div><p class="local"><b>Local compute</b><br>This interface runs on your computer. Keep the server and Ollama running. The public Pages demo contains saved reports only.</p></aside><main><div class="top"><div class="eyebrow">Your research desk</div><span class="badge">LIVE / LOCALHOST ONLY</span></div><h1>A paper. A question. A trail.</h1><p class="intro">Find one research paper, read its briefing, and ask questions with inspectable source quotations.</p><section class="card"><h2>Start with a paper</h2><div class="upload" id="uploadDrop" data-max-bytes="__MAX_UPLOAD_BYTES__"><label for="pdfFile">Drop a research PDF here, or choose a file</label><input id="pdfFile" type="file" accept=".pdf,application/pdf" hidden><button class="primary" id="choosePdf" type="button">Choose PDF</button><p id="uploadName" class="hint">Text-readable PDFs, up to __MAX_UPLOAD_MB__ MB and __MAX_UPLOAD_PAGES__ pages. No scanned or encrypted PDFs.</p><p class="hint">Uploaded files stay in the local data directory. No arXiv search is required.</p></div><form id="digest"><label for="query">Research topic, arXiv ID or arXiv URL</label><div class="row"><input id="query" name="query" required maxlength="1000" placeholder="e.g. 1706.03762 or efficient attention"><button class="primary" id="digestButton" type="submit">Create briefing</button></div></form><p class="hint">A topic search selects one paper. Downloading and local generation may take several minutes.</p></section><p id="status" role="status" aria-live="polite">Ready when you are. Select a saved paper or create a briefing.</p><button id="retryButton" class="textbutton" type="button" hidden>Retry last operation</button><section class="card"><div class="paperhead"><div><div class="eyebrow">Selected paper</div><div id="selected">No paper selected</div><div id="sessionId"></div><button id="resumeButton" class="textbutton" type="button" hidden>Resume saved session</button></div><a id="reportLink" hidden target="_blank" rel="noopener">Open full report</a></div><form id="ask"><label for="question">Ask about this paper</label><div class="row"><input id="question" name="question" required maxlength="1000" placeholder="What evidence supports the main result?"><button class="primary" id="askButton" type="submit" disabled>Ask question</button></div></form><p class="hint">Answers may abstain when retrieved evidence is insufficient. Every exchange is saved.</p></section><div id="empty">Create or select a completed session to view its evidence.</div><iframe id="reader" class="reader" title="Saved briefing and source evidence" sandbox="allow-popups allow-popups-to-escape-sandbox" hidden></iframe><p class="foot">Source quotations establish provenance. Model review adds a fallible support check; neither guarantees scientific correctness.</p></main></div><script>"""
    + SCRIPT
    + "</script></body></html>"
)


class LocalApp:
    def __init__(self, settings, agent_factory=Agent, isolate=None):
        self.settings, self.factory = settings, agent_factory
        self.store = Store(settings.data_dir)
        self.active, self.lock = threading.Lock(), threading.Lock()
        self.upload_lock = threading.Lock()
        self.jobs = {}
        # Production always isolates Agent. In-process injection keeps small HTTP
        # tests deterministic; explicit isolation exercises the real watchdog.
        self.isolate = agent_factory is Agent if isolate is None else isolate
        self.process = None
        self.stopping = threading.Event()
        self.supervisor = None

    def update(self, identity, **fields):
        with self.lock:
            job = self.jobs[identity]
            if fields.get("stage") and fields["stage"] != job["stage"]:
                job["_stage_started"] = time.monotonic()
            job.update(fields)

    def snapshot(self, identity):
        with self.lock:
            job = self.jobs.get(identity)
            if not job:
                return None
            result = {key: value for key, value in job.items() if not key.startswith("_")}
            ended = job.get("_ended", time.monotonic())
            result["elapsed_seconds"] = round(ended - job["_started"], 1)
            result["stage_elapsed_seconds"] = round(ended - job["_stage_started"], 1)
            return result

    def current(self):
        with self.lock:
            identity = next(
                (key for key, job in self.jobs.items() if job["status"] == "running"), None
            )
        return self.snapshot(identity) if identity else None

    def submit(self, action, payload):
        if self.stopping.is_set() or not self.active.acquire(blocking=False):
            return None
        identity = uuid4().hex
        now = time.monotonic()
        with self.lock:
            if len(self.jobs) >= 100:
                self.jobs.pop(next(iter(self.jobs)))
            self.jobs[identity] = dict(
                id=identity,
                status="running",
                stage="initialize",
                stage_status="running",
                detail="Starting isolated worker",
                session=payload.get("session"),
                action=action,
                retry=None
                if action == "upload"
                else {"path": "/api/" + action, "payload": payload},
                filename=payload.get("filename"),
                deadline_seconds=self.settings.operation_timeout,
                timings=[],
                _started=now,
                _stage_started=now,
            )
        job = self.snapshot(identity)
        self.supervisor = threading.Thread(
            target=self.work, args=(identity, action, payload), daemon=True
        )
        self.supervisor.start()
        return job

    def observe(self, identity, message):
        if "event" not in message:
            return message
        event = Event.model_validate(message["event"])
        fields = {"stage": event.node, "stage_status": event.status}
        if event.status == "running":
            fields["_stage_started"] = time.monotonic()
        if SESSION.fullmatch(event.detail):
            fields["session"] = event.detail
            fields["detail"] = ""
        else:
            fields["detail"] = event.detail if event.status == "running" else ""
        with self.lock:
            if event.status != "running":
                self.jobs[identity]["timings"].append(event.model_dump())
        self.update(identity, **fields)
        return None

    def mark_interrupted(self, identity, action, message):
        session = self.snapshot(identity).get("session")
        if not session:
            return
        try:
            # The worker has already exited, so its writer lock is released.
            with self.store.exclusive():
                state = self.store.load(session)
                if action != "ask" and state.status != "ready":
                    state.status, state.error = "failed", message
                state.events.append(
                    Event(
                        node="qa.operation" if action == "ask" else "web.operation",
                        status="failed",
                        seconds=self.snapshot(identity)["elapsed_seconds"],
                        detail=message,
                    )
                )
                self.store.save(state)
        except PaperTrailError:
            # A CLI operation may have acquired the lock after the child exited.
            # Never overwrite its state; the job still exposes the failure.
            pass

    @staticmethod
    def stop_process(process):
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        if process.is_alive():
            raise RuntimeError("The isolated worker could not be stopped.")
        process.join()

    def isolated_work(self, identity, action, payload):
        context = multiprocessing.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(
            target=process_operation,
            args=(self.settings, action, payload, writer, self.factory),
            daemon=True,
        )
        self.process = process
        terminal = None
        deadline = time.monotonic() + self.settings.operation_timeout
        try:
            process.start()
            writer.close()
            while True:
                if time.monotonic() >= deadline or self.stopping.is_set():
                    stage = self.snapshot(identity)["stage"]
                    message = (
                        f"Stopped at {stage} after the {self.settings.operation_timeout:g}s "
                        "operation limit. The worker was stopped and its resources released. "
                        + (
                            "Retry the question; the saved briefing is still available."
                            if action == "ask"
                            else "Resume the saved session to continue from its last checkpoint."
                        )
                    )
                    if self.stopping.is_set():
                        message = "The server stopped this operation. Resume its saved checkpoint."
                    self.stop_process(process)
                    self.mark_interrupted(identity, action, message)
                    return {"status": "failed", "error": message, "timed_out": True}
                if reader.poll(0.1):
                    try:
                        result = self.observe(identity, reader.recv())
                    except EOFError:
                        break
                    if result:
                        terminal = result
                        break
                if not process.is_alive() and not reader.poll():
                    break
            process.join(timeout=2)
            self.stop_process(process)
            if terminal:
                return terminal
            message = "The isolated worker exited before finishing. Resume its saved checkpoint."
            self.mark_interrupted(identity, action, message)
            return {"status": "failed", "error": message}
        finally:
            writer.close()
            reader.close()
            if process.pid is not None:
                self.stop_process(process)
                process.close()
            self.process = None

    def work(self, identity, action, payload):
        try:
            if self.isolate:
                terminal = self.isolated_work(identity, action, payload)
            else:
                terminal = None

                def receive(message):
                    nonlocal terminal
                    result = self.observe(identity, message)
                    if result:
                        terminal = result

                run_operation(self.settings, action, payload, receive, self.factory)
            if terminal["status"] == "failed" and action != "ask":
                session = self.snapshot(identity).get("session")
                if session:
                    try:
                        state = self.store.load(session)
                        if state.node != Node.UNDERSTAND:
                            terminal["retry"] = {
                                "path": "/api/resume",
                                "payload": {"session": session},
                            }
                    except PaperTrailError:
                        pass
            self.update(identity, **terminal, _ended=time.monotonic())
        except Exception:
            self.update(
                identity,
                status="failed",
                error="The local worker could not start or stop cleanly. Restart the server before retrying.",
                _ended=time.monotonic(),
            )
        finally:
            try:
                if action == "upload":
                    Path(payload["path"]).unlink(missing_ok=True)
            except OSError:
                self.update(
                    identity,
                    cleanup_warning="The temporary upload could not be removed from the local data directory.",
                )
                logging.getLogger(__name__).warning(
                    "Temporary upload cleanup failed; a local staging copy remains."
                )
            finally:
                if self.process is None or not self.process.is_alive():
                    self.active.release()

    def close(self):
        self.stopping.set()
        if self.supervisor:
            self.supervisor.join(timeout=6)


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, settings, host="127.0.0.1", port=8765, agent_factory=Agent, isolate=None):
        if host != "127.0.0.1":
            raise ValueError("PaperTrail's web interface binds only to 127.0.0.1.")
        self.app = LocalApp(settings, agent_factory, isolate)
        super().__init__((host, port), Handler)

    def server_close(self):
        self.app.close()
        super().server_close()


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
        charset = "; charset=utf-8" if content_type != "application/pdf" else ""
        self.send_header("Content-Type", content_type + charset)
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
            html = (
                HTML.replace("__MAX_UPLOAD_BYTES__", str(app.settings.max_pdf_bytes))
                .replace("__MAX_UPLOAD_MB__", str(app.settings.max_pdf_bytes // (1024 * 1024)))
                .replace("__MAX_UPLOAD_PAGES__", str(app.settings.max_pages))
            )
            return self.send(200, html.encode(), "text/html")
        if path == "/api/sessions":
            rows = [
                dict(
                    id=s.id,
                    title=s.paper.title if s.paper else None,
                    source=s.paper.source if s.paper else None,
                    document_identity=s.paper.identity if s.paper else None,
                    query=s.query,
                    status=s.status,
                    model=s.model,
                    error=s.error,
                    checkpoint=s.node,
                    can_resume=s.status != "ready" and s.node != Node.UNDERSTAND,
                    report=bool((report := self.report_path(s.id)) and report.is_file()),
                )
                for s in app.store.recent()
            ]
            return self.send(200, {"sessions": rows})
        if path == "/api/active":
            return self.send(200, {"job": app.current()})
        if re.fullmatch(r"/api/jobs/[a-f0-9]{32}", path):
            job = app.snapshot(path.rsplit("/", 1)[1])
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
        if re.fullmatch(r"/document/[a-f0-9]{12}/paper.pdf", path):
            session = path.split("/")[2]
            root = app.settings.data_dir.resolve()
            document = root / "runs" / session / "paper.pdf"
            try:
                state = app.store.load(session)
                if (
                    state.paper
                    and state.paper.source == "upload"
                    and document.resolve() == document
                    and document.is_file()
                    and document.stat().st_size <= app.settings.max_pdf_bytes
                ):
                    return self.send(200, document.read_bytes(), "application/pdf")
            except PaperTrailError:
                pass
        self.send(404, {"error": "Not found."})

    def report_path(self, session):
        root = self.server.app.settings.data_dir.resolve()
        path = root / "runs" / session / "export" / "report.html"
        return path if SESSION.fullmatch(session) and path.resolve() == path else None

    def receive_upload(self):
        """Bound the transfer before handing parsing to the isolated worker."""
        app = self.server.app
        if self.headers.get_content_type() != "application/pdf":
            return self.send(415, {"error": "Choose a PDF file (application/pdf)."})
        lengths = self.headers.get_all("Content-Length", [])
        if (
            self.headers.get("Transfer-Encoding")
            or len(lengths) != 1
            or not re.fullmatch(r"[0-9]{1,12}", lengths[0])
        ):
            return self.send(400, {"error": "A single PDF Content-Length is required."})
        size = int(lengths[0])
        if not 0 < size <= app.settings.max_pdf_bytes:
            limit = app.settings.max_pdf_bytes // (1024 * 1024)
            return self.send(413, {"error": f"Choose a non-empty PDF no larger than {limit} MB."})
        names = self.headers.get_all("X-Filename", [])
        try:
            if len(names) != 1 or len(names[0]) > 2400:
                raise ValueError
            filename = unquote(names[0], errors="strict")
            if (
                not filename.strip()
                or len(filename) > 255
                or any(ord(char) < 32 or ord(char) == 127 for char in filename)
                or not filename.lower().endswith(".pdf")
            ):
                raise ValueError
            # Client labels never become filesystem paths.
            filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
        except (ValueError, UnicodeError):
            return self.send(
                400, {"error": "Provide a valid PDF filename (maximum 255 characters)."}
            )
        if app.current() or not app.upload_lock.acquire(blocking=False):
            return self.send(
                409, {"error": "Another local job or upload is running. Wait for it to finish."}
            )
        staging = None
        accepted = False
        try:
            folder = app.settings.data_dir / "uploads"
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            staging = folder / (uuid4().hex + ".pdf")
            deadline = time.monotonic() + UPLOAD_READ_SECONDS
            remaining = size
            with staging.open("xb") as output:
                staging.chmod(0o600)
                while remaining:
                    seconds_left = deadline - time.monotonic()
                    if seconds_left <= 0:
                        raise TimeoutError
                    self.connection.settimeout(min(10, seconds_left))
                    chunk = self.rfile.read1(min(65536, remaining))
                    if not chunk:
                        raise ValueError("The PDF upload was interrupted. Choose the file again.")
                    output.write(chunk)
                    remaining -= len(chunk)
            with staging.open("rb") as source:
                if not source.read(5).startswith(b"%PDF-"):
                    raise ValueError("This file is not a valid PDF. Choose a text-readable PDF.")
            job = app.submit("upload", {"path": str(staging), "filename": filename})
            if job:
                accepted = True
                return self.send(202, job)
            return self.send(
                409,
                {"error": "Another local job is running. Choose the PDF again after it finishes."},
            )
        except TimeoutError:
            return self.send(
                408, {"error": "PDF upload timed out. Choose the file again to retry."}
            )
        except ValueError as exc:
            return self.send(400, {"error": str(exc)})
        except OSError:
            return self.send(
                500,
                {"error": "The PDF could not be saved locally. Check free disk space and retry."},
            )
        finally:
            try:
                if staging and not accepted:
                    staging.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning(
                    "Interrupted upload cleanup failed; a local staging copy remains."
                )
            finally:
                app.upload_lock.release()

    def do_POST(self):
        if not self.trusted():
            return self.send(403, {"error": "Only this local origin can start jobs."})
        if self.path == "/api/upload":
            return self.receive_upload()
        if self.path not in {"/api/digest", "/api/ask", "/api/resume"}:
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
            keys = {
                "/api/digest": {"query"},
                "/api/ask": {"session", "question"},
                "/api/resume": {"session"},
            }[self.path]
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


def make_server(settings, host="127.0.0.1", port=8765, agent_factory=Agent, isolate=None):
    return LocalServer(settings, host, port, agent_factory, isolate)


def serve(settings, port=8765):
    with make_server(settings, port=port) as server:
        print(f"PaperTrail local compute: http://127.0.0.1:{server.server_port} (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

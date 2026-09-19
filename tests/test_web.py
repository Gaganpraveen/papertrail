import json
import os
import threading
import time
from http.client import HTTPConnection
from types import SimpleNamespace

import pytest

from papertrail.config import Settings
from papertrail.errors import PaperTrailError
from papertrail.schema import Answer, Chunk, Event, Exchange, Node, RunState
from papertrail.storage import Store
from papertrail.web import MAX_BODY, make_server


@pytest.fixture
def web(tmp_path, paper, chunk, briefing):
    gate, calls = threading.Event(), []
    gate.set()
    ready = RunState(
        id="abcdef123456",
        query="attention",
        model="saved-model",
        embedding_model="saved-embed",
        status="ready",
        node=Node.READY,
        paper=paper,
        briefing=briefing,
    )

    class FakeAgent:
        def __init__(self, settings, observer):
            self.store, self.observer = Store(settings.data_dir), observer
            calls.append(("settings", settings.model, settings.embedding_model))

        def new(self, query):
            self.observer(Event(node="understand", status="running", detail=ready.id))
            if not gate.wait(5):
                raise PaperTrailError("Timed out in the test fixture.")
            if query == "fail":
                raise PaperTrailError("Ollama is not reachable. Start ollama serve.")
            if query == "crash":
                raise RuntimeError("secret internal details")
            state = ready.model_copy(deep=True, update={"query": query})
            self.store.save(state)
            return state

        def ask(self, session, question):
            state = self.store.load(session)
            state.exchanges.append(
                Exchange(
                    question=question,
                    retrieval_query=question,
                    answer=Answer(
                        status="insufficient_evidence", claims=[], explanation="No evidence."
                    ),
                    retrieved_ids=[],
                    elapsed_seconds=0,
                )
            )
            self.store.save(state)
            return state

        def resume(self, session):
            calls.append(("resume", session))
            state = self.store.load(session)
            state.status, state.node, state.error = "ready", Node.READY, None
            self.store.save(state)
            return state

        def parsed(self, _state):
            return SimpleNamespace(chunks=[chunk])

        def close(self):
            calls.append(("closed",))

    server = make_server(
        Settings(data_dir=tmp_path, model="new-model"), port=0, agent_factory=FakeAgent
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(server=server, gate=gate, calls=calls, root=tmp_path)
    gate.set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(web, method="GET", path="/", payload=None, headers=None, raw=None):
    port = web.server.server_port
    defaults = {"Host": f"127.0.0.1:{port}"}
    if method == "POST":
        defaults.update({"Origin": f"http://127.0.0.1:{port}", "Content-Type": "application/json"})
    defaults.update(headers or {})
    body = raw if raw is not None else json.dumps(payload) if payload is not None else None
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    connection.request(method, path, body=body, headers=defaults)
    response = connection.getresponse()
    content, metadata = response.read(), dict(response.getheaders())
    connection.close()
    return response.status, content, metadata


def wait_job(web, identity, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, body, _ = request(web, path="/api/jobs/" + identity)
        assert status == 200
        job = json.loads(body)
        if job["status"] != "running":
            return job
        time.sleep(0.01)
    pytest.fail("Background job did not finish")


def test_home_is_local_interface_with_no_external_assets(web):
    status, body, headers = request(web)
    assert status == 200
    assert b"LIVE / LOCALHOST ONLY" in body
    assert b"public Pages demo contains saved reports only" in body
    assert b"onsubmit" in body and b"/api/digest" in body and b"/api/ask" in body
    assert headers["Cache-Control"] == "no-store"
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert "Access-Control-Allow-Origin" not in headers


def test_digest_job_is_nonblocking_and_concurrent_work_is_rejected(web):
    web.gate.clear()
    status, body, _ = request(web, "POST", "/api/digest", {"query": "attention"})
    assert status == 202
    job = json.loads(body)
    assert job["status"] == "running"
    assert request(web, "POST", "/api/digest", {"query": "second"})[0] == 409
    assert request(web, path="/api/sessions")[0] == 200
    web.gate.set()
    completed = wait_job(web, job["id"])
    assert completed["status"] == "succeeded"
    status, report, _ = request(web, path=completed["report"])
    assert status == 200 and b"Attention Is All You Need" in report
    assert ("closed",) in web.calls


def test_ask_uses_saved_model_and_updates_report(web):
    _, body, _ = request(web, "POST", "/api/digest", {"query": "attention"})
    job = wait_job(web, json.loads(body)["id"])
    _, body, _ = request(
        web, "POST", "/api/ask", {"session": job["session"], "question": "Is there evidence?"}
    )
    assert wait_job(web, json.loads(body)["id"])["status"] == "succeeded"
    assert ("settings", "saved-model", "saved-embed") in web.calls
    state = Store(web.root).load(job["session"])
    assert state.exchanges[-1].question == "Is there evidence?"
    assert b"No evidence." in request(web, path=job["report"])[1]
    sessions = json.loads(request(web, path="/api/sessions")[1])["sessions"]
    assert sessions[0]["report"] is True


def test_browser_resume_preserves_session_models_and_checkpoint(web):
    _, body, _ = request(web, "POST", "/api/digest", {"query": "attention"})
    original = wait_job(web, json.loads(body)["id"])
    store = Store(web.root)
    state = store.load(original["session"])
    state.status, state.node, state.error = "failed", Node.BRIEF, "The model timed out."
    store.save(state)
    sessions = json.loads(request(web, path="/api/sessions")[1])["sessions"]
    assert sessions[0]["can_resume"] is True
    assert sessions[0]["checkpoint"] == "brief"
    assert sessions[0]["error"] == "The model timed out."
    status, body, _ = request(web, "POST", "/api/resume", {"session": state.id})
    assert status == 202
    job = wait_job(web, json.loads(body)["id"])
    assert job["status"] == "succeeded" and job["session"] == state.id
    assert ("resume", state.id) in web.calls
    assert ("settings", "saved-model", "saved-embed") in web.calls


def test_refresh_can_reattach_to_active_job_with_elapsed_times(web):
    web.gate.clear()
    _, body, _ = request(web, "POST", "/api/digest", {"query": "attention"})
    original = json.loads(body)
    current = json.loads(request(web, path="/api/active")[1])["job"]
    assert current["id"] == original["id"]
    assert current["status"] == "running"
    assert current["elapsed_seconds"] >= current["stage_elapsed_seconds"] >= 0
    assert current["deadline_seconds"] == 240
    assert "_started" not in current
    web.gate.set()
    wait_job(web, original["id"])
    assert json.loads(request(web, path="/api/active")[1])["job"] is None


def test_repeated_review_stage_resets_clock_and_keeps_metrics_out_of_status(web):
    web.gate.clear()
    _, body, _ = request(web, "POST", "/api/digest", {"query": "attention"})
    identity = json.loads(body)["id"]
    app = web.server.app
    running = Event(node="brief.review", status="running")
    app.observe(identity, {"event": running.model_dump()})
    time.sleep(0.25)
    first_batch = json.loads(request(web, path="/api/jobs/" + identity)[1])
    app.observe(identity, {"event": running.model_dump()})
    second_batch = json.loads(request(web, path="/api/jobs/" + identity)[1])
    assert second_batch["stage_elapsed_seconds"] < first_batch["stage_elapsed_seconds"]
    metrics = '{"server_seconds": 1.2, "output_tokens": 40}'
    app.observe(
        identity,
        {
            "event": Event(
                node="brief.review", status="completed", seconds=1.3, detail=metrics
            ).model_dump()
        },
    )
    completed_batch = json.loads(request(web, path="/api/jobs/" + identity)[1])
    assert completed_batch["detail"] == ""
    assert completed_batch["timings"][-1]["detail"] == metrics
    assert completed_batch["timings"][-1]["seconds"] == 1.3
    web.gate.set()
    wait_job(web, identity)


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "evil.example"},
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
        {"Origin": ""},
    ],
)
def test_cross_origin_and_rebinding_requests_are_rejected(web, headers):
    assert request(web, "POST", "/api/digest", {"query": "attention"}, headers)[0] == 403
    assert web.calls == []


def test_cross_origin_reads_and_missing_post_origin_are_rejected(web):
    assert request(web, headers={"Origin": "https://evil.example"})[0] == 403
    connection = HTTPConnection("127.0.0.1", web.server.server_port)
    connection.request(
        "POST",
        "/api/digest",
        body='{"query":"attention"}',
        headers={"Content-Type": "application/json"},
    )
    assert connection.getresponse().status == 403
    connection.close()


@pytest.mark.parametrize(
    "payload",
    [{}, [], {"query": 3}, {"query": " "}, {"query": "x" * 1001}, {"query": "ok", "extra": True}],
)
def test_invalid_payloads_never_start_work(web, payload):
    assert request(web, "POST", "/api/digest", payload)[0] == 400
    assert web.calls == []


def test_body_bounds_media_type_and_invalid_json(web):
    assert request(web, "POST", "/api/digest", raw="x" * (MAX_BODY + 1))[0] == 413
    assert request(web, "POST", "/api/digest", raw="{")[0] == 400
    assert request(web, "POST", "/api/digest", raw="[" * 1500 + "]" * 1500)[0] == 400
    assert (
        request(web, "POST", "/api/digest", {"query": "attention"}, {"Content-Type": "text/plain"})[
            0
        ]
        == 415
    )


def test_report_path_traversal_and_symlink_escape_are_rejected(web, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-private.html")
    outside.write_text("private")
    folder = web.root / "runs" / "abcdef123456" / "export"
    folder.mkdir(parents=True)
    (folder / "report.html").symlink_to(outside)
    for path in [
        "/report/../../private.html",
        "/report/%2e%2e/private.html",
        "/report/abcdef123456",
    ]:
        assert request(web, path=path)[0] == 404
    assert request(web, "POST", "/api/ask", {"session": "../private", "question": "x"})[0] == 400
    outside.unlink()


def test_duplicate_host_and_body_length_headers_are_rejected(web):
    for duplicate, expected in [("Host", 403), ("Content-Length", 400)]:
        connection = HTTPConnection("127.0.0.1", web.server.server_port)
        connection.putrequest("POST", "/api/digest")
        connection.putheader("Origin", f"http://127.0.0.1:{web.server.server_port}")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "2")
        connection.putheader(
            duplicate, f"127.0.0.1:{web.server.server_port}" if duplicate == "Host" else "2"
        )
        connection.endheaders(b"{}")
        assert connection.getresponse().status == expected
        connection.close()
    assert web.calls == []


@pytest.mark.parametrize(
    "query, message", [("fail", "Ollama is not reachable"), ("crash", "failed unexpectedly")]
)
def test_job_failure_is_reported_and_resources_release(web, query, message):
    _, body, _ = request(web, "POST", "/api/digest", {"query": query})
    job = wait_job(web, json.loads(body)["id"])
    assert job["status"] == "failed" and message in job["error"]
    assert "secret internal details" not in job["error"]
    assert ("closed",) in web.calls
    assert request(web, "POST", "/api/digest", {"query": "again"})[0] == 202


def test_existing_cli_operation_lock_is_respected(web):
    with Store(web.root).exclusive():
        _, body, _ = request(web, "POST", "/api/digest", {"query": "attention"})
        job = wait_job(web, json.loads(body)["id"])
        assert job["status"] == "failed" and "Another PaperTrail operation" in job["error"]
    assert web.calls == []


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "localhost", "192.168.1.2"])
def test_server_refuses_any_bind_except_numeric_loopback(tmp_path, host):
    with pytest.raises(ValueError, match="127.0.0.1"):
        make_server(Settings(data_dir=tmp_path), host=host, port=0)


class ProcessAgent:
    """Spawnable fixture that holds the real writer lock while deliberately stuck."""

    def __init__(self, settings, observer):
        self.store, self.observer, self.settings = Store(settings.data_dir), observer, settings

    def new(self, query):
        state = self.store.load("abcdef123456").model_copy(
            deep=True,
            update={
                "id": "abcdef098765" if query == "hang" else "abcdef876543",
                "query": query,
                "status": "running",
                "node": Node.BRIEF,
            },
        )
        self.store.save(state)
        self.observer(Event(node="brief.generate", status="running", detail=state.id))
        if query == "hang":
            (self.settings.data_dir / "worker.pid").write_text(str(os.getpid()))
            while True:
                time.sleep(0.05)
        state.status, state.node = "ready", Node.READY
        self.store.save(state)
        return state

    def resume(self, session):
        state = self.store.load(session)
        assert self.settings.model == state.model
        assert self.settings.embedding_model == state.embedding_model
        assert state.node == Node.BRIEF
        state.status, state.node, state.error = "ready", Node.READY, None
        self.store.save(state)
        return state

    def ask(self, session, question):
        state = self.store.load(session)
        self.observer(Event(node="qa.generate", status="running", detail=state.id))
        if question == "hang":
            (self.settings.data_dir / "worker.pid").write_text(str(os.getpid()))
            while True:
                time.sleep(0.05)
        return state

    def parsed(self, _state):
        return SimpleNamespace(
            chunks=[Chunk.model_validate_json((self.settings.data_dir / "chunk.json").read_text())]
        )

    def close(self):
        pass


def test_process_deadline_kills_worker_releases_lock_and_allows_resume_and_new_job(
    tmp_path, paper, chunk, briefing
):
    store = Store(tmp_path)
    store.save(
        RunState(
            id="abcdef123456",
            query="original",
            model="saved-model",
            embedding_model="saved-embed",
            status="ready",
            node=Node.READY,
            paper=paper,
            briefing=briefing,
        )
    )
    (tmp_path / "chunk.json").write_text(chunk.model_dump_json())
    server = make_server(
        Settings(data_dir=tmp_path, operation_timeout=3),
        port=0,
        agent_factory=ProcessAgent,
        isolate=True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    web = SimpleNamespace(server=server)
    try:
        _, body, _ = request(web, "POST", "/api/digest", {"query": "hang"})
        job = wait_job(web, json.loads(body)["id"], timeout=8)
        assert job["status"] == "failed" and job["timed_out"] is True
        assert job["stage"] == "brief.generate"
        assert 2.8 <= job["elapsed_seconds"] < 7
        assert job["retry"] == {"path": "/api/resume", "payload": {"session": "abcdef098765"}}
        assert store.load("abcdef098765").status == "failed"
        pid = int((tmp_path / "worker.pid").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        with store.exclusive():
            pass  # OS-level lock released, not merely the browser's active flag.

        _, body, _ = request(web, "POST", "/api/resume", {"session": "abcdef098765"})
        resumed = wait_job(web, json.loads(body)["id"], timeout=8)
        assert resumed["status"] == "succeeded"
        assert resumed["session"] == "abcdef098765"
        assert store.load(resumed["session"]).status == "ready"

        _, body, _ = request(web, "POST", "/api/digest", {"query": "works"})
        next_job = wait_job(web, json.loads(body)["id"], timeout=8)
        assert next_job["status"] == "succeeded"
        assert request(web, path=next_job["report"])[0] == 200

        _, body, _ = request(
            web, "POST", "/api/ask", {"session": next_job["session"], "question": "hang"}
        )
        failed_question = wait_job(web, json.loads(body)["id"], timeout=8)
        assert failed_question["status"] == "failed"
        assert "Retry the question" in failed_question["error"]
        assert store.load(next_job["session"]).status == "ready"
        assert store.load(next_job["session"]).events[-1].node == "qa.operation"
        _, body, _ = request(
            web, "POST", "/api/ask", {"session": next_job["session"], "question": "works"}
        )
        assert wait_job(web, json.loads(body)["id"], timeout=8)["status"] == "succeeded"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_operation_deadline_rejects_invalid_configuration(seconds):
    with pytest.raises(ValueError, match="positive"):
        Settings(operation_timeout=seconds)

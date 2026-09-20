"""An explicit state graph with checkpointed transitions and a persistent QA loop."""

import re
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from papertrail.config import Settings
from papertrail.errors import GroundingError, PaperTrailError
from papertrail.grounding import validate_briefing
from papertrail.llm import Ollama
from papertrail.parsing import parse_pdf, section_label
from papertrail.retrieval import Embedder, HybridIndex, rank_papers
from papertrail.schema import Answer, Event, Exchange, Node, ParsedPaper, RunState
from papertrail.sources import ArxivClient, understand
from papertrail.storage import Store, atomic_json
from papertrail.uploads import copy_upload

EDGES = {
    Node.UNDERSTAND: Node.RETRIEVE,
    Node.RETRIEVE: Node.SELECT,
    Node.SELECT: Node.FETCH,
    Node.FETCH: Node.PARSE,
    Node.PARSE: Node.INDEX,
    Node.INDEX: Node.BRIEF,
    Node.BRIEF: Node.VALIDATE,
    Node.VALIDATE: Node.READY,
}


class Agent:
    def __init__(
        self, settings: Settings, observer=None, source=None, embedder=None, llm=None, index=None
    ):
        self.settings = settings
        self.store = Store(settings.data_dir)
        self.observer = observer or (lambda event: None)
        self.source = source or ArxivClient(
            settings.data_dir / "arxiv-cache", settings.max_pdf_bytes
        )
        self.embedder = embedder or Embedder(settings.embedding_model, settings.data_dir / "models")
        self.index = index or HybridIndex(settings.data_dir / "vectors", self.embedder)
        self.llm = llm or Ollama(settings.ollama_url, settings.model, settings.model_timeout)

    def new(self, query: str) -> RunState:
        state = RunState(
            id=uuid4().hex[:12],
            query=query,
            model=self.settings.model,
            embedding_model=self.settings.embedding_model,
        )
        self.store.save(state)
        return self.run(state)

    def new_upload(self, path: Path, filename: str) -> RunState:
        run_id = uuid4().hex[:12]
        paper, sha256 = copy_upload(
            path,
            self.store.run_dir(run_id) / "paper.pdf",
            filename,
            self.settings.max_pdf_bytes,
        )
        state = RunState(
            id=run_id,
            query=paper.source_filename,
            model=self.settings.model,
            embedding_model=self.settings.embedding_model,
            node=Node.PARSE,
            paper=paper,
            pdf_sha256=sha256,
            warnings=[
                "Uploaded PDF: the filename is a display label. Bibliographic title, authors, publication date and source URL have not been verified."
            ],
        )
        self.store.save(state)
        return self.run(state)

    def resume(self, run_id: str) -> RunState:
        state = self.store.load(run_id)
        if (
            state.model != self.settings.model
            or state.embedding_model != self.settings.embedding_model
        ):
            raise PaperTrailError(
                f"This session uses {state.model} / {state.embedding_model}. Resume with its original model; start a new digest to change models."
            )
        return self.run(state)

    def parsed(self, state: RunState) -> ParsedPaper:
        path = self.store.run_dir(state.id) / "parsed.json"
        if not path.exists():
            raise PaperTrailError(
                "Saved passages are missing. Create a new digest to rebuild this session."
            )
        return ParsedPaper.model_validate_json(path.read_text())

    def run(self, state: RunState) -> RunState:
        state.error = None
        state.status = "running"
        while state.node != Node.READY:
            current = state.node
            self.store.save(state)
            self.observer(Event(node=current, status="running", detail=state.id))
            started = time.monotonic()
            try:
                getattr(self, "_" + current.value)(state)
            except Exception as exc:
                state.status = "failed"
                state.error = (
                    str(exc)
                    if isinstance(exc, PaperTrailError)
                    else f"Unexpected {type(exc).__name__} at {current.value}. Run with --debug for details."
                )
                event = Event(
                    node=current,
                    status="failed",
                    seconds=round(time.monotonic() - started, 3),
                    detail=state.error,
                )
                state.events.append(event)
                self.store.save(state)
                self.observer(event)
                if isinstance(exc, PaperTrailError):
                    recovery = (
                        "Start a new briefing with a corrected ID or topic."
                        if current == Node.UNDERSTAND
                        else "Choose a supported, text-readable PDF to start a new briefing."
                        if current == Node.PARSE and state.paper and state.paper.source == "upload"
                        else f"Retry: papertrail resume {state.id}"
                    )
                    raise PaperTrailError(
                        f"{state.error}\nSaved session: {state.id}. {recovery}"
                    ) from exc
                raise
            event = Event(
                node=current, status="completed", seconds=round(time.monotonic() - started, 3)
            )
            state.events.append(event)
            state.node = EDGES[current]
            self.store.save(state)
            self.observer(event)
        state.status = "ready"
        self.store.save(state)
        return state

    def _understand(self, state):
        state.intent = understand(state.query)

    def _retrieve(self, state):
        state.candidates, warnings = self.source.search(state.intent, self.settings.candidates)
        state.warnings.extend(warnings)

    def _select(self, state):
        state.candidates = rank_papers(state.intent, state.candidates, self.embedder)
        state.paper = state.candidates[0]
        if state.intent.kind == "topic" and state.paper.relevance < 0.4:
            state.warnings.append(
                "The top candidate has weak topic relevance. Inspect the candidate list; try a more specific topic if necessary."
            )

    def _fetch(self, state):
        state.pdf_sha256 = self.source.download(
            state.paper, self.store.run_dir(state.id) / "paper.pdf"
        )

    def _parse(self, state):
        parsed = parse_pdf(
            self.store.run_dir(state.id) / "paper.pdf",
            state.paper.identity,
            self.settings.max_pages,
        )
        if state.pdf_sha256 != parsed.sha256:
            raise PaperTrailError("The saved PDF changed before parsing. Start a new digest.")
        state.pages, state.chunk_count = parsed.pages, len(parsed.chunks)
        state.warnings.extend(parsed.warnings)
        atomic_json(self.store.run_dir(state.id) / "parsed.json", parsed.model_dump())

    def _index(self, state):
        # The same PDF can enter through arXiv and upload with different chunk
        # identities. Keep their vector payloads separate while allowing repeat
        # uploads of identical bytes to reuse a compatible collection.
        cache_key = state.pdf_sha256
        if state.paper.source == "upload":
            cache_key = f"{cache_key}|{state.paper.identity}"
        state.collection = self.index.build(self.parsed(state).chunks, cache_key)

    def briefing_evidence(self, state):
        chunks = self.parsed(state).chunks
        abstracts = [
            c for c in chunks if section_label(c.section) == "abstract" and not c.reference
        ]
        # Some papers print an opening abstract without an Abstract heading.
        # Retain bounded first-page context so a summary is not built solely
        # from methods/results passages; all claims still require quote checks.
        opening = abstracts[:1] or [c for c in chunks if c.page == 1 and not c.reference][:2]
        selected = {c.id: c for c in opening}
        for query, section_pattern, count in (
            ("problem motivation contribution", r"intro|background|problem", 2),
            (
                "proposed method architecture algorithm approach",
                r"method|approach|architecture|attention|encoder|decoder",
                3,
            ),
            (
                "experimental results evaluation performance comparison baseline",
                r"experiment|evaluation|result|translation|performance",
                3,
            ),
            (
                "limitations weaknesses future work constraints failure",
                r"limit|discussion|conclusion|variation|future",
                2,
            ),
        ):
            preferred = [
                c for c in chunks if re.search(section_pattern, c.section, re.I) and not c.reference
            ]
            for hit in self.index.search(state.collection, preferred or chunks, query, limit=count):
                selected.setdefault(hit.chunk.id, hit.chunk)
        return list(selected.values())[:11]

    def _model_progress(self, state, event):
        # Save each completed substage before the next model call. A worker
        # deadline can then recover both the checkpoint and its timings.
        if event.status != "running":
            state.events.append(event)
            self.store.save(state)
        self.observer(event)

    def _prepare_model(self, state):
        self.llm.observer = lambda event: self._model_progress(state, event)
        self.llm.available()
        warmup = getattr(self.llm, "warmup", None)
        if callable(warmup):
            warmup()

    def _brief(self, state):
        self._prepare_model(state)
        evidence = self.briefing_evidence(state)
        state.briefing_evidence_ids = [c.id for c in evidence]
        state.briefing = self.llm.briefing(state.paper, evidence)
        if getattr(self.llm, "extractive_fallbacks", 0):
            state.warnings.append(
                f"Automatic support review replaced {self.llm.extractive_fallbacks} briefing paraphrase(s) with visibly labeled source wording. Inspect those passages; the reviewer is fallible."
            )
        # Explanatory text is controlled by the program; all factual prose is cited.
        state.briefing.limitations_note = (
            "Limitations below are reported in the selected source passages."
            if state.briefing.limitations
            else "No explicit limitations were identified in the selected evidence; this does not establish that the paper has none."
        )

    def _validate(self, state):
        allowed = set(state.briefing_evidence_ids)
        validate_briefing(
            state.briefing, {c.id: c for c in self.parsed(state).chunks if c.id in allowed}
        )

    def ask(self, run_id: str, question: str) -> RunState:
        if not question.strip() or len(question) > 1000:
            raise PaperTrailError("Questions must contain 1-1000 characters.")
        state = self.store.load(run_id)
        if state.status != "ready":
            raise PaperTrailError(f"Session is not ready. First run `papertrail resume {run_id}`.")
        if state.embedding_model != self.embedder.name:
            raise PaperTrailError(
                "Embedding configuration differs from this session; use the original configuration."
            )
        if state.model != self.llm.model:
            raise PaperTrailError(f"This session uses {state.model}. Pass --model {state.model}.")
        started = time.monotonic()
        self.observer(Event(node="qa.retrieve", status="running", detail=state.id))
        # Only expand genuinely referential questions, avoiding unrelated conversation contamination.
        contextual = bool(
            re.search(
                r"\b(it|its|they|their|that|those|this approach|the method)\b", question, re.I
            )
        )
        retrieval_query = question
        if contextual and state.exchanges:
            retrieval_query = state.exchanges[-1].question + "\n" + question
        chunks = self.parsed(state).chunks
        include_references = bool(
            re.search(r"\b(cit(?:e|es|ed|ation(?:s)?)|references?|bibliography)\b", question, re.I)
        )
        hits = self.index.search(
            state.collection, chunks, retrieval_query, include_references=include_references
        )
        self._model_progress(
            state,
            Event(
                node="qa.retrieve",
                status="completed",
                seconds=round(time.monotonic() - started, 3),
                detail=f"{len(hits)} passages",
            ),
        )
        try:
            if not hits:
                answer = Answer(
                    status="insufficient_evidence",
                    explanation="No readable source passages were retrieved.",
                )
            else:
                self._prepare_model(state)
                answer = self.llm.answer(
                    question, [h.chunk for h in hits], [e.question for e in state.exchanges]
                )
        except GroundingError as exc:
            answer = Answer(
                status="insufficient_evidence",
                explanation="A draft answer failed evidence validation; no unverified answer is shown.",
            )
            self._model_progress(
                state, Event(node="qa.validate", status="rejected", detail=str(exc))
            )
        except PaperTrailError as exc:
            self._model_progress(state, Event(node="qa.answer", status="failed", detail=str(exc)))
            self.store.save(state)
            raise
        elapsed = round(time.monotonic() - started, 3)
        state.exchanges.append(
            Exchange(
                question=question,
                retrieval_query=retrieval_query,
                answer=answer,
                retrieved_ids=[h.chunk.id for h in hits],
                elapsed_seconds=elapsed,
            )
        )
        state.events.append(
            Event(node="qa.answer", status="completed", seconds=elapsed, detail=answer.status)
        )
        self.store.save(state)
        return state

    def close(self):
        self.index.close()
        self.source.close()
        self.llm.close()


def settings_for_session(settings: Settings, run_id: str) -> Settings:
    """Read-only commands can open older sessions without guessing their model."""
    state = Store(settings.data_dir).load(run_id)
    return replace(settings, model=state.model, embedding_model=state.embedding_model)

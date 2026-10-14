import hashlib
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.pdfgen import canvas

from papertrail.config import Settings
from papertrail.errors import PaperTrailError, ParseError
from papertrail.graph import Agent
from papertrail.parsing import parse_pdf
from papertrail.retrieval import HybridIndex
from papertrail.schema import Briefing, Claim, Evidence, Node, Paper
from papertrail.uploads import copy_upload, display_filename


def write_pdf(path: Path, text: str = "", encrypt=None) -> None:
    document = canvas.Canvas(str(path), encrypt=encrypt)
    lines = document.beginText(45, 760)
    lines.setFont("Helvetica", 10)
    lines.textLine("Abstract")
    words = (text or "The instrument measures temperature using a calibrated optical sensor. ") * 12
    for offset in range(0, len(words.split()), 10):
        lines.textLine(" ".join(words.split()[offset : offset + 10]))
    document.drawText(lines)
    document.showPage()
    document.save()


class SourceForbidden:
    def search(self, *args):
        pytest.fail("Uploaded documents must not search arXiv")

    def download(self, *args):
        pytest.fail("Uploaded documents must not download from arXiv")

    def close(self):
        pass


def upload_agent(tmp_path) -> Agent:
    def briefing(paper, chunks):
        claim = Claim(
            text="The instrument measures temperature using a calibrated optical sensor.",
            evidence=[Evidence(chunk_id=chunks[0].id, quote=chunks[0].text[:250])],
        )
        return Briefing(
            summary=claim,
            problem=claim,
            method=[claim],
            results=[claim],
            limitations=[],
            limitations_note="",
            follow_up_questions=["What does the instrument measure?", "How was it calibrated?"],
        )

    llm = Mock(model="test-model", extractive_fallbacks=0)
    llm.briefing.side_effect = briefing
    index = Mock()
    index.build.return_value = "upload-collection"
    index.search.return_value = []
    return Agent(
        Settings(data_dir=tmp_path / "state", model="test-model"),
        source=SourceForbidden(),
        index=index,
        llm=llm,
    )


def test_upload_runs_existing_graph_without_arxiv_or_invented_metadata(tmp_path):
    path = tmp_path / "document.pdf"
    write_pdf(path)
    agent = upload_agent(tmp_path)
    state = agent.new_upload(path, "../../private/document.pdf")
    assert state.status == "ready"
    assert state.node == Node.READY
    assert state.intent is None and state.candidates == []
    assert state.paper.source == "upload"
    assert state.paper.identity == f"upload:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    assert state.paper.title == state.paper.source_filename == "document.pdf"
    assert state.paper.arxiv_id == state.paper.url == state.paper.pdf_url == ""
    assert state.paper.published == state.paper.updated == ""
    assert state.paper.authors == []
    assert agent.store.load(state.id) == state
    assert (agent.store.run_dir(state.id) / "paper.pdf").read_bytes() == path.read_bytes()
    assert all(chunk.paper_id == state.paper.identity for chunk in agent.parsed(state).chunks)
    assert {event.node for event in state.events} == {"parse", "index", "brief", "validate"}
    assert agent.index.build.call_args.args[1] == f"{state.pdf_sha256}|{state.paper.identity}"


def test_upload_resume_reuses_saved_pdf_and_parse_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "document.pdf"
    write_pdf(path)
    agent = upload_agent(tmp_path)
    counted_parse = Mock(wraps=parse_pdf)
    monkeypatch.setattr("papertrail.graph.parse_pdf", counted_parse)
    agent.index.build.side_effect = [PaperTrailError("Index unavailable"), "collection"]
    with pytest.raises(PaperTrailError, match="Index unavailable"):
        agent.new_upload(path, "document.pdf")
    failed = agent.store.recent()[0]
    assert failed.node == Node.INDEX
    path.unlink()
    result = agent.resume(failed.id)
    assert result.status == "ready"
    assert counted_parse.call_count == 1
    assert agent.llm.briefing.call_count == 1


def test_upload_identity_is_content_based_and_sessions_are_independent(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    write_pdf(first)
    write_pdf(second, "The instrument measures pressure using a calibrated digital sensor. ")
    agent = upload_agent(tmp_path)
    agent.llm.briefing.side_effect = None
    # Inspect the initial checkpoint without substituting a document's claims.
    agent.run = lambda state: state
    a = agent.new_upload(first, "first.pdf")
    a_again = agent.new_upload(first, "renamed.pdf")
    b = agent.new_upload(second, "first.pdf")
    assert a.paper.identity == a_again.paper.identity
    assert a.paper.identity != b.paper.identity
    assert len({a.id, a_again.id, b.id}) == 3
    assert all(state.exchanges == [] and state.collection is None for state in (a, a_again, b))
    parsed_a = parse_pdf(agent.store.run_dir(a.id) / "paper.pdf", a.paper.identity)
    parsed_b = parse_pdf(agent.store.run_dir(b.id) / "paper.pdf", b.paper.identity)
    assert {c.id for c in parsed_a.chunks}.isdisjoint(c.id for c in parsed_b.chunks)


def test_old_arxiv_records_keep_their_identity(paper):
    old = paper.model_dump(exclude={"source", "document_id", "source_filename"})
    loaded = Paper.model_validate(old)
    assert loaded.source == "arxiv"
    assert loaded.identity == paper.arxiv_id


def test_upload_cannot_reuse_incompatible_arxiv_vector_payloads(tmp_path):
    class Embedder:
        name = "test-vector-isolation"

        def documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def query(self, text):
            return [1.0, 0.0]

    path = tmp_path / "document.pdf"
    write_pdf(path)
    agent = upload_agent(tmp_path)
    agent.run = lambda state: state
    state = agent.new_upload(path, "document.pdf")
    agent._parse(state)
    chunks = agent.parsed(state).chunks
    agent.index = HybridIndex(tmp_path / "vectors", Embedder())
    try:
        arxiv_chunks = [
            chunk.model_copy(update={"id": f"{i:016x}", "paper_id": "1706.03762v7"})
            for i, chunk in enumerate(chunks)
        ]
        arxiv_collection = agent.index.build(arxiv_chunks, state.pdf_sha256)
        agent._index(state)
        assert state.collection != arxiv_collection
        hits = agent.index.search(state.collection, chunks, "temperature", mode="dense")
        assert hits
        assert all(hit.chunk.paper_id == state.paper.identity for hit in hits)
        assert {hit.chunk.id for hit in hits} <= {chunk.id for chunk in chunks}
    finally:
        agent.index.close()


@pytest.mark.parametrize("password", ["", "secret"])
def test_encrypted_upload_rejected_even_with_empty_password(tmp_path, password):
    path = tmp_path / "encrypted.pdf"
    write_pdf(path, encrypt=StandardEncryption(password, ownerPassword="owner-secret"))
    agent = upload_agent(tmp_path)
    with pytest.raises(PaperTrailError, match="Encrypted PDFs are not supported"):
        agent.new_upload(path, "encrypted.pdf")
    failed = agent.store.recent()[0]
    assert failed.node == Node.PARSE and failed.status == "failed"
    agent.index.build.assert_not_called()
    agent.llm.briefing.assert_not_called()


def test_corrupt_upload_has_failed_parse_checkpoint_and_no_model_call(tmp_path):
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"%PDF-1.4\nbroken content")
    agent = upload_agent(tmp_path)
    with pytest.raises(PaperTrailError, match="PDF parsing failed"):
        agent.new_upload(path, "corrupt.pdf")
    assert agent.store.recent()[0].node == Node.PARSE
    agent.index.build.assert_not_called()
    agent.llm.briefing.assert_not_called()


@pytest.mark.parametrize(("pages", "message"), [(1, "little readable"), (2, "limit is 1")])
def test_unreadable_and_over_page_limit_uploads_stop_before_indexing(tmp_path, pages, message):
    path = tmp_path / "unsupported.pdf"
    document = canvas.Canvas(str(path))
    for _ in range(pages):
        document.showPage()
    document.save()
    agent = upload_agent(tmp_path)
    agent.settings = replace(agent.settings, max_pages=1)
    with pytest.raises(PaperTrailError, match=message):
        agent.new_upload(path, "unsupported.pdf")
    assert agent.store.recent()[0].status == "failed"
    agent.index.build.assert_not_called()
    agent.llm.briefing.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "filename", "max_bytes", "message"),
    [
        (b"not PDF", "paper.pdf", 1024, "not a PDF"),
        (b"%PDF-1.4" + b"x" * 100, "paper.pdf", 16, "exceeds"),
        (b"%PDF-1.4", "paper.exe", 1024, "Choose a PDF"),
    ],
)
def test_invalid_upload_does_not_leave_partial_pdf(tmp_path, payload, filename, max_bytes, message):
    source = tmp_path / "source"
    destination = tmp_path / "paper.pdf"
    source.write_bytes(payload)
    with pytest.raises(ParseError, match=message):
        copy_upload(source, destination, filename, max_bytes)
    assert not destination.exists()
    assert not destination.with_suffix(".pdf.tmp").exists()


def test_untrusted_filename_is_only_a_sanitized_display_label():
    assert display_filename("C:\\Users\\private\\paper\x00.pdf") == "paper.pdf"
    assert display_filename("../../report.pdf") == "report.pdf"


def test_long_pdf_filename_preserves_its_extension():
    name = display_filename("x" * 251 + ".pdf")
    assert len(name) == 240
    assert name.endswith(".pdf")

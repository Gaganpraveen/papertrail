"""Offline regressions found during source and recovery boundary review."""

import hashlib
import json

import httpx
import pytest

from papertrail.errors import ModelError, SourceError
from papertrail.llm import DraftAnswer, Ollama, resolve_claim
from papertrail.parsing import chunk_section
from papertrail.sources import ArxivClient, understand


@pytest.mark.parametrize("section", ["8 References", "10 Bibliography", "A References"])
def test_numbered_reference_sections_are_filtered(section):
    chunks = chunk_section(
        "A cited prior paper describes attention and scientific retrieval techniques.",
        "1706.03762v7",
        3,
        section,
    )
    assert chunks and all(chunk.reference for chunk in chunks)


@pytest.mark.parametrize(
    "value",
    [
        "https://arxiv.org:abc/abs/1706.03762",
        "https://[arxiv.org/abs/1706.03762",
    ],
)
def test_malformed_arxiv_urls_have_actionable_errors(value):
    with pytest.raises(SourceError):
        understand(value)


def test_corrupt_metadata_cache_is_refetched(tmp_path):
    params = {"id_list": "1706.03762"}
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    (tmp_path / f"{key}.json").write_text("{")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"/>')

    client = ArxivClient(
        tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)), delay=0
    )
    try:
        with pytest.raises(SourceError, match="No matching"):
            client.search(understand("1706.03762"))
        assert len(requests) == 1
    finally:
        client.close()


def test_fragmented_pdf_header_is_accepted(tmp_path, paper):
    payload = b"%PDF-1.7\n" + b"x" * 120

    class FragmentedStream(httpx.SyncByteStream):
        def __iter__(self):
            yield payload[:2]
            yield payload[2:]

    client = ArxivClient(
        tmp_path,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=FragmentedStream())
            )
        ),
        delay=0,
    )
    try:
        destination = tmp_path / "paper.pdf"
        assert client.download(paper, destination) == hashlib.sha256(payload).hexdigest()
        assert destination.read_bytes() == payload
    finally:
        client.close()


@pytest.mark.parametrize("prompt", ["x" * 28_001, "é" * 14_001])
def test_context_byte_overflow_prevents_http_request(prompt):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(500)

    llm = Ollama(
        "http://localhost:11434",
        "test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        with pytest.raises(ModelError, match="safe model context budget"):
            llm.generate(DraftAnswer, prompt, lambda draft: None)
        assert requests == []
        assert llm.calls == 0
    finally:
        llm.close()


@pytest.mark.parametrize(
    "rejected_draft",
    [
        "not valid JSON",
        json.dumps(
            {
                "status": "answered",
                "claims": [{"evidence_ids": ["E404"], "text": "An unsupported statement."}],
            }
        ),
    ],
)
def test_repair_messages_retain_the_rejected_draft(rejected_draft):
    accepted = DraftAnswer(status="insufficient_evidence")
    replies = iter([rejected_draft, accepted.model_dump_json()])
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": next(replies)}})

    def validate(draft):
        for claim in draft.claims:
            resolve_claim(claim, {})

    llm = Ollama(
        "http://localhost:11434",
        "test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        result = llm.generate(DraftAnswer, "Answer using the supplied evidence only.", validate)
        assert result == accepted
        assert len(requests) == 2
        original_messages = requests[0]["messages"]
        repair_messages = requests[1]["messages"]
        assert repair_messages[:2] == original_messages
        assert repair_messages[2] == {"role": "assistant", "content": rejected_draft}
        assert repair_messages[3]["role"] == "user"
        assert "Validation rejected it:" in repair_messages[3]["content"]
    finally:
        llm.close()


def test_repair_draft_is_included_in_context_budget():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"message": {"content": "invalid JSON " * 500}})

    llm = Ollama(
        "http://localhost:11434",
        "test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    try:
        with pytest.raises(ModelError, match="safe model context budget"):
            llm.generate(DraftAnswer, "x" * 24_000, lambda draft: None)
        assert len(requests) == 1
    finally:
        llm.close()

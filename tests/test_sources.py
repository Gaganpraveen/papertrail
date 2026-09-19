import httpx
import pytest

from papertrail.errors import SourceError
from papertrail.sources import ArxivClient, normalize_id, parse_feed, understand


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1706.03762", "1706.03762"),
        ("arXiv:1706.03762v7", "1706.03762v7"),
        ("https://arxiv.org/pdf/1706.03762v7.pdf", "1706.03762v7"),
        ("https://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
        ("math.GT/0309136", "math.GT/0309136"),
    ],
)
def test_id_formats(value, expected):
    assert normalize_id(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "https://evil.test/abs/1706.03762",
        "https://arxiv.org.evil.test/pdf/1706.03762",
        "http://localhost/private",
        "https://user@arxiv.org/pdf/1706.03762",
        "https://arxiv.org:443/pdf/1706.03762",
        "https://arxiv.org/pdf/1706.03762?next=localhost",
        "",
        "2413.12345",
    ],
)
def test_invalid_input_rejected(value):
    with pytest.raises(SourceError):
        understand(value)


def test_natural_language_query():
    intent = understand("recent work on KV-cache compression for LLMs")
    assert intent.keywords == ["kv-cache", "compression", "llms"]
    assert intent.recent


def test_xml_entities_rejected():
    with pytest.raises(SourceError):
        parse_feed(b'<!DOCTYPE x [<!ENTITY a SYSTEM "file:///etc/passwd">]><feed>&a;</feed>')


def test_retry_and_empty_results(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"/>')

    client = ArxivClient(
        tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)), delay=0
    )
    with pytest.raises(SourceError, match="No matching"):
        client.search(understand("1706.03762"))
    assert len(calls) == 1


def test_topic_broadens_once(tmp_path):
    queries = []

    def handler(request):
        queries.append(request.url.params["search_query"])
        return httpx.Response(200, text='<feed xmlns="http://www.w3.org/2005/Atom"/>')

    client = ArxivClient(
        tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)), delay=0
    )
    with pytest.raises(SourceError):
        client.search(understand("cache compression"))
    assert " AND " in queries[0] and " OR " in queries[1]


@pytest.mark.parametrize(
    "query, limit, expected",
    [
        ("research about electron", 8, {"search_query": "all:electron"}),
        (
            "attention mechanisms for machine translation",
            8,
            {
                "search_query": "all:attention AND all:mechanisms AND all:machine AND all:translation"
            },
        ),
        (
            "recent electron research",
            8,
            {"search_query": "all:electron", "sortBy": "submittedDate", "sortOrder": "descending"},
        ),
        ("electron", 12, {"search_query": "all:electron", "max_results": "12"}),
    ],
)
def test_topic_406_canonical_retry_preserves_intent_and_limit(tmp_path, query, limit, expected):
    requests = []
    body = (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        + "".join(
            f"<entry><id>https://arxiv.org/abs/2601.{i:05d}v1</id>"
            f"<title>Synthetic fixture paper {i}</title></entry>"
            for i in range(max(limit, 10))
        )
        + "</feed>"
    )

    def handler(request):
        requests.append(request)
        return httpx.Response(406) if len(requests) == 1 else httpx.Response(200, text=body)

    client = ArxivClient(
        tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)), delay=0
    )
    try:
        papers, warnings = client.search(understand(query), limit=limit)
        assert len(requests) == 2
        assert dict(requests[1].url.params) == expected
        assert "search_query=all:" in str(requests[1].url)
        assert len(papers) == limit
        assert papers[-1].arxiv_id == f"2601.{limit - 1:05d}v1"
        assert len(warnings) == 1 and "equivalent canonical API query" in warnings[0]
    finally:
        client.close()


def test_topic_406_fallback_stops_after_one_alternate_request(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(406)

    client = ArxivClient(
        tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)), delay=0
    )
    try:
        with pytest.raises(SourceError, match="HTTP 406"):
            client.search(understand("attention mechanisms for machine translation"))
        assert len(requests) == 2
        assert all(" AND " in request.url.params["search_query"] for request in requests)
        assert not list(tmp_path.glob("*.json"))
    finally:
        client.close()


@pytest.mark.parametrize("query, status", [("KV-cache compression", 406), ("electron", 403)])
def test_topic_fallback_never_rewrites_punctuation_or_other_errors(tmp_path, query, status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status)

    client = ArxivClient(
        tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)), delay=0
    )
    try:
        with pytest.raises(SourceError, match=f"HTTP {status}"):
            client.search(understand(query))
        assert len(requests) == 1
    finally:
        client.close()


def test_pdf_size_bound_and_cleanup(tmp_path, paper):
    client = ArxivClient(
        tmp_path,
        max_bytes=20,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"%PDF-" + b"a" * 100)
            )
        ),
        delay=0,
    )
    with pytest.raises(SourceError, match="exceeds"):
        client.download(paper, tmp_path / "paper.pdf")
    assert not (tmp_path / "paper.part").exists()
    assert not (tmp_path / "paper.pdf").exists()


def test_pdf_redirect_is_not_followed(tmp_path, paper, monkeypatch):
    monkeypatch.setattr("papertrail.sources.time.sleep", lambda seconds: None)
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    client = ArxivClient(
        tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)), delay=0
    )
    with pytest.raises(SourceError):
        client.download(paper, tmp_path / "paper.pdf")
    assert all(url.startswith("https://arxiv.org/pdf/") for url in calls)


def test_version_fallback_requires_exact_match(tmp_path, paper, monkeypatch):
    client = ArxivClient(tmp_path, delay=0)
    calls = iter([SourceError("arXiv returned HTTP 406."), [paper]])

    def feed(params):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(client, "_feed", feed)
    papers, warnings = client.search(understand("1706.03762v7"))
    assert papers[0].arxiv_id == "1706.03762v7"
    assert "exact requested revision" in warnings[0]


def test_version_fallback_cannot_substitute_newer_paper(tmp_path, paper, monkeypatch):
    client = ArxivClient(tmp_path, delay=0)
    calls = iter([SourceError("arXiv returned HTTP 406."), [paper]])

    def feed(params):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(client, "_feed", feed)
    with pytest.raises(SourceError, match="different revision"):
        client.search(understand("1706.03762v2"))

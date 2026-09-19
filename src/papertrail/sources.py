"""Official arXiv API only; bounded, serialized requests and validated download URLs."""

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlencode, urlparse

import httpx
from defusedxml import ElementTree

from papertrail.errors import SourceError
from papertrail.schema import Intent, Paper
from papertrail.storage import atomic_json

ID = re.compile(r"(?:\d{2}(?:0[1-9]|1[0-2])\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v[1-9]\d*)?")
STOP = set(
    "a an the on of for with and or in to from about work research paper papers recent latest new advances study studies please find me explain how what is are does this that their using".split()
)
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def normalize_id(value: str) -> str | None:
    text = value.strip()
    if text.lower().startswith("arxiv:"):
        text = text[6:]
    if "://" in text:
        try:
            parsed = urlparse(text)
            port = parsed.port
        except ValueError as exc:
            raise SourceError(
                "The arXiv URL is malformed. Use an arXiv ID or a plain https://arxiv.org/abs/ URL."
            ) from exc
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
            or port
            or parsed.username
            or parsed.password
        ):
            raise SourceError("Only arXiv IDs and arxiv.org paper URLs are accepted.")
        match = re.fullmatch(r"/(?:abs|pdf|html)/(.+?)(?:\.pdf)?/?", unquote(parsed.path))
        if not match or parsed.query or parsed.fragment:
            raise SourceError(
                "Use a plain arXiv /abs/ or /pdf/ paper URL without query parameters."
            )
        text = match[1]
    return text if ID.fullmatch(text) else None


def understand(value: str) -> Intent:
    value = value.strip()
    if not value or len(value) > 500:
        raise SourceError("Enter an arXiv ID or a research topic between 1 and 500 characters.")
    paper_id = normalize_id(value)
    if paper_id:
        return Intent(kind="paper", value=paper_id)
    if "://" in value or re.match(r"^(?:arxiv:)?\d{4}\.", value, re.I):
        raise SourceError(
            "This looks like an invalid arXiv ID. Example: 1706.03762 or 1706.03762v7."
        )
    keywords = list(
        dict.fromkeys(
            x.lower()
            for x in re.findall(r"[A-Za-z0-9]+(?:[-][A-Za-z0-9]+)*", value)
            if x.lower() not in STOP and len(x) > 1
        )
    )[:8]
    if not keywords:
        raise SourceError("Please give a more specific topic, such as 'KV-cache compression'.")
    return Intent(
        kind="topic",
        value=value,
        keywords=keywords,
        recent=bool(re.search(r"\b(recent|latest|new)\b", value, re.I)),
    )


def parse_feed(body: bytes) -> list[Paper]:
    try:
        root = ElementTree.fromstring(body)
    except Exception as exc:
        raise SourceError("arXiv returned an invalid Atom feed. Retry later.") from exc
    result = []
    for entry in root.findall("a:entry", ATOM):

        def field(name: str, entry=entry) -> str:
            return " ".join((entry.findtext(f"a:{name}", "", ATOM)).split())

        raw_id = field("id")
        if "/api/errors" in raw_id:
            raise SourceError(f"arXiv rejected the query: {field('summary')}")
        paper_id = normalize_id(raw_id)
        if paper_id is None:
            continue
        result.append(
            Paper(
                arxiv_id=paper_id,
                title=field("title"),
                abstract=field("summary"),
                published=field("published"),
                updated=field("updated"),
                authors=[
                    " ".join(a.findtext("a:name", "", ATOM).split())
                    for a in entry.findall("a:author", ATOM)
                ],
                categories=[a.attrib.get("term", "") for a in entry.findall("a:category", ATOM)],
                url=f"https://arxiv.org/abs/{paper_id}",
                pdf_url=f"https://arxiv.org/pdf/{paper_id}",
            )
        )
    return result


class ArxivClient:
    def __init__(
        self, cache: Path, max_bytes: int = 30 * 1024 * 1024, client=None, delay: float = 3.0
    ):
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.client = client or httpx.Client(
            timeout=45,
            follow_redirects=False,
            headers={"User-Agent": "PaperTrail/0.1 (arXiv research digest; single-user CLI)"},
        )
        self.delay = delay

    def _throttle(self):
        timestamp = self.cache / "last-request.txt"
        # CLI holds a data-directory lock, so rate limits also apply across processes.
        try:
            previous = float(timestamp.read_text())
        except (OSError, ValueError):
            previous = 0
        wait = self.delay - (time.time() - previous)
        if wait > 0:
            time.sleep(wait)
        timestamp.write_text(str(time.time()))

    def _feed(self, params: dict, *, literal_syntax: bool = False) -> list[Paper]:
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        cached = self.cache / f"{key}.json"
        if cached.exists() and time.time() - cached.stat().st_mtime < 86400:
            try:
                return [Paper.model_validate(x) for x in json.loads(cached.read_text())]
            except (ValueError, TypeError, OSError):
                # Treat malformed cache entries as misses; a successful fetch
                # atomically replaces them, preserving the official source.
                pass
        for attempt in range(3):
            self._throttle()
            try:
                endpoint = "https://export.arxiv.org/api/query"
                if literal_syntax:
                    # Preserve the documented field-prefix colon. This is an
                    # equivalent URL spelling, not a different search source.
                    response = self.client.get(endpoint + "?" + urlencode(params, safe=":"))
                else:
                    response = self.client.get(endpoint, params=params)
                response.raise_for_status()
                if len(response.content) > 2_000_000:
                    raise SourceError("Unexpectedly large metadata response from arXiv.")
                papers = parse_feed(response.content)
                atomic_json(cached, [p.model_dump() for p in papers])
                return papers
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise SourceError(f"arXiv returned HTTP {exc.response.status_code}.") from exc
                if attempt == 2:
                    raise SourceError(
                        "arXiv is unavailable after three attempts. Your session is saved; resume it later."
                    ) from exc
                time.sleep(2**attempt)
        return []

    def search(self, intent: Intent, limit: int = 8) -> tuple[list[Paper], list[str]]:
        warnings = []
        if intent.kind == "paper":
            try:
                papers = self._feed({"id_list": intent.value})
            except SourceError as exc:
                if "HTTP 406" not in str(exc) or not re.search(r"v\d+$", intent.value):
                    raise
                papers = self._feed({"id_list": re.sub(r"v\d+$", "", intent.value)})
                if not papers or papers[0].arxiv_id != intent.value:
                    raise SourceError(
                        "arXiv rejected the version-specific lookup and the base lookup returned a different revision. Refusing to substitute versions."
                    ) from exc
                warnings.append(
                    "Version-specific metadata lookup returned HTTP 406; the base lookup was verified to refer to the exact requested revision."
                )
            if re.search(r"v\d+$", intent.value) and any(
                p.arxiv_id != intent.value for p in papers
            ):
                raise SourceError(
                    "arXiv returned a different paper revision. Refusing to substitute versions."
                )
        else:

            def topic_feed(operator: str) -> list[Paper]:
                params = {
                    "search_query": f" {operator} ".join(f'all:"{k}"' for k in intent.keywords),
                    "max_results": limit,
                    "sortBy": "submittedDate" if intent.recent else "relevance",
                    "sortOrder": "descending",
                }
                try:
                    return self._feed(params)[:limit]
                except SourceError as exc:
                    if "HTTP 406" not in str(exc) or not all(
                        re.fullmatch(r"[A-Za-z0-9]+", term) for term in intent.keywords
                    ):
                        raise
                    # Some arXiv/CDN routes reject a fully parameterized URL
                    # while serving the equivalent documented minimal query.
                    # Single alphanumeric terms need no phrase quotes; preserve
                    # every term and Boolean operator. Defaults are relevance
                    # order and ten results, so trim locally for smaller limits.
                    canonical = {
                        "search_query": f" {operator} ".join(f"all:{k}" for k in intent.keywords)
                    }
                    if intent.recent:
                        canonical.update(sortBy="submittedDate", sortOrder="descending")
                    if limit > 10:
                        canonical["max_results"] = limit
                    papers = self._feed(canonical, literal_syntax=True)[:limit]
                    warning = (
                        "arXiv returned HTTP 406 for the parameterized topic request; "
                        "an equivalent canonical API query succeeded with the same terms and sort intent."
                    )
                    if warning not in warnings:
                        warnings.append(warning)
                    return papers

            papers = topic_feed("AND")
            if not papers and len(intent.keywords) > 1:
                papers = topic_feed("OR")
                warnings.append(
                    "No papers matched every keyword. Search broadened to any keyword; inspect the candidate ranking."
                )
        if not papers:
            raise SourceError(
                "No matching arXiv papers found. Check the ID or try a more specific/reworded topic."
            )
        return papers, warnings

    def download(self, paper: Paper, destination: Path) -> str:
        expected = f"https://arxiv.org/pdf/{paper.arxiv_id}"
        if normalize_id(paper.arxiv_id) != paper.arxiv_id or paper.pdf_url != expected:
            raise SourceError("Refusing an untrusted PDF URL.")
        if destination.exists():
            data = destination.read_bytes()
            if data.startswith(b"%PDF-") and len(data) <= self.max_bytes:
                return hashlib.sha256(data).hexdigest()
        temporary = destination.with_suffix(".part")
        for attempt in range(3):
            self._throttle()
            try:
                with self.client.stream("GET", expected) as response:
                    response.raise_for_status()
                    total = 0
                    digest = hashlib.sha256()
                    with temporary.open("wb") as stream:
                        for block in response.iter_bytes(chunk_size=65536):
                            if total == 0 and not block.startswith(b"%PDF-"):
                                raise SourceError(
                                    "The download is not a PDF. arXiv may be temporarily blocking requests."
                                )
                            total += len(block)
                            if total > self.max_bytes:
                                raise SourceError(
                                    f"PDF exceeds the {self.max_bytes // (1024 * 1024)} MB limit."
                                )
                            stream.write(block)
                            digest.update(block)
                if total < 100:
                    raise SourceError("arXiv returned an empty or incomplete PDF.")
                temporary.replace(destination)
                return digest.hexdigest()
            except (httpx.HTTPError, OSError) as exc:
                if attempt == 2:
                    raise SourceError(
                        "PDF download failed after three attempts. Resume this session to retry."
                    ) from exc
                time.sleep(2**attempt)
            finally:
                temporary.unlink(missing_ok=True)
        raise SourceError("PDF download failed.")

    def close(self):
        self.client.close()

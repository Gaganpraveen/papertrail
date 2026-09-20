"""Shared, validated data contracts. Generated metadata never overwrites source metadata."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def now() -> str:
    return datetime.now(UTC).isoformat()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paper(Record):
    arxiv_id: str = ""
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    published: str = ""
    updated: str = ""
    categories: list[str] = Field(default_factory=list)
    url: str = ""
    pdf_url: str = ""
    relevance: float = 0.0
    source: Literal["arxiv", "upload"] = "arxiv"
    document_id: str = ""
    source_filename: str = ""

    @property
    def identity(self) -> str:
        return self.document_id if self.source == "upload" else self.arxiv_id


class Intent(Record):
    kind: Literal["paper", "topic"]
    value: str
    keywords: list[str] = Field(default_factory=list)
    recent: bool = False


class Chunk(Record):
    id: str
    paper_id: str
    page: int
    section: str
    text: str
    start: int
    end: int
    reference: bool = False


class ParsedPaper(Record):
    sha256: str
    pages: int
    chunks: list[Chunk]
    warnings: list[str] = Field(default_factory=list)


class Evidence(Record):
    chunk_id: str
    quote: str = Field(min_length=15, max_length=1200)


class Claim(Record):
    text: str = Field(min_length=1, max_length=2400)
    evidence: list[Evidence] = Field(min_length=1, max_length=4)


class Briefing(Record):
    summary: Claim
    problem: Claim
    method: list[Claim] = Field(min_length=1, max_length=5)
    results: list[Claim] = Field(min_length=1, max_length=5)
    limitations: list[Claim] = Field(default_factory=list, max_length=5)
    limitations_note: str
    follow_up_questions: list[str] = Field(min_length=2, max_length=5)


class Answer(Record):
    status: Literal["answered", "insufficient_evidence"]
    claims: list[Claim] = Field(default_factory=list, max_length=5)
    explanation: str


class Hit(Record):
    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None


class Exchange(Record):
    question: str
    retrieval_query: str
    answer: Answer
    retrieved_ids: list[str]
    elapsed_seconds: float
    created_at: str = Field(default_factory=now)


class Node(StrEnum):
    UNDERSTAND = "understand"
    RETRIEVE = "retrieve"
    SELECT = "select"
    FETCH = "fetch"
    PARSE = "parse"
    INDEX = "index"
    BRIEF = "brief"
    VALIDATE = "validate"
    READY = "ready"


class Event(Record):
    node: str
    status: str
    seconds: float = 0.0
    detail: str = ""
    at: str = Field(default_factory=now)


class RunState(Record):
    schema_version: int = 1
    id: str
    query: str
    model: str
    embedding_model: str
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)
    node: Node = Node.UNDERSTAND
    status: Literal["running", "failed", "ready"] = "running"
    intent: Intent | None = None
    candidates: list[Paper] = Field(default_factory=list)
    paper: Paper | None = None
    collection: str | None = None
    pdf_sha256: str | None = None
    pages: int = 0
    chunk_count: int = 0
    briefing: Briefing | None = None
    briefing_evidence_ids: list[str] = Field(default_factory=list)
    exchanges: list[Exchange] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    error: str | None = None

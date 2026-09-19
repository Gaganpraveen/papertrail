import pytest

from papertrail.schema import Briefing, Chunk, Claim, Evidence, Paper


@pytest.fixture
def paper():
    return Paper(
        arxiv_id="1706.03762v7",
        title="Attention Is All You Need",
        authors=["Test Author"],
        abstract="A test abstract.",
        published="2017-06-12T00:00:00Z",
        updated="2023-08-02T00:00:00Z",
        categories=["cs.CL"],
        url="https://arxiv.org/abs/1706.03762v7",
        pdf_url="https://arxiv.org/pdf/1706.03762v7",
    )


@pytest.fixture
def chunk():
    return Chunk(
        id="a" * 16,
        paper_id="1706.03762v7",
        page=2,
        section="Results",
        text="The model achieves 28.4 BLEU on the translation benchmark. Training takes 3.5 days on eight GPUs.",
        start=0,
        end=99,
    )


@pytest.fixture
def claim(chunk):
    return Claim(
        text="The model achieves 28.4 BLEU.",
        evidence=[
            Evidence(
                chunk_id=chunk.id,
                quote="The model achieves 28.4 BLEU on the translation benchmark.",
            )
        ],
    )


@pytest.fixture
def briefing(claim):
    return Briefing(
        summary=claim,
        problem=claim,
        method=[claim],
        results=[claim],
        limitations=[],
        limitations_note="No explicit limitations in the selected evidence.",
        follow_up_questions=["What is the method?", "How was it evaluated?"],
    )

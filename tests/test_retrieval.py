from papertrail.retrieval import HybridIndex, bm25
from papertrail.schema import Chunk


class TinyEmbedder:
    """Deterministic test double; never used by the application or live evaluations."""

    name = "test-only"

    def documents(self, texts):
        return [[1.0, 0.0] if "attention" in t.lower() else [0.0, 1.0] for t in texts]

    def query(self, text):
        return [1.0, 0.0]


def test_bm25_recognizes_exact_term():
    scores = bm25(
        "KV cache compression",
        [
            "KV cache compression uses quantization",
            "translation through attention",
            "a different topic",
        ],
    )
    assert scores[0] > scores[1] == scores[2] == 0


def test_persistent_vectors_and_reference_filter(tmp_path):
    chunks = [
        Chunk(
            id=str(i) * 16,
            paper_id="1706.03762v7",
            page=i + 1,
            section="References" if i == 1 else "Method",
            text="attention retrieval passage" if i < 2 else "different approach",
            start=0,
            end=24,
            reference=i == 1,
        )
        for i in range(3)
    ]
    index = HybridIndex(tmp_path / "vectors", TinyEmbedder())
    collection = index.build(chunks, "abc")
    assert index.build(chunks, "abc") == collection
    index.close()
    reopened = HybridIndex(tmp_path / "vectors", TinyEmbedder())
    hits = reopened.search(collection, chunks, "attention", limit=3)
    assert hits[0].chunk.id == chunks[0].id
    assert all(not hit.chunk.reference for hit in hits)
    hits_with_refs = reopened.search(
        collection, chunks, "attention", limit=3, include_references=True
    )
    assert any(h.chunk.reference for h in hits_with_refs)
    reopened.close()


def test_collections_isolate_papers(tmp_path):
    index = HybridIndex(tmp_path / "vectors", TinyEmbedder())
    chunk = Chunk(
        id="a" * 16,
        paper_id="1706.03762",
        page=1,
        section="Method",
        text="attention method",
        start=0,
        end=16,
    )
    assert index.build([chunk], "first") != index.build([chunk], "second")
    index.close()


def test_subset_is_applied_before_vector_search(tmp_path):
    index = HybridIndex(tmp_path / "vectors", TinyEmbedder())
    chunks = [
        Chunk(
            id=str(i) * 16,
            paper_id="1706.03762",
            page=i + 1,
            section="Intro" if i else "Method",
            text="attention" if i == 0 else "unrelated background",
            start=0,
            end=20,
        )
        for i in range(3)
    ]
    collection = index.build(chunks, "subset")
    hits = index.search(collection, chunks[1:], "attention", limit=1, mode="dense")
    assert hits and hits[0].chunk.id in {c.id for c in chunks[1:]}
    index.close()

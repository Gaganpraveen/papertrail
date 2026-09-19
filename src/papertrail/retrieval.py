"""Dense search + BM25, fused by reciprocal rank, with source diversity."""

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient, models

from papertrail.errors import PaperTrailError
from papertrail.parsing import PARSER_VERSION
from papertrail.schema import Chunk, Hit, Intent, Paper


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def bm25(query: str, documents: list[str]) -> list[float]:
    """Okapi BM25 (k1=1.5, b=.75). Small per-paper corpus; no extra service."""
    if not documents:
        return []
    terms = [Counter(tokens(document)) for document in documents]
    lengths = [sum(term.values()) for term in terms]
    mean_length = max(sum(lengths) / len(lengths), 1)
    frequency = Counter(word for term in terms for word in term)
    scores = []
    for term, length in zip(terms, lengths, strict=True):
        score = 0.0
        for word in set(tokens(query)):
            df, tf = frequency[word], term[word]
            idf = math.log(1 + (len(terms) - df + 0.5) / (df + 0.5))
            score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / mean_length))
        scores.append(score)
    return scores


class Embedder:
    def __init__(self, name: str, cache: Path):
        self.name, self.cache, self._model = name, cache, None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            try:
                self._model = TextEmbedding(self.name, cache_dir=str(self.cache), threads=2)
            except Exception as exc:
                raise PaperTrailError(
                    "Could not load the embedding model. First run needs internet access to download it; run `papertrail doctor --warmup`."
                ) from exc
        return self._model

    def documents(self, texts: list[str]) -> list[list[float]]:
        return [x.tolist() for x in self.model.embed(texts, batch_size=32)]

    def query(self, text: str) -> list[float]:
        return next(self.model.query_embed(text)).tolist()


def rank_papers(intent: Intent, papers: list[Paper], embedder: Embedder) -> list[Paper]:
    if intent.kind == "paper":
        return papers
    docs = [p.title + ". " + p.abstract for p in papers]
    embeddings = np.array(embedder.documents(docs))
    query = np.array(embedder.query(intent.value))
    dense = (
        embeddings @ query / (np.linalg.norm(embeddings, axis=1) * max(np.linalg.norm(query), 1e-9))
    )
    lexical = bm25(
        " ".join(intent.keywords), [p.title + " " + p.title + " " + p.abstract for p in papers]
    )
    maximum = max(max(lexical), 1e-9)
    ranked = [
        p.model_copy(update={"relevance": round(0.7 * float(d) + 0.3 * b / maximum, 4)})
        for p, d, b in zip(papers, dense, lexical, strict=True)
    ]
    return sorted(ranked, key=lambda p: p.relevance, reverse=True)


class HybridIndex:
    def __init__(self, path: Path, embedder: Embedder):
        self.path, self.embedder, self._client = path, embedder, None

    @property
    def client(self):
        if self._client is None:
            self._client = QdrantClient(path=str(self.path))
        return self._client

    def build(self, chunks: list[Chunk], pdf_sha: str) -> str:
        signature = f"{pdf_sha}|{self.embedder.name}|{PARSER_VERSION}"
        collection = "paper_" + hashlib.sha256(signature.encode()).hexdigest()[:24]
        if self.client.collection_exists(collection) and self.client.get_collection(
            collection
        ).points_count == len(chunks):
            return collection
        vectors = self.embedder.documents([c.section + "\n" + c.text for c in chunks])
        if not vectors:
            raise PaperTrailError("No passages were extracted for indexing.")
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]), distance=models.Distance.COSINE
                ),
            )
        self.client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, chunk.id)),
                    vector=vector,
                    payload=chunk.model_dump(),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            wait=True,
        )
        return collection

    def search(
        self,
        collection: str,
        chunks: list[Chunk],
        query: str,
        limit: int = 7,
        mode: str = "hybrid",
        include_references: bool = False,
    ) -> list[Hit]:
        eligible = {c.id: c for c in chunks if include_references or not c.reference}
        if not eligible:
            return []
        dense: dict[str, float] = {}
        lexical: dict[str, float] = {}
        if mode in {"hybrid", "dense"}:
            if not self.client.collection_exists(collection):
                raise PaperTrailError(
                    "The saved vector index is missing. Start a new digest to rebuild it."
                )
            # Filter before nearest-neighbor search. Post-filtering a global top-k
            # can discard all results when briefing retrieval targets a section.
            query_filter = models.Filter(
                must=[models.FieldCondition(key="id", match=models.MatchAny(any=list(eligible)))]
            )
            found = self.client.query_points(
                collection,
                query=self.embedder.query(query),
                query_filter=query_filter,
                limit=min(len(eligible), 24),
                with_payload=True,
            ).points
            dense = {
                p.payload["id"]: float(p.score)
                for p in found
                if p.payload and p.payload["id"] in eligible
            }
        if mode in {"hybrid", "bm25"}:
            scores = bm25(query, [c.section + " " + c.text for c in eligible.values()])
            lexical = dict(
                sorted(
                    (
                        (c.id, score)
                        for c, score in zip(eligible.values(), scores, strict=True)
                        if score > 0
                    ),
                    key=lambda pair: pair[1],
                    reverse=True,
                )[:24]
            )
        fused: dict[str, float] = {}
        for ranking in (dense, lexical):
            for rank, identity in enumerate(ranking, 1):
                fused[identity] = fused.get(identity, 0) + 1 / (60 + rank)
        selected = []
        for identity, score in sorted(fused.items(), key=lambda pair: pair[1], reverse=True):
            chunk = eligible[identity]
            # Suppress windows with excessive overlap, while allowing multiple passages per page.
            if any(
                chunk.page == hit.chunk.page
                and chunk.section == hit.chunk.section
                and len(set(tokens(chunk.text)) & set(tokens(hit.chunk.text)))
                / max(len(set(tokens(chunk.text))), 1)
                > 0.82
                for hit in selected
            ):
                continue
            selected.append(
                Hit(
                    chunk=chunk,
                    score=score,
                    dense_score=dense.get(identity),
                    lexical_score=lexical.get(identity),
                )
            )
            if len(selected) >= limit:
                break
        return selected

    def close(self):
        if self._client is not None:
            self._client.close()

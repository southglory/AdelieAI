from typing import AsyncIterator

import pytest

from core.retrieval.chunker import RecursiveTextSplitter
from core.retrieval.document_store import SqlDocumentStore
from core.retrieval.ingest import DenseRetriever, IngestService
from core.schemas.retrieval import Chunk, RetrievedChunk


class FakeEmbedder:
    """Deterministic stub embedder — bag-of-chars vector for tiny dim.
    Same text → same vector. Avoids loading real model in unit tests.
    """

    model_id = "fake-bag-of-chars"
    dim = 26

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 26
        for ch in text.lower():
            if "a" <= ch <= "z":
                v[ord(ch) - ord("a")] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._vec(query)


class FakeVectorStore:
    def __init__(self) -> None:
        self.entries: dict[str, tuple[Chunk, list[float]]] = {}

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    async def upsert(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.entries[c.id] = (c, c.embedding or [])

    async def search(
        self, query_vec: list[float], k: int, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        scored = [
            (chunk, self._cosine(query_vec, vec))
            for chunk, vec in self.entries.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(chunk=chunk, score=score) for chunk, score in scored[:k]
        ]

    async def delete_by_doc(self, doc_id: str) -> None:
        self.entries = {
            cid: ce for cid, ce in self.entries.items() if ce[0].doc_id != doc_id
        }


@pytest.fixture
async def stack() -> AsyncIterator[
    tuple[IngestService, DenseRetriever, SqlDocumentStore, FakeVectorStore]
]:
    doc_store = SqlDocumentStore.from_url("sqlite+aiosqlite:///:memory:")
    await doc_store.init_schema()
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    chunker = RecursiveTextSplitter(chunk_size=120, chunk_overlap=20)

    ingest = IngestService(chunker, embedder, doc_store, vector_store)
    retriever = DenseRetriever(embedder, vector_store, doc_store)
    yield ingest, retriever, doc_store, vector_store
    await doc_store.dispose()


async def test_ingest_persists_doc_chunks_and_vectors(stack) -> None:
    ingest, retriever, doc_store, vector_store = stack
    doc, chunks = await ingest.ingest(
        title="company handbook",
        source="memory",
        content="alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota",
    )
    assert len(chunks) > 0
    assert all(c.embedding is not None for c in chunks)

    persisted = await doc_store.get(doc.id)
    assert persisted is not None
    persisted_chunks = await doc_store.list_chunks(doc.id)
    assert len(persisted_chunks) == len(chunks)
    assert len(vector_store.entries) == len(chunks)


async def test_dense_retrieve_returns_relevant_chunks(stack) -> None:
    ingest, retriever, _, _ = stack
    await ingest.ingest(
        title="d1",
        source="t",
        content="apple banana cherry\n\ndatabase engineering framework\n\nzebra yak xenon",
    )
    ctx = await retriever.retrieve("database", k=3)
    assert ctx.method == "dense"
    assert ctx.query == "database"
    assert len(ctx.results) >= 1
    top = ctx.results[0].chunk.text
    assert "database" in top.lower() or "engineering" in top.lower()


async def test_delete_removes_from_both_stores(stack) -> None:
    ingest, _, doc_store, vector_store = stack
    doc, _ = await ingest.ingest(
        title="d1", source="t", content="hello world"
    )
    assert await doc_store.get(doc.id) is not None
    assert len(vector_store.entries) > 0

    await ingest.delete(doc.id)
    assert await doc_store.get(doc.id) is None
    assert len(vector_store.entries) == 0


async def test_empty_content_skips_vectors(stack) -> None:
    ingest, _, doc_store, vector_store = stack
    doc, chunks = await ingest.ingest(title="empty", source="t", content="")
    assert chunks == []
    assert await doc_store.get(doc.id) is not None
    assert len(vector_store.entries) == 0

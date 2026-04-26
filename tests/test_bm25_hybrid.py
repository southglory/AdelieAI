import pytest

from core.retrieval.bm25 import InMemoryBM25, tokenize
from core.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from core.schemas.retrieval import Chunk, RetrievedChunk


def _chunk(cid: str, doc_id: str, text: str, position: int = 0) -> Chunk:
    return Chunk(id=cid, doc_id=doc_id, position=position, text=text)


def test_tokenize_handles_korean_and_english() -> None:
    tokens = tokenize("Hello WORLD 안녕 hello-world test_case")
    assert "hello" in tokens
    assert "world" in tokens
    assert tokens.count("hello") == 2


async def test_bm25_returns_relevant_chunks() -> None:
    bm = InMemoryBM25()
    await bm.add(
        [
            _chunk("c1", "d1", "the quick brown fox jumps over the lazy dog"),
            _chunk("c2", "d1", "FastAPI is a Python web framework"),
            _chunk("c3", "d2", "machine learning models for natural language"),
        ]
    )
    hits = await bm.search("FastAPI Python framework", k=2)
    assert hits
    assert hits[0].chunk.id == "c2"


async def test_bm25_remove_by_doc() -> None:
    bm = InMemoryBM25()
    await bm.add(
        [
            _chunk("c1", "d1", "alpha bravo"),
            _chunk("c2", "d2", "alpha charlie"),
        ]
    )
    await bm.remove_by_doc("d1")
    hits = await bm.search("alpha", k=5)
    assert all(h.chunk.doc_id == "d2" for h in hits)
    assert bm.size() == 1


def test_rrf_simple_combination() -> None:
    a = [
        RetrievedChunk(chunk=_chunk("x", "d", "x"), score=10.0),
        RetrievedChunk(chunk=_chunk("y", "d", "y"), score=9.0),
    ]
    b = [
        RetrievedChunk(chunk=_chunk("y", "d", "y"), score=8.0),
        RetrievedChunk(chunk=_chunk("z", "d", "z"), score=7.0),
    ]
    fused = reciprocal_rank_fusion([a, b], k=3)
    assert [r.chunk.id for r in fused] == ["y", "x", "z"]


def test_rrf_uses_reciprocal_rank_not_score() -> None:
    a = [
        RetrievedChunk(chunk=_chunk("x", "d", "x"), score=1000.0),
        RetrievedChunk(chunk=_chunk("y", "d", "y"), score=999.0),
    ]
    b = [
        RetrievedChunk(chunk=_chunk("y", "d", "y"), score=0.001),
    ]
    fused = reciprocal_rank_fusion([a, b], k=3, rrf_k=1)
    # y appears at rank 2 in a (1/3) and rank 1 in b (1/2) → 0.833...
    # x appears at rank 1 in a only (1/2) → 0.5
    assert fused[0].chunk.id == "y"


@pytest.mark.asyncio
async def test_hybrid_retriever_dense_plus_bm25_no_reranker(monkeypatch) -> None:
    """End-to-end on hybrid pipeline using fakes from test_retrieval_ingest."""
    from tests.test_retrieval_ingest import FakeEmbedder, FakeVectorStore
    from core.retrieval.chunker import RecursiveTextSplitter
    from core.retrieval.document_store import SqlDocumentStore
    from core.retrieval.ingest import IngestService

    doc_store = SqlDocumentStore.from_url("sqlite+aiosqlite:///:memory:")
    await doc_store.init_schema()
    vector = FakeVectorStore()
    bm25 = InMemoryBM25()
    embedder = FakeEmbedder()
    chunker = RecursiveTextSplitter(chunk_size=200, chunk_overlap=20)
    ingest = IngestService(chunker, embedder, doc_store, vector, bm25=bm25)
    await ingest.ingest(
        title="t",
        source="s",
        content=(
            "fastapi pydantic uvicorn web framework\n\n"
            "natural language processing transformers\n\n"
            "database relational tables postgres"
        ),
    )

    hybrid = HybridRetriever(
        embedder=embedder,
        vector_store=vector,
        bm25=bm25,
        doc_store=doc_store,
        reranker=None,
        candidate_pool=10,
    )
    ctx = await hybrid.retrieve("fastapi framework", k=2)
    assert ctx.method == "hybrid"
    assert len(ctx.results) >= 1
    top_text = ctx.results[0].chunk.text.lower()
    assert "framework" in top_text or "fastapi" in top_text
    await doc_store.dispose()


async def test_warm_bm25_from_doc_store_after_restart() -> None:
    """Ingest into one stack, then build a fresh BM25 with the same
    doc_store and verify the warm step rebuilds the index from disk.
    """
    from tests.test_retrieval_ingest import FakeEmbedder, FakeVectorStore
    from core.retrieval.chunker import RecursiveTextSplitter
    from core.retrieval.document_store import SqlDocumentStore
    from core.retrieval.ingest import IngestService

    db_url = "sqlite+aiosqlite:///:memory:"
    doc_store = SqlDocumentStore.from_url(db_url)
    await doc_store.init_schema()
    bm25_first = InMemoryBM25()
    ingest_first = IngestService(
        RecursiveTextSplitter(chunk_size=200),
        FakeEmbedder(),
        doc_store,
        FakeVectorStore(),
        bm25=bm25_first,
    )
    await ingest_first.ingest(
        title="t", source="s", content="alpha\n\nbeta\n\ngamma"
    )
    assert bm25_first.size() >= 1

    # New BM25 starts empty until warm
    bm25_second = InMemoryBM25()
    ingest_second = IngestService(
        RecursiveTextSplitter(chunk_size=200),
        FakeEmbedder(),
        doc_store,
        FakeVectorStore(),
        bm25=bm25_second,
    )
    assert bm25_second.size() == 0
    warmed = await ingest_second.warm_bm25_from_doc_store()
    assert warmed >= 1
    assert bm25_second.size() == warmed
    await doc_store.dispose()

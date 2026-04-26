import pytest
from fastapi.testclient import TestClient

from core.agent.rag import format_context, retrieval_event_payload
from core.agent.runner import run_session, stream_session
from core.api.app import build_app
from core.retrieval.chunker import RecursiveTextSplitter
from core.retrieval.document_store import SqlDocumentStore
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.bm25 import InMemoryBM25
from core.retrieval.ingest import IngestService
from core.schemas.agent import EventType, SessionState
from core.schemas.retrieval import (
    Chunk,
    RetrievedChunk,
    RetrievedContext,
)
from core.serving.protocols import GenerationParams
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore
from tests.test_retrieval_ingest import FakeEmbedder, FakeVectorStore


def _chunk(text: str, doc_title: str = "src") -> Chunk:
    return Chunk(
        id=f"c-{text[:6]}",
        doc_id="d1",
        position=0,
        text=text,
        metadata={"doc_title": doc_title},
    )


def test_format_context_passthrough_when_empty() -> None:
    ctx = RetrievedContext(method="dense", results=[], query="q")
    out = format_context("ask", ctx)
    assert out == "ask"


def test_format_context_includes_numbered_citations() -> None:
    ctx = RetrievedContext(
        method="dense",
        query="q",
        results=[
            RetrievedChunk(chunk=_chunk("first chunk text"), score=0.9),
            RetrievedChunk(chunk=_chunk("second chunk text"), score=0.8),
        ],
    )
    out = format_context("Question?", ctx)
    assert "[1]" in out
    assert "[2]" in out
    assert "first chunk text" in out
    assert "Question?" in out


def test_retrieval_event_payload_shape() -> None:
    ctx = RetrievedContext(
        method="hybrid",
        query="q",
        results=[
            RetrievedChunk(chunk=_chunk("hello"), score=0.5),
        ],
    )
    payload = retrieval_event_payload(ctx)
    assert payload["method"] == "hybrid"
    assert payload["query"] == "q"
    assert len(payload["results"]) == 1
    r = payload["results"][0]
    assert {"score", "chunk_id", "doc_id", "preview", "doc_title"} <= set(r.keys())


@pytest.fixture
async def stack():
    doc_store = SqlDocumentStore.from_url("sqlite+aiosqlite:///:memory:")
    await doc_store.init_schema()
    vector = FakeVectorStore()
    bm25 = InMemoryBM25()
    embedder = FakeEmbedder()
    chunker = RecursiveTextSplitter(chunk_size=200, chunk_overlap=20)
    ingest = IngestService(chunker, embedder, doc_store, vector, bm25=bm25)
    await ingest.ingest(
        title="db handbook",
        source="t",
        content="alpha beta\n\nThe FastAPI framework for Python web apps\n\ngamma delta",
    )
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector,
        bm25=bm25,
        doc_store=doc_store,
        candidate_pool=10,
    )
    yield ingest, retriever
    await doc_store.dispose()


async def test_run_session_with_retrieval_records_event(stack) -> None:
    _, retriever = stack
    store = InMemorySessionStore()
    llm = StubLLMClient()
    session = await store.create("alice", "FastAPI framework", "stub")

    result = await run_session(
        store, llm, session.id, "alice",
        params=GenerationParams(retrieval_k=3),
        retriever=retriever,
    )
    assert result.state == SessionState.COMPLETED

    events = await store.events(session.id, "alice")
    types = [e.event_type for e in events]
    assert EventType.RETRIEVAL in types
    retrieval_event = next(e for e in events if e.event_type == EventType.RETRIEVAL)
    assert "results" in retrieval_event.payload


async def test_run_session_skips_retrieval_when_k_zero(stack) -> None:
    _, retriever = stack
    store = InMemorySessionStore()
    llm = StubLLMClient()
    session = await store.create("alice", "FastAPI", "stub")

    await run_session(
        store, llm, session.id, "alice",
        params=GenerationParams(retrieval_k=0),
        retriever=retriever,
    )
    events = await store.events(session.id, "alice")
    assert all(e.event_type != EventType.RETRIEVAL for e in events)


async def test_stream_session_with_retrieval(stack) -> None:
    _, retriever = stack
    store = InMemorySessionStore()
    llm = StubLLMClient()
    session = await store.create("alice", "FastAPI", "stub")

    chunks = []
    async for ev in stream_session(
        store, llm, session.id, "alice",
        params=GenerationParams(retrieval_k=2),
        retriever=retriever,
    ):
        if ev.type == "chunk":
            chunks.append(ev.text or "")
    assert chunks
    events = await store.events(session.id, "alice")
    assert any(e.event_type == EventType.RETRIEVAL for e in events)


async def test_run_session_no_retriever_means_plain_run(stack) -> None:
    """If retriever is None, retrieval_k>0 silently skips retrieval —
    runner doesn't fail.
    """
    store = InMemorySessionStore()
    llm = StubLLMClient()
    session = await store.create("alice", "g", "stub")
    result = await run_session(
        store, llm, session.id, "alice",
        params=GenerationParams(retrieval_k=5),
        retriever=None,
    )
    assert result.state == SessionState.COMPLETED
    events = await store.events(session.id, "alice")
    assert all(e.event_type != EventType.RETRIEVAL for e in events)


def test_web_run_form_accepts_retrieval_k(stack) -> None:
    ingest, retriever = stack
    app = build_app(
        store=InMemorySessionStore(),
        llm=StubLLMClient(),
        ingest=ingest,
        retriever=retriever,
    )
    client = TestClient(app)
    create = client.post(
        "/web/sessions", data={"goal": "FastAPI", "model_spec": "stub"}
    )
    sid = create.text.split('id="session-')[1].split('"')[0]
    r = client.post(
        f"/web/sessions/{sid}/run", data={"retrieval_k": "2"}
    )
    assert r.status_code == 200
    assert 'class="pill completed"' in r.text

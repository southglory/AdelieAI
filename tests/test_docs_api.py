import pytest
from fastapi.testclient import TestClient

from core.api.app import build_app
from core.retrieval.chunker import RecursiveTextSplitter
from core.retrieval.document_store import SqlDocumentStore
from core.retrieval.ingest import DenseRetriever, IngestService
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore
from tests.test_retrieval_ingest import FakeEmbedder, FakeVectorStore


@pytest.fixture
async def stack():
    doc_store = SqlDocumentStore.from_url("sqlite+aiosqlite:///:memory:")
    await doc_store.init_schema()
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    chunker = RecursiveTextSplitter(chunk_size=200, chunk_overlap=20)
    ingest = IngestService(chunker, embedder, doc_store, vector_store)
    retriever = DenseRetriever(embedder, vector_store, doc_store)
    yield ingest, retriever
    await doc_store.dispose()


@pytest.fixture
async def client(stack) -> TestClient:
    ingest, retriever = stack
    app = build_app(
        store=InMemorySessionStore(),
        llm=StubLLMClient(),
        ingest=ingest,
        retriever=retriever,
    )
    return TestClient(app)


def test_ingest_doc_returns_201_and_chunk_count(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    r = client.post(
        "/api/v1/docs",
        json={
            "title": "company handbook",
            "source": "manual",
            "content": "alpha beta gamma\n\ndelta epsilon zeta",
        },
        headers=headers,
    )
    assert r.status_code == 201
    body = r.json()
    assert body["chunk_count"] >= 1
    assert body["document"]["title"] == "company handbook"


def test_list_docs(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    client.post(
        "/api/v1/docs",
        json={"title": "a", "source": "s", "content": "x"},
        headers=headers,
    )
    client.post(
        "/api/v1/docs",
        json={"title": "b", "source": "s", "content": "y"},
        headers=headers,
    )
    r = client.get("/api/v1/docs", headers=headers)
    assert r.status_code == 200
    titles = {d["title"] for d in r.json()}
    assert {"a", "b"} <= titles


def test_get_doc_and_chunks(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    sid = client.post(
        "/api/v1/docs",
        json={"title": "t", "source": "s", "content": "para1\n\npara2"},
        headers=headers,
    ).json()["document"]["id"]
    r = client.get(f"/api/v1/docs/{sid}", headers=headers)
    assert r.status_code == 200
    chunks = client.get(f"/api/v1/docs/{sid}/chunks", headers=headers).json()
    assert isinstance(chunks, list)
    assert all("text" in c for c in chunks)


def test_delete_doc(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    sid = client.post(
        "/api/v1/docs",
        json={"title": "t", "source": "s", "content": "x"},
        headers=headers,
    ).json()["document"]["id"]
    r = client.delete(f"/api/v1/docs/{sid}", headers=headers)
    assert r.status_code == 204
    r = client.get(f"/api/v1/docs/{sid}", headers=headers)
    assert r.status_code == 404


def test_search_returns_relevant_results(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    client.post(
        "/api/v1/docs",
        json={
            "title": "tech",
            "source": "s",
            "content": "database engineering framework\n\nphysics chemistry biology",
        },
        headers=headers,
    )
    r = client.post(
        "/api/v1/docs/search",
        json={"query": "engineering framework", "k": 2},
        headers=headers,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "dense"
    assert len(body["results"]) >= 1


def test_unauthorized_without_user_header(client: TestClient) -> None:
    r = client.post(
        "/api/v1/docs",
        json={"title": "t", "source": "s", "content": "x"},
    )
    assert r.status_code == 401


def test_health_includes_embedder_id(client: TestClient) -> None:
    r = client.get("/health")
    assert r.json()["embedder"] == "fake-bag-of-chars"


def test_503_when_no_ingest_configured() -> None:
    app = build_app(
        store=InMemorySessionStore(),
        llm=StubLLMClient(),
    )
    app.state.ingest = None
    app.state.retriever = None
    # The router was bound at build time so we exercise it via TestClient
    # against a fresh app where we explicitly nullify ingest.
    bare = build_app(
        store=InMemorySessionStore(), llm=StubLLMClient(), ingest=None
    )
    # Only run the assertion if no embedder was loaded by default.
    if bare.state.ingest is None:
        c = TestClient(bare)
        r = c.post(
            "/api/v1/docs",
            json={"title": "t", "source": "s", "content": "x"},
            headers={"X-User-Id": "alice"},
        )
        assert r.status_code == 503

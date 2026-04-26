from typing import Callable

import pytest
from fastapi.testclient import TestClient

from core.agent.agentic_runner import run_agentic_session
from core.agent.graph import build_agent_graph
from core.agent.nodes import parse_plan
from core.agent.state import AgentState, Plan
from core.api.app import build_app
from core.retrieval.bm25 import InMemoryBM25
from core.retrieval.chunker import RecursiveTextSplitter
from core.retrieval.document_store import SqlDocumentStore
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.ingest import IngestService
from core.schemas.agent import EventType, SessionState
from core.serving.protocols import GenerationParams, GenerationResult
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore
from tests.test_retrieval_ingest import FakeEmbedder, FakeVectorStore


# --- plan parsing ---


def test_parse_plan_valid_json() -> None:
    raw = '{"skip_retrieval": false, "search_queries": ["hello"], "rationale": "needs lookup"}'
    plan = parse_plan(raw, goal="x")
    assert plan.skip_retrieval is False
    assert plan.search_queries == ["hello"]
    assert plan.rationale == "needs lookup"


def test_parse_plan_with_surrounding_text() -> None:
    raw = (
        "Here is my plan:\n"
        '{"skip_retrieval": true, "search_queries": [], "rationale": "no docs"}\n'
        "End."
    )
    plan = parse_plan(raw, goal="x")
    assert plan.skip_retrieval is True


def test_parse_plan_falls_back_on_garbage() -> None:
    plan = parse_plan("not json at all", goal="my goal")
    assert plan.search_queries == ["my goal"]
    assert "fallback" in plan.rationale.lower() or "parseable" in plan.rationale.lower()


def test_parse_plan_drops_non_string_queries() -> None:
    raw = '{"skip_retrieval": false, "search_queries": ["ok", 123, null, "also ok"]}'
    plan = parse_plan(raw, goal="x")
    assert plan.search_queries == ["ok", "also ok"]


# --- scripted LLM that returns a JSON plan, then a normal answer ---


class ScriptedLLM:
    """Returns plan JSON for the first call (planner) and a fixed
    string for subsequent calls (reasoner). Lets us drive the
    LangGraph deterministically without running real generation.
    """

    model_id = "scripted-test-1"

    def __init__(self, plan_json: str, answer: str = "OK answer") -> None:
        self.plan_json = plan_json
        self.answer = answer
        self.calls: list[str] = []

    async def generate(self, prompt: str, params: GenerationParams | None = None):
        params = params or GenerationParams()
        self.calls.append(prompt)
        is_planner = params.system and "planner" in params.system.lower()
        text = self.plan_json if is_planner else self.answer
        return GenerationResult(
            text=text,
            tokens_in=len(prompt) // 4,
            tokens_out=len(text) // 4,
            latency_ms=1,
            model_id=self.model_id,
            params=params,
        )

    async def astream(self, prompt: str, params: GenerationParams | None = None):
        result = await self.generate(prompt, params)
        from core.serving.protocols import StreamEvent

        for ch in result.text:
            yield StreamEvent(type="chunk", text=ch)
        yield StreamEvent(
            type="done",
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            latency_ms=result.latency_ms,
        )


# --- end-to-end agentic flow ---


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
        title="d1", source="t",
        content="alpha\n\nbeta gamma\n\ndelta epsilon zeta",
    )
    retriever = HybridRetriever(
        embedder=embedder, vector_store=vector, bm25=bm25, doc_store=doc_store,
        candidate_pool=10,
    )
    yield ingest, retriever
    await doc_store.dispose()


async def test_agentic_runs_all_four_nodes_with_retrieval(stack) -> None:
    _, retriever = stack
    store = InMemorySessionStore()
    llm = ScriptedLLM(
        plan_json='{"skip_retrieval": false, "search_queries": ["alpha"], "rationale": "needs"}',
        answer="answer with [1] citation",
    )
    session = await store.create("alice", "what about alpha?", "scripted")

    result = await run_agentic_session(
        store, llm, session.id, "alice", retriever=retriever, retrieval_k=2,
    )
    assert result.state == SessionState.COMPLETED

    events = await store.events(session.id, "alice")
    nodes_seen = [e.payload.get("node") for e in events if "node" in (e.payload or {})]
    assert "plan" in nodes_seen
    assert "retrieve" in nodes_seen
    assert "reason" in nodes_seen
    assert "report" in nodes_seen

    final_event = next(e for e in events if e.event_type == EventType.FINAL)
    assert "answer" in final_event.payload
    assert final_event.payload["mode"] == "agentic"
    assert final_event.payload["plan"] is not None


async def test_agentic_skip_retrieval_when_planner_says(stack) -> None:
    _, retriever = stack
    store = InMemorySessionStore()
    llm = ScriptedLLM(
        plan_json='{"skip_retrieval": true, "search_queries": [], "rationale": "general chitchat"}',
        answer="hi",
    )
    session = await store.create("alice", "hi", "scripted")

    await run_agentic_session(
        store, llm, session.id, "alice", retriever=retriever,
    )
    events = await store.events(session.id, "alice")
    retrieve_events = [
        e for e in events
        if (e.payload or {}).get("node") == "retrieve_skip"
        or (e.payload or {}).get("node") == "retrieve"
    ]
    assert any(
        (e.payload or {}).get("node") == "retrieve_skip" for e in retrieve_events
    )


async def test_agentic_no_retriever_skips_retrieval_silently() -> None:
    store = InMemorySessionStore()
    llm = ScriptedLLM(
        plan_json='{"skip_retrieval": false, "search_queries": ["x"]}',
        answer="ok",
    )
    session = await store.create("alice", "g", "scripted")
    result = await run_agentic_session(
        store, llm, session.id, "alice", retriever=None,
    )
    assert result.state == SessionState.COMPLETED


async def test_agentic_session_rejects_non_pending(stack) -> None:
    _, retriever = stack
    store = InMemorySessionStore()
    llm = ScriptedLLM(plan_json="{}")
    session = await store.create("alice", "g", "scripted")
    await store.transition(session.id, "alice", SessionState.RUNNING)

    from core.agent.runner import SessionNotRunnable

    with pytest.raises(SessionNotRunnable):
        await run_agentic_session(store, llm, session.id, "alice")


def test_agentic_endpoint(stack) -> None:
    ingest, retriever = stack
    llm = ScriptedLLM(
        plan_json='{"skip_retrieval": false, "search_queries": ["alpha"]}',
        answer="agentic answer",
    )
    app = build_app(
        store=InMemorySessionStore(), llm=llm, ingest=ingest, retriever=retriever,
    )
    client = TestClient(app)
    headers = {"X-User-Id": "alice"}
    sid = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "tell me about alpha", "model_spec": "scripted"},
        headers=headers,
    ).json()["id"]
    r = client.post(
        f"/api/v1/agents/sessions/{sid}/run/agentic?retrieval_k=2", headers=headers
    )
    assert r.status_code == 200
    assert r.json()["state"] == "completed"

    events = client.get(
        f"/api/v1/agents/sessions/{sid}/events", headers=headers
    ).json()
    nodes = [e["payload"].get("node") for e in events if "node" in e.get("payload", {})]
    assert {"plan", "retrieve", "reason", "report"} <= set(nodes)


def test_web_agentic_form(stack) -> None:
    ingest, retriever = stack
    llm = ScriptedLLM(
        plan_json='{"skip_retrieval": false, "search_queries": ["alpha"]}',
        answer="ok",
    )
    app = build_app(
        store=InMemorySessionStore(), llm=llm, ingest=ingest, retriever=retriever,
    )
    client = TestClient(app)
    create = client.post("/web/sessions", data={"goal": "g", "model_spec": "x"})
    sid = create.text.split('id="session-')[1].split('"')[0]
    r = client.post(
        f"/web/sessions/{sid}/run/agentic", data={"retrieval_k": "2"}
    )
    assert r.status_code == 200
    assert 'class="pill completed"' in r.text

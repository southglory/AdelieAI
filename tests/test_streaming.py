import pytest
from fastapi.testclient import TestClient

from core.agent.runner import SessionNotRunnable, stream_session
from core.api.app import build_app
from core.schemas.agent import EventType, SessionState
from core.serving.protocols import GenerationParams, StreamEvent
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore, SessionNotFound


async def test_stub_astream_yields_chunks_then_done() -> None:
    stub = StubLLMClient()
    events = []
    async for ev in stub.astream("hi", params=GenerationParams(max_new_tokens=10)):
        events.append(ev)

    chunks = [e for e in events if e.type == "chunk"]
    dones = [e for e in events if e.type == "done"]
    assert len(chunks) > 0
    assert len(dones) == 1
    assert dones[0].tokens_out is not None
    assert all(isinstance(e, StreamEvent) for e in events)


async def test_stream_session_full_lifecycle() -> None:
    store = InMemorySessionStore()
    llm = StubLLMClient()
    session = await store.create("alice", "summarize", "stub")

    events = []
    async for ev in stream_session(store, llm, session.id, "alice"):
        events.append(ev)

    chunks = [e for e in events if e.type == "chunk"]
    done = [e for e in events if e.type == "done"]
    assert len(chunks) > 0
    assert len(done) == 1

    final_session = await store.get(session.id, "alice")
    assert final_session.state == SessionState.COMPLETED

    persisted = await store.events(session.id, "alice")
    types = [e.event_type for e in persisted]
    assert EventType.LLM_CALL in types
    assert EventType.FINAL in types
    final_event = next(e for e in persisted if e.event_type == EventType.FINAL)
    assert final_event.payload["answer"]
    llm_event = next(e for e in persisted if e.event_type == EventType.LLM_CALL)
    assert llm_event.payload.get("streamed") is True


async def test_stream_session_rejects_non_pending() -> None:
    store = InMemorySessionStore()
    llm = StubLLMClient()
    session = await store.create("alice", "g", "m")
    await store.transition(session.id, "alice", SessionState.RUNNING)

    with pytest.raises(SessionNotRunnable):
        async for _ in stream_session(store, llm, session.id, "alice"):
            pass


async def test_stream_session_unknown_id_raises() -> None:
    store = InMemorySessionStore()
    llm = StubLLMClient()
    with pytest.raises(SessionNotFound):
        async for _ in stream_session(store, llm, "nope", "alice"):
            pass


@pytest.fixture
def client() -> TestClient:
    app = build_app(store=InMemorySessionStore(), llm=StubLLMClient())
    return TestClient(app)


def test_stream_endpoint_json(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    r = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "hi", "model_spec": "x"},
        headers=headers,
    )
    sid = r.json()["id"]

    with client.stream(
        "POST", f"/api/v1/agents/sessions/{sid}/run/stream", headers=headers
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        body = "".join(r.iter_text())
    assert "data:" in body
    assert '"type":"chunk"' in body
    assert '"type":"done"' in body


def test_stream_endpoint_web_form(client: TestClient) -> None:
    create = client.post(
        "/web/sessions", data={"goal": "g", "model_spec": "m"}
    )
    sid = create.text.split('id="session-')[1].split('"')[0]
    with client.stream(
        "POST",
        f"/web/sessions/{sid}/run/stream",
        data={"max_new_tokens": "10", "temperature": "0.0"},
    ) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert '"type":"chunk"' in body


def test_stream_endpoint_409_for_already_running(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    r = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "g", "model_spec": "m"},
        headers=headers,
    )
    sid = r.json()["id"]
    client.post(
        f"/api/v1/agents/sessions/{sid}/transition",
        json={"to": "running"},
        headers=headers,
    )

    with client.stream(
        "POST", f"/api/v1/agents/sessions/{sid}/run/stream", headers=headers
    ) as r:
        body = "".join(r.iter_text())
    assert "error" in body
    assert "expected pending" in body

import pytest
from fastapi.testclient import TestClient

from core.agent.runner import SessionNotRunnable, run_session
from core.api.app import build_app
from core.schemas.agent import EventType, SessionState
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore, SessionNotFound


async def test_run_session_pending_to_completed() -> None:
    store = InMemorySessionStore()
    llm = StubLLMClient()
    session = await store.create("alice", "summarize", "stub-deterministic-1")

    result = await run_session(store, llm, session.id, "alice")
    assert result.state == SessionState.COMPLETED

    events = await store.events(session.id, "alice")
    types = [e.event_type for e in events]
    assert EventType.STATE_TRANSITION in types  # → RUNNING
    assert EventType.LLM_CALL in types
    assert EventType.FINAL in types
    assert types.count(EventType.STATE_TRANSITION) == 2  # → RUNNING, → COMPLETED

    final = next(e for e in events if e.event_type == EventType.FINAL)
    assert "answer" in final.payload
    assert final.payload["model_id"] == "stub-deterministic-1"


async def test_run_session_rejects_non_pending() -> None:
    store = InMemorySessionStore()
    llm = StubLLMClient()
    session = await store.create("alice", "g", "m")
    await store.transition(session.id, "alice", SessionState.RUNNING)

    with pytest.raises(SessionNotRunnable):
        await run_session(store, llm, session.id, "alice")


async def test_run_session_unknown_id_raises() -> None:
    store = InMemorySessionStore()
    llm = StubLLMClient()
    with pytest.raises(SessionNotFound):
        await run_session(store, llm, "nonexistent", "alice")


@pytest.fixture
def client() -> TestClient:
    app = build_app(store=InMemorySessionStore(), llm=StubLLMClient())
    return TestClient(app)


def test_run_endpoint_json(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    r = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "demo", "model_spec": "stub"},
        headers=headers,
    )
    sid = r.json()["id"]

    r = client.post(f"/api/v1/agents/sessions/{sid}/run", headers=headers)
    assert r.status_code == 200
    assert r.json()["state"] == "completed"

    r = client.get(f"/api/v1/agents/sessions/{sid}/events", headers=headers)
    events = r.json()
    final = next(e for e in events if e["event_type"] == "final")
    assert "answer" in final["payload"]


def test_run_endpoint_web_returns_row(client: TestClient) -> None:
    create = client.post(
        "/web/sessions", data={"goal": "demo", "model_spec": "stub"}
    )
    sid = create.text.split('id="session-')[1].split('"')[0]

    r = client.post(f"/web/sessions/{sid}/run")
    assert r.status_code == 200
    assert 'class="pill completed"' in r.text


def test_run_endpoint_409_if_already_running(client: TestClient) -> None:
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

    r = client.post(f"/api/v1/agents/sessions/{sid}/run", headers=headers)
    assert r.status_code == 409


def test_health_includes_llm_id(client: TestClient) -> None:
    r = client.get("/health")
    assert r.json()["llm"] == "stub-deterministic-1"


def test_run_with_custom_params_propagate(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    r = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "g", "model_spec": "m"},
        headers=headers,
    )
    sid = r.json()["id"]
    r = client.post(
        f"/api/v1/agents/sessions/{sid}/run",
        json={
            "max_new_tokens": 128,
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 20,
            "system": "you are concise",
        },
        headers=headers,
    )
    assert r.status_code == 200

    events = client.get(
        f"/api/v1/agents/sessions/{sid}/events", headers=headers
    ).json()
    final = next(e for e in events if e["event_type"] == "final")
    p = final["payload"]["params"]
    assert p["max_new_tokens"] == 128
    assert p["temperature"] == 0.3
    assert p["system"] == "you are concise"


def test_stub_echoes_params() -> None:
    from core.serving.protocols import GenerationParams
    from core.serving.stub_client import StubLLMClient
    import asyncio

    params = GenerationParams(max_new_tokens=42, temperature=0.5, system="sys")
    result = asyncio.run(StubLLMClient().generate("hi", params=params))
    assert "max_new=42" in result.text
    assert "temp=0.5" in result.text
    assert "system=3 chars" in result.text
    assert result.params.max_new_tokens == 42


def test_web_run_with_form_params(client: TestClient) -> None:
    create = client.post(
        "/web/sessions", data={"goal": "g", "model_spec": "m"}
    )
    sid = create.text.split('id="session-')[1].split('"')[0]
    r = client.post(
        f"/web/sessions/{sid}/run",
        data={
            "max_new_tokens": "64",
            "temperature": "0.0",
            "system": "be brief",
        },
    )
    assert r.status_code == 200
    assert 'class="pill completed"' in r.text

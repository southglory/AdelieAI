import pytest
from fastapi.testclient import TestClient

from core.api.app import build_app
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore


@pytest.fixture
def client() -> TestClient:
    app = build_app(store=InMemorySessionStore(), llm=StubLLMClient())
    return TestClient(app)


def test_root_redirects_to_sessions(client: TestClient) -> None:
    r = client.get("/web/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert "/web/sessions" in r.headers["location"]


def test_sessions_page_renders(client: TestClient) -> None:
    r = client.get("/web/sessions")
    assert r.status_code == 200
    assert "Agent Sessions" in r.text
    assert "demo" in r.text


def test_nav_shows_active_llm_id(client: TestClient) -> None:
    r = client.get("/web/sessions")
    assert "stub-deterministic-1" in r.text


def test_create_form_default_is_active_llm(client: TestClient) -> None:
    r = client.get("/web/sessions")
    assert 'name="model_spec" value="stub-deterministic-1"' in r.text


def test_create_via_form_returns_row_partial(client: TestClient) -> None:
    r = client.post(
        "/web/sessions",
        data={"goal": "summarize Q3", "model_spec": "claude-opus-4-7"},
    )
    assert r.status_code == 200
    assert "summarize Q3" in r.text
    assert 'class="pill pending"' in r.text
    assert "<html" not in r.text  # row partial only


def test_transition_via_form(client: TestClient) -> None:
    create = client.post(
        "/web/sessions",
        data={"goal": "g", "model_spec": "m"},
    )
    sid = create.text.split('id="session-')[1].split('"')[0]

    r = client.post(
        f"/web/sessions/{sid}/transition",
        data={"to": "running"},
    )
    assert r.status_code == 200
    assert 'class="pill running"' in r.text


def test_invalid_transition_409(client: TestClient) -> None:
    create = client.post(
        "/web/sessions", data={"goal": "g", "model_spec": "m"}
    )
    sid = create.text.split('id="session-')[1].split('"')[0]
    r = client.post(
        f"/web/sessions/{sid}/transition", data={"to": "completed"}
    )
    assert r.status_code == 409


def test_whoami_sets_cookie(client: TestClient) -> None:
    r = client.post(
        "/web/whoami", data={"user_id": "alice"}, follow_redirects=False
    )
    assert r.status_code == 303
    assert "user_id=alice" in r.headers["set-cookie"]


def test_user_isolation_via_cookie(client: TestClient) -> None:
    client.post("/web/whoami", data={"user_id": "alice"}, follow_redirects=False)
    r = client.post(
        "/web/sessions", data={"goal": "alice goal", "model_spec": "m"}
    )
    assert "alice goal" in r.text

    client.post("/web/whoami", data={"user_id": "bob"}, follow_redirects=False)
    r = client.get("/web/sessions")
    assert "alice goal" not in r.text


def test_events_partial_after_transition(client: TestClient) -> None:
    create = client.post(
        "/web/sessions", data={"goal": "g", "model_spec": "m"}
    )
    sid = create.text.split('id="session-')[1].split('"')[0]
    client.post(f"/web/sessions/{sid}/transition", data={"to": "running"})

    r = client.get(f"/web/sessions/{sid}/events")
    assert r.status_code == 200
    assert "pending → running" in r.text

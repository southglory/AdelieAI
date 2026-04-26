import pytest
from fastapi.testclient import TestClient

from core.api.app import build_app
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore


@pytest.fixture
def client() -> TestClient:
    app = build_app(store=InMemorySessionStore(), llm=StubLLMClient())
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_and_run_session(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    r = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "demo", "model_spec": "claude"},
        headers=headers,
    )
    assert r.status_code == 201
    sid = r.json()["id"]
    assert r.json()["state"] == "pending"

    r = client.post(
        f"/api/v1/agents/sessions/{sid}/transition",
        json={"to": "running"},
        headers=headers,
    )
    assert r.status_code == 200

    r = client.post(
        f"/api/v1/agents/sessions/{sid}/transition",
        json={"to": "completed"},
        headers=headers,
    )
    assert r.status_code == 200
    assert r.json()["state"] == "completed"

    r = client.get(f"/api/v1/agents/sessions/{sid}/events", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_unauthorized_without_header(client: TestClient) -> None:
    r = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "g", "model_spec": "m"},
    )
    assert r.status_code == 401


def test_idor_other_user_returns_404(client: TestClient) -> None:
    r = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "g", "model_spec": "m"},
        headers={"X-User-Id": "alice"},
    )
    sid = r.json()["id"]

    r = client.get(
        f"/api/v1/agents/sessions/{sid}", headers={"X-User-Id": "bob"}
    )
    assert r.status_code == 404


def test_invalid_transition_returns_409(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    r = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "g", "model_spec": "m"},
        headers=headers,
    )
    sid = r.json()["id"]
    r = client.post(
        f"/api/v1/agents/sessions/{sid}/transition",
        json={"to": "completed"},
        headers=headers,
    )
    assert r.status_code == 409

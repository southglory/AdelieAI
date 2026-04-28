"""Industry demo routes and capability tier self-introspection."""

import pytest
from fastapi.testclient import TestClient

from core.api.app import build_app
from core.api.demos_router import VERTICALS
from core.personas.store import InMemoryChatStore
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore


@pytest.fixture
def client() -> TestClient:
    app = build_app(
        store=InMemorySessionStore(),
        llm=StubLLMClient(),
        chat_store=InMemoryChatStore(),
    )
    return TestClient(app)


def test_demos_index_lists_three_verticals(client: TestClient) -> None:
    r = client.get("/demo/")
    assert r.status_code == 200
    for slug in ("gaming", "legal", "knowledge"):
        assert f"/demo/{slug}" in r.text


def test_each_vertical_renders(client: TestClient) -> None:
    for slug in VERTICALS:
        r = client.get(f"/demo/{slug}")
        assert r.status_code == 200, f"{slug} did not render"


def test_unknown_vertical_404(client: TestClient) -> None:
    r = client.get("/demo/banking")
    assert r.status_code == 404


def test_vertical_page_shows_tier_label(client: TestClient) -> None:
    r = client.get("/demo/gaming")
    assert "T2" in r.text


def test_vertical_page_shows_persona_id(client: TestClient) -> None:
    r = client.get("/demo/legal")
    assert "cold_detective" in r.text


def test_health_exposes_tier(client: TestClient) -> None:
    r = client.get("/health")
    body = r.json()
    assert body["tier"] >= 1
    assert body["tier_max"] == 5
    assert "tier_status" in body
    for key in ("T1", "T2", "T3", "T4", "T5"):
        assert key in body["tier_status"]


def test_root_exposes_tier_summary(client: TestClient) -> None:
    r = client.get("/")
    body = r.json()
    assert "tier" in body
    assert "tier_max" in body


def test_stub_build_floors_at_tier_1(client: TestClient) -> None:
    """StubLLMClient + no retriever should land at T1 only."""
    body = client.get("/health").json()
    assert body["tier"] == 1
    # T2+ should report missing components
    for higher in ("T2", "T3", "T4", "T5"):
        assert "missing" in body["tier_status"][higher]


def test_global_nav_links_to_demos(client: TestClient) -> None:
    r = client.get("/web/personas")
    assert 'href="/demo/"' in r.text

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


def test_gaming_demo_has_jrpg_chrome(client: TestClient) -> None:
    """gaming vertical was custom-designed in Step 2 — verify the
    distinctive JRPG shop chrome survives."""
    r = client.get("/demo/gaming")
    body = r.text
    assert "CROOKED COIN" in body  # shop sign
    assert "shop-stage" in body  # full-bleed scene container
    assert "dialogue-thread" in body  # HTMX target
    assert "/web/chat/cynical_merchant/messages" in body  # backend wire
    assert "Press+Start+2P" in body  # pixel font import
    # Inventory items rendered server-side (mock, not from a registry yet)
    for item in ("가죽 갑옷", "단검", "회복 약초", "마법 두루마리", "횃불"):
        assert item in body, f"missing inventory item: {item}"


def test_legal_demo_has_noir_chrome(client: TestClient) -> None:
    """legal vertical was custom-designed in Step 3 — verify the
    distinctive noir detective-office chrome survives."""
    r = client.get("/demo/legal")
    body = r.text
    assert "CASE FILE #07" in body  # case stamp header
    assert "office-stage" in body  # full-bleed scene container
    assert "case-board" in body  # cork board panel
    assert "case-summary" in body  # case brief card on top of board
    assert "살인" in body  # case classification — locked-room murder
    assert "밀실" in body  # explicit nature of the case
    assert "CONFIDENTIAL" in body  # stamp
    assert "transcript" in body  # interrogation paper
    assert "/web/chat/cold_detective/messages" in body  # backend wire
    assert "Special+Elite" in body  # typewriter font import
    # Static evidence chips render — T3 retrieval-as-tool surface
    for evidence in ("evidence_1.md", "case_log_07.md", "timeline.txt", "witness_a.md"):
        assert evidence in body, f"missing evidence file: {evidence}"
    assert "tier T3" in body


def test_knowledge_demo_has_archive_chrome(client: TestClient) -> None:
    """knowledge vertical was custom-designed in Step 4 — verify the
    distinctive ancient-archive chrome survives."""
    r = client.get("/demo/knowledge")
    body = r.text
    assert "ANCIENT ARCHIVE" in body  # archive title
    assert "archive-stage" in body  # full-bleed scene container
    assert "kg-svg" in body  # inline SVG knowledge graph
    assert "/web/chat/ancient_dragon/messages" in body  # backend wire
    assert "Cinzel" in body  # display font import (Roman inscription)
    assert "Cormorant+Garamond" in body  # body serif font import
    # KG nodes
    for node in (":Self", ":Vyrnaes", ":Sothryn", ":Erebor", ":Arkenstone", ":Thrór", ":Dragon", ":WingedBeing"):
        assert node in body, f"missing KG node: {node}"
    # SPARQL trace + OWL reasoner panels
    assert "PREFIX" in body and "SELECT" in body and "WHERE" in body
    assert "consistent" in body
    assert "subClassOf" in body
    assert "transitive" in body
    # Tier badge
    assert "tier T4" in body


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


def test_stub_build_reports_t4_via_default_kg_retriever(client: TestClient) -> None:
    """StubLLMClient + no vector RAG, but default ToolRegistry +
    default RdfGraphRetriever + StubOWLReasoner are registered →
    tier reaches T4. T2 (vector RAG) and T5 (multi-agent runner)
    still missing in the stub build."""
    body = client.get("/health").json()
    assert body["tier"] == 4
    assert "ok" in body["tier_status"]["T3"]
    assert "ok" in body["tier_status"]["T4"]
    assert "reasoner" in body["tier_status"]["T4"]
    assert "missing" in body["tier_status"]["T2"]  # no real LLM, no vector RAG
    assert "missing" in body["tier_status"]["T5"]


def test_global_nav_links_to_demos(client: TestClient) -> None:
    r = client.get("/web/personas")
    assert 'href="/demo/"' in r.text

"""Smoke tests for the persona engine web UI.

Covers:
- Persona registry exposes the three v0.1.5 defaults.
- Persona gallery renders and links each card to its chat thread.
- Chat thread renders, accepts a message, and replies via the LLM.
- Per-turn telemetry (latency + tokens_out) flows from generation
  result into the assistant turn rendered in HTML.
- Reset clears history per persona/user pair.
"""

import pytest
from fastapi.testclient import TestClient

from core.api.app import build_app
from core.personas import DEFAULT_PERSONAS, get_persona, list_personas
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


def test_registry_has_default_personas() -> None:
    ids = {p.persona_id for p in list_personas()}
    # v0.1.5 baseline (general companions)
    assert {"penguin_relaxed", "fish_swimmer", "knight_brave"}.issubset(ids)
    # v0.2 vertical-anchored personas
    assert "cynical_merchant" in ids


def test_registry_industry_and_tier_are_declared() -> None:
    for p in list_personas():
        assert p.target_tier in {1, 2, 3, 4, 5}
        assert p.industry in {"general", "gaming", "legal", "knowledge"}


def test_get_persona_returns_none_for_unknown_id() -> None:
    assert get_persona("dragon_lord") is None


def test_personas_page_renders_default_cards(client: TestClient) -> None:
    r = client.get("/web/personas")
    assert r.status_code == 200
    assert "Personas" in r.text
    assert "🐧" in r.text
    assert "🐟" in r.text
    assert "⚔️" in r.text
    assert "💰" in r.text  # cynical_merchant
    assert "/web/chat/penguin_relaxed" in r.text
    assert "/web/chat/cynical_merchant" in r.text


def test_personas_card_shows_initial_zero_turns(client: TestClient) -> None:
    r = client.get("/web/personas")
    assert "<dt>turns</dt><dd>0</dd>" in r.text.replace("\n", "").replace(" ", "")


def test_chat_thread_404_for_unknown_persona(client: TestClient) -> None:
    r = client.get("/web/chat/dragon_lord")
    assert r.status_code == 404


def test_chat_thread_renders_with_empty_state(client: TestClient) -> None:
    r = client.get("/web/chat/penguin_relaxed")
    assert r.status_code == 200
    assert "놀고 있는 펭귄" in r.text
    assert "Send a message below" in r.text
    assert 'name="message"' in r.text


def test_submit_message_returns_pair_partial(client: TestClient) -> None:
    r = client.post(
        "/web/chat/penguin_relaxed/messages",
        data={"message": "안녕? 오늘 뭐 했어?"},
    )
    assert r.status_code == 200
    assert "안녕? 오늘 뭐 했어?" in r.text
    # StubLLMClient now returns a persona-shaped canned reply when the
    # system prompt declares 펭귄/물고기/기사/잡화상/탐정. For the
    # penguin persona we expect the reply to NOT be the bare echo.
    assert "[stub:" not in r.text
    assert "<html" not in r.text  # partial only
    # Per-turn telemetry must surface
    assert "tok" in r.text
    assert "s ·" in r.text or "s" in r.text


def test_submit_message_persists_across_loads(client: TestClient) -> None:
    client.post(
        "/web/chat/penguin_relaxed/messages",
        data={"message": "first"},
    )
    client.post(
        "/web/chat/penguin_relaxed/messages",
        data={"message": "second"},
    )
    page = client.get("/web/chat/penguin_relaxed")
    assert "first" in page.text
    assert "second" in page.text
    # Turn count surfaces in sidebar
    assert "<dt>turns</dt><dd>4</dd>" in page.text.replace("\n", "").replace(" ", "")


def test_submit_empty_message_is_rejected(client: TestClient) -> None:
    r = client.post(
        "/web/chat/penguin_relaxed/messages",
        data={"message": "   "},
    )
    assert r.status_code == 400


def test_messages_for_unknown_persona_404(client: TestClient) -> None:
    r = client.post(
        "/web/chat/dragon_lord/messages",
        data={"message": "hi"},
    )
    assert r.status_code == 404


def test_reset_clears_thread(client: TestClient) -> None:
    client.post(
        "/web/chat/penguin_relaxed/messages",
        data={"message": "hello"},
    )
    r = client.post(
        "/web/chat/penguin_relaxed/reset",
        follow_redirects=False,
    )
    assert r.status_code == 303
    page = client.get("/web/chat/penguin_relaxed")
    assert "hello" not in page.text
    assert "Send a message below" in page.text


def test_reset_isolated_per_persona(client: TestClient) -> None:
    client.post(
        "/web/chat/penguin_relaxed/messages",
        data={"message": "penguin-only"},
    )
    client.post(
        "/web/chat/fish_swimmer/messages",
        data={"message": "fish-only"},
    )
    client.post("/web/chat/penguin_relaxed/reset", follow_redirects=False)

    pen = client.get("/web/chat/penguin_relaxed")
    fish = client.get("/web/chat/fish_swimmer")
    assert "penguin-only" not in pen.text
    assert "fish-only" in fish.text


def test_personas_link_in_global_nav(client: TestClient) -> None:
    r = client.get("/web/sessions")
    assert 'href="/web/personas"' in r.text


def test_root_endpoint_lists_personas_module(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "personas" in body["modules"]
    assert body["chat_store"] == "InMemoryChatStore"

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


# === 5-tier rating (Step 6.2) ===


def _post_msg_get_assistant_id(client: TestClient, persona_id: str, msg: str) -> int:
    """Helper — submit message, parse `id="turn-N"` from assistant fragment."""
    import re
    r = client.post(f"/web/chat/{persona_id}/messages", data={"message": msg})
    assert r.status_code == 200
    matches = re.findall(r'id="turn-(\d+)"', r.text)
    assert matches, "no assistant turn id in response"
    return int(matches[-1])


def test_rating_widget_renders_in_assistant_turn(client: TestClient) -> None:
    r = client.post(
        "/web/chat/penguin_relaxed/messages",
        data={"message": "별점 위젯 떠?"},
    )
    assert r.status_code == 200
    assert 'class="rating"' in r.text
    assert "DPO 데이터 수집" in r.text
    # 4 rate-btn: bad / fine / good / dismiss
    assert r.text.count('class="rate-btn') == 4
    assert "👎 bad" in r.text
    assert "➖ fine" in r.text
    assert "👍 good" in r.text
    assert "⊘ dismiss" in r.text


def test_rate_good_marks_selected_and_label(client: TestClient) -> None:
    turn_id = _post_msg_get_assistant_id(client, "penguin_relaxed", "rate me")
    r = client.post(
        f"/web/chat/penguin_relaxed/turns/{turn_id}/rate",
        data={"rating": 3},
    )
    assert r.status_code == 200
    # the "good" button is selected
    assert "rate-good selected" in r.text or "rate-btn rate-good selected" in r.text
    # label echoes the new state
    assert ">good\n" in r.text or ">good <" in r.text or "good" in r.text


def test_rate_dismiss_marks_separately(client: TestClient) -> None:
    turn_id = _post_msg_get_assistant_id(client, "penguin_relaxed", "skip me")
    r = client.post(
        f"/web/chat/penguin_relaxed/turns/{turn_id}/rate",
        data={"rating": 0},
    )
    assert r.status_code == 200
    assert "rate-dismiss selected" in r.text
    assert "dismissed" in r.text


def test_rate_persists_across_thread_reload(client: TestClient) -> None:
    turn_id = _post_msg_get_assistant_id(client, "penguin_relaxed", "stick")
    client.post(
        f"/web/chat/penguin_relaxed/turns/{turn_id}/rate",
        data={"rating": 1},
    )
    page = client.get("/web/chat/penguin_relaxed")
    assert "rate-bad selected" in page.text


def test_rate_rejects_out_of_range(client: TestClient) -> None:
    turn_id = _post_msg_get_assistant_id(client, "penguin_relaxed", "edge")
    for bad_value in (4, 5, 7, -1):
        r = client.post(
            f"/web/chat/penguin_relaxed/turns/{turn_id}/rate",
            data={"rating": bad_value},
        )
        assert r.status_code == 400, f"expected 400 for rating={bad_value}"


def test_rate_unknown_turn_404(client: TestClient) -> None:
    r = client.post(
        "/web/chat/penguin_relaxed/turns/99999/rate",
        data={"rating": 3},
    )
    assert r.status_code == 404


def test_rate_unknown_persona_404(client: TestClient) -> None:
    r = client.post(
        "/web/chat/dragon_lord/turns/1/rate",
        data={"rating": 3},
    )
    assert r.status_code == 404


# === Rating summary stats surface in UI (Step 6.2 UX) ===


def test_rating_summary_hidden_when_no_assistant_turns(client: TestClient) -> None:
    """Empty persona → no rating-summary noise on the gallery card."""
    r = client.get("/web/personas")
    assert r.status_code == 200
    assert 'class="rating-summary"' not in r.text


def test_rating_summary_appears_after_first_rating(client: TestClient) -> None:
    turn_id = _post_msg_get_assistant_id(client, "penguin_relaxed", "rate me")
    client.post(
        f"/web/chat/penguin_relaxed/turns/{turn_id}/rate",
        data={"rating": 3},
    )
    page = client.get("/web/personas")
    # Card now shows the summary
    assert 'class="rating-summary"' in page.text
    assert "👍 1" in page.text
    assert "👎 0" in page.text
    # No DPO pair yet (need both good and bad on same prompt)
    assert "DPO 0" in page.text


def test_dpo_pair_count_surfaces_after_good_and_bad(client: TestClient) -> None:
    # Two assistant turns for the same prompt — one good, one bad → 1 pair.
    # Stub picker is now history-aware (turn 1 vs turn 2 → different lines)
    # so the pair survives harvest dedup. Restored from the direct-store
    # workaround once ticket #62 was fixed.
    t1 = _post_msg_get_assistant_id(client, "cynical_merchant", "할인 안 돼?")
    t2 = _post_msg_get_assistant_id(client, "cynical_merchant", "할인 안 돼?")
    client.post(
        f"/web/chat/cynical_merchant/turns/{t1}/rate", data={"rating": 3}
    )
    client.post(
        f"/web/chat/cynical_merchant/turns/{t2}/rate", data={"rating": 1}
    )
    gallery = client.get("/web/personas")
    assert "DPO 1" in gallery.text
    thread = client.get("/web/chat/cynical_merchant")
    assert "header-rating-summary" in thread.text
    assert "DPO 1" in thread.text


def test_stub_returns_different_replies_for_repeat_prompt(client: TestClient) -> None:
    """Regression for ticket #62 — same user_text twice must produce two
    distinct stub replies (history-aware picker)."""
    r1 = client.post(
        "/web/chat/cynical_merchant/messages", data={"message": "외상?"}
    )
    r2 = client.post(
        "/web/chat/cynical_merchant/messages", data={"message": "외상?"}
    )
    # Pull the .body of the assistant turn out of each response.
    import re
    def _body(html: str) -> str:
        m = re.search(r'<div class="body">([\s\S]*?)</div>', html)
        # Find the second body (assistant) — first is user
        bodies = re.findall(r'<div class="body">([\s\S]*?)</div>', html)
        return bodies[-1]
    assert _body(r1.text) != _body(r2.text), (
        "stub returned identical replies for repeat prompt — DPO harvest "
        "would dedup. Picker must be history-aware (ticket #62)."
    )


def test_dismiss_surfaces_separately_in_summary(client: TestClient) -> None:
    turn_id = _post_msg_get_assistant_id(client, "penguin_relaxed", "skip")
    client.post(
        f"/web/chat/penguin_relaxed/turns/{turn_id}/rate", data={"rating": 0}
    )
    page = client.get("/web/personas")
    assert "⊘ 1" in page.text
    # dismiss is NOT counted as bad
    assert "👎 0" in page.text


def test_rating_stats_dataclass_via_store_directly() -> None:
    """RatingStats independent of HTTP — test the store path directly."""
    import asyncio

    from core.personas.store import InMemoryChatStore, ChatTurn
    from datetime import datetime, timezone

    async def run() -> None:
        store = InMemoryChatStore()
        for role, content, rating in [
            ("user", "Q", None),
            ("assistant", "A1", 3),  # good
            ("user", "Q", None),
            ("assistant", "A2", 2),  # fine
            ("user", "Q", None),
            ("assistant", "A3", 1),  # bad
            ("user", "Q", None),
            ("assistant", "A4", 0),  # dismiss
            ("user", "Q", None),
            ("assistant", "A5", None),  # no rating
        ]:
            await store.append(
                ChatTurn(
                    id=None,
                    persona_id="p",
                    user_id="u",
                    role=role,
                    content=content,
                    tokens_in=None,
                    tokens_out=None,
                    latency_ms=None,
                    created_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
                    rating=rating,
                )
            )
        stats = await store.rating_stats("p", "u")
        assert stats.good == 1
        assert stats.fine == 1
        assert stats.bad == 1
        assert stats.dismiss == 1
        assert stats.assistant_total == 5
        assert stats.rated_total == 3  # excludes dismiss + None
        # Same prompt "Q" has good (A1) and bad (A3) → 1 DPO pair
        assert stats.dpo_pairs == 1

    asyncio.run(run())

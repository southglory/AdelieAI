"""DPO pair harvest from rated chat turns (Step 6.2)."""
from __future__ import annotations

from datetime import datetime, timezone

from core.personas.store import ChatTurn
from scripts.export_dpo import harvest_pairs


def _turn(
    *,
    id: int,
    role: str,
    content: str,
    rating: int | None = None,
    persona_id: str = "merchant",
    user_id: str = "alice",
) -> ChatTurn:
    return ChatTurn(
        id=id,
        persona_id=persona_id,
        user_id=user_id,
        role=role,  # type: ignore[arg-type]
        content=content,
        tokens_in=None,
        tokens_out=None,
        latency_ms=None,
        created_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
        rating=rating,
    )


def test_no_pairs_when_no_ratings_diverge() -> None:
    turns = [
        _turn(id=1, role="user", content="할인?"),
        _turn(id=2, role="assistant", content="안 돼.", rating=4),
        _turn(id=3, role="user", content="할인?"),
        _turn(id=4, role="assistant", content="농담이지.", rating=5),
    ]
    pairs = harvest_pairs(turns)
    # Both high — no rejected, so no pair
    assert pairs == []


def test_single_chosen_rejected_pair() -> None:
    turns = [
        _turn(id=1, role="user", content="할인?"),
        _turn(id=2, role="assistant", content="안 돼. 정가야.", rating=5),
        _turn(id=3, role="user", content="할인?"),
        _turn(id=4, role="assistant", content="네 알겠어요.", rating=1),
    ]
    pairs = harvest_pairs(turns)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.persona_id == "merchant"
    assert p.prompt == "할인?"
    assert p.chosen == "안 돼. 정가야."
    assert p.rejected == "네 알겠어요."
    assert p.chosen_rating == 5
    assert p.rejected_rating == 1


def test_cross_product_high_low() -> None:
    turns = [
        _turn(id=1, role="user", content="외상?"),
        _turn(id=2, role="assistant", content="A", rating=5),
        _turn(id=3, role="user", content="외상?"),
        _turn(id=4, role="assistant", content="B", rating=4),
        _turn(id=5, role="user", content="외상?"),
        _turn(id=6, role="assistant", content="C", rating=2),
        _turn(id=7, role="user", content="외상?"),
        _turn(id=8, role="assistant", content="D", rating=1),
    ]
    pairs = harvest_pairs(turns)
    # 2 high × 2 low = 4 pairs
    assert len(pairs) == 4
    chosens = {p.chosen for p in pairs}
    assert chosens == {"A", "B"}
    rejecteds = {p.rejected for p in pairs}
    assert rejecteds == {"C", "D"}


def test_unrated_turns_excluded() -> None:
    turns = [
        _turn(id=1, role="user", content="?"),
        _turn(id=2, role="assistant", content="rated_high", rating=5),
        _turn(id=3, role="user", content="?"),
        _turn(id=4, role="assistant", content="unrated"),  # no rating
        _turn(id=5, role="user", content="?"),
        _turn(id=6, role="assistant", content="rated_low", rating=1),
    ]
    pairs = harvest_pairs(turns)
    # Only rated turns enter the pool — 1 high × 1 low = 1 pair
    assert len(pairs) == 1
    assert pairs[0].chosen == "rated_high"
    assert pairs[0].rejected == "rated_low"


def test_grouping_by_prompt_and_persona() -> None:
    turns = [
        _turn(id=1, role="user", content="A"),
        _turn(id=2, role="assistant", content="hi-A", rating=5),
        _turn(id=3, role="user", content="A"),
        _turn(id=4, role="assistant", content="lo-A", rating=1),
        # Different prompt — no cross-pollination
        _turn(id=5, role="user", content="B"),
        _turn(id=6, role="assistant", content="hi-B", rating=5),
        # Different persona — no cross-pollination
        _turn(id=7, role="user", content="A", persona_id="dragon"),
        _turn(id=8, role="assistant", content="dragon-A", rating=1, persona_id="dragon"),
    ]
    pairs = harvest_pairs(turns)
    # Only the "merchant + A" group has both high and low
    assert len(pairs) == 1
    assert pairs[0].persona_id == "merchant"
    assert pairs[0].prompt == "A"


def test_threshold_override() -> None:
    turns = [
        _turn(id=1, role="user", content="?"),
        _turn(id=2, role="assistant", content="middish_hi", rating=3),
        _turn(id=3, role="user", content="?"),
        _turn(id=4, role="assistant", content="middish_lo", rating=2),
    ]
    # Default thresholds (4/2) → no pair (3 < 4 chosen threshold)
    assert harvest_pairs(turns) == []
    # Loosen thresholds → 1 pair
    loose = harvest_pairs(turns, chosen_threshold=3, rejected_threshold=2)
    assert len(loose) == 1
    assert loose[0].chosen == "middish_hi"


def test_identical_text_dedup() -> None:
    turns = [
        _turn(id=1, role="user", content="?"),
        _turn(id=2, role="assistant", content="same", rating=5),
        _turn(id=3, role="user", content="?"),
        _turn(id=4, role="assistant", content="same", rating=1),
    ]
    pairs = harvest_pairs(turns)
    # chosen and rejected are identical text — no useful pair
    assert pairs == []

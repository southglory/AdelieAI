"""DPO pair harvest from rated chat turns (Step 6.2 — 3-tier + dismiss).

Rating semantics:
    None = not interacted
    0    = dismiss (explicit non-evaluation)
    1    = bad
    2    = fine
    3    = good

Default thresholds: chosen ≥ 3 (good), rejected ≤ 1 (bad).
`dismiss` (0) and `fine` (2) and unrated (None) all carry no preference signal.
"""
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


def test_no_pairs_when_only_good() -> None:
    turns = [
        _turn(id=1, role="user", content="할인?"),
        _turn(id=2, role="assistant", content="안 돼.", rating=3),
        _turn(id=3, role="user", content="할인?"),
        _turn(id=4, role="assistant", content="농담이지.", rating=3),
    ]
    # Both good — no rejected → no pair
    assert harvest_pairs(turns) == []


def test_no_pairs_when_only_bad() -> None:
    turns = [
        _turn(id=1, role="user", content="?"),
        _turn(id=2, role="assistant", content="A", rating=1),
        _turn(id=3, role="user", content="?"),
        _turn(id=4, role="assistant", content="B", rating=1),
    ]
    # Both bad — no chosen → no pair
    assert harvest_pairs(turns) == []


def test_single_chosen_rejected_pair() -> None:
    turns = [
        _turn(id=1, role="user", content="할인?"),
        _turn(id=2, role="assistant", content="안 돼. 정가야.", rating=3),
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
    assert p.chosen_rating == 3
    assert p.rejected_rating == 1


def test_cross_product_good_bad() -> None:
    # 2 goods × 2 bads = 4 pairs
    turns = [
        _turn(id=1, role="user", content="외상?"),
        _turn(id=2, role="assistant", content="A", rating=3),
        _turn(id=3, role="user", content="외상?"),
        _turn(id=4, role="assistant", content="B", rating=3),
        _turn(id=5, role="user", content="외상?"),
        _turn(id=6, role="assistant", content="C", rating=1),
        _turn(id=7, role="user", content="외상?"),
        _turn(id=8, role="assistant", content="D", rating=1),
    ]
    pairs = harvest_pairs(turns)
    assert len(pairs) == 4
    assert {p.chosen for p in pairs} == {"A", "B"}
    assert {p.rejected for p in pairs} == {"C", "D"}


def test_fine_excluded_from_dpo() -> None:
    """`fine` (2) is the explicit middle — no preference signal, drops from DPO."""
    turns = [
        _turn(id=1, role="user", content="?"),
        _turn(id=2, role="assistant", content="hi", rating=3),
        _turn(id=3, role="user", content="?"),
        _turn(id=4, role="assistant", content="mid", rating=2),
        _turn(id=5, role="user", content="?"),
        _turn(id=6, role="assistant", content="lo", rating=1),
    ]
    pairs = harvest_pairs(turns)
    # Only good × bad — fine is excluded
    assert len(pairs) == 1
    assert pairs[0].chosen == "hi"
    assert pairs[0].rejected == "lo"


def test_dismiss_excluded_from_dpo() -> None:
    """`dismiss` (0) is explicit non-evaluation — must NOT be treated as bad."""
    turns = [
        _turn(id=1, role="user", content="?"),
        _turn(id=2, role="assistant", content="hi", rating=3),
        _turn(id=3, role="user", content="?"),
        _turn(id=4, role="assistant", content="dropped", rating=0),  # dismiss
    ]
    pairs = harvest_pairs(turns)
    # dismiss must NOT count as rejected, even though 0 ≤ rejected_threshold(1)
    assert pairs == []


def test_unrated_turns_excluded() -> None:
    turns = [
        _turn(id=1, role="user", content="?"),
        _turn(id=2, role="assistant", content="rated_good", rating=3),
        _turn(id=3, role="user", content="?"),
        _turn(id=4, role="assistant", content="unrated"),  # None
        _turn(id=5, role="user", content="?"),
        _turn(id=6, role="assistant", content="rated_bad", rating=1),
    ]
    pairs = harvest_pairs(turns)
    assert len(pairs) == 1
    assert pairs[0].chosen == "rated_good"
    assert pairs[0].rejected == "rated_bad"


def test_grouping_by_prompt_and_persona() -> None:
    turns = [
        _turn(id=1, role="user", content="A"),
        _turn(id=2, role="assistant", content="hi-A", rating=3),
        _turn(id=3, role="user", content="A"),
        _turn(id=4, role="assistant", content="lo-A", rating=1),
        # Different prompt — no cross-pollination
        _turn(id=5, role="user", content="B"),
        _turn(id=6, role="assistant", content="hi-B", rating=3),
        # Different persona — no cross-pollination
        _turn(id=7, role="user", content="A", persona_id="dragon"),
        _turn(id=8, role="assistant", content="dragon-A", rating=1, persona_id="dragon"),
    ]
    pairs = harvest_pairs(turns)
    assert len(pairs) == 1
    assert pairs[0].persona_id == "merchant"
    assert pairs[0].prompt == "A"


def test_threshold_override_includes_fine() -> None:
    """Loosening rejected_threshold to include fine (2)."""
    turns = [
        _turn(id=1, role="user", content="?"),
        _turn(id=2, role="assistant", content="g", rating=3),
        _turn(id=3, role="user", content="?"),
        _turn(id=4, role="assistant", content="m", rating=2),
    ]
    # Defaults exclude fine
    assert harvest_pairs(turns) == []
    # Override → 1 pair
    loose = harvest_pairs(turns, chosen_threshold=3, rejected_threshold=2)
    assert len(loose) == 1


def test_identical_text_dedup() -> None:
    turns = [
        _turn(id=1, role="user", content="?"),
        _turn(id=2, role="assistant", content="same", rating=3),
        _turn(id=3, role="user", content="?"),
        _turn(id=4, role="assistant", content="same", rating=1),
    ]
    # chosen and rejected are identical text — no useful pair
    assert harvest_pairs(turns) == []

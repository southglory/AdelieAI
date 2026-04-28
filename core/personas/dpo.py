"""DPO pair harvesting from rated chat turns (Step 6.2).

Rating semantics (see `core/personas/store.py`):
    None = not interacted
    0    = dismiss (explicit non-evaluation)
    1    = bad
    2    = fine
    3    = good

Default thresholds: chosen ≥ 3 (good), rejected ≤ 1 (bad).
`dismiss` (0) and `fine` (2) and unrated (None) all carry no preference signal.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from core.personas.store import ChatTurn


@dataclass(frozen=True)
class DPOPair:
    persona_id: str
    prompt: str
    chosen: str
    rejected: str
    chosen_rating: int
    rejected_rating: int


def harvest_pairs(
    turns: list[ChatTurn],
    *,
    chosen_threshold: int = 3,
    rejected_threshold: int = 1,
) -> list[DPOPair]:
    """Walk a chronologically ordered list of turns, group assistant
    replies by their preceding user turn (same persona+user), and emit
    chosen/rejected pairs from the rating divergence.

    Defaults match 3-tier semantics: chosen ≥ 3 (good), rejected ≤ 1 (bad).
    """
    by_prompt: dict[tuple[str, str, str], list[ChatTurn]] = defaultdict(list)
    pending_user: dict[tuple[str, str], str | None] = {}

    for t in turns:
        key = (t.persona_id, t.user_id)
        if t.role == "user":
            pending_user[key] = t.content
        elif t.role == "assistant":
            user_text = pending_user.get(key)
            if user_text is None or t.rating is None or t.rating == 0:
                # None=not interacted, 0=dismiss — both lack preference signal
                continue
            by_prompt[(t.persona_id, t.user_id, user_text)].append(t)

    pairs: list[DPOPair] = []
    for (persona_id, _, prompt), replies in by_prompt.items():
        highs = [
            r for r in replies
            if r.rating is not None and r.rating >= chosen_threshold
        ]
        lows = [
            r for r in replies
            if r.rating is not None and r.rating > 0 and r.rating <= rejected_threshold
        ]
        for h in highs:
            for low in lows:
                if h.content == low.content:
                    continue
                pairs.append(
                    DPOPair(
                        persona_id=persona_id,
                        prompt=prompt,
                        chosen=h.content,
                        rejected=low.content,
                        chosen_rating=h.rating,  # type: ignore[arg-type]
                        rejected_rating=low.rating,  # type: ignore[arg-type]
                    )
                )
    return pairs

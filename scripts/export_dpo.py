"""Harvest DPO training pairs from rated chat turns (Step 6.2).

Usage:
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
        scripts/export_dpo.py \
        --db sqlite+aiosqlite:///data/adelie.db \
        --persona cynical_merchant \
        --out data/dpo/cynical_merchant.jsonl

Logic:
    For each (persona_id, prompt) where the same user_text was answered
    multiple times with at least one *high* rating (>= chosen_threshold)
    AND at least one *low* rating (<= rejected_threshold), emit one
    `{prompt, chosen, rejected}` pair per (high × low) cross-product.
    Higher rating → chosen, lower → rejected.

    Default thresholds: chosen >= 3 (good), rejected <= 1 (bad). Rating
    semantics: 0=dismiss · 1=bad · 2=fine · 3=good · None=not interacted.
    `dismiss` and `fine` are both excluded from DPO (no preference signal).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from core.personas.store import ChatTurn, SqlChatStore


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

    Default thresholds match 3-tier semantics: chosen ≥ 3 (good),
    rejected ≤ 1 (bad). Rating 0 (dismiss) and 2 (fine) carry no
    preference signal — both are *explicitly* excluded.

    Notes
    -----
    - We key by (persona_id, user_id, user_text). Two assistants for the
      *same prompt across separate sessions* count if they share user_id.
    - Unrated (None) turns are skipped — same as dismiss.
    - Identical chosen and rejected text is skipped (e.g., user double-rated).
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
        highs = [r for r in replies if r.rating is not None and r.rating >= chosen_threshold]
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


async def _fetch_all_turns(store: SqlChatStore) -> list[ChatTurn]:
    """Lightweight scan — read every turn ordered by id ascending. We
    don't filter at the SQL layer because we want to keep the harvest
    in one place (and the chat-turn table is small enough)."""
    from sqlalchemy import select
    from core.personas.store import _ChatTurnRow

    async with store._sessionmaker() as session:  # noqa: SLF001 — internal access
        stmt = select(_ChatTurnRow).order_by(_ChatTurnRow.id.asc())
        rows = (await session.execute(stmt)).scalars().all()
        return [r.to_dataclass() for r in rows]


async def main_async(args: argparse.Namespace) -> int:
    db_url = args.db or os.environ.get(
        "ADELIE_DB_URL", "sqlite+aiosqlite:///data/adelie.db"
    )
    store = SqlChatStore.from_url(db_url)
    try:
        await store.init_schema()
        turns = await _fetch_all_turns(store)
    finally:
        await store.dispose()

    if args.persona:
        turns = [t for t in turns if t.persona_id == args.persona]

    pairs = harvest_pairs(
        turns,
        chosen_threshold=args.chosen_threshold,
        rejected_threshold=args.rejected_threshold,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {
                        "persona_id": p.persona_id,
                        "prompt": p.prompt,
                        "chosen": p.chosen,
                        "rejected": p.rejected,
                        "chosen_rating": p.chosen_rating,
                        "rejected_rating": p.rejected_rating,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"=== DPO export ===")
    print(f"  db                : {db_url}")
    print(f"  persona_filter    : {args.persona or '(all)'}")
    print(f"  thresholds        : chosen ≥ {args.chosen_threshold}, "
          f"rejected ≤ {args.rejected_threshold}")
    print(f"  rated turns       : {sum(1 for t in turns if t.rating is not None)}")
    print(f"  pairs emitted     : {len(pairs)}")
    print(f"  output            : {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harvest DPO chosen/rejected pairs from rated chat turns."
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLAlchemy URL. Defaults to $ADELIE_DB_URL or sqlite+aiosqlite:///data/adelie.db",
    )
    parser.add_argument("--persona", default=None, help="Filter by persona_id.")
    parser.add_argument(
        "--out",
        default="data/dpo/pairs.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--chosen-threshold", type=int, default=3)
    parser.add_argument("--rejected-threshold", type=int, default=1)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

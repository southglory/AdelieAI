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
from pathlib import Path

# harvest logic lives in core/personas/dpo.py so the API layer
# (RatingStats / personas gallery) can reuse it without scripts/ imports.
from core.personas.dpo import DPOPair, harvest_pairs  # noqa: F401 — re-export
from core.personas.store import ChatTurn, SqlChatStore


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

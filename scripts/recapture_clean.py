"""Clean re-capture against the *real* trained model — for README freshness.

Replaces the legacy `capture_screenshots.py` and `capture_step6_screenshots.py`
runs whose outputs accumulated stale chat state (test1 / cp949 artifacts /
pending sessions piled up across runs).

This script:
  1. Targets a *fresh* set of SQLite databases (`shot_chats.db`, `shot_sessions.db`)
     — the running server must point at them via env vars.
  2. Resets each persona's chat history before seeding (POST /reset).
  3. Drives a *meaningful* conversation per persona — not "test1" — so the
     screenshot shows the actual trained voice.
  4. Captures 4 frames in one Playwright run:
        02_chat_thread.png         (merchant chat, 3 turns, real model voice,
                                    rating widget visible under each turn)
        03_sessions.png            (clean empty-state sessions page)
        31_personas_with_dpo.png   (gallery after rating divergence → DPO N)
        32_metrics_dashboard.png   (per-persona activity rollup)
     We dropped 01_personas (covered by 31) and 30_rating_widget (was a
     duplicate of 02) — see commit history for the README cleanup.

Usage (against a server already running with the real LoRA mounted):
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 \\
        scripts/recapture_clean.py \\
        --url http://127.0.0.1:8765 \\
        --out docs/screenshots
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import requests  # type: ignore[import-untyped]
from playwright.sync_api import Page, sync_playwright


# Real-model takes ~3-5s per turn on RTX 3090, so explicit waits prevent
# capturing mid-stream HTML.
GENERATE_TIMEOUT = 90  # seconds for a single generate to settle


def reset_persona(s: requests.Session, base: str, persona_id: str) -> None:
    s.post(
        f"{base}/web/chat/{persona_id}/reset",
        allow_redirects=False,
    )


def submit(s: requests.Session, base: str, persona_id: str, msg: str) -> int:
    r = s.post(
        f"{base}/web/chat/{persona_id}/messages",
        data={"message": msg},
        timeout=GENERATE_TIMEOUT,
    )
    r.raise_for_status()
    ids = [int(x) for x in re.findall(r'id="turn-(\d+)"', r.text)]
    if not ids:
        raise RuntimeError(f"no assistant turn id in response for {persona_id}")
    return ids[-1]


def rate(s: requests.Session, base: str, persona_id: str, turn_id: int, rating: int) -> None:
    s.post(
        f"{base}/web/chat/{persona_id}/turns/{turn_id}/rate",
        data={"rating": rating},
        timeout=10,
    )


def shot(page: Page, out: Path, name: str) -> None:
    page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    print(f"  saved {name}.png")


def seed_phase_1_baseline(base: str) -> None:
    """Reset everything → empty state for 01 / 03."""
    s = requests.Session()
    for pid in [
        "penguin_relaxed",
        "fish_swimmer",
        "knight_brave",
        "cynical_merchant",
        "cold_detective",
        "ancient_dragon",
    ]:
        reset_persona(s, base, pid)


def seed_phase_2_conversations(base: str) -> tuple[int, int, int]:
    """Drive real-model conversations + ratings.

    Returns merchant assistant turn ids (3 of them) — the chat thread
    screenshot scrolls past these. Strategy:
      - merchant: 3 messages of *the same prompt* — gets 3 distinct
        replies (real model sampling) → rate 2 good + 1 bad → 2 DPO pairs
      - detective: 2 messages, single good rating
      - dragon: 1 message, single good rating
      - penguin: 2 different prompts, 1 good 1 dismiss
    """
    s = requests.Session()

    # Merchant — three turns on the same prompt to harvest DPO pairs
    print("  seeding merchant (3 turns, same prompt)...")
    m_ids: list[int] = []
    for _ in range(3):
        m_ids.append(submit(s, base, "cynical_merchant", "할인 좀 안 돼?"))
    rate(s, base, "cynical_merchant", m_ids[0], 3)  # good
    rate(s, base, "cynical_merchant", m_ids[1], 3)  # good
    rate(s, base, "cynical_merchant", m_ids[2], 1)  # bad → 2 DPO pairs

    # Detective — single demonstration
    print("  seeding detective...")
    d_id = submit(s, base, "cold_detective", "유리 조각이 어디서 깨졌지?")
    rate(s, base, "cold_detective", d_id, 3)

    # Dragon — single demonstration
    print("  seeding dragon...")
    dr_id = submit(s, base, "ancient_dragon", "너의 어미는?")
    rate(s, base, "ancient_dragon", dr_id, 3)

    # Penguin — different prompts (no DPO pair, just summary surface)
    print("  seeding penguin...")
    p_a = submit(s, base, "penguin_relaxed", "안녕! 오늘 뭐 했어?")
    rate(s, base, "penguin_relaxed", p_a, 3)
    p_b = submit(s, base, "penguin_relaxed", "춤추자")
    rate(s, base, "penguin_relaxed", p_b, 0)  # dismiss

    return tuple(m_ids)  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default="docs/screenshots")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Health check
    try:
        h = requests.get(f"{args.url}/health", timeout=10).json()
        print(f"server health: llm={h.get('llm')}")
    except Exception as e:
        print(f"❌ server not reachable at {args.url}: {e}", file=sys.stderr)
        return 1

    print("\n=== Phase 1: clean state for 01 / 03 ===")
    seed_phase_1_baseline(args.url)
    time.sleep(0.3)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()

        # 03 — empty sessions page (no pending crud)
        page.goto(f"{args.url}/web/sessions", wait_until="networkidle")
        shot(page, out, "03_sessions")

        # === Phase 2: real-model conversations ===
        print("\n=== Phase 2: real-model chats + ratings ===")
        seed_phase_2_conversations(args.url)

        # 02 — merchant chat thread, full page (rating widget visible per turn)
        page.goto(f"{args.url}/web/chat/cynical_merchant", wait_until="networkidle")
        shot(page, out, "02_chat_thread")

        # 31 — gallery with rating-summary + DPO badges
        page.goto(f"{args.url}/web/personas", wait_until="networkidle")
        shot(page, out, "31_personas_with_dpo")

        # 32 — metrics dashboard
        page.goto(f"{args.url}/web/metrics", wait_until="networkidle")
        shot(page, out, "32_metrics_dashboard")

        browser.close()

    print("\n🎉 done. Refreshed 4 PNGs under docs/screenshots/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

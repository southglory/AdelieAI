"""One-off screenshot capture for Step 6.2 + A5 UI surfaces.

Seeds a local server with merchant chat + ratings, then captures:
  30_rating_widget.png       — chat thread w/ 3-tier rating + dismiss
  31_personas_with_dpo.png   — gallery w/ rating-summary + DPO badge
  32_metrics_dashboard.png   — /web/metrics rollup table

Usage:
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 \\
        scripts/capture_step6_screenshots.py \\
        --url http://127.0.0.1:8765 --out docs/screenshots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import requests  # type: ignore[import-untyped]
from playwright.sync_api import sync_playwright


def seed(base: str) -> None:
    """Drive chat turns + ratings via HTTP. Uses cookie session for the
    'demo' user (matches default in personas_web.py).

    Strategy: send the *same prompt* multiple times to each persona so
    the stub picker rotates through different canned lines, then rate
    them divergently — that's how DPO pairs appear in the badges.
    """
    import re
    s = requests.Session()

    def submit(persona: str, msg: str) -> int:
        r = s.post(f"{base}/web/chat/{persona}/messages", data={"message": msg})
        r.raise_for_status()
        ids = [int(x) for x in re.findall(r'id="turn-(\d+)"', r.text)]
        return ids[-1]

    def rate(persona: str, turn_id: int, rating: int) -> None:
        s.post(f"{base}/web/chat/{persona}/turns/{turn_id}/rate", data={"rating": rating})

    # Merchant — same prompt 3x → rotates through canned lines.
    # Rate 2 good + 1 bad + + 1 dismiss → 2 DPO pairs (2 good × 1 bad).
    m_ids = [submit("cynical_merchant", "할인 안 돼?") for _ in range(3)]
    m_ids.append(submit("cynical_merchant", "이 검 얼마야?"))  # dismiss target
    rate("cynical_merchant", m_ids[0], 3)  # good
    rate("cynical_merchant", m_ids[1], 3)  # good
    rate("cynical_merchant", m_ids[2], 1)  # bad → 2 pairs harvested
    rate("cynical_merchant", m_ids[3], 0)  # dismiss

    # Detective — same prompt 2x → 1 good 1 bad → 1 pair
    d_ids = [submit("cold_detective", "유리 조각이 어디서 깨졌지?") for _ in range(2)]
    rate("cold_detective", d_ids[0], 3)
    rate("cold_detective", d_ids[1], 1)

    # Penguin — single chat, mixed signal but no pair (different prompts)
    p_ids = [
        submit("penguin_relaxed", "안녕!"),
        submit("penguin_relaxed", "오늘 뭐 했어?"),
    ]
    rate("penguin_relaxed", p_ids[0], 3)
    rate("penguin_relaxed", p_ids[1], 2)  # fine


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default="docs/screenshots")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"seeding {args.url} ...")
    seed(args.url)
    print("done")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()

        # 30 — chat thread with rating widget
        page.goto(f"{args.url}/web/chat/cynical_merchant", wait_until="networkidle")
        page.screenshot(
            path=str(out / "30_rating_widget.png"),
            full_page=True,
        )
        print("saved 30_rating_widget.png")

        # 31 — gallery with rating-summary
        page.goto(f"{args.url}/web/personas", wait_until="networkidle")
        page.screenshot(
            path=str(out / "31_personas_with_dpo.png"),
            full_page=True,
        )
        print("saved 31_personas_with_dpo.png")

        # 32 — metrics dashboard
        page.goto(f"{args.url}/web/metrics", wait_until="networkidle")
        page.screenshot(
            path=str(out / "32_metrics_dashboard.png"),
            full_page=True,
        )
        print("saved 32_metrics_dashboard.png")

        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

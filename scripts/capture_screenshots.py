"""Headless Chromium walks the running console and saves PNGs.

Unlike tests/e2e/test_flow.py this script never asserts — it just
navigates and captures. Useful for refreshing docs/screenshots/ on
demand without fighting model loading edge cases.

Usage:
    .venv/Scripts/python -m playwright install chromium  (once)
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 scripts/capture_screenshots.py \\
        --url http://localhost:8770 --out docs/screenshots
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from playwright.sync_api import Page, sync_playwright


def shot(page: Page, out: Path, name: str) -> None:
    page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    print(f"  saved {name}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8770")
    parser.add_argument("--out", default="docs/screenshots")
    parser.add_argument("--user", default="penguin-watcher")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        # Set the user identity once (no separate "before" capture
        # — the empty-sessions and user-changed views look identical).
        page.goto(f"{args.url}/web/sessions")
        page.locator("input[name='user_id']").fill(args.user)
        page.locator("nav form button[type='submit']").click()
        page.wait_for_url(f"{args.url}/web/sessions")
        time.sleep(0.3)

        # 01 — sessions list with three KO role-play sessions
        for goal in [
            "놀고있는 펭귄으로서 지금 할 말을 해줘",
            "헤엄치는 물고기가 상어를 만났을 때 할 말을 해줘",
            "용감한 기사로서 용 앞에서 한마디",
        ]:
            page.locator("input[name='goal']").fill(goal)
            page.locator(".create-form button[type='submit']").click()
            time.sleep(0.5)
        shot(page, out, "01_sessions")

        # 02 — open the first session, run-config form pre-filled with
        # custom sampling values for the screenshot.
        page.locator("#session-list a").first.click()
        time.sleep(0.4)
        page.locator("input[name='retrieval_k']").fill("3")
        page.locator("input[name='temperature']").fill("0.85")
        time.sleep(0.2)
        shot(page, out, "02_run_config")

        # 03 — after generate (stub LLM completes instantly so no
        # mid-flight streaming frame to capture).
        page.locator("#generate-btn").click()
        page.wait_for_load_state("networkidle", timeout=30_000)
        time.sleep(0.6)
        shot(page, out, "03_answer")

        # 04 — graceful 503 page when no embedder is loaded
        page.goto(f"{args.url}/web/docs")
        time.sleep(0.4)
        shot(page, out, "04_docs_unavailable")

        # 05 — /health JSON
        page.goto(f"{args.url}/health")
        time.sleep(0.2)
        shot(page, out, "05_health")

        # 06 — Swagger
        page.goto(f"{args.url}/docs")
        time.sleep(1.5)
        shot(page, out, "06_swagger")

        browser.close()
    print(f"\nall screenshots saved to {out.resolve()}")


if __name__ == "__main__":
    main()

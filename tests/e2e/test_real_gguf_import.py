"""Opt-in real-model golden path for Adelie Drop.

Run against a server booted with a GGUF MODEL_PATH by setting E2E_BASE_URL.
The normal Stub E2E skips this test honestly.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from playwright.sync_api import Page, expect


def test_character_card_generates_with_real_gguf(
    page: Page,
    base_url: str,
    screenshots: Path,
    tmp_path: Path,
) -> None:
    health = httpx.get(f"{base_url}/health", timeout=10).json()
    if str(health["llm"]).startswith("stub-"):
        pytest.skip("requires a server booted with a real GGUF MODEL_PATH")

    console_errors: list[str] = []
    failed_requests: list[str] = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("requestfailed", lambda request: failed_requests.append(request.url))

    card = tmp_path / "mara-merchant.json"
    card.write_text(
        json.dumps(
            {
                "spec": "chara_card_v2",
                "data": {
                    "name": "Mara Merchant",
                    "description": (
                        "A terse fantasy shopkeeper. Never breaks character and "
                        "never gives discounts. Answers in one short English sentence."
                    ),
                    "personality": "Dry, practical, unimpressed by adventurers.",
                    "scenario": "A crowded supply shop before the city gates close.",
                    "mes_example": (
                        "{{user}}: Can I get a discount?\n"
                        "{{char}}: The price is already kinder than I am."
                    ),
                },
            }
        ),
        encoding="utf-8",
    )

    page.set_viewport_size({"width": 1440, "height": 960})
    page.goto(f"{base_url}/web/personas/import")
    expect(page.locator(".runtime-now")).to_contain_text(str(health["llm"]))
    page.locator("input[name='persona_file']").set_input_files(card)
    page.get_by_role("button", name="Import & open chat").click()
    page.wait_for_url(f"{base_url}/web/chat/mara_merchant?imported=1")

    page.locator("input[name='message']").fill("Can I get a discount?")
    page.get_by_role("button", name="send").click()
    assistant = page.locator(".turn.assistant").last
    expect(assistant).to_be_visible(timeout=120_000)
    expect(assistant).not_to_contain_text("[stub:")
    expect(assistant.locator(".stat")).to_contain_text("tok")
    page.screenshot(
        path=str(screenshots / "36_adelie_drop_real_gguf.png"),
        full_page=True,
    )

    assert console_errors == []
    assert failed_requests == []

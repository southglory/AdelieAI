"""Browser golden path: Character Card file to a live local chat."""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import Page, expect


def test_character_card_import_to_chat(
    page: Page,
    base_url: str,
    screenshots: Path,
    tmp_path: Path,
) -> None:
    console_errors: list[str] = []
    failed_requests: list[str] = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("requestfailed", lambda request: failed_requests.append(request.url))
    card = tmp_path / "mira.json"
    card.write_text(
        json.dumps(
            {
                "spec": "chara_card_v2",
                "data": {
                    "name": "Mira",
                    "description": "A cartographer who trusts coastlines, not rumors.",
                    "personality": "Precise, patient, quietly adventurous.",
                    "scenario": "A map room overlooking a stormy harbor.",
                    "mes_example": "{{user}}: I am lost.\n{{char}}: Find the coastline first.",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    page.set_viewport_size({"width": 1440, "height": 960})
    page.goto(f"{base_url}/web/personas/import")
    expect(page.get_by_text("Bring a character.")).to_be_visible()
    page.screenshot(path=str(screenshots / "33_adelie_drop_import.png"), full_page=True)

    page.locator("input[name='persona_file']").set_input_files(card)
    expect(page.locator("#file-label")).to_have_text("mira.json")
    page.get_by_role("button", name="Import & open chat").click()

    page.wait_for_url(f"{base_url}/web/chat/mira?imported=1")
    expect(page.locator(".chat-header .who")).to_contain_text("Mira")
    expect(page.locator(".chat-meta code", has_text="active-runtime")).to_be_visible()

    page.locator("input[name='message']").fill("Show me the north road from the harbor.")
    page.get_by_role("button", name="send").click()
    expect(page.get_by_text("Show me the north road from the harbor.", exact=True)).to_be_visible()
    expect(page.locator(".turn.assistant").last).to_be_visible()
    assert console_errors == []
    assert failed_requests == []

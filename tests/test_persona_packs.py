"""Portable persona import, validation, discovery, and web activation."""

from __future__ import annotations

import json
import base64
import struct
import zipfile
import zlib
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from core.api.app import build_app
from core.personas.packs import (
    PackValidationError,
    PersonaImportService,
    load_persona_pack,
)
from core.personas.registry import get_persona, list_personas
from core.personas.store import InMemoryChatStore
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore


def _card(name: str = "Mira") -> bytes:
    return json.dumps(
        {
            "spec": "chara_card_v2",
            "data": {
                "name": name,
                "description": "A cartographer who trusts coastlines, not rumors.",
                "personality": "Precise, patient, quietly adventurous.",
                "scenario": "A map room overlooking a stormy harbor.",
                "mes_example": "{{user}}: 길을 잃었어.\n{{char}}: 먼저 해안선을 찾죠.",
            },
        },
        ensure_ascii=False,
    ).encode()


def _png_card(card: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    metadata = b"chara\x00" + base64.b64encode(card)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"tEXt", metadata) + chunk(b"IEND", b"")


def test_character_card_import_normalizes_to_runnable_pack(tmp_path) -> None:
    loaded = PersonaImportService(tmp_path).install("mira.json", _card())

    assert loaded.persona.persona_id == "mira"
    assert loaded.persona.display_name == "Mira"
    assert loaded.persona.source_format == "character-card-v2"
    assert loaded.root.name == "mira.adelie"
    assert "cartographer" in loaded.persona.system_prompt
    assert (loaded.root / "MANIFEST.json").is_file()
    assert (loaded.root / "system_prompt.md").is_file()


def test_pack_loader_reports_all_actionable_file_issues(tmp_path) -> None:
    root = tmp_path / "broken.adelie"
    root.mkdir()
    (root / "MANIFEST.json").write_text(
        json.dumps(
            {
                "persona_id": "broken",
                "display_name": "Broken",
                "system_prompt": "missing.md",
                "rag": {"enabled": True, "corpus_path": "missing-corpus"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(PackValidationError) as error:
        load_persona_pack(root)

    assert "missing.md is missing" in str(error.value)
    assert "RAG corpus" in str(error.value)


def test_imported_pack_is_discovered_without_global_registry_mutation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADELIE_PACKS_DIR", str(tmp_path))
    PersonaImportService(tmp_path).install("mira.json", _card())

    assert "mira" in {persona.persona_id for persona in list_personas()}
    assert get_persona("mira").pack_path is not None


def test_duplicate_import_is_blocked_without_overwrite(tmp_path) -> None:
    service = PersonaImportService(tmp_path)
    service.install("mira.json", _card())

    with pytest.raises(PackValidationError, match="already installed"):
        service.install("mira.json", _card())


def test_png_character_card_metadata_is_imported(tmp_path) -> None:
    loaded = PersonaImportService(tmp_path).install("mira.png", _png_card(_card()))
    assert loaded.persona.persona_id == "mira"
    assert loaded.source_format == "character-card-v2"


def test_root_level_zipped_adelie_pack_is_normalized_safely(tmp_path) -> None:
    manifest = {
        "persona_id": "archived",
        "display_name": "Archived",
        "system_prompt": "system_prompt.md",
    }
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("MANIFEST.json", json.dumps(manifest))
        zf.writestr("system_prompt.md", "You are a careful archivist.")

    loaded = PersonaImportService(tmp_path).install("archived.adelie", archive.getvalue())
    assert loaded.persona.persona_id == "archived"
    assert loaded.root == (tmp_path / "archived.adelie").resolve()


def test_zipped_pack_rejects_path_traversal(tmp_path) -> None:
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../MANIFEST.json", "{}")

    with pytest.raises(PackValidationError, match="unsafe archive member"):
        PersonaImportService(tmp_path).install("unsafe.adelie", archive.getvalue())


def test_import_page_and_upload_reach_real_chat_route(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADELIE_PACKS_DIR", str(tmp_path))
    app = build_app(
        store=InMemorySessionStore(),
        llm=StubLLMClient(),
        chat_store=InMemoryChatStore(),
    )
    client = TestClient(app)

    page = client.get("/web/personas/import")
    assert page.status_code == 200
    assert "ADELIE DROP" in page.text
    assert "Character Card V2" in page.text

    response = client.post(
        "/web/personas/import",
        files={"persona_file": ("mira.json", _card(), "application/json")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/web/chat/mira?imported=1"
    chat = client.get(response.headers["location"])
    assert chat.status_code == 200
    assert "Mira" in chat.text


def test_malformed_import_surfaces_safe_error_in_ui(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ADELIE_PACKS_DIR", str(tmp_path))
    app = build_app(
        store=InMemorySessionStore(),
        llm=StubLLMClient(),
        chat_store=InMemoryChatStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/web/personas/import",
        files={"persona_file": ("bad.json", b"not-json", "application/json")},
    )

    assert response.status_code == 400
    assert "Import blocked" in response.text
    assert "invalid" in response.text

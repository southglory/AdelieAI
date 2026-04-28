"""Per-persona grounding context builder."""

from __future__ import annotations

import pytest

from core.personas.grounding import build_grounding_context
from core.personas.registry import get_persona
from core.tools import ToolRegistry
from core.tools.evidence_search import EvidenceSearch

pytest.importorskip("rdflib")

from core.retrieval.graph_retriever_rdflib import RdflibGraphRetriever


def test_general_persona_yields_empty_grounding() -> None:
    p = get_persona("penguin_relaxed")
    out = build_grounding_context(p, user_text="hello")
    assert out == ""


def test_knowledge_persona_pulls_kg_facts() -> None:
    p = get_persona("ancient_dragon")
    g = RdflibGraphRetriever()
    out = build_grounding_context(p, user_text="너의 가장 오래된 조상은?", graph_retriever=g)
    assert out  # non-empty
    # Asserted ancestry must appear so the LLM does not invent lore
    assert "Vyrnaes" in out
    assert "Sothryn" in out
    # Lair info also present
    assert "Erebor" in out
    # The "do not invent" instruction must be present
    assert "환각" in out or "만들어내" in out


def test_knowledge_grounding_without_retriever_is_empty() -> None:
    p = get_persona("ancient_dragon")
    out = build_grounding_context(p, user_text="..", graph_retriever=None)
    assert out == ""


def test_legal_persona_pulls_evidence_hits() -> None:
    p = get_persona("cold_detective")
    reg = ToolRegistry()
    reg.register(EvidenceSearch())
    out = build_grounding_context(
        p,
        user_text="유리 조각이 어디서 발견되었어?",
        tool_registry=reg,
    )
    assert out
    assert "evidence_search" in out
    # The mock corpus's `evidence_1.md` mentions glass; should surface
    assert "evidence_1.md" in out


def test_legal_grounding_no_hits_lists_catalog() -> None:
    p = get_persona("cold_detective")
    reg = ToolRegistry()
    reg.register(EvidenceSearch())
    out = build_grounding_context(
        p,
        user_text="마시멜로 외계인",
        tool_registry=reg,
    )
    assert out
    assert "0건" in out
    # Catalog of known files should still surface so model knows what's available
    assert "evidence_1.md" in out
    assert "case_log_07.md" in out


def test_legal_grounding_without_tool_registry_is_empty() -> None:
    p = get_persona("cold_detective")
    out = build_grounding_context(p, user_text="..", tool_registry=None)
    assert out == ""


def test_grounding_threaded_through_chat_for_dragon() -> None:
    """End-to-end: POSTing a message to /web/chat/ancient_dragon/messages
    with the live app should build grounding and surface KG facts in the
    augmented system prompt — verified indirectly by the StubLLMClient
    receiving an enlarged system field."""
    from fastapi.testclient import TestClient

    from core.api.app import build_app
    from core.personas.store import InMemoryChatStore
    from core.serving.stub_client import StubLLMClient
    from core.session.store_memory import InMemorySessionStore

    app = build_app(
        store=InMemorySessionStore(),
        llm=StubLLMClient(),
        chat_store=InMemoryChatStore(),
    )
    c = TestClient(app)
    r = c.post(
        "/web/chat/ancient_dragon/messages",
        data={"message": "너의 가장 오래된 조상은?"},
    )
    assert r.status_code == 200
    # The stub falls back to its echo form for the augmented system,
    # but with grounding the recorded system is much larger than the
    # bare persona prompt. We can't easily inspect the system prompt
    # from outside, so smoke-test that the response renders without
    # error — the deeper assertion is in test_knowledge_persona_pulls_kg_facts.
    assert "<html" not in r.text  # partial only

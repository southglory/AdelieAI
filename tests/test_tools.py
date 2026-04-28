"""Tool protocol + evidence_search stub + tier introspection."""

import pytest
from fastapi.testclient import TestClient

from core.api.app import build_app
from core.personas.store import InMemoryChatStore
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore
from core.tools import Tool, ToolRegistry
from core.tools.evidence_search import EvidenceSearch


@pytest.fixture
def client() -> TestClient:
    app = build_app(
        store=InMemorySessionStore(),
        llm=StubLLMClient(),
        chat_store=InMemoryChatStore(),
    )
    return TestClient(app)


def test_evidence_search_satisfies_protocol() -> None:
    assert isinstance(EvidenceSearch(), Tool)


def test_evidence_search_metadata_is_complete() -> None:
    tool = EvidenceSearch()
    assert tool.name == "evidence_search"
    assert tool.description
    assert tool.input_schema["type"] == "object"
    assert "query" in tool.input_schema["properties"]
    assert "query" in tool.input_schema["required"]


def test_evidence_search_returns_hits_for_known_query() -> None:
    tool = EvidenceSearch()
    out = tool.call({"query": "유리"})
    assert out["n_hits"] >= 1
    assert any("evidence_1.md" == h["path"] for h in out["hits"])


def test_evidence_search_no_hits_for_unknown_query() -> None:
    tool = EvidenceSearch()
    out = tool.call({"query": "마시멜로 외계인 비행기"})
    assert out["n_hits"] == 0
    assert out["hits"] == []


def test_evidence_search_rejects_empty_query() -> None:
    tool = EvidenceSearch()
    out = tool.call({"query": "  "})
    assert out["hits"] == []
    assert out["error"]


def test_tool_registry_rejects_duplicate_name() -> None:
    reg = ToolRegistry()
    reg.register(EvidenceSearch())
    with pytest.raises(ValueError, match="already registered"):
        reg.register(EvidenceSearch())


def test_tool_registry_emits_function_schemas() -> None:
    reg = ToolRegistry()
    reg.register(EvidenceSearch())
    schemas = reg.schemas()
    assert len(schemas) == 1
    s = schemas[0]
    assert s["name"] == "evidence_search"
    assert "parameters" in s


def test_default_app_registers_evidence_search(client: TestClient) -> None:
    """build_app should register the evidence_search stub by default —
    this drives /health to declare T3 ok."""
    body = client.get("/health").json()
    assert body["tier"] >= 3
    assert "ok" in body["tier_status"]["T3"]
    assert "1 tool" in body["tier_status"]["T3"]


def test_detective_persona_declares_tier_3() -> None:
    from core.personas.registry import get_persona
    p = get_persona("cold_detective")
    assert p is not None
    assert p.target_tier == 3
    assert p.industry == "legal"
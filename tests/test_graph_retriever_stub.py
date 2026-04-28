"""GraphRetriever / OWLReasoner stub + tier introspection."""

import pytest
from fastapi.testclient import TestClient

from core.api.app import build_app
from core.personas.store import InMemoryChatStore
from core.retrieval.graph_retriever import GraphRetriever, OWLReasoner
from core.retrieval.graph_retriever_stub import RdfGraphRetriever, StubOWLReasoner
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore


@pytest.fixture
def client() -> TestClient:
    app = build_app(
        store=InMemorySessionStore(),
        llm=StubLLMClient(),
        chat_store=InMemoryChatStore(),
    )
    return TestClient(app)


def test_retriever_satisfies_protocol() -> None:
    assert isinstance(RdfGraphRetriever(), GraphRetriever)


def test_reasoner_satisfies_protocol() -> None:
    assert isinstance(StubOWLReasoner(), OWLReasoner)


def test_query_returns_hits_for_known_uri() -> None:
    r = RdfGraphRetriever()
    hits = r.query("SELECT * WHERE { :Self ?p ?o }")
    assert len(hits) == 1
    assert any(t.subject == ":Self" for t in hits[0].triples)


def test_query_returns_empty_for_unknown_uri() -> None:
    r = RdfGraphRetriever()
    hits = r.query("SELECT * WHERE { :NonExistentEntity ?p ?o }")
    assert hits == []


def test_expand_finds_neighborhood() -> None:
    r = RdfGraphRetriever()
    hits = r.expand("Erebor")
    assert len(hits) == 1
    triples = hits[0].triples
    # Erebor appears as subject (hostsRace, containsTreasure, wasAttackedBy)
    # and as object (Self lairIn Erebor) → at least 4 triples
    assert len(triples) >= 4


def test_expand_returns_empty_for_unknown() -> None:
    r = RdfGraphRetriever()
    assert r.expand("Marshmallow") == []


def test_reasoner_reports_consistent_and_inferred() -> None:
    rs = StubOWLReasoner()
    assert rs.consistent() is True
    inferred = rs.infer()
    assert len(inferred) >= 1
    assert all(t.inferred for t in inferred)
    assert all(t.source for t in inferred)


def test_default_app_registers_kg_retriever_and_reasoner(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["tier"] >= 4
    assert "ok" in body["tier_status"]["T4"]
    assert "reasoner" in body["tier_status"]["T4"]


def test_dragon_persona_declares_tier_4() -> None:
    from core.personas.registry import get_persona
    p = get_persona("ancient_dragon")
    assert p is not None
    assert p.target_tier == 4
    assert p.industry == "knowledge"

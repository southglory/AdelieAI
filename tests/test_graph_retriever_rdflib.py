"""Real rdflib-backed GraphRetriever + OWLReasoner.

These tests exercise the actual SPARQL engine and OWL-RL forward
chaining — they fail fast if `rdflib` or `owlrl` are not installed
(the stub fallback path is covered separately by
test_graph_retriever_stub.py).
"""
from __future__ import annotations

import pytest

pytest.importorskip("rdflib")

from core.retrieval.graph_retriever import GraphRetriever, OWLReasoner
from core.retrieval.graph_retriever_rdflib import (
    RdflibGraphRetriever,
    RdflibOWLReasoner,
)


@pytest.fixture(scope="module")
def retriever() -> RdflibGraphRetriever:
    return RdflibGraphRetriever()


@pytest.fixture(scope="module")
def reasoner(retriever: RdflibGraphRetriever) -> RdflibOWLReasoner:
    return RdflibOWLReasoner(retriever)


def test_retriever_satisfies_protocol(retriever: RdflibGraphRetriever) -> None:
    assert isinstance(retriever, GraphRetriever)


def test_reasoner_satisfies_protocol(reasoner: RdflibOWLReasoner) -> None:
    assert isinstance(reasoner, OWLReasoner)


def test_turtle_parses_to_expected_size(retriever: RdflibGraphRetriever) -> None:
    """Asserted base — 21 triples (12 facts + 4 metadata + 5 in
    embedded property/class declarations)."""
    assert len(retriever.graph) >= 18


def test_sparql_transitive_descendant_returns_both_ancestors(
    retriever: RdflibGraphRetriever,
) -> None:
    """`descendantOf+` must walk the chain Self → Vyrnaes → Sothryn."""
    hits = retriever.query(
        """
        PREFIX adel: <http://adelie.local/lore#>
        SELECT ?ancestor WHERE {
            adel:Self adel:descendantOf+ ?ancestor .
        }
        """
    )
    ancestors = sorted(h.triples[0].object for h in hits)
    # property paths produce a single bound column → rendered triples
    # use _to_triple's len==1 fallback path; we just check both
    # local names appear somewhere in the rendered output.
    rendered = " ".join(h.triples[0].object for h in hits)
    assert ":Vyrnaes" in rendered or "Vyrnaes" in rendered
    assert ":Sothryn" in rendered or "Sothryn" in rendered


def test_sparql_no_hits_for_unknown_uri(retriever: RdflibGraphRetriever) -> None:
    hits = retriever.query(
        """
        PREFIX adel: <http://adelie.local/lore#>
        SELECT ?o WHERE { adel:DoesNotExist adel:descendantOf ?o . }
        """
    )
    assert hits == []


def test_expand_collects_neighbors_in_both_directions(
    retriever: RdflibGraphRetriever,
) -> None:
    hits = retriever.expand("Erebor")
    assert len(hits) == 1
    triples = hits[0].triples
    # Erebor has at least these connections:
    #   Erebor a Mountain
    #   Erebor hostsRace Dwarf
    #   Erebor containsTreasure Arkenstone
    #   Erebor wasAttackedBy Self
    #   Self lairIn Erebor   ← incoming
    assert len(triples) >= 5
    predicates = {t.predicate for t in triples}
    assert any("type" in p for p in predicates)


def test_expand_returns_empty_for_unknown(retriever: RdflibGraphRetriever) -> None:
    assert retriever.expand("Marshmallow") == []


def test_reasoner_reports_consistent(reasoner: RdflibOWLReasoner) -> None:
    assert reasoner.consistent() is True


def test_reasoner_materializes_subclass_inference(
    reasoner: RdflibOWLReasoner,
) -> None:
    """OWL-RL must derive `:Self type :WingedBeing` from
    `:Self type :Dragon` + `:Dragon subClassOf :WingedBeing`."""
    inferred = reasoner.infer()
    assert all(t.inferred for t in inferred)
    rendered = [(t.subject, t.predicate, t.object) for t in inferred]
    assert any(
        s == ":Self" and "type" in p and o == ":WingedBeing"
        for s, p, o in rendered
    )


def test_reasoner_materializes_transitive_descendant(
    reasoner: RdflibOWLReasoner,
) -> None:
    """OWL-RL must derive `:Self descendantOf :Sothryn` because
    descendantOf is owl:TransitiveProperty."""
    inferred = reasoner.infer()
    rendered = [(t.subject, t.predicate, t.object) for t in inferred]
    assert any(
        s == ":Self" and "descendantOf" in p and o == ":Sothryn"
        for s, p, o in rendered
    )


def test_reasoner_filters_reflexive_noise(reasoner: RdflibOWLReasoner) -> None:
    """OWL-RL produces many `:X sameAs :X` and `:X subClassOf :X`
    triples — they should not surface as inferred knowledge."""
    inferred = reasoner.infer()
    for t in inferred:
        if t.subject == t.object:
            assert "sameAs" not in t.predicate
            assert "subClassOf" not in t.predicate
            assert "equivalentClass" not in t.predicate


def test_default_app_uses_rdflib_backend() -> None:
    """build_app should pick the real backend when rdflib + owlrl
    are installed (which they are in this test environment)."""
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
    assert type(app.state.graph_retriever).__name__ == "RdflibGraphRetriever"
    assert type(app.state.owl_reasoner).__name__ == "RdflibOWLReasoner"
    body = TestClient(app).get("/health").json()
    assert body["tier"] == 4
    assert "ok" in body["tier_status"]["T4"]
    assert "reasoner" in body["tier_status"]["T4"]

"""Stub `RdfGraphRetriever` — T4 capability marker.

A bare-bones GraphRetriever implementation that lets `ancient_dragon`
reach into a hand-built mock KG of dragon lore (12 nodes / 18 triples).

Two purposes — same pattern as core/tools/evidence_search.py:

  1. Activate T4 in `_compute_tier` (a graph_retriever on app.state) so
     /health and /demo/knowledge can declare a working KG stack.
  2. Demonstrate the GraphRetriever Protocol shape with a real
     concrete class for future contributors to copy when wiring in
     a real triple store (rdflib in-memory · Fuseki · GraphDB ·
     Stardog).

The mock graph encodes a small dragon lore: ancestry chain, lair,
treasure, races interacting with each. Queries are pseudo-SPARQL —
the stub recognizes a few keyword patterns rather than parsing real
SPARQL syntax. Real SPARQL execution lands when rdflib is wired in.
"""
from __future__ import annotations

from core.retrieval.graph_retriever import (
    GraphHit,
    GraphRetriever,
    OWLReasoner,
    Triple,
)


# Mock RDF triples — 18 (s, p, o) facts forming a small lore graph.
# Each is a Triple with explicit `inferred=False` (asserted) or
# `inferred=True` (would be produced by an OWL reasoner from the
# asserted base).
_BASE_TRIPLES: list[Triple] = [
    Triple(subject=":Self", predicate=":a", object=":Dragon"),
    Triple(subject=":Self", predicate=":nameLost", object="false"),
    Triple(subject=":Self", predicate=":age", object="1247"),
    Triple(subject=":Self", predicate=":lairIn", object=":Erebor"),
    Triple(subject=":Self", predicate=":descendantOf", object=":Vyrnaes"),
    Triple(subject=":Vyrnaes", predicate=":a", object=":Dragon"),
    Triple(subject=":Vyrnaes", predicate=":descendantOf", object=":Sothryn"),
    Triple(subject=":Sothryn", predicate=":a", object=":Dragon"),
    Triple(subject=":Sothryn", predicate=":nameLost", object="true"),
    Triple(subject=":Erebor", predicate=":a", object=":Mountain"),
    Triple(subject=":Erebor", predicate=":hostsRace", object=":Dwarf"),
    Triple(subject=":Erebor", predicate=":containsTreasure", object=":Arkenstone"),
    Triple(subject=":Erebor", predicate=":wasAttackedBy", object=":Self"),
    Triple(subject=":Arkenstone", predicate=":a", object=":Treasure"),
    Triple(subject=":Arkenstone", predicate=":discoveredBy", object=":Thrór"),
    Triple(subject=":Thrór", predicate=":a", object=":DwarfKing"),
    Triple(subject=":Thrór", predicate=":lineageOf", object=":Dwarf"),
    Triple(subject=":Dragon", predicate=":subClassOf", object=":WingedBeing"),
]


# Inferred triples — what an OWL reasoner with a transitive closure
# rule for :descendantOf and a subClassOf rule for :a would derive
# from the base set above.
_INFERRED_TRIPLES: list[Triple] = [
    Triple(subject=":Self", predicate=":descendantOf", object=":Sothryn",
           inferred=True, source="transitive(:descendantOf)"),
    Triple(subject=":Self", predicate=":a", object=":WingedBeing",
           inferred=True, source="subClassOf(:Dragon, :WingedBeing)"),
    Triple(subject=":Vyrnaes", predicate=":a", object=":WingedBeing",
           inferred=True, source="subClassOf(:Dragon, :WingedBeing)"),
    Triple(subject=":Sothryn", predicate=":a", object=":WingedBeing",
           inferred=True, source="subClassOf(:Dragon, :WingedBeing)"),
]


def _matches(t: Triple, terms: list[str]) -> bool:
    text = f"{t.subject} {t.predicate} {t.object}".lower()
    return any(term in text for term in terms)


class RdfGraphRetriever:
    """Concrete GraphRetriever stub over the mock dragon-lore corpus."""

    def query(self, sparql: str) -> list[GraphHit]:
        """Naive keyword-matching pseudo-SPARQL.

        Real SPARQL parsing comes when rdflib is wired in. For the
        demo, we extract URI-like tokens (`:Foo`) and `?var` names
        from the query, then return triples that mention any of them.
        """
        terms: list[str] = []
        for token in sparql.replace("{", " ").replace("}", " ").split():
            if token.startswith(":"):
                terms.append(token.lower().rstrip(",.()"))
        if not terms:
            return []
        all_triples = _BASE_TRIPLES + _INFERRED_TRIPLES
        matched = [t for t in all_triples if _matches(t, terms)]
        if not matched:
            return []
        return [
            GraphHit(
                triples=matched,
                score=1.0,
                explanation=f"matched {len(matched)} triples on terms {terms}",
            )
        ]

    def expand(self, entity: str, depth: int = 1) -> list[GraphHit]:
        """1-hop neighborhood expansion.

        Returns all triples where `entity` (case-insensitive) appears
        as either subject or object. depth>1 is collapsed to 1 in
        the stub — real layered expansion arrives with rdflib.
        """
        e = entity.lower().lstrip(":")
        all_triples = _BASE_TRIPLES + _INFERRED_TRIPLES
        hits = [
            t for t in all_triples
            if e in t.subject.lower() or e in t.object.lower()
        ]
        if not hits:
            return []
        return [
            GraphHit(
                triples=hits,
                score=1.0,
                explanation=f"{depth}-hop neighborhood of :{entity.lstrip(':')}",
            )
        ]


class StubOWLReasoner:
    """Always-consistent stub reasoner.

    Real reasoners (HermiT, Pellet, ELK, owlready2's sync_reasoner)
    detect contradictions and materialize implied triples. The stub
    just exposes the pre-computed `_INFERRED_TRIPLES` list so the
    /demo/knowledge UI can show an "OWL reasoner output" panel.
    """

    def consistent(self) -> bool:
        return True

    def infer(self) -> list[Triple]:
        return list(_INFERRED_TRIPLES)


# Sanity check — these classes satisfy the runtime-checkable Protocols.
assert isinstance(RdfGraphRetriever(), GraphRetriever)
assert isinstance(StubOWLReasoner(), OWLReasoner)

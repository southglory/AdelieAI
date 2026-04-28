"""T4 — graph (KG) retrieval Protocol.

Skeleton only — no SPARQL endpoint or rdflib in-memory graph yet.

Why a separate Protocol from `Retriever` (vector RAG):
  * Vector retrievers return text chunks ranked by similarity.
  * Graph retrievers return *triples* (or expanded subgraphs)
    selected by structural query (SPARQL / Cypher).
  * Downstream prompt-builders treat them differently — KG hits
    become structured "facts" injected with provenance, not free
    text mixed into a context window.

A T4-aware persona may use both: the vector retriever finds
candidate entity mentions, the graph retriever expands their 1-2
hop neighborhoods.

See docs/CAPABILITY_TIERS.md, tier T4.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class Triple(BaseModel):
    """An (s, p, o) RDF triple. Curie-form is allowed in any slot."""

    model_config = ConfigDict(frozen=True)

    subject: str
    predicate: str
    object: str
    # Optional provenance / inferred-from-axiom marker.
    inferred: bool = False
    source: str | None = None


class GraphHit(BaseModel):
    """A SPARQL result row, normalized as bound triples + score."""

    model_config = ConfigDict(frozen=True)

    triples: list[Triple]
    score: float = 1.0
    explanation: str | None = None  # natural-language gloss for the LLM


@runtime_checkable
class GraphRetriever(Protocol):
    """A KG-backed retriever.

    Two query modes:
      * `query(sparql)` — raw SPARQL for the LLM-as-SPARQL-generator path
      * `expand(entity, depth)` — neighborhood expansion for
        candidate-entity retrieval
    """

    def query(self, sparql: str) -> list[GraphHit]: ...

    def expand(self, entity: str, depth: int = 1) -> list[GraphHit]: ...


@runtime_checkable
class OWLReasoner(Protocol):
    """Optional layer above GraphRetriever.

    Materializes inferred triples (transitive closure, equivalent
    classes, property chains) and lets the persona answer questions
    that aren't directly stated in the asserted KG.

    Concrete implementations: HermiT (Java), Pellet, ELK, owlready2's
    sync_reasoner.
    """

    def consistent(self) -> bool: ...

    def infer(self) -> list[Triple]: ...

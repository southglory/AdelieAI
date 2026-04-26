from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from core.schemas.retrieval import RetrievedChunk, RetrievedContext


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    skip_retrieval: bool = False
    search_queries: list[str] = Field(default_factory=list)
    rationale: str = ""

    @classmethod
    def fallback(cls, goal: str) -> "Plan":
        return cls(
            skip_retrieval=False,
            search_queries=[goal],
            rationale="planner failed to produce a parseable plan; using goal as single query",
        )


def _merge_retrievals(
    a: list[RetrievedChunk], b: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    seen: set[str] = set()
    merged: list[RetrievedChunk] = []
    for h in [*a, *b]:
        if h.chunk.id in seen:
            continue
        seen.add(h.chunk.id)
        merged.append(h)
    return merged


class AgentState(TypedDict, total=False):
    """LangGraph state — accumulates as nodes run.

    'merged_results' uses a custom reducer so retriever_node can be
    called multiple times (one per search query) and have results
    accumulate without duplicates instead of overwriting.
    """

    goal: str
    plan: Plan | None
    retrieval_k: int
    contexts: Annotated[list[RetrievedContext], list.__add__]
    merged_results: Annotated[list[RetrievedChunk], _merge_retrievals]
    answer: str
    final_report: str

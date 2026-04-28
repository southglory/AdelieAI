"""Stub `evidence_search` tool — T3 capability marker.

A bare-bones Tool implementation that lets `cold_detective` reach into
a tiny mock case file. The point of this stub is twofold:

  1. Activate T3 in `_compute_tier` (a non-empty ToolRegistry) so
     /health and /demo/legal can declare a working tool stack.
  2. Demonstrate the Tool Protocol shape with a real concrete class
     for future contributors to copy.

The mock corpus is a fixed dict, NOT a real RAG. When AdelieAI grows
a `RagAsTool` adapter, this module is expected to be replaced or
upgraded — same Tool name + input schema, real retriever underneath.
"""
from __future__ import annotations

from typing import Any

from core.tools.protocols import Tool


# Tiny mock corpus — keyword-indexed for demo purposes only.
# Each "file" is a string with simulated case-file content.
_CASE_FILES: dict[str, str] = {
    "evidence_1.md": (
        "현장에서 발견된 유리 조각. 안쪽 방향으로 떨어져 있었음. "
        "추정 충격 위치는 방 내부에서 외부로 향한 힘. "
        "참고: case_log_07 의 23:00 진술과 어긋남."
    ),
    "case_log_07.md": (
        "23:00 — 용의자 A 의 진술: 사건 당시 가게에 있었음. "
        "23:30 — 동일 인물의 두 번째 진술: 사건 당시 거리에 있었음. "
        "두 진술 모두 같은 시각에 대한 것이며, 위치만 다름. 위증 의심."
    ),
    "timeline.txt": (
        "11:04 — 멈춘 시계의 시각 (조작 추정). "
        "11:30 — 실제 사건 시각. "
        "11:45 — 이웃이 비명 소리를 들음. "
        "12:10 — 경찰 도착."
    ),
    "witness_a.md": (
        "증인 A: 11시경 가게 앞을 지나갔다고 진술. "
        "보강 증언: 같은 시각 거리에 있었다는 다른 증인 진술 1건 존재. "
        "신뢰도 중간 — 시간 인식이 흐릿함."
    ),
}


def _search_keywords(query: str) -> list[dict[str, Any]]:
    """Naive keyword overlap. Returns hits with file path + snippet."""
    query_terms = [t.lower().strip() for t in query.replace(",", " ").split() if t.strip()]
    hits = []
    for path, body in _CASE_FILES.items():
        score = sum(body.lower().count(t) for t in query_terms)
        if score > 0:
            hits.append({"path": path, "score": score, "snippet": body})
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits


class EvidenceSearch:
    """Concrete Tool — `evidence_search` over the mock case corpus."""

    name = "evidence_search"
    description = (
        "Search the active case file for evidence matching a natural-language query. "
        "Returns a ranked list of file paths with snippets. Use when the persona "
        "needs to ground a deduction in the recorded evidence."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of what to find — keywords, names, times.",
            }
        },
        "required": ["query"],
    }

    def call(self, arguments: dict[str, Any]) -> Any:
        query = arguments.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return {"hits": [], "error": "query must be a non-empty string"}
        hits = _search_keywords(query)
        return {
            "query": query,
            "n_hits": len(hits),
            "hits": hits[:4],  # top_k mirrors the persona's recommended default
        }


# Sanity check: this class satisfies the runtime-checkable Tool Protocol.
assert isinstance(EvidenceSearch(), Tool)

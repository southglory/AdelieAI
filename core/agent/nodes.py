import json
import re
from typing import Callable

from core.agent.rag import format_context
from core.agent.state import AgentState, Plan
from core.logging import get_logger
from core.retrieval.protocols import Retriever
from core.schemas.retrieval import RetrievedContext
from core.serving.protocols import GenerationParams, LLMClient

log = get_logger("differentia.agent.nodes")


PLANNER_SYSTEM = (
    "당신은 RAG 시스템의 planner입니다. 사용자 goal을 보고 다음을 결정하세요.\n"
    "1) skip_retrieval: 단순한 인사·일반 상식·산수 등 문서 조회가 필요 없으면 true.\n"
    "2) search_queries: 1~3개의 구체적 검색어 리스트 (skip_retrieval=false일 때). "
    "사용자 goal과 동일한 언어로 작성하세요.\n"
    "3) rationale: 결정의 근거 한 줄.\n\n"
    "반드시 JSON만 출력하세요. 예:\n"
    '{"skip_retrieval": false, "search_queries": ["...", "..."], "rationale": "..."}'
)


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def parse_plan(raw: str, *, goal: str) -> Plan:
    match = _JSON_BLOCK.search(raw)
    if not match:
        return Plan.fallback(goal)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Plan.fallback(goal)
    try:
        return Plan(
            skip_retrieval=bool(data.get("skip_retrieval", False)),
            search_queries=[
                q for q in data.get("search_queries", []) if isinstance(q, str)
            ],
            rationale=str(data.get("rationale", ""))[:500],
        )
    except Exception:
        return Plan.fallback(goal)


def make_planner_node(
    llm: LLMClient,
    *,
    on_event: Callable[[str, dict], None] | None = None,
) -> Callable:
    async def planner(state: AgentState) -> dict:
        goal = state["goal"]
        params = GenerationParams(
            max_new_tokens=200, temperature=0.0, system=PLANNER_SYSTEM
        )
        result = await llm.generate(goal, params=params)
        plan = parse_plan(result.text, goal=goal)
        if on_event is not None:
            on_event(
                "plan",
                {
                    "model_id": result.model_id,
                    "plan": plan.model_dump(),
                    "raw": result.text[:400],
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "latency_ms": result.latency_ms,
                },
            )
        log.info(
            "planner_done",
            extra={
                "queries": len(plan.search_queries),
                "skip_retrieval": plan.skip_retrieval,
            },
        )
        return {"plan": plan}
    return planner


def make_retriever_node(
    retriever: Retriever | None,
    *,
    on_event: Callable[[str, dict], None] | None = None,
) -> Callable:
    async def retrieve(state: AgentState) -> dict:
        plan = state.get("plan")
        if retriever is None or plan is None or plan.skip_retrieval:
            if on_event is not None:
                on_event(
                    "retrieve_skip",
                    {
                        "reason": "no retriever or planner asked to skip",
                        "plan": plan.model_dump() if plan is not None else None,
                    },
                )
            return {"contexts": [], "merged_results": []}

        k = state.get("retrieval_k", 5) or 5
        queries = plan.search_queries or [state["goal"]]
        contexts: list[RetrievedContext] = []
        merged_ids: dict[str, float] = {}
        merged: list = []
        for q in queries:
            ctx = await retriever.retrieve(q, k=k)
            contexts.append(ctx)
            for r in ctx.results:
                cid = r.chunk.id
                if cid not in merged_ids:
                    merged_ids[cid] = r.score
                    merged.append(r)
                elif r.score > merged_ids[cid]:
                    merged_ids[cid] = r.score
        merged.sort(key=lambda r: merged_ids[r.chunk.id], reverse=True)
        merged = merged[:k]

        if on_event is not None:
            on_event(
                "retrieve",
                {
                    "queries": queries,
                    "k": k,
                    "method": contexts[0].method if contexts else None,
                    "merged": [
                        {
                            "score": merged_ids[r.chunk.id],
                            "doc_id": r.chunk.doc_id,
                            "doc_title": r.chunk.metadata.get("doc_title"),
                            "preview": r.chunk.text[:200],
                        }
                        for r in merged
                    ],
                },
            )
        log.info(
            "retriever_done",
            extra={"queries": len(queries), "merged_hits": len(merged)},
        )
        return {"contexts": contexts, "merged_results": merged}
    return retrieve


REASONER_SYSTEM = (
    "당신은 도메인 전문가입니다. 주어진 컨텍스트를 활용해 사용자 질문에 답하세요. "
    "컨텍스트가 답을 직접 포함하지 않으면 알지 못한다고 정직하게 말하고, 추측하지 마세요. "
    "근거 인용은 [1], [2] 같이 컨텍스트 번호로 표기하세요."
)


def make_reasoner_node(
    llm: LLMClient,
    *,
    on_event: Callable[[str, dict], None] | None = None,
) -> Callable:
    async def reason(state: AgentState) -> dict:
        merged = state.get("merged_results", [])
        ctx = RetrievedContext(
            method="hybrid", query=state["goal"], results=merged
        )
        prompt = format_context(state["goal"], ctx)
        params = GenerationParams(
            max_new_tokens=512, temperature=0.3, system=REASONER_SYSTEM
        )
        result = await llm.generate(prompt, params=params)
        if on_event is not None:
            on_event(
                "reason",
                {
                    "model_id": result.model_id,
                    "preview": result.text[:200],
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "latency_ms": result.latency_ms,
                    "context_chunks": len(merged),
                },
            )
        log.info(
            "reasoner_done",
            extra={
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
            },
        )
        return {"answer": result.text}
    return reason


def make_reporter_node(
    *,
    on_event: Callable[[str, dict], None] | None = None,
) -> Callable:
    async def report(state: AgentState) -> dict:
        answer = state.get("answer", "")
        merged = state.get("merged_results", [])
        if not merged:
            final = answer
        else:
            sources = "\n".join(
                f"[{i+1}] {r.chunk.metadata.get('doc_title', r.chunk.doc_id[:8])}"
                for i, r in enumerate(merged)
            )
            final = f"{answer}\n\n---\nSources:\n{sources}"
        if on_event is not None:
            on_event(
                "report",
                {
                    "answer_chars": len(final),
                    "sources_appended": bool(merged),
                },
            )
        return {"final_report": final}
    return report

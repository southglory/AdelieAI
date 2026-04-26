import time

from core.agent.graph import build_agent_graph
from core.agent.runner import SessionNotRunnable
from core.logging import get_logger
from core.retrieval.protocols import Retriever
from core.schemas.agent import AgentSession, EventType, SessionState
from core.serving.protocols import LLMClient
from core.session.events import build_event
from core.session.protocols import SessionStore
from core.session.store_memory import SessionNotFound

log = get_logger("differentia.agent.agentic")


_NODE_EVENT_TYPE = {
    "plan": EventType.LLM_CALL,
    "retrieve": EventType.RETRIEVAL,
    "retrieve_skip": EventType.RETRIEVAL,
    "reason": EventType.LLM_CALL,
    "report": EventType.TOOL_CALL,
}


async def run_agentic_session(
    store: SessionStore,
    llm: LLMClient,
    session_id: str,
    user_id: str,
    *,
    retriever: Retriever | None = None,
    retrieval_k: int = 5,
) -> AgentSession:
    """Run the LangGraph 4-node agentic flow against an existing session.

    Per-node events are appended as they happen so the UI's events
    timeline shows each step (plan / retrieve / reason / report).
    """
    session = await store.get(session_id, user_id)
    if session is None:
        raise SessionNotFound(session_id)
    if session.state != SessionState.PENDING:
        raise SessionNotRunnable(
            f"session is {session.state.value}, expected pending"
        )

    await store.transition(session_id, user_id, SessionState.RUNNING)
    log.info(
        "session_started",
        extra={
            "session_id": session_id,
            "user_id": user_id,
            "mode": "agentic",
            "retrieval_k": retrieval_k,
        },
    )

    pending_events: list[tuple[str, dict]] = []

    def collect(node: str, payload: dict) -> None:
        pending_events.append((node, payload))

    async def flush() -> None:
        while pending_events:
            node, payload = pending_events.pop(0)
            event_type = _NODE_EVENT_TYPE.get(node, EventType.TOOL_CALL)
            tokens_in = int(payload.get("tokens_in") or 0)
            tokens_out = int(payload.get("tokens_out") or 0)
            latency_ms = int(payload.get("latency_ms") or 0)
            await store.append_event(
                build_event(
                    session_id=session_id,
                    event_type=event_type,
                    payload={"node": node, **payload},
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_ms=latency_ms,
                )
            )

    graph = build_agent_graph(llm=llm, retriever=retriever, on_event=collect)

    t0 = time.perf_counter()
    try:
        final_state = await graph.ainvoke(
            {"goal": session.goal, "retrieval_k": retrieval_k}
        )
    except Exception as e:
        log.exception(
            "session_failed",
            extra={
                "session_id": session_id,
                "user_id": user_id,
                "error_type": type(e).__name__,
            },
        )
        await flush()
        await store.append_event(
            build_event(
                session_id=session_id,
                event_type=EventType.ERROR,
                payload={"error": str(e), "type": type(e).__name__},
            )
        )
        await store.transition(session_id, user_id, SessionState.FAILED)
        raise

    await flush()
    total_latency = int((time.perf_counter() - t0) * 1000)

    final_text = final_state.get("final_report") or final_state.get("answer", "")
    plan = final_state.get("plan")

    await store.append_event(
        build_event(
            session_id=session_id,
            event_type=EventType.FINAL,
            payload={
                "answer": final_text,
                "model_id": llm.model_id,
                "mode": "agentic",
                "plan": plan.model_dump() if plan is not None else None,
                "total_latency_ms": total_latency,
            },
        )
    )
    await store.transition(session_id, user_id, SessionState.COMPLETED)
    log.info(
        "session_completed",
        extra={
            "session_id": session_id,
            "user_id": user_id,
            "mode": "agentic",
            "total_latency_ms": total_latency,
        },
    )
    return await store.get(session_id, user_id)

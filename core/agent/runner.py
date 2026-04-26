import time
from typing import AsyncIterator

from core.agent.rag import format_context, retrieval_event_payload
from core.logging import get_logger
from core.retrieval.protocols import Retriever
from core.schemas.agent import AgentSession, EventType, SessionState
from core.serving.protocols import (
    GenerationParams,
    LLMClient,
    StreamEvent,
)
from core.session.events import build_event
from core.session.protocols import SessionStore
from core.session.store_memory import SessionNotFound

log = get_logger("differentia.agent")


class SessionNotRunnable(Exception):
    pass


async def _maybe_retrieve(
    *,
    store: SessionStore,
    retriever: Retriever | None,
    session_id: str,
    goal: str,
    k: int,
) -> str:
    """If a retriever and k>0 are present, run retrieval and persist a
    RETRIEVAL event. Returns the prompt to send to the LLM (augmented
    with context, or plain goal if no retrieval ran).
    """
    if retriever is None or k <= 0:
        return goal
    ctx = await retriever.retrieve(goal, k=k)
    await store.append_event(
        build_event(
            session_id=session_id,
            event_type=EventType.RETRIEVAL,
            payload=retrieval_event_payload(ctx),
        )
    )
    log.info(
        "retrieval_done",
        extra={
            "session_id": session_id,
            "method": ctx.method,
            "hits": len(ctx.results),
            "k": k,
        },
    )
    return format_context(goal, ctx)


async def run_session(
    store: SessionStore,
    llm: LLMClient,
    session_id: str,
    user_id: str,
    *,
    params: GenerationParams | None = None,
    retriever: Retriever | None = None,
) -> AgentSession:
    params = params or GenerationParams()
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
            "mode": "sync",
            "retrieval_k": params.retrieval_k,
        },
    )

    try:
        prompt = await _maybe_retrieve(
            store=store,
            retriever=retriever,
            session_id=session_id,
            goal=session.goal,
            k=params.retrieval_k,
        )
        result = await llm.generate(prompt, params=params)
    except Exception as e:
        log.exception(
            "session_failed",
            extra={
                "session_id": session_id,
                "user_id": user_id,
                "error_type": type(e).__name__,
            },
        )
        await store.append_event(
            build_event(
                session_id=session_id,
                event_type=EventType.ERROR,
                payload={"error": str(e), "type": type(e).__name__},
            )
        )
        await store.transition(session_id, user_id, SessionState.FAILED)
        raise

    log.info(
        "session_completed",
        extra={
            "session_id": session_id,
            "user_id": user_id,
            "mode": "sync",
            "model_id": result.model_id,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "latency_ms": result.latency_ms,
        },
    )

    await store.append_event(
        build_event(
            session_id=session_id,
            event_type=EventType.LLM_CALL,
            payload={
                "model_id": result.model_id,
                "preview": result.text[:200],
                "params": result.params.model_dump(),
            },
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            latency_ms=result.latency_ms,
        )
    )
    await store.append_event(
        build_event(
            session_id=session_id,
            event_type=EventType.FINAL,
            payload={
                "answer": result.text,
                "model_id": result.model_id,
                "params": result.params.model_dump(),
            },
        )
    )
    return await store.transition(session_id, user_id, SessionState.COMPLETED)


async def stream_session(
    store: SessionStore,
    llm: LLMClient,
    session_id: str,
    user_id: str,
    *,
    params: GenerationParams | None = None,
    retriever: Retriever | None = None,
) -> AsyncIterator[StreamEvent]:
    params = params or GenerationParams()
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
            "mode": "stream",
            "retrieval_k": params.retrieval_k,
        },
    )

    chunks: list[str] = []
    t0 = time.perf_counter()
    tokens_in = 0
    tokens_out = 0

    try:
        prompt = await _maybe_retrieve(
            store=store,
            retriever=retriever,
            session_id=session_id,
            goal=session.goal,
            k=params.retrieval_k,
        )
        async for ev in llm.astream(prompt, params=params):
            if ev.type == "chunk" and ev.text:
                chunks.append(ev.text)
                yield ev
            elif ev.type == "done":
                tokens_in = ev.tokens_in or 0
                tokens_out = ev.tokens_out or 0
    except Exception as e:
        log.exception(
            "session_failed",
            extra={
                "session_id": session_id,
                "user_id": user_id,
                "mode": "stream",
                "error_type": type(e).__name__,
            },
        )
        await store.append_event(
            build_event(
                session_id=session_id,
                event_type=EventType.ERROR,
                payload={"error": str(e), "type": type(e).__name__},
            )
        )
        await store.transition(session_id, user_id, SessionState.FAILED)
        yield StreamEvent(type="error", error=str(e))
        return

    text = "".join(chunks)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    await store.append_event(
        build_event(
            session_id=session_id,
            event_type=EventType.LLM_CALL,
            payload={
                "model_id": llm.model_id,
                "preview": text[:200],
                "params": params.model_dump(),
                "streamed": True,
            },
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
    )
    await store.append_event(
        build_event(
            session_id=session_id,
            event_type=EventType.FINAL,
            payload={
                "answer": text,
                "model_id": llm.model_id,
                "params": params.model_dump(),
            },
        )
    )
    await store.transition(session_id, user_id, SessionState.COMPLETED)
    log.info(
        "session_completed",
        extra={
            "session_id": session_id,
            "user_id": user_id,
            "mode": "stream",
            "model_id": llm.model_id,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
        },
    )

    yield StreamEvent(
        type="done",
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
    )

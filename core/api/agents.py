import json

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.agent.agentic_runner import run_agentic_session
from core.agent.runner import SessionNotRunnable, run_session, stream_session
from core.retrieval.protocols import Retriever
from core.schemas.agent import AgentEvent, AgentSession, SessionState
from core.serving.protocols import GenerationParams, LLMClient
from core.session.protocols import SessionStore
from core.session.state_machine import InvalidTransition
from core.session.store_memory import SessionNotFound


class CreateSessionRequest(BaseModel):
    goal: str = Field(min_length=1)
    model_spec: str = Field(min_length=1)


class TransitionRequest(BaseModel):
    to: SessionState


def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-Id header required",
        )
    return x_user_id


def build_router(
    store: SessionStore,
    llm: LLMClient,
    retriever: Retriever | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/agents", tags=["agents"])

    @router.post(
        "/sessions",
        response_model=AgentSession,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session(
        body: CreateSessionRequest,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> AgentSession:
        user_id = _require_user(x_user_id)
        return await store.create(user_id, body.goal, body.model_spec)

    @router.get("/sessions", response_model=list[AgentSession])
    async def list_sessions(
        limit: int = 50,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> list[AgentSession]:
        user_id = _require_user(x_user_id)
        return await store.list_sessions(user_id, limit=limit)

    @router.get("/sessions/{session_id}", response_model=AgentSession)
    async def get_session(
        session_id: str,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> AgentSession:
        user_id = _require_user(x_user_id)
        session = await store.get(session_id, user_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return session

    @router.post("/sessions/{session_id}/transition", response_model=AgentSession)
    async def transition_session(
        session_id: str,
        body: TransitionRequest,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> AgentSession:
        user_id = _require_user(x_user_id)
        try:
            return await store.transition(session_id, user_id, body.to)
        except SessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        except InvalidTransition as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(e)
            )

    @router.post("/sessions/{session_id}/run", response_model=AgentSession)
    async def run_session_ep(
        session_id: str,
        params: GenerationParams | None = None,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> AgentSession:
        user_id = _require_user(x_user_id)
        try:
            return await run_session(
                store, llm, session_id, user_id, params=params, retriever=retriever
            )
        except SessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        except SessionNotRunnable as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(e)
            )

    @router.post("/sessions/{session_id}/run/agentic", response_model=AgentSession)
    async def run_agentic_ep(
        session_id: str,
        retrieval_k: int = 5,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> AgentSession:
        user_id = _require_user(x_user_id)
        try:
            return await run_agentic_session(
                store, llm, session_id, user_id,
                retriever=retriever, retrieval_k=retrieval_k,
            )
        except SessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        except SessionNotRunnable as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(e)
            )

    @router.post("/sessions/{session_id}/run/stream")
    async def run_session_stream(
        session_id: str,
        params: GenerationParams | None = None,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> StreamingResponse:
        user_id = _require_user(x_user_id)

        async def event_source():
            try:
                async for ev in stream_session(
                    store, llm, session_id, user_id, params=params, retriever=retriever
                ):
                    yield f"data: {ev.model_dump_json()}\n\n"
            except SessionNotFound:
                yield f"data: {json.dumps({'type':'error','error':'session not found'})}\n\n"
            except SessionNotRunnable as e:
                yield f"data: {json.dumps({'type':'error','error':str(e)})}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @router.get("/sessions/{session_id}/events", response_model=list[AgentEvent])
    async def get_events(
        session_id: str,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> list[AgentEvent]:
        user_id = _require_user(x_user_id)
        session = await store.get(session_id, user_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return await store.events(session_id, user_id)

    @router.delete("/sessions/{session_id}", response_model=AgentSession)
    async def delete_session(
        session_id: str,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> AgentSession:
        user_id = _require_user(x_user_id)
        try:
            return await store.soft_delete(session_id, user_id)
        except SessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        except InvalidTransition as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=str(e)
            )

    return router

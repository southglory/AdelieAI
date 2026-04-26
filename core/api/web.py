import json
from pathlib import Path

from fastapi import APIRouter, Cookie, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from core.agent.agentic_runner import run_agentic_session
from core.agent.runner import SessionNotRunnable, run_session, stream_session
from core.retrieval.protocols import Retriever
from core.schemas.agent import SessionState
from core.serving.protocols import GenerationParams, LLMClient
from core.session.protocols import SessionStore
from core.session.state_machine import InvalidTransition
from core.session.store_memory import SessionNotFound

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

DEFAULT_USER = "demo"


def _user(user_id: str | None) -> str:
    return user_id or DEFAULT_USER


def build_web_router(
    store: SessionStore,
    llm: LLMClient,
    retriever: Retriever | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/web", tags=["web"])

    @router.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/web/sessions")

    @router.get("/sessions", response_class=HTMLResponse, include_in_schema=False)
    async def sessions_page(
        request: Request,
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        sessions = await store.list_sessions(uid)
        return templates.TemplateResponse(
            request,
            "sessions/list.html",
            {"sessions": sessions, "user_id": uid},
        )

    @router.post("/sessions", response_class=HTMLResponse, include_in_schema=False)
    async def sessions_create(
        request: Request,
        goal: str = Form(...),
        model_spec: str = Form(...),
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        session = await store.create(uid, goal, model_spec)
        return templates.TemplateResponse(
            request, "sessions/_row.html", {"s": session, "user_id": uid}
        )

    @router.post(
        "/sessions/{sid}/transition",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def sessions_transition(
        request: Request,
        sid: str,
        to: str = Form(...),
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        try:
            session = await store.transition(sid, uid, SessionState(to))
        except SessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        except InvalidTransition as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        return templates.TemplateResponse(
            request, "sessions/_row.html", {"s": session, "user_id": uid}
        )

    @router.delete(
        "/sessions/{sid}",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def sessions_cancel(
        request: Request,
        sid: str,
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        try:
            session = await store.soft_delete(sid, uid)
        except SessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        except InvalidTransition as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        return templates.TemplateResponse(
            request, "sessions/_row.html", {"s": session, "user_id": uid}
        )

    @router.get(
        "/sessions/{sid}", response_class=HTMLResponse, include_in_schema=False
    )
    async def session_detail(
        request: Request,
        sid: str,
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        session = await store.get(sid, uid)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        events = await store.events(sid, uid)
        return templates.TemplateResponse(
            request,
            "sessions/detail.html",
            {"s": session, "events": events, "user_id": uid},
        )

    @router.post(
        "/sessions/{sid}/run",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def sessions_run(
        request: Request,
        sid: str,
        system: str | None = Form(default=None),
        max_new_tokens: int = Form(default=256),
        temperature: float = Form(default=0.7),
        top_p: float = Form(default=0.9),
        top_k: int = Form(default=50),
        retrieval_k: int = Form(default=0),
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            system=system or None,
            retrieval_k=retrieval_k,
        )
        try:
            session = await run_session(
                store, llm, sid, uid, params=params, retriever=retriever
            )
        except SessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        except SessionNotRunnable as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        return templates.TemplateResponse(
            request, "sessions/_row.html", {"s": session, "user_id": uid}
        )

    @router.post(
        "/sessions/{sid}/run/agentic",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def sessions_run_agentic(
        request: Request,
        sid: str,
        retrieval_k: int = Form(default=5),
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        try:
            session = await run_agentic_session(
                store, llm, sid, uid,
                retriever=retriever, retrieval_k=retrieval_k,
            )
        except SessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        except SessionNotRunnable as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
        return templates.TemplateResponse(
            request, "sessions/_row.html", {"s": session, "user_id": uid}
        )

    @router.post(
        "/sessions/{sid}/run/stream",
        include_in_schema=False,
    )
    async def sessions_run_stream(
        sid: str,
        system: str | None = Form(default=None),
        max_new_tokens: int = Form(default=256),
        temperature: float = Form(default=0.7),
        top_p: float = Form(default=0.9),
        top_k: int = Form(default=50),
        retrieval_k: int = Form(default=0),
        user_id: str | None = Cookie(default=None),
    ) -> StreamingResponse:
        uid = _user(user_id)
        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            system=system or None,
            retrieval_k=retrieval_k,
        )

        async def event_source():
            try:
                async for ev in stream_session(
                    store, llm, sid, uid, params=params, retriever=retriever
                ):
                    yield f"data: {ev.model_dump_json()}\n\n"
            except SessionNotFound:
                yield f"data: {json.dumps({'type':'error','error':'session not found'})}\n\n"
            except SessionNotRunnable as e:
                yield f"data: {json.dumps({'type':'error','error':str(e)})}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @router.get(
        "/sessions/{sid}/events",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def session_events_partial(
        request: Request,
        sid: str,
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        events = await store.events(sid, uid)
        return templates.TemplateResponse(
            request, "events/_list.html", {"events": events}
        )

    @router.post("/whoami", include_in_schema=False)
    async def whoami(user_id: str = Form(...)) -> RedirectResponse:
        resp = RedirectResponse(
            url="/web/sessions", status_code=status.HTTP_303_SEE_OTHER
        )
        resp.set_cookie("user_id", user_id, httponly=True, samesite="lax")
        return resp

    return router

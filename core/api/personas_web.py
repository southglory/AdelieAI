"""HTMX-driven web UI for the persona engine.

Routes:
- GET  /web/personas               — gallery
- GET  /web/chat/{persona_id}      — chat thread for one persona
- POST /web/chat/{persona_id}/messages
- POST /web/chat/{persona_id}/reset
"""

from pathlib import Path

from fastapi import APIRouter, Cookie, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core.personas.chat import submit_chat_turn
from core.personas.grounding import build_grounding_context
from core.personas.registry import get_persona, list_personas
from core.personas.store import ChatStore
from core.serving.protocols import GenerationParams, LLMClient

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

DEFAULT_USER = "demo"


def _user(user_id: str | None) -> str:
    return user_id or DEFAULT_USER


def build_personas_web_router(
    chat_store: ChatStore,
    llm: LLMClient,
) -> APIRouter:
    router = APIRouter(prefix="/web", tags=["personas-web"])

    @router.get(
        "/personas",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def personas_page(
        request: Request,
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        personas = list_personas()
        counts = {
            p.persona_id: await chat_store.turn_count(p.persona_id, uid)
            for p in personas
        }
        return templates.TemplateResponse(
            request,
            "personas/list.html",
            {"personas": personas, "counts": counts, "user_id": uid},
        )

    @router.get(
        "/chat/{persona_id}",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def chat_thread(
        request: Request,
        persona_id: str,
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        persona = get_persona(persona_id)
        if persona is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        turns = await chat_store.list_turns(persona_id, uid)
        return templates.TemplateResponse(
            request,
            "chat/thread.html",
            {
                "persona": persona,
                "turns": turns,
                "user_id": uid,
            },
        )

    @router.post(
        "/chat/{persona_id}/messages",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def chat_submit(
        request: Request,
        persona_id: str,
        message: str = Form(...),
        max_new_tokens: int = Form(default=256),
        temperature: float = Form(default=0.7),
        top_p: float = Form(default=0.9),
        top_k: int = Form(default=50),
        user_id: str | None = Cookie(default=None),
    ):
        uid = _user(user_id)
        persona = get_persona(persona_id)
        if persona is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        if not message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="message is empty",
            )
        params = GenerationParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        # Per-turn grounding: knowledge personas get KG triples for
        # the speaker, legal personas get evidence_search hits keyed
        # off the user's message. Avoids hallucinated lore without
        # requiring full LLM tool-calling wiring.
        grounding = build_grounding_context(
            persona,
            user_text=message,
            graph_retriever=getattr(request.app.state, "graph_retriever", None),
            tool_registry=getattr(request.app.state, "tool_registry", None),
        )
        user_turn, assistant_turn = await submit_chat_turn(
            chat_store=chat_store,
            llm=llm,
            persona=persona,
            user_id=uid,
            user_text=message,
            params=params,
            grounding_context=grounding,
        )
        return templates.TemplateResponse(
            request,
            "chat/_pair.html",
            {
                "persona": persona,
                "user_turn": user_turn,
                "assistant_turn": assistant_turn,
            },
        )

    @router.post(
        "/chat/{persona_id}/reset",
        include_in_schema=False,
    )
    async def chat_reset(
        persona_id: str,
        user_id: str | None = Cookie(default=None),
    ) -> RedirectResponse:
        uid = _user(user_id)
        persona = get_persona(persona_id)
        if persona is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        await chat_store.reset(persona_id, uid)
        return RedirectResponse(
            url=f"/web/chat/{persona_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    @router.post(
        "/chat/{persona_id}/turns/{turn_id}/rate",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def chat_rate(
        request: Request,
        persona_id: str,
        turn_id: int,
        rating: int = Form(...),
    ):
        persona = get_persona(persona_id)
        if persona is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        # Toggle: clicking the same rating again clears it.
        if not (1 <= rating <= 5):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="rating must be 1..5",
            )
        existing = await chat_store.rate(turn_id, rating)
        if existing is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="turn not found",
            )
        # Return the assistant turn fragment with updated rating widget.
        return templates.TemplateResponse(
            request,
            "chat/_turn_assistant.html",
            {"persona": persona, "t": existing},
        )

    return router

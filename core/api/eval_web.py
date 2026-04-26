from fastapi import APIRouter, Cookie, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core.api.web import TEMPLATES_DIR
from core.eval.runner import evaluate_session
from core.serving.protocols import LLMClient
from core.session.protocols import SessionStore
from core.session.store_memory import SessionNotFound

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def build_eval_web_router(store: SessionStore, llm: LLMClient) -> APIRouter:
    router = APIRouter(prefix="/web", tags=["web"])

    @router.post(
        "/sessions/{sid}/evaluate",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def evaluate_session_web(
        request: Request,
        sid: str,
        user_id: str | None = Cookie(default=None),
    ):
        uid = user_id or "demo"
        try:
            result = await evaluate_session(
                store=store, llm=llm, session_id=sid, user_id=uid,
            )
        except SessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return templates.TemplateResponse(
            request, "sessions/_eval.html", {"r": result}
        )

    return router

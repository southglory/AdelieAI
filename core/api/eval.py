from fastapi import APIRouter, Header, HTTPException, status

from core.eval.runner import evaluate_session
from core.schemas.eval import EvalResult
from core.serving.protocols import LLMClient
from core.session.protocols import SessionStore
from core.session.store_memory import SessionNotFound


def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-Id header required",
        )
    return x_user_id


def build_eval_router(store: SessionStore, llm: LLMClient) -> APIRouter:
    router = APIRouter(prefix="/api/v1/eval", tags=["eval"])

    @router.post("/{session_id}", response_model=EvalResult)
    async def evaluate_one(
        session_id: str,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> EvalResult:
        user_id = _require_user(x_user_id)
        try:
            return await evaluate_session(
                store=store,
                llm=llm,
                session_id=session_id,
                user_id=user_id,
            )
        except SessionNotFound:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    return router

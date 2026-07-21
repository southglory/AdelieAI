"""Visible import surface for portable personas."""

from __future__ import annotations

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core.api.personas_web import DEFAULT_USER, TEMPLATES_DIR, _user
from core.personas.packs import MAX_IMPORT_BYTES, PackValidationError, PersonaImportService


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def build_persona_import_router(service: PersonaImportService) -> APIRouter:
    router = APIRouter(prefix="/web/personas", tags=["persona-import"])

    @router.get("/import", response_class=HTMLResponse, include_in_schema=False)
    async def import_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "personas/import.html",
            {"user_id": _user(request.cookies.get("user_id")), "error": None},
        )

    @router.post("/import", response_class=HTMLResponse, include_in_schema=False)
    async def import_persona(
        request: Request,
        persona_file: UploadFile = File(...),
    ):
        payload = await persona_file.read(MAX_IMPORT_BYTES + 1)
        try:
            loaded = service.install(persona_file.filename or "persona.json", payload)
        except PackValidationError as exc:
            return templates.TemplateResponse(
                request,
                "personas/import.html",
                {
                    "user_id": _user(request.cookies.get("user_id")) or DEFAULT_USER,
                    "error": str(exc),
                },
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        return RedirectResponse(
            url=f"/web/chat/{loaded.persona.persona_id}?imported=1",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    return router


__all__ = ["build_persona_import_router"]

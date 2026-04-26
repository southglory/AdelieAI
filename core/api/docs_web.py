from fastapi import APIRouter, Cookie, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from core.api.web import TEMPLATES_DIR
from core.retrieval.ingest import DenseRetriever, IngestService

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def build_docs_web_router(
    ingest: IngestService | None,
    retriever: DenseRetriever | None,
) -> APIRouter:
    router = APIRouter(prefix="/web/docs", tags=["web"])

    def _require_ingest() -> IngestService:
        if ingest is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="retrieval not available — embedder failed to load",
            )
        return ingest

    @router.get(
        "", response_class=HTMLResponse, include_in_schema=False
    )
    async def docs_page(
        request: Request,
        user_id: str | None = Cookie(default=None),
    ):
        if ingest is None:
            return templates.TemplateResponse(
                request,
                "docs/unavailable.html",
                {"user_id": user_id or "demo"},
            )
        docs = await ingest.doc_store.list_docs()
        return templates.TemplateResponse(
            request,
            "docs/list.html",
            {"docs": docs, "user_id": user_id or "demo"},
        )

    @router.post(
        "", response_class=HTMLResponse, include_in_schema=False
    )
    async def ingest_form(
        request: Request,
        title: str = Form(..., min_length=1),
        source: str = Form(default="manual"),
        content: str = Form(..., min_length=1),
        user_id: str | None = Cookie(default=None),
    ):
        svc = _require_ingest()
        doc, chunks = await svc.ingest(
            title=title, source=source, content=content
        )
        return templates.TemplateResponse(
            request,
            "docs/_row.html",
            {"d": doc, "chunk_count": len(chunks)},
        )

    @router.get(
        "/{doc_id}", response_class=HTMLResponse, include_in_schema=False
    )
    async def doc_detail(
        request: Request,
        doc_id: str,
        user_id: str | None = Cookie(default=None),
    ):
        svc = _require_ingest()
        doc = await svc.doc_store.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        chunks = await svc.doc_store.list_chunks(doc_id)
        return templates.TemplateResponse(
            request,
            "docs/detail.html",
            {"d": doc, "chunks": chunks, "user_id": user_id or "demo"},
        )

    @router.delete(
        "/{doc_id}", response_class=HTMLResponse, include_in_schema=False
    )
    async def delete_doc(
        doc_id: str,
        user_id: str | None = Cookie(default=None),
    ) -> HTMLResponse:
        svc = _require_ingest()
        await svc.delete(doc_id)
        return HTMLResponse("")

    @router.post(
        "/search", response_class=HTMLResponse, include_in_schema=False
    )
    async def search_form(
        request: Request,
        query: str = Form(..., min_length=1),
        k: int = Form(default=5),
        user_id: str | None = Cookie(default=None),
    ):
        if retriever is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="retrieval not available",
            )
        ctx = await retriever.retrieve(query, k=k)
        return templates.TemplateResponse(
            request, "docs/_search_results.html", {"ctx": ctx}
        )

    return router

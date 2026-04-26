from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from core.retrieval.ingest import DenseRetriever, IngestService
from core.schemas.retrieval import (
    Chunk,
    Document,
    RetrievedContext,
)


class IngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=512)
    source: str = Field(default="manual", max_length=512)
    content: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document: Document
    chunk_count: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=50)


def _require_user(x_user_id: str | None) -> str:
    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-Id header required",
        )
    return x_user_id


def build_docs_router(
    ingest: IngestService | None,
    retriever: DenseRetriever | None,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/docs", tags=["docs"])

    def _require_ingest() -> IngestService:
        if ingest is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "retrieval not available — embedder failed to load "
                    "(check EMBEDDING_MODEL_PATH and models/upstream/)"
                ),
            )
        return ingest

    def _require_retriever() -> DenseRetriever:
        if retriever is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="retrieval not available",
            )
        return retriever

    @router.post(
        "",
        response_model=IngestResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def ingest_doc(
        body: IngestRequest,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> IngestResponse:
        _require_user(x_user_id)
        svc = _require_ingest()
        doc, chunks = await svc.ingest(
            title=body.title,
            source=body.source,
            content=body.content,
            metadata=body.metadata,
        )
        return IngestResponse(document=doc, chunk_count=len(chunks))

    @router.get("", response_model=list[Document])
    async def list_docs(
        limit: int = 100,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> list[Document]:
        _require_user(x_user_id)
        svc = _require_ingest()
        return await svc.doc_store.list_docs(limit=limit)

    @router.get("/{doc_id}", response_model=Document)
    async def get_doc(
        doc_id: str,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> Document:
        _require_user(x_user_id)
        svc = _require_ingest()
        doc = await svc.doc_store.get(doc_id)
        if doc is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        return doc

    @router.get("/{doc_id}/chunks", response_model=list[Chunk])
    async def get_chunks(
        doc_id: str,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> list[Chunk]:
        _require_user(x_user_id)
        svc = _require_ingest()
        chunks = await svc.doc_store.list_chunks(doc_id)
        return chunks

    @router.delete(
        "/{doc_id}", status_code=status.HTTP_204_NO_CONTENT
    )
    async def delete_doc(
        doc_id: str,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> None:
        _require_user(x_user_id)
        svc = _require_ingest()
        await svc.delete(doc_id)

    @router.post("/search", response_model=RetrievedContext)
    async def search(
        body: SearchRequest,
        x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    ) -> RetrievedContext:
        _require_user(x_user_id)
        r = _require_retriever()
        return await r.retrieve(body.query, k=body.k)

    return router

from core.logging import get_logger
from core.retrieval.document_store import build_document
from core.retrieval.protocols import (
    BM25Index,
    Chunker,
    DocumentStore,
    Embedder,
    VectorStore,
)
from core.schemas.retrieval import (
    Chunk,
    Document,
    RetrievedChunk,
    RetrievedContext,
)

log = get_logger("differentia.retrieval")


class IngestService:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        doc_store: DocumentStore,
        vector_store: VectorStore,
        *,
        bm25: BM25Index | None = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.bm25 = bm25

    async def ingest(
        self,
        *,
        title: str,
        source: str,
        content: str,
        metadata: dict | None = None,
    ) -> tuple[Document, list[Chunk]]:
        doc = build_document(
            title=title,
            source=source,
            content=content,
            metadata=metadata or {},
        )
        raw_chunks = self.chunker.split(doc)
        if not raw_chunks:
            await self.doc_store.add(doc, [])
            log.info(
                "ingest_empty",
                extra={"doc_id": doc.id, "title": title, "chunks": 0},
            )
            return doc, []

        vectors = await self.embedder.embed_passages([c.text for c in raw_chunks])
        embedded = [
            Chunk(
                id=c.id,
                doc_id=c.doc_id,
                position=c.position,
                text=c.text,
                metadata=c.metadata,
                embedding=v,
            )
            for c, v in zip(raw_chunks, vectors)
        ]
        await self.doc_store.add(doc, embedded)
        await self.vector_store.upsert(embedded)
        if self.bm25 is not None:
            await self.bm25.add(embedded)
        log.info(
            "ingest_done",
            extra={
                "doc_id": doc.id,
                "title": title,
                "chunks": len(embedded),
                "embed_model": getattr(self.embedder, "model_id", "?"),
            },
        )
        return doc, embedded

    async def delete(self, doc_id: str) -> None:
        await self.doc_store.delete(doc_id)
        await self.vector_store.delete_by_doc(doc_id)
        if self.bm25 is not None:
            await self.bm25.remove_by_doc(doc_id)
        log.info("ingest_deleted", extra={"doc_id": doc_id})

    async def warm_bm25_from_doc_store(self) -> int:
        """Rebuild the in-memory BM25 index from the persistent document
        store — call once on app startup since BM25 has no on-disk form.
        """
        if self.bm25 is None:
            return 0
        all_chunks = []
        if hasattr(self.doc_store, "all_chunks"):
            all_chunks = await self.doc_store.all_chunks()
        if all_chunks:
            await self.bm25.add(all_chunks)
        log.info("bm25_warmed", extra={"chunks": len(all_chunks)})
        return len(all_chunks)


class DenseRetriever:
    """Single-channel dense retriever — vector search only. Hybrid +
    reranker variants live in Phase 2.2.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        doc_store: DocumentStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.doc_store = doc_store

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None,
    ) -> RetrievedContext:
        if not query.strip():
            return RetrievedContext(method="dense", results=[], query=query)
        vec = await self.embedder.embed_query(query)
        hits = await self.vector_store.search(vec, k, filters=filters)
        if not hits:
            return RetrievedContext(method="dense", results=[], query=query)

        chunk_ids = [h.chunk.id for h in hits]
        full = {c.id: c for c in await self.doc_store.get_chunks(chunk_ids)}
        results = [
            RetrievedChunk(chunk=full.get(h.chunk.id, h.chunk), score=h.score)
            for h in hits
        ]
        return RetrievedContext(method="dense", results=results, query=query)

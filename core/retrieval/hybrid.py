import asyncio
from typing import Literal

from core.retrieval.protocols import (
    BM25Index,
    DocumentStore,
    Embedder,
    Reranker,
    VectorStore,
)
from core.schemas.retrieval import Chunk, RetrievedChunk, RetrievedContext


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]],
    k: int,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Standard Reciprocal Rank Fusion.

    score(c) = Σ_r 1 / (rrf_k + rank_r(c))
    where rank_r(c) is the (1-indexed) rank of chunk c in ranking r.
    rrf_k = 60 is the canonical default from Cormack et al., 2009.
    """
    fused: dict[str, float] = {}
    representatives: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            if cid not in representatives:
                representatives[cid] = hit
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [
        RetrievedChunk(chunk=representatives[cid].chunk, score=score)
        for cid, score in ordered[:k]
    ]


class HybridRetriever:
    """Dense + BM25 with Reciprocal Rank Fusion, optional cross-encoder
    rerank on the fused top-N. Standard 2024 RAG retrieval pipeline.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25: BM25Index,
        doc_store: DocumentStore,
        *,
        reranker: Reranker | None = None,
        candidate_pool: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.doc_store = doc_store
        self.reranker = reranker
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None,
    ) -> RetrievedContext:
        if not query.strip():
            return RetrievedContext(method="hybrid", results=[], query=query)

        pool = max(self.candidate_pool, k)
        dense_task = self._dense_search(query, pool, filters)
        bm25_task = self.bm25.search(query, pool, filters=filters)
        dense_hits, bm25_hits = await asyncio.gather(dense_task, bm25_task)

        fused = reciprocal_rank_fusion(
            [dense_hits, bm25_hits], k=pool, rrf_k=self.rrf_k
        )
        method: Literal["bm25", "dense", "hybrid", "reranked"] = "hybrid"

        await self._hydrate(fused)

        if self.reranker is not None and fused:
            fused = await self.reranker.rerank(query, fused, top_k=k)
            method = "reranked"
        else:
            fused = fused[:k]

        return RetrievedContext(method=method, results=fused, query=query)

    async def _dense_search(
        self, query: str, k: int, filters: dict | None
    ) -> list[RetrievedChunk]:
        vec = await self.embedder.embed_query(query)
        return await self.vector_store.search(vec, k, filters=filters)

    async def _hydrate(self, hits: list[RetrievedChunk]) -> None:
        """Replace any stub chunks (vector store may strip metadata) with
        the full Chunk loaded from the document store.
        """
        ids = [h.chunk.id for h in hits]
        if not ids:
            return
        full = {c.id: c for c in await self.doc_store.get_chunks(ids)}
        for i, h in enumerate(hits):
            replacement: Chunk | None = full.get(h.chunk.id)
            if replacement is not None:
                hits[i] = RetrievedChunk(chunk=replacement, score=h.score)

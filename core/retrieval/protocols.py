from typing import Protocol, runtime_checkable

from core.schemas.retrieval import Chunk, Document, RetrievedChunk, RetrievedContext


@runtime_checkable
class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dim: int

    async def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, chunks: list[Chunk]) -> None: ...

    async def search(
        self,
        query_vec: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]: ...

    async def delete_by_doc(self, doc_id: str) -> None: ...


@runtime_checkable
class BM25Index(Protocol):
    async def add(self, chunks: list[Chunk]) -> None: ...

    async def search(
        self, query: str, k: int, filters: dict | None = None
    ) -> list[RetrievedChunk]: ...

    async def remove_by_doc(self, doc_id: str) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    model_id: str

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class DocumentStore(Protocol):
    async def add(self, doc: Document, chunks: list[Chunk]) -> None: ...

    async def get(self, doc_id: str) -> Document | None: ...

    async def list_docs(self, limit: int = 100) -> list[Document]: ...

    async def list_chunks(self, doc_id: str) -> list[Chunk]: ...

    async def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]: ...

    async def delete(self, doc_id: str) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    async def retrieve(
        self, query: str, k: int = 5, filters: dict | None = None
    ) -> RetrievedContext: ...

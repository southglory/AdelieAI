import asyncio
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from core.schemas.retrieval import Chunk, RetrievedChunk


class ChromaVectorStore:
    def __init__(
        self,
        persist_dir: str | Path,
        *,
        collection_name: str = "chunks",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _upsert_sync(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        missing = [c.id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(
                f"vectorstore.upsert requires precomputed embeddings; "
                f"missing for {len(missing)} chunks (e.g. {missing[0]})"
            )
        ids = [c.id for c in chunks]
        embeddings = [c.embedding for c in chunks]
        documents = [c.text for c in chunks]
        metadatas: list[dict[str, Any]] = []
        for c in chunks:
            md = {k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))}
            md["doc_id"] = c.doc_id
            md["position"] = c.position
            metadatas.append(md)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def upsert(self, chunks: list[Chunk]) -> None:
        await asyncio.to_thread(self._upsert_sync, chunks)

    def _search_sync(
        self, query_vec: list[float], k: int, filters: dict | None
    ) -> list[RetrievedChunk]:
        where = filters if filters else None
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            where=where,
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        out: list[RetrievedChunk] = []
        for cid, text, md, dist in zip(ids, docs, metadatas, distances):
            md = dict(md or {})
            chunk = Chunk(
                id=cid,
                doc_id=md.pop("doc_id", ""),
                position=int(md.pop("position", 0)),
                text=text or "",
                metadata=md,
            )
            score = 1.0 - float(dist)
            out.append(RetrievedChunk(chunk=chunk, score=score))
        return out

    async def search(
        self,
        query_vec: list[float],
        k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._search_sync, query_vec, k, filters)

    def _delete_by_doc_sync(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    async def delete_by_doc(self, doc_id: str) -> None:
        await asyncio.to_thread(self._delete_by_doc_sync, doc_id)

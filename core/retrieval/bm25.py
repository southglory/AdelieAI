import asyncio
import re
from threading import RLock

from rank_bm25 import BM25Okapi

from core.schemas.retrieval import Chunk, RetrievedChunk

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Whitespace + word-character tokenizer.

    Note: this is a baseline that works for Latin scripts and decent on
    Korean character-level. Production-grade Korean BM25 should swap to
    konlpy / mecab-ko, and English to a sklearn-style analyzer with
    stemming. We expose the function so callers can override.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class InMemoryBM25:
    """Simple BM25 over an in-memory corpus. Rebuilds the BM25Okapi
    index lazily on next search after any add/remove. Fine for small
    corpora (<100k chunks); production should swap to Elasticsearch
    or OpenSearch for incremental indexing.
    """

    def __init__(self, tokenizer=tokenize) -> None:
        self._tokenize = tokenizer
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._index: BM25Okapi | None = None
        self._lock = RLock()

    def _invalidate(self) -> None:
        self._index = None

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        if not self._tokens:
            self._index = None
            return
        self._index = BM25Okapi(self._tokens)

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self._lock:
            for c in chunks:
                self._chunks.append(c)
                self._tokens.append(self._tokenize(c.text))
            self._invalidate()

    async def remove_by_doc(self, doc_id: str) -> None:
        with self._lock:
            kept_chunks: list[Chunk] = []
            kept_tokens: list[list[str]] = []
            for c, t in zip(self._chunks, self._tokens):
                if c.doc_id != doc_id:
                    kept_chunks.append(c)
                    kept_tokens.append(t)
            self._chunks = kept_chunks
            self._tokens = kept_tokens
            self._invalidate()

    def _search_sync(
        self, query: str, k: int, filters: dict | None
    ) -> list[RetrievedChunk]:
        with self._lock:
            self._ensure_index()
            if self._index is None or not self._chunks:
                return []
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []
            scores = self._index.get_scores(query_tokens)

            indices = list(range(len(self._chunks)))
            if filters:
                indices = [
                    i
                    for i in indices
                    if all(self._chunks[i].metadata.get(k) == v for k, v in filters.items())
                ]
            indices.sort(key=lambda i: scores[i], reverse=True)
            top = indices[:k]
            results: list[RetrievedChunk] = []
            for i in top:
                if scores[i] <= 0:
                    continue
                results.append(
                    RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]))
                )
            return results

    async def search(
        self, query: str, k: int, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self._search_sync, query, k, filters)

    def size(self) -> int:
        with self._lock:
            return len(self._chunks)

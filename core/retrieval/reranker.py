import asyncio
import json
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder

from core.schemas.retrieval import RetrievedChunk


class CrossEncoderReranker:
    """Cross-encoder rerank — re-scores (query, chunk) pairs jointly.
    Standard pattern used by RAG pipelines: dense+sparse retrieve a
    candidate pool, cross-encoder rescores the pool for final ordering.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        manifest = self.model_path / "MANIFEST.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.model_id = data.get("model_id", str(self.model_path))
        else:
            self.model_id = str(self.model_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CrossEncoder(
            str(self.model_path), device=device, max_length=max_length
        )

    def _rerank_sync(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [[query, c.chunk.text] for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        scored = list(zip(candidates, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(chunk=c.chunk, score=s) for c, s in scored[:top_k]
        ]

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        return await asyncio.to_thread(self._rerank_sync, query, candidates, top_k)

import asyncio
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


class TransformersEmbedder:
    """sentence-transformers wrapper. Reads model_id from MANIFEST.json
    when available. Applies E5-family role prefixes when enabled — the
    multilingual-e5-* models expect 'query:' and 'passage:' prefixes
    and omitting them measurably degrades recall.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        use_e5_prefix: bool | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        manifest = self.model_path / "MANIFEST.json"
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.model_id = data.get("model_id", str(self.model_path))
        else:
            self.model_id = str(self.model_path)

        if use_e5_prefix is None:
            use_e5_prefix = "e5" in self.model_id.lower()
        self._use_e5_prefix = use_e5_prefix

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(str(self.model_path), device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def _prefixed(self, texts: list[str], role: str) -> list[str]:
        if not self._use_e5_prefix:
            return texts
        return [f"{role}: {t}" for t in texts]

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    async def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(
            self._embed_sync, self._prefixed(texts, "passage")
        )

    async def embed_query(self, query: str) -> list[float]:
        vectors = await asyncio.to_thread(
            self._embed_sync, self._prefixed([query], "query")
        )
        return vectors[0]

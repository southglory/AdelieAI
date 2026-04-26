import uuid

from core.schemas.retrieval import Chunk, Document


class RecursiveTextSplitter:
    """Standard recursive text splitter — tries large separators first, falls
    back to smaller ones when a piece is still too long. Same algorithm
    LangChain's RecursiveCharacterTextSplitter uses.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, doc: Document) -> list[Chunk]:
        pieces = self._split_text(doc.content, self.separators)
        merged = self._merge(pieces)
        cleaned = [m.strip() for m in merged if m.strip()]
        return [
            Chunk(
                id=str(uuid.uuid4()),
                doc_id=doc.id,
                position=i,
                text=text,
                metadata={
                    **doc.metadata,
                    "doc_title": doc.title,
                    "doc_source": doc.source,
                },
            )
            for i, text in enumerate(cleaned)
        ]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep, *rest = separators
        if sep == "":
            # Hard split into chunk_size pieces — last resort.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        if sep not in text:
            return self._split_text(text, rest)

        out: list[str] = []
        for piece in text.split(sep):
            if not piece:
                continue
            piece_with_sep = piece + (sep if sep != "\n\n" else "\n\n")
            if len(piece_with_sep) <= self.chunk_size:
                out.append(piece_with_sep)
            else:
                out.extend(self._split_text(piece_with_sep, rest))
        return out

    def _merge(self, pieces: list[str]) -> list[str]:
        merged: list[str] = []
        current = ""
        for piece in pieces:
            if not current:
                current = piece
                continue
            candidate = current + piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                merged.append(current)
                tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = tail + piece
        if current:
            merged.append(current)
        return merged

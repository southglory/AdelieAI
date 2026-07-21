"""Corpus-backed evidence retrieval for the legal persona.

``EvidenceSearch`` keeps the small synchronous Tool contract used by the
grounding layer.  Retrieval itself sits behind ``EvidenceSearchPort`` so a
vector or remote backend can replace the filesystem adapter without changing
the tool name, input schema, or output shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.tools.protocols import Tool


DEFAULT_CORPUS_DIR = (
    Path(__file__).resolve().parents[2]
    / "personas"
    / "cold_detective"
    / "rag_corpus"
)
SUPPORTED_SUFFIXES = frozenset({".md", ".txt"})


class EvidenceCorpusError(RuntimeError):
    """The configured evidence source cannot be searched."""


@dataclass(frozen=True)
class EvidenceHit:
    path: str
    score: float
    snippet: str


@dataclass(frozen=True)
class EvidenceSearchResult:
    hits: tuple[EvidenceHit, ...]
    total_hits: int
    source: dict[str, Any]


class EvidenceSearchPort(Protocol):
    """Replaceable synchronous retrieval boundary used by the Tool facade."""

    backend_name: str

    def search(self, query: str, *, top_k: int) -> EvidenceSearchResult: ...


def _terms(text: str) -> list[str]:
    """Tokenize Latin, numeric, and Korean text without extra dependencies."""
    return [term for term in re.findall(r"[\w가-힣]+", text.lower()) if term]


class FileCorpusEvidenceSearch:
    """Deterministic keyword search over UTF-8 Markdown/text case files."""

    backend_name = "filesystem_keyword"

    def __init__(self, corpus_dir: str | Path = DEFAULT_CORPUS_DIR) -> None:
        self.corpus_dir = Path(corpus_dir)

    def _documents(self) -> list[tuple[str, str]]:
        if not self.corpus_dir.is_dir():
            raise EvidenceCorpusError(
                f"evidence corpus directory not found: {self.corpus_dir}. "
                "Create the directory with UTF-8 .md/.txt case files or pass "
                "EvidenceSearch(corpus_dir=...)."
            )

        paths = sorted(
            path
            for path in self.corpus_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not paths:
            raise EvidenceCorpusError(
                f"evidence corpus contains no supported files: {self.corpus_dir}. "
                "Add at least one UTF-8 .md or .txt case file."
            )

        documents: list[tuple[str, str]] = []
        for path in paths:
            try:
                body = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as exc:
                raise EvidenceCorpusError(
                    f"could not read evidence file {path}: {exc}. "
                    "Ensure the file is readable UTF-8 text."
                ) from exc
            if body:
                documents.append((path.relative_to(self.corpus_dir).as_posix(), body))

        if not documents:
            raise EvidenceCorpusError(
                f"evidence corpus has no non-empty files: {self.corpus_dir}. "
                "Add evidence text before searching."
            )
        return documents

    def search(self, query: str, *, top_k: int) -> EvidenceSearchResult:
        documents = self._documents()
        query_terms = _terms(query)
        ranked: list[EvidenceHit] = []
        for relative_path, body in documents:
            searchable = body.lower()
            term_score = sum(searchable.count(term) for term in query_terms)
            phrase_bonus = (
                2
                if len(query_terms) > 1 and query.lower().strip() in searchable
                else 0
            )
            score = float(term_score + phrase_bonus)
            if score > 0:
                ranked.append(
                    EvidenceHit(path=relative_path, score=score, snippet=body)
                )

        ranked.sort(key=lambda hit: (-hit.score, hit.path))
        return EvidenceSearchResult(
            hits=tuple(ranked[:top_k]),
            total_hits=len(ranked),
            source={
                "type": "directory",
                "path": str(self.corpus_dir),
                "files_indexed": len(documents),
            },
        )


class EvidenceSearch:
    """Tool facade for searching the active case corpus."""

    name = "evidence_search"
    description = (
        "Search the active case file for evidence matching a natural-language query. "
        "Returns a ranked list of file paths with snippets. Use when the persona "
        "needs to ground a deduction in the recorded evidence."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of what to find — keywords, names, times.",
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        backend: EvidenceSearchPort | None = None,
        *,
        corpus_dir: str | Path | None = None,
        top_k: int = 4,
    ) -> None:
        if backend is not None and corpus_dir is not None:
            raise ValueError("pass backend or corpus_dir, not both")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.backend = backend or FileCorpusEvidenceSearch(corpus_dir or DEFAULT_CORPUS_DIR)
        self.top_k = top_k

    def call(self, arguments: dict[str, Any]) -> Any:
        query = arguments.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return {
                "hits": [],
                "error": "query must be a non-empty string",
                "backend": self.backend.backend_name,
                "source": None,
            }

        try:
            result = self.backend.search(query.strip(), top_k=self.top_k)
        except EvidenceCorpusError as exc:
            return {
                "query": query,
                "n_hits": 0,
                "hits": [],
                "error": str(exc),
                "backend": self.backend.backend_name,
                "source": {"type": "unavailable"},
            }

        return {
            "query": query,
            "n_hits": result.total_hits,
            "hits": [
                {"path": hit.path, "score": hit.score, "snippet": hit.snippet}
                for hit in result.hits
            ],
            "backend": self.backend.backend_name,
            "source": result.source,
        }


assert isinstance(EvidenceSearch(), Tool)

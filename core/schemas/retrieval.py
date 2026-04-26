from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    source: str
    content: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: str
    position: int
    text: str
    metadata: dict = Field(default_factory=dict)
    embedding: list[float] | None = None


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


class RetrievedContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["bm25", "dense", "hybrid", "reranked"] = "dense"
    results: list[RetrievedChunk] = Field(default_factory=list)
    query: str = ""

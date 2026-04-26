from typing import AsyncIterator, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class GenerationParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_new_tokens: int = 256
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    system: str | None = None
    retrieval_k: int = Field(default=0, ge=0, le=20)


class GenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    model_id: str
    params: GenerationParams


class StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["chunk", "done", "error"]
    text: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    error: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    model_id: str

    async def generate(
        self, prompt: str, params: GenerationParams | None = None
    ) -> GenerationResult: ...

    def astream(
        self, prompt: str, params: GenerationParams | None = None
    ) -> AsyncIterator[StreamEvent]: ...

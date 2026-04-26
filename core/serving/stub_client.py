import asyncio
import time
from typing import AsyncIterator

from core.serving.protocols import GenerationParams, GenerationResult, StreamEvent


class StubLLMClient:
    model_id = "stub-deterministic-1"

    def _format_text(self, prompt: str, params: GenerationParams) -> str:
        return (
            f"[stub:{self.model_id}] received {len(prompt)} chars"
            + (f" with system={len(params.system)} chars" if params.system else "")
            + f". params: max_new={params.max_new_tokens}"
            f" temp={params.temperature} top_p={params.top_p} top_k={params.top_k}."
        )

    async def generate(
        self, prompt: str, params: GenerationParams | None = None
    ) -> GenerationResult:
        params = params or GenerationParams()
        t0 = time.perf_counter()
        text = self._format_text(prompt, params)
        return GenerationResult(
            text=text,
            tokens_in=len(prompt) // 4,
            tokens_out=len(text) // 4,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            model_id=self.model_id,
            params=params,
        )

    async def astream(
        self, prompt: str, params: GenerationParams | None = None
    ) -> AsyncIterator[StreamEvent]:
        params = params or GenerationParams()
        text = self._format_text(prompt, params)
        t0 = time.perf_counter()
        for ch in text:
            yield StreamEvent(type="chunk", text=ch)
            await asyncio.sleep(0.005)
        yield StreamEvent(
            type="done",
            tokens_in=len(prompt) // 4,
            tokens_out=len(text) // 4,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

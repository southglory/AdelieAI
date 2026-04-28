"""ScriptedLLMClient — queue-based mock LLM for tests with explicit
control over response sequence.

Use this when a test needs *guaranteed* distinct or scripted replies
(e.g., DPO data harvesting flow, voice-comparison fixtures). Stub mode
gives best-effort variety but no guarantees, so tests that assert
specific sequences should use this client.

Decision tree (see also `docs/serving/README.md`):

    test asserts a specific reply  ......  ScriptedLLMClient
    test only needs persona-shaped voice .  StubLLMClient
    integration with a real model  ......  TransformersClient / GGUFClient

Example:
    llm = ScriptedLLMClient([
        "또 왔어? 살 거면 사라.",   # answer to first prompt
        "흠, 뭘 원해? 그냥 비켜.",   # answer to second
    ])
    # 3rd call raises — exhaustion is loud, not silent
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Iterable

from core.serving.protocols import (
    GenerationParams,
    GenerationResult,
    StreamEvent,
)


class ScriptedExhausted(RuntimeError):
    """Raised when the script runs out — surfaces test setup mismatch
    (e.g., test expected N calls but supplied fewer replies)."""


class ScriptedLLMClient:
    """Returns the next scripted reply per `generate()` call.

    Optional `cycle=True` makes the script wrap around instead of
    raising on exhaustion — useful for soak tests, less useful when
    the test wants exact sequence control.
    """

    model_id = "scripted-mock-1"

    def __init__(
        self,
        replies: Iterable[str],
        *,
        cycle: bool = False,
    ) -> None:
        self._replies: list[str] = list(replies)
        if not self._replies:
            raise ValueError("ScriptedLLMClient requires at least one reply")
        self._cursor = 0
        self._cycle = cycle

    def reset(self) -> None:
        """Rewind cursor to 0 — useful for fixtures shared across tests
        that don't want one test's consumption affecting another."""
        self._cursor = 0

    @property
    def remaining(self) -> int:
        if self._cycle:
            return len(self._replies)  # never exhausts
        return max(0, len(self._replies) - self._cursor)

    def _next(self) -> str:
        if self._cursor >= len(self._replies):
            if self._cycle:
                self._cursor = 0
            else:
                raise ScriptedExhausted(
                    f"ScriptedLLMClient exhausted after "
                    f"{len(self._replies)} replies. Test setup likely "
                    f"supplied fewer scripted lines than the test issues."
                )
        text = self._replies[self._cursor]
        self._cursor += 1
        return text

    async def generate(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> GenerationResult:
        params = params or GenerationParams()
        t0 = time.perf_counter()
        text = self._next()
        return GenerationResult(
            text=text,
            tokens_in=len(prompt) // 4,
            tokens_out=len(text) // 4,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            model_id=self.model_id,
            params=params,
        )

    async def astream(
        self,
        prompt: str,
        params: GenerationParams | None = None,
    ) -> AsyncIterator[StreamEvent]:
        params = params or GenerationParams()
        text = self._next()
        t0 = time.perf_counter()
        for ch in text:
            yield StreamEvent(type="chunk", text=ch)
            await asyncio.sleep(0)  # yield to loop without artificial delay
        yield StreamEvent(
            type="done",
            tokens_in=len(prompt) // 4,
            tokens_out=len(text) // 4,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

"""ScriptedLLMClient — queue-based mock for tests with exact sequence control."""
from __future__ import annotations

import pytest

from core.serving.protocols import GenerationParams
from core.serving.scripted_client import ScriptedExhausted, ScriptedLLMClient


async def test_returns_replies_in_order() -> None:
    llm = ScriptedLLMClient(["A", "B", "C"])
    r1 = await llm.generate("anything")
    r2 = await llm.generate("anything")
    r3 = await llm.generate("anything")
    assert (r1.text, r2.text, r3.text) == ("A", "B", "C")


async def test_exhaustion_raises_loud_error() -> None:
    llm = ScriptedLLMClient(["only"])
    await llm.generate("p")
    with pytest.raises(ScriptedExhausted):
        await llm.generate("p")


async def test_cycle_mode_wraps_around() -> None:
    llm = ScriptedLLMClient(["A", "B"], cycle=True)
    seen = [(await llm.generate("p")).text for _ in range(5)]
    assert seen == ["A", "B", "A", "B", "A"]


async def test_reset_rewinds_cursor() -> None:
    llm = ScriptedLLMClient(["A", "B"])
    await llm.generate("p")
    llm.reset()
    second_after_reset = await llm.generate("p")
    assert second_after_reset.text == "A"


async def test_remaining_decrements_per_call() -> None:
    llm = ScriptedLLMClient(["A", "B", "C"])
    assert llm.remaining == 3
    await llm.generate("p")
    assert llm.remaining == 2
    await llm.generate("p")
    assert llm.remaining == 1


async def test_cycle_mode_remaining_constant() -> None:
    llm = ScriptedLLMClient(["A", "B"], cycle=True)
    for _ in range(5):
        await llm.generate("p")
    assert llm.remaining == 2  # never exhausts


def test_empty_replies_rejects_construction() -> None:
    with pytest.raises(ValueError):
        ScriptedLLMClient([])


async def test_passes_through_generation_params() -> None:
    llm = ScriptedLLMClient(["X"])
    params = GenerationParams(max_new_tokens=42, temperature=0.1)
    r = await llm.generate("p", params)
    assert r.params.max_new_tokens == 42
    assert r.params.temperature == pytest.approx(0.1)


async def test_astream_emits_chunks_then_done() -> None:
    llm = ScriptedLLMClient(["abc"])
    events = []
    async for ev in llm.astream("prompt"):
        events.append(ev)
    chunks = [e for e in events if e.type == "chunk"]
    done = [e for e in events if e.type == "done"]
    assert "".join(c.text for c in chunks) == "abc"
    assert len(done) == 1

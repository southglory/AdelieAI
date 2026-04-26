import asyncio
import json
from pathlib import Path

from core.eval.compare import (
    AdapterRunResult,
    ComparisonPrompt,
    ComparisonReport,
    DEFAULT_PROMPTS,
    compare_adapters,
    save_report,
)
from core.serving.protocols import GenerationParams, GenerationResult


class FixedClient:
    """Returns a configured response. For deterministic tests."""

    def __init__(self, model_id: str, response: str) -> None:
        self.model_id = model_id
        self._response = response

    async def generate(self, prompt: str, params: GenerationParams | None = None):
        return GenerationResult(
            text=self._response,
            tokens_in=10,
            tokens_out=len(self._response) // 4,
            latency_ms=50,
            model_id=self.model_id,
            params=params or GenerationParams(),
        )

    async def astream(self, prompt: str, params: GenerationParams | None = None):
        from core.serving.protocols import StreamEvent
        result = await self.generate(prompt, params)
        for ch in result.text:
            yield StreamEvent(type="chunk", text=ch)
        yield StreamEvent(
            type="done",
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            latency_ms=result.latency_ms,
        )


def test_default_prompts_cover_both_kinds() -> None:
    kinds = {p.kind for p in DEFAULT_PROMPTS}
    assert "roleplay" in kinds
    assert "general" in kinds


async def test_compare_adapters_runs_grid() -> None:
    base = FixedClient("base-stub", "base-answer")
    v1 = FixedClient("v1-stub", "v1-answer")
    v2 = FixedClient("v2-stub", "v2-answer")
    judge = FixedClient("judge-stub", "0.85")

    prompts = [
        ComparisonPrompt("p1", "sys", "q1?", "general"),
        ComparisonPrompt("p2", "sys", "q2?", "roleplay"),
    ]
    report = await compare_adapters(
        candidates=[("base", base), ("v1", v1), ("v2", v2)],
        prompts=prompts,
        judge=judge,
    )
    assert len(report.runs) == 6  # 3 candidates × 2 prompts
    assert report.judge_model_id == "judge-stub"
    assert all(r.relevance == 0.85 for r in report.runs)


async def test_save_report_writes_json_and_markdown(tmp_path: Path) -> None:
    base = FixedClient("base", "Hello world.")
    v1 = FixedClient("v1", "Annyeong!")
    judge = FixedClient("judge", "0.9")
    prompts = [ComparisonPrompt("greet", "sys", "say hi", "general")]
    report = await compare_adapters(
        candidates=[("base", base), ("v1", v1)],
        prompts=prompts,
        judge=judge,
    )
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"
    save_report(report, json_path=json_path, markdown_path=md_path)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data["runs"]) == 2
    assert data["judge_model_id"] == "judge"

    md = md_path.read_text(encoding="utf-8")
    assert "Adapter comparison" in md
    assert "base" in md and "v1" in md
    assert "0.90" in md


async def test_judge_optional() -> None:
    base = FixedClient("base", "x")
    report = await compare_adapters(
        candidates=[("base", base)],
        prompts=[ComparisonPrompt("p", "sys", "q?", "general")],
        judge=None,
    )
    assert report.judge_model_id is None
    assert report.runs[0].relevance is None


def test_markdown_handles_empty_runs() -> None:
    report = ComparisonReport(
        started_at="2026-04-26T00:00:00Z",
        judge_model_id=None,
        prompts=[ComparisonPrompt("p", "sys", "q", "general")],
    )
    md = report.to_markdown()
    assert "Adapter comparison" in md

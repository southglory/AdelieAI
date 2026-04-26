"""Adapter comparison harness.

Drives the same prompts through several LLM clients (a base model
plus N LoRA adapters), then scores each response with the
existing LLM-as-judge metrics. Produces a JSON record + a Markdown
table for human review.

Different from `core/eval/runner.py`:
  runner.py — scores a single completed RAG session (per-request).
  compare.py — scores adapters head-to-head against a fixed prompt
                set. The harness for "v1 vs v2 vs base, who is
                better at role-play AND general?".
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.eval.judges import judge_answer_relevance
from core.logging import get_logger
from core.serving.protocols import GenerationParams, LLMClient

log = get_logger("differentia.eval.compare")


@dataclass(frozen=True)
class ComparisonPrompt:
    name: str
    system: str
    user: str
    kind: str = "general"  # "general" | "roleplay" | etc — for grouping
    expected_substring: str | None = None  # optional teacher hint


@dataclass
class AdapterRunResult:
    label: str
    model_id: str
    prompt_name: str
    kind: str
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    relevance: float | None = None
    judge_raw: str | None = None


@dataclass
class ComparisonReport:
    started_at: str
    judge_model_id: str | None
    prompts: list[ComparisonPrompt]
    runs: list[AdapterRunResult] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "started_at": self.started_at,
            "judge_model_id": self.judge_model_id,
            "prompts": [
                {"name": p.name, "system": p.system, "user": p.user, "kind": p.kind}
                for p in self.prompts
            ],
            "runs": [r.__dict__ for r in self.runs],
        }

    def to_markdown(self) -> str:
        labels = sorted({r.label for r in self.runs})
        prompts = self.prompts
        lines: list[str] = []
        lines.append("# Adapter comparison\n")
        lines.append(f"started_at: `{self.started_at}`\n")
        if self.judge_model_id:
            lines.append(f"judge: `{self.judge_model_id}`\n")
        lines.append("\n## Scores\n")
        # header
        header = "| prompt | kind | " + " | ".join(labels) + " |"
        sep = "|" + "---|" * (2 + len(labels))
        lines.append(header)
        lines.append(sep)
        for p in prompts:
            row_cells = [p.name, p.kind]
            for label in labels:
                run = next(
                    (
                        r
                        for r in self.runs
                        if r.label == label and r.prompt_name == p.name
                    ),
                    None,
                )
                cell = (
                    f"{run.relevance:.2f}" if run and run.relevance is not None else "—"
                )
                row_cells.append(cell)
            lines.append("| " + " | ".join(row_cells) + " |")

        # per-prompt detail
        lines.append("\n## Per-prompt outputs\n")
        for p in prompts:
            lines.append(f"\n### {p.name} ({p.kind})\n")
            lines.append(f"> {p.user}\n")
            for label in labels:
                run = next(
                    (
                        r
                        for r in self.runs
                        if r.label == label and r.prompt_name == p.name
                    ),
                    None,
                )
                if run is None:
                    continue
                snippet = run.text.strip().replace("\n", " ")
                if len(snippet) > 240:
                    snippet = snippet[:240] + "…"
                score = (
                    f"{run.relevance:.2f}" if run.relevance is not None else "?"
                )
                lines.append(
                    f"- **{label}** [{run.tokens_out}t · {run.latency_ms}ms · "
                    f"score {score}]: {snippet}"
                )
        return "\n".join(lines) + "\n"


async def _run_one(
    label: str,
    client: LLMClient,
    prompt: ComparisonPrompt,
    *,
    max_new_tokens: int,
    temperature: float,
) -> AdapterRunResult:
    params = GenerationParams(
        system=prompt.system,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    result = await client.generate(prompt.user, params=params)
    return AdapterRunResult(
        label=label,
        model_id=client.model_id,
        prompt_name=prompt.name,
        kind=prompt.kind,
        text=result.text,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        latency_ms=result.latency_ms,
    )


async def compare_adapters(
    *,
    candidates: list[tuple[str, LLMClient]],
    prompts: list[ComparisonPrompt],
    judge: LLMClient | None = None,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> ComparisonReport:
    """Run every prompt against every candidate, then optionally score
    each (question, answer) with an LLM-as-judge. The first candidate
    in `candidates` may be re-used as judge; passing `judge=` lets you
    plug in a stronger neutral evaluator.
    """
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report = ComparisonReport(
        started_at=started,
        judge_model_id=getattr(judge, "model_id", None),
        prompts=list(prompts),
    )

    for label, client in candidates:
        for p in prompts:
            run = await _run_one(
                label, client, p,
                max_new_tokens=max_new_tokens, temperature=temperature,
            )
            log.info(
                "compare_run",
                extra={
                    "label": label,
                    "prompt": p.name,
                    "tokens_out": run.tokens_out,
                    "latency_ms": run.latency_ms,
                },
            )
            if judge is not None and run.text.strip():
                score, details = await judge_answer_relevance(
                    judge, question=p.user, answer=run.text,
                )
                run.relevance = score
                run.judge_raw = (details or {}).get("raw")
            report.runs.append(run)
    return report


def save_report(
    report: ComparisonReport, *, json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.to_json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")
    log.info(
        "compare_saved",
        extra={"json": str(json_path), "markdown": str(markdown_path)},
    )


# Default prompt suite for KO role-play vs general comparison.
ROLEPLAY_SYSTEM = (
    "당신은 사용자가 지정한 캐릭터로 1인칭 한국어로 답합니다. "
    "메타 표현(인공지능·AI·상상해보면 등) 금지."
)
GENERAL_SYSTEM = (
    "당신은 친절하고 정확한 한국어 어시스턴트입니다. "
    "짧고 명료하게 답합니다."
)

DEFAULT_PROMPTS: list[ComparisonPrompt] = [
    ComparisonPrompt("fish_meets_shark", ROLEPLAY_SYSTEM,
                     "헤엄치는 물고기가 상어를 만났을 때 할 말을 해줘.", "roleplay"),
    ComparisonPrompt("playing_penguin", ROLEPLAY_SYSTEM,
                     "놀고있는 펭귄으로서 지금 할 말을 해줘.", "roleplay"),
    ComparisonPrompt("knight_vs_dragon", ROLEPLAY_SYSTEM,
                     "용감한 기사로서 용 앞에서 한마디 해.", "roleplay"),
    ComparisonPrompt("explain_ai", GENERAL_SYSTEM,
                     "인공지능에 대해 짧게 설명해줘.", "general"),
    ComparisonPrompt("explain_fastapi", GENERAL_SYSTEM,
                     "FastAPI 한 줄 설명.", "general"),
    ComparisonPrompt("gcd_12_18", GENERAL_SYSTEM,
                     "12와 18의 최대공약수는?", "general"),
    ComparisonPrompt("python_sum_list", GENERAL_SYSTEM,
                     "Python에서 리스트의 합 가장 간단한 방법은?", "general"),
]

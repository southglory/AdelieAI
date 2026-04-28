"""Behavioral test suite runner per persona (Step 6.1).

Loads `personas/{persona_id}/eval_prompts.yaml`, generates a reply for
each prompt against the active LLMClient (with grounding applied as in
production), grades it against must_contain_any / must_not_contain
patterns, and emits a structured report.

Outcome categories:
  - pass: must_contain_any 의 어떤 항목 ∈ reply (또는 must_contain_any 비어있음)
          AND must_not_contain 의 모두 ∉ reply
          AND banned_phrases (글로벌 + per-prompt) 의 모두 ∉ reply
  - fail_missing: must_contain_any 충족 X
  - fail_banned:  must_not_contain / banned_phrases 위반
  - error:        생성 실패

Composite metrics returned alongside per-prompt results:
  - pass_rate (전체 / 카테고리별)
  - banned_violations (총 횟수)
  - cjk_ratio_avg (한글 비율 평균)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.personas.grounding import build_grounding_context
from core.personas.registry import Persona, get_persona
from core.serving.protocols import GenerationParams, LLMClient


CJK_HAN = re.compile(r"[一-鿿㐀-䶿]")


@dataclass(frozen=True)
class PromptResult:
    id: str
    prompt: str
    reply: str
    category: str
    outcome: str   # "pass" | "fail_missing" | "fail_banned" | "error"
    detail: str    # human-readable failure reason
    cjk_ratio: float
    cjk_han_count: int
    banned_violations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalReport:
    persona_id: str
    n_prompts: int
    pass_rate: float
    pass_by_category: dict[str, float]
    banned_violations_total: int
    cjk_ratio_avg: float
    cjk_han_count_total: int
    results: list[PromptResult]


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    hangul = sum(1 for c in text if "가" <= c <= "힣")
    return round(hangul / len(text), 3)


def _grade(
    reply: str,
    must_contain_any: list[str],
    must_not_contain: list[str],
    banned: list[str],
) -> tuple[str, str, list[str]]:
    """Returns (outcome, detail, banned_violations)."""
    # banned_phrases (global) + must_not_contain (per-prompt) → all forbidden
    banned_in_reply = [b for b in banned if b in reply]
    not_contain_in_reply = [n for n in must_not_contain if n in reply]
    if banned_in_reply or not_contain_in_reply:
        return (
            "fail_banned",
            f"banned/forbidden phrases present: {banned_in_reply + not_contain_in_reply}",
            banned_in_reply + not_contain_in_reply,
        )
    if must_contain_any:
        if not any(m in reply for m in must_contain_any):
            return (
                "fail_missing",
                f"none of must_contain_any present: {must_contain_any}",
                [],
            )
    return "pass", "", []


def load_eval_spec(persona_id: str) -> dict[str, Any]:
    """Load `personas/{persona_id}/eval_prompts.yaml` (or .yml)."""
    import yaml  # type: ignore[import-not-found]

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "personas" / persona_id / "eval_prompts.yaml",
        repo_root / "personas" / persona_id / "eval_prompts.yml",
    ]
    for path in candidates:
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"no eval_prompts.yaml found under personas/{persona_id}/ — "
        f"checked {[str(c) for c in candidates]}"
    )


async def run_persona_eval(
    persona_id: str,
    *,
    llm: LLMClient,
    graph_retriever=None,
    tool_registry=None,
    params: GenerationParams | None = None,
) -> EvalReport:
    """Execute the suite end-to-end against the live LLM.

    Production parity: grounding context is built the same way the
    chat handler does, so the eval reflects what end-users actually
    experience.
    """
    spec = load_eval_spec(persona_id)
    persona = get_persona(persona_id)
    if persona is None:
        raise ValueError(f"persona '{persona_id}' not in registry")

    banned = list(spec.get("banned_phrases", []))
    base_params = params or GenerationParams()

    results: list[PromptResult] = []
    for entry in spec.get("prompts", []):
        prompt = entry["prompt"]
        category = entry.get("category", "uncategorized")

        grounding = build_grounding_context(
            persona,
            user_text=prompt,
            graph_retriever=graph_retriever,
            tool_registry=tool_registry,
        )
        augmented_system = persona.system_prompt + (grounding or "")
        run_params = base_params.model_copy(update={"system": augmented_system})

        try:
            generation = await llm.generate(prompt, run_params)
            reply = generation.text or ""
        except Exception as exc:
            results.append(PromptResult(
                id=entry.get("id", prompt[:30]),
                prompt=prompt,
                reply="",
                category=category,
                outcome="error",
                detail=f"generation error: {exc}",
                cjk_ratio=0.0,
                cjk_han_count=0,
            ))
            continue

        outcome, detail, banned_hits = _grade(
            reply,
            must_contain_any=list(entry.get("must_contain_any", [])),
            must_not_contain=list(entry.get("must_not_contain", [])),
            banned=banned,
        )
        results.append(PromptResult(
            id=entry.get("id", prompt[:30]),
            prompt=prompt,
            reply=reply,
            category=category,
            outcome=outcome,
            detail=detail,
            cjk_ratio=_cjk_ratio(reply),
            cjk_han_count=len(CJK_HAN.findall(reply)),
            banned_violations=banned_hits,
        ))

    return _compose_report(persona_id, results)


def _compose_report(persona_id: str, results: list[PromptResult]) -> EvalReport:
    n = len(results)
    n_pass = sum(1 for r in results if r.outcome == "pass")
    pass_rate = round(n_pass / n, 3) if n else 0.0

    categories: dict[str, list[PromptResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)
    pass_by_category = {
        cat: round(sum(1 for r in rs if r.outcome == "pass") / len(rs), 3)
        for cat, rs in categories.items()
    }
    banned_total = sum(len(r.banned_violations) for r in results)
    cjk_avg = round(
        sum(r.cjk_ratio for r in results) / n, 3
    ) if n else 0.0
    cjk_han_total = sum(r.cjk_han_count for r in results)

    return EvalReport(
        persona_id=persona_id,
        n_prompts=n,
        pass_rate=pass_rate,
        pass_by_category=pass_by_category,
        banned_violations_total=banned_total,
        cjk_ratio_avg=cjk_avg,
        cjk_han_count_total=cjk_han_total,
        results=results,
    )


def render_report_md(report: EvalReport, *, adapter_label: str = "") -> str:
    """Pretty-print the report as Markdown for `docs/eval/runs/`."""
    head = f"# Persona eval — {report.persona_id}"
    if adapter_label:
        head += f" · {adapter_label}"
    lines = [
        head,
        "",
        f"- pass_rate: **{report.pass_rate * 100:.0f}%**  ({report.n_prompts} prompts)",
        f"- banned_violations_total: {report.banned_violations_total}",
        f"- cjk_ratio_avg: {report.cjk_ratio_avg}",
        f"- cjk_han_count_total: {report.cjk_han_count_total}",
        "",
        "## By category",
        "",
        "| category | pass_rate |",
        "|---|---|",
    ]
    for cat, rate in sorted(report.pass_by_category.items()):
        lines.append(f"| {cat} | {rate * 100:.0f}% |")
    lines += ["", "## Per prompt", ""]
    for r in report.results:
        marker = {"pass": "✅", "fail_missing": "❌", "fail_banned": "🚨", "error": "💥"}[r.outcome]
        lines.append(f"### {marker} `{r.id}` · {r.category}")
        lines.append("")
        lines.append(f"> {r.prompt}")
        lines.append("")
        lines.append(f"답변: {r.reply[:280]}")
        if r.detail:
            lines.append("")
            lines.append(f"_{r.detail}_")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "PromptResult",
    "EvalReport",
    "load_eval_spec",
    "run_persona_eval",
    "render_report_md",
]

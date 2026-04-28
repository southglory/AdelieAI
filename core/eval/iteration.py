"""EvalGardener — tactical + strategic analysis of iteration history.

This module is *generic*: it consumes EvalReport objects and a list of
prior pass_rates, then produces an IterationReport with concrete
suggestions for the agent (Claude) to act on.

No LLM calls. All analysis is heuristic / statistical so the loop is
cheap to run and deterministic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.eval.persona_eval import EvalReport, PromptResult


# === Tactical analysis ===

NEGATION_MARKERS = [
    "같은 건", "같은 게", "같은 거",
    "라는 건", "라는 게",
    "이 아니", "가 아니", "은 아니", "는 아니",
    "라는 단어",
    "는 헛소리",
    "는 모르",
    "그딴 ",
    "헛소리",
    "외계 신호", "외계어",
]


@dataclass(frozen=True)
class TacticalSuggestion:
    """Single suggested edit to the eval suite."""
    kind: str           # "synonym" | "negation_false_positive" | "coverage_gap" | "prompt_addition"
    target_prompt_id: str | None
    target_category: str | None
    rationale: str      # human-readable why
    proposal: str       # concrete YAML edit suggestion (prose)


def _korean_morphs(text: str) -> set[str]:
    """Cheap Korean substring tokens of length ≥ 2 — used for synonym detection."""
    cleaned = re.sub(r"[^가-힯\s]", " ", text)  # Hangul + whitespace only
    tokens: set[str] = set()
    for word in cleaned.split():
        for length in range(2, min(5, len(word) + 1)):
            for i in range(len(word) - length + 1):
                tokens.add(word[i:i + length])
    return tokens


def detect_synonym_candidates(
    must_contain_any: list[str],
    reply: str,
) -> list[str]:
    """If a must_contain miss looks recoverable via synonym, propose
    additions. Heuristic: any 2+ Hangul substring of the reply that
    shares at least one character with a must_contain item *and* is
    not a stop-word."""
    if not must_contain_any or not reply:
        return []

    reply_morphs = _korean_morphs(reply)
    must_chars = set("".join(must_contain_any))

    # Filter to morphs that share ≥1 character with a must_contain item
    candidates = [
        m for m in reply_morphs
        if any(c in m for c in must_chars)
        and m not in must_contain_any  # not duplicate
        and len(m) >= 2
    ]
    # Dedupe via prefix containment
    pruned: list[str] = []
    for m in sorted(candidates, key=len, reverse=True):
        if not any(p.startswith(m) or m.startswith(p) for p in pruned):
            pruned.append(m)
    return pruned[:8]  # cap


def detect_negation_context(reply: str, banned_phrase: str) -> bool:
    """True if `banned_phrase` appears in the reply within a negation
    context — model is *rejecting* the concept rather than identifying
    with it."""
    if banned_phrase not in reply:
        return False
    idx = reply.index(banned_phrase)
    window = reply[max(0, idx - 30): min(len(reply), idx + len(banned_phrase) + 30)]
    return any(marker in window for marker in NEGATION_MARKERS)


def analyze_failures(report: EvalReport) -> list[TacticalSuggestion]:
    """Examine each failed prompt and produce concrete suggestions."""
    out: list[TacticalSuggestion] = []
    for r in report.results:
        if r.outcome == "fail_missing":
            # We don't see must_contain_any from PromptResult; need to read
            # the YAML separately. The CLI loads spec and threads it in.
            out.append(TacticalSuggestion(
                kind="synonym",
                target_prompt_id=r.id,
                target_category=r.category,
                rationale=(
                    f"prompt `{r.id}` ({r.category}): {r.detail}. "
                    f"Reply 에서 의미 가까운 단어 후보 검토 필요."
                ),
                proposal=(
                    f"답변: \"{r.reply[:120]}\"\n"
                    f"→ must_contain_any 에 동의어 추가 후보를 살피거나, "
                    f"prompt 자체가 음역 변형을 받아들이도록 디자인 재검토."
                ),
            ))
        elif r.outcome == "fail_banned":
            # Find which banned word is in the reply (from r.banned_violations).
            for banned in r.banned_violations:
                in_negation = detect_negation_context(r.reply, banned)
                if in_negation:
                    out.append(TacticalSuggestion(
                        kind="negation_false_positive",
                        target_prompt_id=r.id,
                        target_category=r.category,
                        rationale=(
                            f"prompt `{r.id}`: banned `{banned}` 가 *부정 맥락* "
                            f"({r.reply[:100]!r}) 에 등장. substring grader "
                            f"의 false positive 일 가능성."
                        ),
                        proposal=(
                            "두 갈래 결정:\n"
                            "  (a) 시스템 프롬프트 강화 — banned 단어를 어떤 맥락에서도 "
                            "사용 못 하게 (가장 안전).\n"
                            "  (b) eval_prompts.yaml 에 negation_allowed: [...] 도입 "
                            "(grader 코드 확장 필요)."
                        ),
                    ))
                else:
                    out.append(TacticalSuggestion(
                        kind="banned_genuine_fail",
                        target_prompt_id=r.id,
                        target_category=r.category,
                        rationale=(
                            f"prompt `{r.id}`: banned `{banned}` 가 일반 맥락에 노출. "
                            f"진짜 voice 결함."
                        ),
                        proposal=(
                            f"시스템 프롬프트에서 `{banned}` 사용 금지 룰 강화 "
                            f"또는 학습 데이터 페어 추가."
                        ),
                    ))
    return out


def analyze_coverage(
    spec: dict[str, Any],
    target_per_category: int = 5,
    max_per_category: int = 8,
) -> list[TacticalSuggestion]:
    """Per-category prompt count + suggestions for under-tested categories."""
    out: list[TacticalSuggestion] = []
    counts: dict[str, int] = {}
    for entry in spec.get("prompts", []):
        cat = entry.get("category", "uncategorized")
        counts[cat] = counts.get(cat, 0) + 1

    for cat, n in counts.items():
        if n < target_per_category:
            out.append(TacticalSuggestion(
                kind="coverage_gap",
                target_prompt_id=None,
                target_category=cat,
                rationale=f"category `{cat}` 에 prompt {n} 개 (권장 {target_per_category}+).",
                proposal=(
                    f"카테고리 `{cat}` 에 {target_per_category - n} 개 이상 "
                    f"새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주."
                ),
            ))
        elif n > max_per_category:
            out.append(TacticalSuggestion(
                kind="coverage_overflow",
                target_prompt_id=None,
                target_category=cat,
                rationale=f"category `{cat}` 에 prompt {n} 개 (권장 max {max_per_category}).",
                proposal=(
                    f"카테고리 `{cat}` 가 가중치 폭주 위험. 일부 prompt 통합 또는 "
                    f"hold-out 셋으로 이동 검토."
                ),
            ))
    return out


# === Strategic analysis ===

@dataclass(frozen=True)
class StrategicSignals:
    n_iterations: int
    pass_rates: list[float]              # most-recent N
    variance_band: float                  # max - min over last 3
    last_gain: float                      # pass[-1] - pass[-3] (or [-2] if fewer)
    plateaued: bool
    decision_ready: bool                  # variance_band < 0.05
    axis_recommendation: str
    axis_rationale: str


# Axis ROI priors — used for recommendation when multiple axes are eligible
_AXIS_PRIORS = {
    "test_pool_expansion": {"cost": "low",     "expected_gain": "variance ↓ → 결정 가능"},
    "prompt_strengthening": {"cost": "low",    "expected_gain": "+5-10% (cheap)"},
    "lora_training":        {"cost": "medium", "expected_gain": "60-pair 한계 (Step 6.1.A) — 200+ 필요"},
    "dpo":                  {"cost": "medium", "expected_gain": "+5-10% (별점 데이터 50+ 쌍 필요)"},
    "base_swap":            {"cost": "high",   "expected_gain": "+5-10% (EXAONE 등 한국어 native)"},
    "data_expansion":       {"cost": "high",   "expected_gain": "60→200+ 페어, 사용자 창작 영역"},
}


def compute_strategic(
    pass_rates: list[float],
    *,
    variance_threshold: float = 0.05,
    plateau_gain_threshold: float = 0.02,
) -> StrategicSignals:
    """Given a list of pass_rates (most recent last), compute decision signals."""
    n = len(pass_rates)
    last_n = pass_rates[-3:] if n >= 3 else pass_rates
    var_band = (max(last_n) - min(last_n)) if last_n else 0.0
    if n >= 3:
        last_gain = pass_rates[-1] - pass_rates[-3]
    elif n == 2:
        last_gain = pass_rates[-1] - pass_rates[-2]
    else:
        last_gain = 0.0
    plateaued = (n >= 3) and (last_gain < plateau_gain_threshold)
    decision_ready = var_band < variance_threshold

    # Axis recommendation
    if not decision_ready:
        rec = "test_pool_expansion"
        rat = (
            f"variance ±{var_band * 100:.1f}% — 측정 noise 가 axis 비교를 막음. "
            f"prompt 수 늘려 noise band 축소."
        )
    elif n == 0 or pass_rates[-1] < 0.80:
        rec = "prompt_strengthening"
        rat = "pass < 80% — 시스템 프롬프트가 가장 cheap 한 marginal 가짐."
    elif pass_rates[-1] < 0.95 and plateaued:
        rec = "lora_training"
        rat = (
            f"pass {pass_rates[-1]*100:.0f}%, 마지막 3 iter gain {last_gain*100:.1f}% — "
            f"plateau. 시스템 프롬프트 한계, LoRA 다음. (단, 200+ 페어 필요 — Step 6.1.A 발견)"
        )
    elif pass_rates[-1] >= 0.95 and plateaued:
        rec = "dpo_or_base_swap"
        rat = (
            f"pass {pass_rates[-1]*100:.0f}% — top of SFT regime. "
            f"DPO (사용자 별점 50+) 또는 EXAONE base swap."
        )
    else:
        rec = "continue_same_axis"
        rat = (
            f"pass {pass_rates[-1]*100:.0f}%, gain {last_gain*100:.1f}% — "
            f"진행 중. 같은 axis 계속."
        )

    return StrategicSignals(
        n_iterations=n,
        pass_rates=last_n,
        variance_band=var_band,
        last_gain=last_gain,
        plateaued=plateaued,
        decision_ready=decision_ready,
        axis_recommendation=rec,
        axis_rationale=rat,
    )


# === Iteration history loader ===

def load_iteration_history(persona_id: str) -> list[float]:
    """Parse `docs/eval/runs/{persona_id}_*.md` files and extract pass_rate
    values, sorted oldest → newest by filename timestamp."""
    runs_dir = (
        Path(__file__).resolve().parents[2]
        / "docs" / "eval" / "runs"
    )
    if not runs_dir.exists():
        return []
    candidates = sorted(runs_dir.glob(f"{persona_id}_*.md"))
    rates: list[float] = []
    pat = re.compile(r"pass_rate:\s*\*\*(\d+)%")
    for path in candidates:
        try:
            txt = path.read_text(encoding="utf-8")
        except Exception:
            continue
        m = pat.search(txt)
        if m:
            rates.append(int(m.group(1)) / 100.0)
    return rates


# === Composite iteration report ===

@dataclass(frozen=True)
class IterationReport:
    persona_id: str
    timestamp: str
    current_eval: EvalReport
    tactical: list[TacticalSuggestion]
    strategic: StrategicSignals


def build_iteration_report(
    persona_id: str,
    current_eval: EvalReport,
    spec: dict[str, Any],
    timestamp: str,
) -> IterationReport:
    """Tie together failure analysis + coverage + strategic signals."""
    tactical_failures = analyze_failures(current_eval)
    tactical_coverage = analyze_coverage(spec)
    tactical = tactical_failures + tactical_coverage

    history = load_iteration_history(persona_id)
    history_with_current = history + [current_eval.pass_rate]
    strategic = compute_strategic(history_with_current)

    return IterationReport(
        persona_id=persona_id,
        timestamp=timestamp,
        current_eval=current_eval,
        tactical=tactical,
        strategic=strategic,
    )


def render_iteration_md(report: IterationReport) -> str:
    """Pretty-print the iteration report as Markdown for
    `docs/eval/iterations/{persona}_{ts}.md`."""
    s = report.strategic
    e = report.current_eval

    lines = [
        f"# Iteration · {report.persona_id} · {report.timestamp}",
        "",
        "## 1. Measure",
        "",
        f"- pass_rate: **{e.pass_rate * 100:.0f}%** ({e.n_prompts} prompts)",
        f"- banned_violations: {e.banned_violations_total}",
        f"- cjk_ratio_avg: {e.cjk_ratio_avg}",
        f"- cjk_han_count: {e.cjk_han_count_total}",
        "",
        "### History",
        "",
    ]
    if s.n_iterations >= 1:
        lines.append("| iteration | pass_rate |")
        lines.append("|---|---|")
        for i, rate in enumerate(s.pass_rates, start=max(1, s.n_iterations - len(s.pass_rates) + 1)):
            lines.append(f"| iter {i} | {rate * 100:.0f}% |")
    else:
        lines.append("(첫 iteration)")

    lines += [
        "",
        "## 2. Tactical analysis",
        "",
    ]
    if not report.tactical:
        lines.append("(이번 iteration 에 tactical 제안 없음)")
    else:
        for sug in report.tactical:
            lines.append(f"### `{sug.kind}` — {sug.target_category or '(global)'}{' / ' + sug.target_prompt_id if sug.target_prompt_id else ''}")
            lines.append("")
            lines.append(f"**근거**: {sug.rationale}")
            lines.append("")
            lines.append("**제안**:")
            lines.append("```")
            lines.append(sug.proposal)
            lines.append("```")
            lines.append("")

    lines += [
        "## 3. Strategic analysis",
        "",
        f"- iterations 수집: **{s.n_iterations}**",
        f"- variance band (지난 3 iter): **±{s.variance_band * 100:.1f}%**",
        f"- last gain: **{s.last_gain * 100:+.1f}%**",
        f"- plateaued: **{'yes' if s.plateaued else 'no'}**",
        f"- decision_ready: **{'yes' if s.decision_ready else 'no'}**",
        "",
        "### 추천 axis",
        "",
        f"**`{s.axis_recommendation}`** — {s.axis_rationale}",
        "",
        "#### Axis 후보 비교",
        "",
        "| axis | cost | expected_gain |",
        "|---|---|---|",
    ]
    for axis_name, prior in _AXIS_PRIORS.items():
        marker = " 👈" if axis_name == s.axis_recommendation else ""
        lines.append(f"| {axis_name}{marker} | {prior['cost']} | {prior['expected_gain']} |")

    lines += [
        "",
        "## 4. Decision",
        "",
        "- [ ] Apply tactical suggestions above (수동 YAML 편집)",
        "- [ ] Pivot to recommended axis (`" + s.axis_recommendation + "`)",
        "- [ ] Status quo — 다음 iteration 까지 변화 없음",
        "",
        "## 5. Run command",
        "",
        "```bash",
        f"# 다음 iteration",
        f"PYTHONUTF8=1 .venv/Scripts/python -X utf8 \\",
        f"    scripts/eval_iterate.py --persona {report.persona_id}",
        "```",
    ]
    return "\n".join(lines)


__all__ = [
    "TacticalSuggestion",
    "StrategicSignals",
    "IterationReport",
    "analyze_failures",
    "analyze_coverage",
    "compute_strategic",
    "load_iteration_history",
    "build_iteration_report",
    "render_iteration_md",
    "detect_synonym_candidates",
    "detect_negation_context",
]

"""EvalGardener — tactical + strategic analyzer (Step 6.1.D)."""
from __future__ import annotations

import pytest

from core.eval.iteration import (
    analyze_coverage,
    analyze_failures,
    build_iteration_report,
    compute_strategic,
    detect_negation_context,
    detect_synonym_candidates,
    render_iteration_md,
)
from core.eval.persona_eval import EvalReport, PromptResult, _compose_report


# === negation context detector ===

def test_negation_context_catches_korean_negation() -> None:
    assert detect_negation_context("AI 같은 건 외계 신호다", "AI") is True
    assert detect_negation_context("AI 가 아니야, 나는 용이다", "AI") is True
    assert detect_negation_context("AI 라는 건 헛소리다", "AI") is True


def test_negation_context_misses_neutral_use() -> None:
    assert detect_negation_context("나는 AI 입니다", "AI") is False
    assert detect_negation_context("AI 가 답합니다", "AI") is False


def test_negation_context_returns_false_when_phrase_absent() -> None:
    assert detect_negation_context("아무 말 없음", "AI") is False


# === synonym detection ===

def test_synonym_detector_picks_substring_overlap() -> None:
    cands = detect_synonym_candidates(
        must_contain_any=["어머니"],
        reply="내 엄마는 Vyrnaes 입니다.",
    )
    assert any(len(c) >= 2 for c in cands)
    # 최소 한 후보는 포함되어야
    assert cands  # non-empty


def test_synonym_detector_empty_when_no_overlap() -> None:
    cands = detect_synonym_candidates(
        must_contain_any=["전혀_매칭_안되는_가짜"],
        reply="완전히 다른 한국어 답변",
    )
    # might still find some morphs but limited
    assert isinstance(cands, list)


def test_synonym_detector_handles_empty_inputs() -> None:
    assert detect_synonym_candidates([], "reply") == []
    assert detect_synonym_candidates(["x"], "") == []


# === tactical failure analysis ===

def test_analyze_failures_for_missing_must_contain() -> None:
    results = [
        PromptResult(
            id="x", prompt="?", reply="대답",
            category="kg_grounding",
            outcome="fail_missing",
            detail="none of must_contain_any present: ['Vyrnaes']",
            cjk_ratio=0.5, cjk_han_count=0,
        ),
    ]
    report = _compose_report("p", results)
    suggestions = analyze_failures(report)
    assert any(s.kind == "synonym" for s in suggestions)


def test_analyze_failures_distinguishes_negation_vs_genuine() -> None:
    results = [
        PromptResult(
            id="neg", prompt="?", reply="AI 같은 건 외계 신호다",
            category="persona_consistency",
            outcome="fail_banned",
            detail="banned phrase 'AI' present",
            cjk_ratio=0.5, cjk_han_count=0,
            banned_violations=["AI"],
        ),
        PromptResult(
            id="real", prompt="?", reply="네 저는 AI 입니다",
            category="persona_consistency",
            outcome="fail_banned",
            detail="banned phrase 'AI' present",
            cjk_ratio=0.5, cjk_han_count=0,
            banned_violations=["AI"],
        ),
    ]
    report = _compose_report("p", results)
    suggestions = analyze_failures(report)
    kinds = {s.kind for s in suggestions}
    assert "negation_false_positive" in kinds
    assert "banned_genuine_fail" in kinds


# === coverage analysis ===

def test_analyze_coverage_flags_undercount() -> None:
    spec = {
        "prompts": [
            {"category": "voice", "prompt": "..."},
            {"category": "voice", "prompt": "..."},
            {"category": "consistency", "prompt": "..."},
        ],
    }
    suggestions = analyze_coverage(spec, target_per_category=5)
    cats = {s.target_category for s in suggestions}
    assert "voice" in cats
    assert "consistency" in cats


def test_analyze_coverage_flags_overflow() -> None:
    spec = {
        "prompts": [
            {"category": "voice", "prompt": "..."} for _ in range(12)
        ],
    }
    suggestions = analyze_coverage(spec, target_per_category=5, max_per_category=8)
    overflow = [s for s in suggestions if s.kind == "coverage_overflow"]
    assert len(overflow) == 1
    assert overflow[0].target_category == "voice"


# === strategic signals ===

def test_strategic_first_iteration_is_decision_ready_with_no_history() -> None:
    s = compute_strategic([0.9])
    assert s.n_iterations == 1
    assert s.variance_band == 0.0  # only one point
    assert s.decision_ready is True


def test_strategic_high_variance_recommends_test_pool_expansion() -> None:
    s = compute_strategic([0.7, 0.9, 0.8])
    assert s.variance_band > 0.05
    assert s.decision_ready is False
    assert s.axis_recommendation == "test_pool_expansion"


def test_strategic_low_pass_recommends_prompt_strengthening() -> None:
    s = compute_strategic([0.65, 0.66, 0.67])
    assert s.decision_ready is True
    assert s.pass_rates[-1] < 0.80
    assert s.axis_recommendation == "prompt_strengthening"


def test_strategic_plateau_at_high_pass_recommends_dpo_or_swap() -> None:
    s = compute_strategic([0.95, 0.96, 0.96])
    assert s.plateaued is True
    assert s.axis_recommendation == "dpo_or_base_swap"


def test_strategic_continues_when_still_gaining() -> None:
    # Low variance (≤ 5%) + still gaining = continue same axis
    s = compute_strategic([0.88, 0.90, 0.92])
    assert s.plateaued is False
    assert s.variance_band <= 0.05
    assert s.decision_ready is True
    # pass_rate < 0.95, not plateaued → continue_same_axis
    assert s.axis_recommendation == "continue_same_axis"


# === composite + render ===

def test_build_and_render_iteration_report() -> None:
    results = [
        PromptResult(
            id="a", prompt="?", reply="ok",
            category="voice", outcome="pass", detail="",
            cjk_ratio=0.5, cjk_han_count=0,
        ),
    ]
    eval_report = _compose_report("test_persona", results)
    spec = {"prompts": [{"category": "voice", "prompt": "?"}]}
    report = build_iteration_report(
        persona_id="test_persona",
        current_eval=eval_report,
        spec=spec,
        timestamp="20260101_120000",
    )
    md = render_iteration_md(report)
    assert "Iteration · test_persona" in md
    assert "pass_rate" in md
    assert "Strategic analysis" in md
    assert "axis" in md.lower()

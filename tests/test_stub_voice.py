"""StubLLMClient persona-aware canned voice."""

import asyncio

import pytest

from core.serving.protocols import GenerationParams
from core.serving.stub_client import StubLLMClient


def _gen(system: str | None, prompt: str) -> str:
    client = StubLLMClient()
    params = GenerationParams(system=system) if system is not None else GenerationParams()
    out = asyncio.run(client.generate(prompt, params=params))
    return out.text


def test_stub_falls_back_to_echo_with_no_system() -> None:
    text = _gen(None, "hello")
    assert "[stub:" in text


def test_stub_falls_back_to_echo_for_unrecognized_persona() -> None:
    text = _gen("당신은 외계 종족 X12 입니다.", "안녕")
    assert "[stub:" in text


@pytest.mark.parametrize(
    "keyword",
    ["펭귄", "물고기", "기사", "잡화상", "탐정"],
)
def test_stub_returns_persona_voice_when_system_contains_keyword(keyword: str) -> None:
    system = f"당신은 {keyword}입니다. 1인칭으로 답하세요."
    text = _gen(system, "안녕? 오늘 뭐 했어?")
    assert "[stub:" not in text  # not the echo
    assert text.strip()  # non-empty


def test_stub_voice_is_deterministic_per_prompt() -> None:
    system = "당신은 냉소적인 잡화상 주인입니다."
    a = _gen(system, "할인 좀 안 돼요?")
    b = _gen(system, "할인 좀 안 돼요?")
    assert a == b


def test_stub_voice_varies_across_prompts() -> None:
    system = "당신은 잡화상 주인입니다. 신용 거래는 안 받습니다."
    seen = {_gen(system, p) for p in [
        "안녕?",
        "이 검 얼마예요?",
        "할인 되나요?",
        "외상 가능?",
        "이거 가품이죠?",
        "장사 어때요?",
    ]}
    # 5 canned merchant lines; with 6 different prompts we expect
    # at least 2 distinct replies (collisions on hash() are possible
    # but we should not collapse to a single line).
    assert len(seen) >= 2


def test_detective_keyword_picks_detective_voice() -> None:
    system = "당신은 도시 형사 사무소의 냉정한 탐정입니다."
    text = _gen(system, "사건 상황은?")
    assert "[stub:" not in text
    # Detective canned lines are deduction-flavored (사실 / 추정 /
    # 증거 등) — verify at least one detective fingerprint appears
    # over a small batch to avoid hash-collision flakiness.
    batch = " ".join(_gen(system, p) for p in [
        "이 사건 어떻게 봐?",
        "범인이 누구야?",
        "당신 탐정 맞아?",
        "유리 조각이 안에서 깨졌어.",
    ])
    assert any(kw in batch for kw in ["사실", "추정", "탐정", "모순", "사건"])
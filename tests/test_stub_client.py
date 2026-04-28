"""StubLLMClient picker — best-effort variety for repeat prompts.

Stub mode is a *dev-time voice preview* (no real model loaded), and
provides best-effort response variety. Tests that require *guaranteed*
reply sequences must use `core.serving.scripted_client.ScriptedLLMClient`.

The contract this picker upholds (ticket #62 root fix):

  1. *Stable across runs* — same prompt → same line, regardless of
     PYTHONHASHSEED. Achieved via hashlib.sha256 (not Python's hash()).
  2. *Repeat-prompt rotation* — the same `user_text` submitted on N
     consecutive turns walks through N distinct lines (mod len(lines)),
     so DPO data harvesting in stub mode actually finds chosen/rejected
     pairs.
"""
from __future__ import annotations

from core.serving.stub_client import _PERSONA_VOICES, _pick


def _lines_for(keyword: str) -> list[str]:
    for kw, lines in _PERSONA_VOICES:
        if kw == keyword:
            return lines
    raise KeyError(keyword)


def test_stable_across_runs_same_prompt_same_line() -> None:
    lines = _lines_for("잡화상")
    a = _pick(lines, "User: hi\nAssistant: ")
    b = _pick(lines, "User: hi\nAssistant: ")
    assert a == b


def test_different_user_text_different_lines() -> None:
    lines = _lines_for("잡화상")
    # Different first prompts → different starting lines (high probability).
    # Loose assert: at least 2 of 3 distinct user_texts pick different lines.
    picks = {
        _pick(lines, f"User: {q}\nAssistant: ")
        for q in ["할인?", "외상?", "수표?"]
    }
    assert len(picks) >= 2


def test_repeat_prompt_walks_all_lines_in_a_row() -> None:
    """Core ticket #62 contract — N consecutive identical user_texts
    produce N distinct replies (rotation guarantee)."""
    lines = _lines_for("잡화상")  # 5 canned lines
    n = len(lines)

    prompt = "User: 외상?\nAssistant: "
    seen: list[str] = []
    for _ in range(n):
        reply = _pick(lines, prompt)
        seen.append(reply)
        # Simulate the chat history growing with each turn.
        prompt += f"{reply}\nUser: 외상?\nAssistant: "

    assert len(set(seen)) == n, (
        f"stub picker collided on repeat prompt — {len(set(seen))}/{n} "
        f"distinct.\n  picks: {seen}"
    )


def test_unknown_persona_falls_through_to_stub_voice() -> None:
    # _pick is called only when _select_voice returns a list, but check
    # the function's behavior on an arbitrary 1-line list (no rotation
    # possible, must return that line).
    assert _pick(["only"], "anything") == "only"

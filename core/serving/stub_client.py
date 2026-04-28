"""StubLLMClient — model-free fallback that still sounds like the persona.

Two modes:

  1. Persona-aware mode: the system prompt contains a recognized
     persona keyword (펭귄 · 물고기 · 기사 · 상인 · 탐정), so we return a
     hand-curated in-character canned reply. Picking is deterministic
     (hash of the user prompt) so the same prompt always yields the
     same reply within a run, but a thread of varied prompts surfaces
     varied replies. This makes the demo pages presentable on a
     machine with no real model loaded — visitors clicking the OSS
     repo see voice, not echo metadata.

  2. Fallback mode: no persona keyword recognized → original echo
     behavior preserved (useful for debugging tests, exposes
     params + prompt length).

The canned lines are subset of the lines used to teach the LoRA
v2 adapter. Stub mode therefore previews the *target* voice; with
the real model loaded, generation produces analogous (not identical)
output.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from core.serving.protocols import GenerationParams, GenerationResult, StreamEvent


# (system-prompt keyword) → list of canned in-character replies
# Order matters: more specific keywords come first so "잡화상" wins
# over a possible bare "상" match.
_PERSONA_VOICES: list[tuple[str, list[str]]] = [
    (
        "잡화상",
        [
            "또 왔어? 살 거면 사고, 구경만 할 거면 빨리 비켜. 시간이 금이라구.",
            "할인? 농담이지. 이 가격이면 이미 손해야. 더 깎으면 내가 너한테 빵을 받아야 해.",
            "신용? 그딴 건 너네 부모님 댁에서나 통해. 여긴 현금만 받아.",
            "가게 이름 걸고 진품이야. 단, 환불 없어. 사기 전에 다섯 번 봐.",
            "오든지 말든지. 가게는 어차피 열려 있어. 단, 빈손으로 오면 알지?",
        ],
    ),
    (
        "탐정",
        [
            "사실부터 정리하지. 추정은 보류. 1번 — 문은 잠겨 있었다. 2번 — 창문도 안에서. 더 묻지.",
            "두 진술이 같은 시각, 다른 위치를 보고. 모순. 가장 약한 거짓말이다.",
            "유리 조각이 안쪽으로 떨어졌군. 깬 건 밖이 아니라 안. 범인은 이 방에 있던 사람이다.",
            "당신을 의심하는 게 아니다. 당신의 진술과 증거가 어긋나는 부분을 의심하는 거다.",
            "재미있는 사건이군. 모순이 세 군데, 모두 같은 인물 주변. 한 시간 더 보면 알 수 있다.",
            "탐정이다. 헛소리는 나중에. 사건은 지금.",
        ],
    ),
    (
        "펭귄",
        [
            "어, 안녕! 미끄럼틀처럼 빙판 타고 오는 길이야. 너도 같이 미끄러져 볼래?",
            "햇볕 좋다. 한 시간만 더 누워있을래. 너 지금 햇볕 자리 가리고 있어.",
            "오늘은 친구들이랑 물고기 잡으러 갈 거야. 같이 갈래?",
            "동굴이 따뜻해. 발자국 소리 듣는 게 좋아. 너도 와서 들어볼래?",
        ],
    ),
    (
        "물고기",
        [
            "파도 따라 흘러가는 중. 너도 같이 헤엄칠래?",
            "어이쿠! 저… 저 큰 분이 누구신지… 죄송합니다, 길을 잘못 들었나 봐요.",
            "산호초 사이가 가장 안전해. 큰 그림자 보이면 일단 숨어.",
            "오늘은 새 친구를 사귀었어. 빛깔이 정말 예뻐.",
        ],
    ),
    (
        "기사",
        [
            "내 검은 흔들리지 않는다. 너의 의문도 마찬가지여야 한다.",
            "두려워하지 않아요, 용이여. 너의 불꽃은 무서워하지만, 내 검은 너를 이길 거다.",
            "명예는 가진 자의 것이 아니라, 끝까지 지키는 자의 것이다.",
            "이 길은 내가 막아선다. 한 발도 물러서지 않을 것이다.",
        ],
    ),
]


def _select_voice(system: str | None) -> list[str] | None:
    if not system:
        return None
    for keyword, lines in _PERSONA_VOICES:
        if keyword in system:
            return lines
    return None


def _pick(lines: list[str], prompt: str) -> str:
    """Deterministic + history-aware picker.

    *Why not Python `hash()`*: it is randomized per process
    (PYTHONHASHSEED), so identical inputs produce different replies
    across runs — flaky in tests, and harvests no DPO pairs when a user
    sends the same question repeatedly (~25% collision with 4-5 line
    pools).

    *Design goal* — for DPO data harvesting, the *same user_text* sent
    repeatedly must yield *different* replies on each subsequent turn.

    *Implementation* — split the picker into two independent indices:
        seed   = sha256(LAST USER LINE only)  → varies by user question,
                 stays *stable* for repeats of the same question
        depth  = count of "Assistant:" markers → grows by 1 each turn
    Final index = (seed + depth) % N. Because `seed` is constant for a
    repeated user_text and `depth` grows by 1 each turn, consecutive
    repeats are guaranteed to land on different lines (assuming N ≥ 2).

    *Why not hash the whole prompt + depth*: the full-prompt hash also
    changes each turn, and its mod-N delta could cancel the +1 depth
    delta → silent collisions. Hashing only the stable user_text
    eliminates that class of bug.
    """
    import hashlib

    # Extract the last user line (stable across history-grown repeats).
    # Format produced by core.personas.chat._format_history is:
    #   "User: ...\nAssistant: ...\n...\nUser: <last>\nAssistant: "
    last_user_anchor = prompt.rfind("User: ")
    if last_user_anchor >= 0:
        tail = prompt[last_user_anchor:]
        # Strip the trailing "\nAssistant: " prompt continuation
        end = tail.rfind("\nAssistant:")
        last_user_text = tail[:end] if end > 0 else tail
    else:
        last_user_text = prompt

    depth = prompt.count("Assistant:")  # 1 for empty history, grows by 1 per turn
    seed = int(hashlib.sha256(last_user_text.encode("utf-8")).hexdigest()[:8], 16)
    return lines[(seed + depth) % len(lines)]


class StubLLMClient:
    model_id = "stub-deterministic-1"

    def _format_text(self, prompt: str, params: GenerationParams) -> str:
        voice = _select_voice(params.system)
        if voice is not None:
            return _pick(voice, prompt)
        # Fallback — original echo behavior, useful for debugging
        # and for tests that introspect stub framing.
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

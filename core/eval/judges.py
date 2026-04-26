import re

from core.serving.protocols import GenerationParams, LLMClient

_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_score(text: str) -> float | None:
    """Extract the first non-negative float in the LLM output and clamp
    to [0, 1]. Returns None if nothing parses. Treats bare integers
    1..10 as an x/10 scale (so an LLM that outputs "8" still works);
    fractional values like 1.5 just clamp.
    """
    for match in _FLOAT_RE.findall(text):
        try:
            value = float(match)
        except ValueError:
            continue
        if value < 0:
            continue
        if "." not in match and 1 <= value <= 10:
            value = value / 10.0
        return min(1.0, max(0.0, value))
    return None


_FAITHFULNESS_SYSTEM = (
    "당신은 답변 평가자입니다. 주어진 컨텍스트 안에서 답변이 얼마나 충실한지(faithfulness)를 0.0~1.0 사이 숫자로만 출력하세요.\n"
    "- 답변의 모든 사실 주장이 컨텍스트로 뒷받침되면 1.0\n"
    "- 일부만 뒷받침 → 비례 점수\n"
    "- 컨텍스트와 무관·모순 → 0.0\n"
    "출력은 숫자 한 개. 어떤 텍스트도 추가하지 말 것."
)


_RELEVANCE_SYSTEM = (
    "당신은 답변 평가자입니다. 답변이 질문에 얼마나 직접적으로 답하는지(answer_relevance)를 0.0~1.0 사이 숫자로만 출력하세요.\n"
    "- 답변이 질문의 핵심을 직접 다루면 1.0\n"
    "- 부분적으로 다루거나 우회 → 비례 점수\n"
    "- 동문서답 → 0.0\n"
    "출력은 숫자 한 개. 어떤 텍스트도 추가하지 말 것."
)


async def judge_faithfulness(
    llm: LLMClient,
    *,
    question: str,
    answer: str,
    contexts: list[str],
) -> tuple[float | None, dict]:
    if not contexts:
        return (None, {"reason": "no contexts"})
    body = "\n\n".join(f"[{i+1}] {c[:1500]}" for i, c in enumerate(contexts))
    prompt = (
        f"Question: {question}\n\nAnswer: {answer}\n\nContexts:\n{body}\n\nScore:"
    )
    result = await llm.generate(
        prompt,
        params=GenerationParams(
            temperature=0.0, max_new_tokens=12, system=_FAITHFULNESS_SYSTEM
        ),
    )
    score = _parse_score(result.text)
    return (score, {"raw": result.text[:80], "evaluator_model": result.model_id})


async def judge_answer_relevance(
    llm: LLMClient,
    *,
    question: str,
    answer: str,
) -> tuple[float | None, dict]:
    prompt = f"Question: {question}\n\nAnswer: {answer}\n\nScore:"
    result = await llm.generate(
        prompt,
        params=GenerationParams(
            temperature=0.0, max_new_tokens=12, system=_RELEVANCE_SYSTEM
        ),
    )
    score = _parse_score(result.text)
    return (score, {"raw": result.text[:80], "evaluator_model": result.model_id})

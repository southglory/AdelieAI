"""Per-persona grounding context for chat.

The persona's system prompt declares the *voice* (tone, register,
forbidden words). This module supplies the *facts* — KG triples for
knowledge personas, evidence-file hits for legal personas — and
appends them to the system prompt for a single turn.

Why this matters: without grounding, the LLM frequently hallucinates
domain-specific lore (the dragon inventing "yellow crow ancestors"
instead of the asserted Vyrnaes / Sothryn lineage). With grounding,
the system prompt carries the asserted facts directly, dramatically
reducing fabrication.

This is the cheap version of T3/T4 tool-use — "retrieval as system
prompt" rather than mid-generation function calls. Real LLM-driven
tool calling (Qwen2.5's <tool_call> format) lands in a future
milestone; this file is the bridge.
"""
from __future__ import annotations

from core.personas.registry import Persona


def build_grounding_context(
    persona: Persona,
    *,
    user_text: str,
    graph_retriever=None,
    tool_registry=None,
    max_triples: int = 16,
    max_evidence: int = 3,
) -> str:
    """Build a grounding suffix for the persona's system prompt.

    Returns an empty string when no grounding source is available
    (e.g. industry="general", or the appropriate retriever/registry
    is not registered on app.state).
    """
    if persona.industry == "knowledge" and graph_retriever is not None:
        return _knowledge_grounding(graph_retriever, max_triples)
    if persona.industry == "legal" and tool_registry is not None:
        return _legal_grounding(tool_registry, user_text, max_evidence)
    return ""


def _knowledge_grounding(graph_retriever, max_triples: int) -> str:
    """For dragon-class personas: expand the speaker (`:Self`) and
    render the resulting triples as Korean prose. RDF turtle is hostile
    to LLM interpretation; natural-language facts are followed.

    Skips entailment-housekeeping predicates (subClassOf, equivalentClass)
    and class-membership of class entities — these don't help the LLM
    answer "who is your mother".
    """
    hits = graph_retriever.expand("Self", depth=2)
    if not hits:
        return ""
    raw = hits[0].triples[:max_triples * 2]  # over-fetch, then prune
    sentences: list[str] = []
    seen: set[str] = set()
    for t in raw:
        s = _render_fact(t)
        if s and s not in seen:
            sentences.append(s)
            seen.add(s)
    if not sentences:
        return ""
    body = "\n".join("  · " + s for s in sentences[:max_triples])
    return (
        "\n\n[당신이 알고 있는 사실 — 이것 외에는 모릅니다]\n"
        + body
        + "\n\n중요한 규칙:\n"
        + "  · 위 사실에 적힌 인물·장소만 답에 등장시키세요.\n"
        + "  · 새로운 이름·종족·장소를 만들어내지 마세요.\n"
        + "  · 모르는 것은 '기록되지 않았다' 또는 '추정 (uncertain)' 으로 표시하세요.\n"
        + "  · 답할 때 위 사실을 자연스럽게 인용하되, RDF/그래프 문법은 노출하지 마세요."
    )


# === KG triple → Korean prose ===

# URI local-name → Korean phrase (so "Dragon" appears as "용",
# "Mountain" as "산" etc. in the LLM-facing fact list).
_KO_GLOSS = {
    "Self":        "당신",
    "Dragon":      "용",
    "WingedBeing": "날개 달린 존재",
    "Mountain":    "산",
    "Treasure":    "보물",
    "Dwarf":       "드워프",
    "DwarfKing":   "드워프 왕",
    "Class":       "분류",
    "Thing":       "존재",
}


def _ko(uri_or_name: str) -> str:
    """Korean rendering of a URI local-name; falls through to the
    raw local-name (proper nouns like Vyrnaes, Sothryn, Erebor,
    Arkenstone, Thrór keep their original spellings)."""
    name = uri_or_name.lstrip(":") if uri_or_name.startswith(":") else uri_or_name
    return _KO_GLOSS.get(name, name)


_PREDICATE_MAP_SELF = {
    # Subject is :Self → render in 2nd-person ("당신은 ...")
    ":type":         lambda o: f"당신은 {_ko(o)} 입니다.",
    ":a":            lambda o: f"당신은 {_ko(o)} 입니다.",
    ":age":          lambda o: f"당신의 나이는 {o} 살입니다.",
    ":nameLost":     lambda o: (
        "당신의 이름은 아직 잊히지 않았습니다."
        if o.lower() in {"false", "0"}
        else "당신의 이름은 잊혔습니다."
    ),
    ":lairIn":       lambda o: f"당신의 거처는 {_ko(o)} 산입니다.",
    ":descendantOf": lambda o: f"당신의 어미는 {_ko(o)} 입니다.",
}

_PREDICATE_MAP_3RD = {
    ":descendantOf":     lambda s, o: f"{_ko(s)} 의 어미는 {_ko(o)} 입니다.",
    ":lairIn":           lambda s, o: f"{_ko(s)} 의 거처는 {_ko(o)} 입니다.",
    ":type":             lambda s, o: (
        None if o.lstrip(":") in {"Class", "Thing"}
        else f"{_ko(s)} 는 {_ko(o)} 입니다."
    ),
    ":a":                lambda s, o: (
        None if o.lstrip(":") in {"Class", "Thing"}
        else f"{_ko(s)} 는 {_ko(o)} 입니다."
    ),
    ":containsTreasure": lambda s, o: f"{_ko(s)} 에는 보물 {_ko(o)} 이 있습니다.",
    ":hostsRace":        lambda s, o: f"{_ko(s)} 에는 {_ko(o)} 종족이 살고 있습니다.",
    ":discoveredBy":     lambda s, o: f"{_ko(s)} 은 {_ko(o)} 에 의해 발견되었습니다.",
    ":wasAttackedBy":    lambda s, o: (
        # When the speaker (:Self) is the attacker, render in 1인칭.
        f"당신은 800년 전 {_ko(s)} 을(를) 공격한 적이 있습니다."
        if o.lstrip(":") == "Self"
        else f"{_ko(s)} 은 {_ko(o)} 에게 공격받았습니다."
    ),
}

# Predicates that just clutter the LLM's input and don't help answer
# user-facing questions.
_SKIP_PREDICATES = {":subClassOf", ":equivalentClass", ":sameAs"}


def _render_fact(triple) -> str | None:
    """Render a single Triple as a Korean sentence, or None to skip."""
    s, p, o = triple.subject, triple.predicate, triple.object
    if p in _SKIP_PREDICATES:
        return None
    if s == ":Self" and p in _PREDICATE_MAP_SELF:
        return _PREDICATE_MAP_SELF[p](o)
    if p in _PREDICATE_MAP_3RD:
        return _PREDICATE_MAP_3RD[p](s, o)
    return None


def _legal_grounding(tool_registry, user_text: str, max_evidence: int) -> str:
    """For detective-class personas: pre-run evidence_search with the
    user's message and inject top hits."""
    tool = tool_registry.get("evidence_search")
    if tool is None:
        return ""
    try:
        result = tool.call({"query": user_text})
    except Exception:
        return ""
    hits = result.get("hits", []) if isinstance(result, dict) else []
    catalog = "evidence_1.md, case_log_07.md, timeline.txt, witness_a.md"
    if not hits:
        return (
            "\n\n[evidence_search 결과: 사건 파일에서 일치 0건]\n"
            f"현재 알려진 사건 파일: {catalog}\n"
            "사건 파일에 없는 사실을 만들어내지 마세요. "
            "추론은 '추정' 표지를 붙여 명시하세요."
        )
    formatted = "\n".join(
        f"  [{h.get('path', '?')}] " + (h.get('snippet') or '').strip()[:240]
        for h in hits[:max_evidence]
    )
    return (
        f"\n\n[evidence_search('{user_text[:60]}') 결과 — 사건 파일 발췌]\n"
        + formatted
        + "\n\n중요한 규칙:\n"
        + "  · 위 발췌의 사실만 답에 사용하세요. 다른 시간·인물·장소를 만들지 마세요.\n"
        + "  · 시간, 인물 이름, 위치를 답에 인용할 때는 위 발췌에 적힌 그대로만 쓰세요.\n"
        + "  · 발췌에 없는 사실을 추론할 때는 '추정 (uncertain)' 표지를 명시하세요.\n"
        + "  · 답할 때 자료 출처 (괄호 안의 파일명) 를 한 번이라도 언급하면 신뢰도가 올라갑니다."
    )


__all__ = ["build_grounding_context"]

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
prompt" rather than mid-generation function calls.

# Architecture: data ↔ code separation

This module is **generic** — it knows how to walk a graph, render
triples, and assemble a prompt suffix, but it does NOT hardcode any
domain vocabulary. All Korean labels (`어미`, `용`, `보물` …) and
predicate templates live in `personas/{persona_id}/grounding_templates.yaml`.

Adding a new knowledge-vertical persona (e.g. medical advisor) means:
  1. Author its `grounding_templates.yaml` with the right vocabulary
     (`{s} 의 주치의는 {o}` instead of `{s} 의 어미는 {o}` etc.)
  2. Reuse this module unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.personas.registry import Persona


def _adelie_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_templates(persona_id: str) -> dict[str, Any] | None:
    """Load `personas/{persona_id}/grounding_templates.yaml` if present."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return None
    path = _adelie_root() / "personas" / persona_id / "grounding_templates.yaml"
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
        templates = _load_templates(persona.persona_id) or _DEFAULT_KG_TEMPLATES
        return _knowledge_grounding(graph_retriever, max_triples, templates)
    if persona.industry == "legal" and tool_registry is not None:
        return _legal_grounding(tool_registry, user_text, max_evidence)
    return ""


# === Generic KG → prose renderer ===

def _local(uri: str) -> str:
    return uri.lstrip(":") if uri.startswith(":") else uri


def _gloss(name: str, glossary: dict[str, str]) -> str:
    """Korean rendering of a URI local-name; falls through to the
    raw local-name (proper nouns kept verbatim)."""
    n = _local(name)
    return glossary.get(n, n)


def _render_fact(triple, templates: dict[str, Any]) -> str | None:
    """Render a single Triple as a Korean sentence using templates."""
    s_local = _local(triple.subject)
    p_local = _local(triple.predicate)
    o_local = _local(triple.object)

    skip_pred = set(templates.get("skip_predicates", []))
    if p_local in skip_pred:
        return None
    skip_type_obj = set(templates.get("skip_type_objects", []))
    if p_local in {"type", "a"} and o_local in skip_type_obj:
        return None

    glossary = templates.get("class_glossary", {})
    s_ko = _gloss(s_local, glossary)
    o_ko = _gloss(o_local, glossary)

    self_t = templates.get("self_templates", {})
    third_t = templates.get("third_templates", {})

    # Self-anchored facts → 2인칭
    if s_local == "Self":
        # boolean-typed predicates can have value-suffixed templates
        # (e.g. nameLost_false / nameLost_true)
        suffixed = f"{p_local}_{triple.object.lower()}"
        if suffixed in self_t:
            return self_t[suffixed].format(s=s_ko, o=o_ko)
        if p_local in self_t:
            return self_t[p_local].format(s=s_ko, o=o_ko)

    # 3rd-person: also support special "_self" suffix when object is Self
    if o_local == "Self":
        suffixed = f"{p_local}_self"
        if suffixed in third_t:
            return third_t[suffixed].format(s=s_ko, o=o_ko)
    if p_local in third_t:
        return third_t[p_local].format(s=s_ko, o=o_ko)
    return None


def _knowledge_grounding(
    graph_retriever, max_triples: int, templates: dict[str, Any]
) -> str:
    """Generic KG-RAG grounding using per-persona templates."""
    hits = graph_retriever.expand("Self", depth=2)
    if not hits:
        return ""
    raw = hits[0].triples[:max_triples * 2]
    sentences: list[str] = []
    seen: set[str] = set()
    for t in raw:
        s = _render_fact(t, templates)
        if s and s not in seen:
            sentences.append(s)
            seen.add(s)
    if not sentences:
        return ""

    body = "\n".join("  · " + s for s in sentences[:max_triples])
    chain_summary = _chain_summary(graph_retriever, templates)

    header = templates.get("header", "[알려진 사실]")
    rules = templates.get("rules", "")
    return f"\n\n{header}\n{body}{chain_summary or ''}{rules}"


def _chain_summary(graph_retriever, templates: dict[str, Any]) -> str:
    """Walk the configured chain (e.g. descendantOf from Self) and emit
    a Korean summary so the LLM gets the lineage right."""
    chain_cfg = templates.get("chain")
    if not chain_cfg:
        return ""
    pred = chain_cfg["predicate"]
    start = chain_cfg["start"]
    glossary = templates.get("class_glossary", {})
    chain_templates = chain_cfg.get("templates", {})

    chain: list[str] = []
    current = start
    visited: set[str] = set()
    for _ in range(8):
        if current in visited:
            break
        visited.add(current)
        hits = graph_retriever.expand(current, depth=1)
        if not hits:
            break
        next_ancestor = None
        for t in hits[0].triples:
            if (_local(t.subject) == current
                    and _local(t.predicate) == pred
                    and not t.inferred):
                next_ancestor = _local(t.object)
                break
        if not next_ancestor:
            break
        chain.append(next_ancestor)
        current = next_ancestor
    if not chain:
        return ""

    header = chain_cfg.get("header", "[체인]")
    if len(chain) == 1:
        return (
            f"\n\n{header}\n"
            + chain_templates.get("direct", "  · {direct}").format(direct=_gloss(chain[0], glossary))
        )

    deepest = _gloss(chain[-1], glossary)
    parent_of_deepest = _gloss(chain[-2], glossary)
    direct = _gloss(chain[0], glossary)
    chain_with_labels = " → ".join(
        f"{_gloss(name, glossary)}" + (
            f" ({chain_cfg['labels']['direct']})" if i == 0
            else (f" ({chain_cfg['labels']['deepest']})" if i == len(chain) - 1
                  else "")
        )
        for i, name in enumerate(chain)
    )
    lines = [
        chain_templates.get("deepest", "  · 가장 깊은: {deepest}").format(
            deepest=deepest, parent_of_deepest=parent_of_deepest
        ),
        chain_templates.get("direct", "  · 직계: {direct}").format(direct=direct),
        chain_templates.get("full", "  · 전체: {chain_with_labels}").format(
            chain_with_labels=chain_with_labels
        ),
    ]
    return "\n\n" + header + "\n" + "\n".join(lines)


# === Legal grounding (no per-persona template needed yet) ===

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


# === Default KG templates (fallback when persona has no yaml) ===

_DEFAULT_KG_TEMPLATES: dict[str, Any] = {
    "class_glossary": {"Self": "당신"},
    "self_templates": {
        "type": "당신은 {o} 입니다.",
        "a": "당신은 {o} 입니다.",
    },
    "third_templates": {
        "type": "{s} 는 {o} 입니다.",
        "a": "{s} 는 {o} 입니다.",
    },
    "skip_predicates": ["subClassOf", "equivalentClass", "sameAs"],
    "skip_type_objects": ["Class", "Thing"],
    "header": "[알려진 사실]",
    "rules": (
        "\n\n중요한 규칙:\n"
        "  · 위 사실에 없는 정보를 만들지 마세요.\n"
        "  · 모르는 것은 '기록되지 않았다' 라고 답하세요."
    ),
}


__all__ = ["build_grounding_context"]

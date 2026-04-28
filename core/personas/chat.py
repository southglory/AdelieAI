"""Persona-aware chat generation.

Builds a chat-style prompt out of the persona's system prompt and the
prior conversation, calls the active LLM, and returns both the new
user turn and the freshly-generated assistant turn so the UI can
append both at once.
"""

from datetime import datetime, timezone

from core.personas.registry import Persona
from core.personas.store import ChatStore, ChatTurn
from core.serving.protocols import GenerationParams, LLMClient


def _format_history(history: list[ChatTurn], user_text: str) -> str:
    """Render the conversation as a single prompt string.

    The active LLM client may apply its own chat template; we still
    include role markers so the model sees a multi-turn shape even
    when the client treats the input as a flat string.
    """
    parts: list[str] = []
    for turn in history:
        if turn.role == "user":
            parts.append(f"User: {turn.content}\n")
        else:
            parts.append(f"Assistant: {turn.content}\n")
    parts.append(f"User: {user_text}\nAssistant: ")
    return "".join(parts)


async def submit_chat_turn(
    *,
    chat_store: ChatStore,
    llm: LLMClient,
    persona: Persona,
    user_id: str,
    user_text: str,
    params: GenerationParams | None = None,
    grounding_context: str | None = None,
) -> tuple[ChatTurn, ChatTurn]:
    """Persist the user message, run the LLM, persist the assistant
    reply, return both turns in order.

    `grounding_context` (optional) is appended to `persona.system_prompt`
    for this turn's generation only. It carries per-turn facts retrieved
    from a KG (knowledge personas) or a tool registry (legal personas)
    so the LLM does not have to fabricate domain-specific lore.
    """
    user_text = user_text.strip()
    if not user_text:
        raise ValueError("user_text is empty")

    base_params = params or GenerationParams()
    augmented_system = persona.system_prompt
    if grounding_context:
        augmented_system = persona.system_prompt + grounding_context
    persona_params = base_params.model_copy(
        update={"system": augmented_system}
    )

    history = await chat_store.list_turns(persona.persona_id, user_id)
    prompt = _format_history(history, user_text)

    user_turn = await chat_store.append(
        ChatTurn(
            id=None,
            persona_id=persona.persona_id,
            user_id=user_id,
            role="user",
            content=user_text,
            tokens_in=None,
            tokens_out=None,
            latency_ms=None,
            created_at=datetime.now(timezone.utc),
        )
    )

    result = await llm.generate(prompt, persona_params)

    assistant_turn = await chat_store.append(
        ChatTurn(
            id=None,
            persona_id=persona.persona_id,
            user_id=user_id,
            role="assistant",
            content=result.text,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            latency_ms=result.latency_ms,
            created_at=datetime.now(timezone.utc),
        )
    )

    return user_turn, assistant_turn

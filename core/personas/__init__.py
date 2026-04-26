"""Persona engine — pick a character, hold a conversation, see the cost.

A persona = (display_name, emoji, system_prompt, optional adapter hint).
v0.1.5 ships three role-play personas backed by the active LLM via
system-prompt switching. The full persona-pack format (.adelie) lands
later — this module is the runtime that consumes it.
"""

from core.personas.registry import (
    DEFAULT_PERSONAS,
    Persona,
    get_persona,
    list_personas,
)

__all__ = [
    "DEFAULT_PERSONAS",
    "Persona",
    "get_persona",
    "list_personas",
]

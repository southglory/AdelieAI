"""Protocols for tool-use (T3).

Skeleton only — no concrete implementations yet. The shape mirrors
the OpenAI / Anthropic function-calling pattern (name + JSON schema +
synchronous call) so future LLM clients can negotiate tools without
custom adapters.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class ToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    model_config = ConfigDict(frozen=True)

    name: str
    arguments: dict[str, Any]
    call_id: str  # echoed back in ToolResult for correlation


class ToolResult(BaseModel):
    """The result of one ToolCall."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    output: Any
    error: str | None = None


@runtime_checkable
class Tool(Protocol):
    """A side-effect-bearing capability the LLM can opt to invoke.

    Implementers expose:
      * name            — stable identifier (what the LLM types in tool calls)
      * description     — short human-readable purpose
      * input_schema    — JSON-schema-style dict describing arguments

    Most tools are synchronous because they wrap fast operations
    (lookup, calc). Long-running tools (web search, external API)
    should still expose a synchronous facade — concurrency is the
    runner's responsibility, not the tool's.
    """

    name: str
    description: str
    input_schema: dict[str, Any]

    def call(self, arguments: dict[str, Any]) -> Any: ...


class ToolRegistry:
    """In-memory registry. A vertical (legal / gaming / knowledge)
    constructs one and hands it to the persona's runner.

    Keeping this concrete (vs Protocol) because the registry shape is
    not a swap point — only its members are.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI-function-style descriptors for prompt injection."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)

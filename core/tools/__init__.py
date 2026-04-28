"""T3 — tool-use protocols.

A tool is something the LLM can *decide* to call mid-generation:
retrieval, calculator, web search, internal API, etc. The Tool
protocol below is intentionally minimal — it leaves serialization
of arguments / results to each implementation.

This module ships the Protocol only. Concrete tools land per
persona / vertical (e.g. legal vertical's evidence search).

See docs/CAPABILITY_TIERS.md, tier T3.
"""

from core.tools.protocols import Tool, ToolCall, ToolResult, ToolRegistry

__all__ = ["Tool", "ToolCall", "ToolResult", "ToolRegistry"]

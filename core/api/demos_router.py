"""Industry-vertical demo pages.

Each vertical mounts at `/demo/{vertical}` and showcases the persona
engine through a domain-shaped UX. Backend is the same engine — the
demos differ only in framing, layout, and which persona / tier they
foreground.

Verticals (initial set):
  /demo/gaming     → cynical_merchant     (T2)
  /demo/legal      → cold_detective       (T3)
  /demo/knowledge  → ancient_dragon       (T4)

The template under `templates/demos/{vertical}/index.html` is the
landing page. Step 1 ships placeholders; Step 2 replaces gaming with
the frontend-design pass; Steps 3+ extend to legal and knowledge.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from core.personas.registry import get_persona

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Vertical → (persona_id, tier_label, blurb).
# When a persona moves out of placeholder state, just update its
# entry in registry.py — this map is the authoritative *route*
# registration, not the persona definition.
VERTICALS: dict[str, dict[str, str]] = {
    "gaming": {
        "persona_id": "cynical_merchant",
        "tier_label": "T2 — Standard NPC (LoRA + vector RAG)",
        "blurb": "RPG game NPC vertical. Voice + lore RAG, no tools required.",
    },
    "legal": {
        "persona_id": "cold_detective",
        "tier_label": "T3 — Vertical Advisor (+ tool-use, citations)",
        "blurb": (
            "Investigation / legal advisor. Evidence sidebar, citation chips, "
            "retrieval surfaced as tool calls."
        ),
    },
    "knowledge": {
        "persona_id": "ancient_dragon",
        "tier_label": "T4 — Domain Expert (+ RDF/OWL KG, OWL reasoner)",
        "blurb": (
            "Domain wiki / lore advisor. Graph visualization, SPARQL trace, "
            "OWL-validated answers."
        ),
    },
}


def build_demos_router() -> APIRouter:
    router = APIRouter(prefix="/demo", tags=["demos"])

    @router.get(
        "/",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def demo_index(request: Request) -> HTMLResponse:
        """Landing page listing all available verticals."""
        return templates.TemplateResponse(
            request,
            "demos/index.html",
            {
                "verticals": [
                    {"slug": slug, **meta, "persona": get_persona(meta["persona_id"])}
                    for slug, meta in VERTICALS.items()
                ],
            },
        )

    @router.get(
        "/{vertical}",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    async def demo_vertical(request: Request, vertical: str) -> HTMLResponse:
        meta = VERTICALS.get(vertical)
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown vertical: {vertical}",
            )
        persona = get_persona(meta["persona_id"])  # may be None until persona is added
        return templates.TemplateResponse(
            request,
            f"demos/{vertical}/index.html",
            {
                "vertical": vertical,
                "tier_label": meta["tier_label"],
                "blurb": meta["blurb"],
                "persona": persona,
                "persona_id": meta["persona_id"],
            },
        )

    return router

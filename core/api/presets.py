from fastapi import APIRouter, HTTPException, status

from core.agent.presets import PRESETS, Preset, list_presets


def build_presets_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1/presets", tags=["presets"])

    @router.get("", response_model=list[Preset])
    async def get_all() -> list[Preset]:
        return list_presets()

    @router.get("/{name}", response_model=Preset)
    async def get_one(name: str) -> Preset:
        preset = PRESETS.get(name)
        if preset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown preset: {name}",
            )
        return preset

    return router

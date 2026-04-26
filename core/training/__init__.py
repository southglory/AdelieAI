"""Training pipeline — LoRA fine-tuning for Qwen2.5 family.

Production owners: us. We download base weights from HF (with
MANIFEST.json provenance) and emit our own adapters into
models/ours/{name}/ with their own MANIFEST + recipe.md.
"""

from core.training.dataset import (
    GENERAL_PAIRS,
    GENERAL_SYSTEM,
    ROLEPLAY_PAIRS,
    ROLEPLAY_SYSTEM,
    build_mixed_dataset,
    build_roleplay_dataset,
)
from core.training.lora_config import (
    QWEN_LORA_TARGETS,
    default_lora_config,
)

__all__ = [
    "GENERAL_PAIRS",
    "GENERAL_SYSTEM",
    "QWEN_LORA_TARGETS",
    "ROLEPLAY_PAIRS",
    "ROLEPLAY_SYSTEM",
    "build_mixed_dataset",
    "build_roleplay_dataset",
    "default_lora_config",
]

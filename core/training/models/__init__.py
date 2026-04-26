"""From-scratch model implementations.

mission/02_llm-tuning-training: "작은 모델(예: guppy LLM)부터
시작해 점점 큰 모델로 직접 파인튜닝·훈련하며, 아키텍처를 수정하는
연구·실험을 병행한다."

This package is the *we built it ourselves* track. transformers
library is not a dependency of any model class here — only torch.
"""

from core.training.models.nano_gpt import (
    NanoGPT,
    NanoGPTConfig,
    causal_mask,
)

__all__ = [
    "NanoGPT",
    "NanoGPTConfig",
    "causal_mask",
]

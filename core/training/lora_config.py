"""Standard LoRA target spec for Qwen2 family.

Targets all 7 attention + MLP projection blocks (q,k,v,o + gate,up,down).
For Qwen2-7B at r=16 this is ~50M trainable params (~0.65% of base).
"""

QWEN_LORA_TARGETS: list[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def default_lora_config(
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
):
    """Returns a peft.LoraConfig. Imported lazily so the module is
    importable even when peft isn't installed (test envs).
    """
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=QWEN_LORA_TARGETS,
    )

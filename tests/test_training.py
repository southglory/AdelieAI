"""Training pipeline unit tests — no GPU / no actual training."""

import pytest

from core.training.dataset import (
    ROLEPLAY_PAIRS,
    ROLEPLAY_SYSTEM,
    build_roleplay_dataset,
    dataset_stats,
)
from core.training.lora_config import QWEN_LORA_TARGETS


def test_dataset_has_meaningful_size() -> None:
    assert len(ROLEPLAY_PAIRS) >= 50
    assert all("user" in p and "assistant" in p for p in ROLEPLAY_PAIRS)


def test_dataset_pairs_are_unique_user_prompts() -> None:
    users = [p["user"] for p in ROLEPLAY_PAIRS]
    assert len(users) == len(set(users)), "duplicate user prompts found"


def test_dataset_pairs_are_pure_korean() -> None:
    """Sanity check that every assistant turn we trained on is Korean —
    we explicitly want the model to learn 한국어 강제. Allow basic
    punctuation, ASCII numbers, and short ASCII words like onomatopoeia.
    """
    for p in ROLEPLAY_PAIRS:
        text = p["assistant"]
        non_korean_letters = [
            ch for ch in text if ch.isalpha() and not ("가" <= ch <= "힯")
        ]
        # at most a handful of latin chars per pair (e.g., proper nouns)
        assert len(non_korean_letters) <= 8, (
            f"too many non-Korean letters in: {text!r}"
        )


def test_dataset_excludes_meta_phrases() -> None:
    banned = ["AI", "인공지능", "상상해보면", "사실 저는", "실제로는"]
    for p in ROLEPLAY_PAIRS:
        for phrase in banned:
            assert phrase not in p["assistant"], (
                f"banned phrase {phrase!r} in: {p['assistant']!r}"
            )


def test_system_prompt_is_short_enough() -> None:
    assert 0 < len(ROLEPLAY_SYSTEM) < 300


def test_lora_targets_cover_attention_and_mlp() -> None:
    assert "q_proj" in QWEN_LORA_TARGETS
    assert "k_proj" in QWEN_LORA_TARGETS
    assert "v_proj" in QWEN_LORA_TARGETS
    assert "o_proj" in QWEN_LORA_TARGETS
    assert "gate_proj" in QWEN_LORA_TARGETS
    assert "up_proj" in QWEN_LORA_TARGETS
    assert "down_proj" in QWEN_LORA_TARGETS


def test_dataset_stats_shape() -> None:
    s = dataset_stats()
    assert s["n_pairs"] == len(ROLEPLAY_PAIRS)
    assert s["user_chars_avg"] > 0
    assert s["assistant_chars_avg"] > 0


def test_build_dataset_returns_messages_format() -> None:
    pytest.importorskip("datasets")
    ds = build_roleplay_dataset()
    assert len(ds) == len(ROLEPLAY_PAIRS)
    row = ds[0]
    assert "messages" in row
    msgs = row["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"
    assert msgs[0]["content"] == ROLEPLAY_SYSTEM


def test_lora_config_factory_buildable_when_peft_installed() -> None:
    pytest.importorskip("peft")
    from core.training.lora_config import default_lora_config

    cfg = default_lora_config()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert cfg.task_type == "CAUSAL_LM"

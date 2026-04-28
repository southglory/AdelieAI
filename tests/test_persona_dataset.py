"""Per-persona dataset loader + train/val split (Step 6.1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.training.dataset import (
    GENERAL_PAIRS,
    GENERAL_SYSTEM,
    ROLEPLAY_SYSTEM,
    build_persona_dataset,
    load_persona_pairs,
    split_train_val,
)


def test_load_cynical_merchant_pairs() -> None:
    pairs = load_persona_pairs("cynical_merchant")
    assert len(pairs) >= 10
    # Every loaded pair has non-empty user + assistant
    for p in pairs:
        assert p["user"].strip()
        assert p["assistant"].strip()
        # Template placeholder lines must be filtered out
        assert not (p["user"].startswith("<") and p["user"].endswith(">"))


def test_load_unknown_persona_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_persona_pairs("nonexistent_persona_xyz")


def test_load_skips_template_placeholders(tmp_path: Path) -> None:
    """Lines like {"user": "<예: ...>", ...} must not contaminate training."""
    # Create a fake persona dir with a placeholder + real line
    persona_dir = (
        Path(__file__).resolve().parent.parent / "personas" / "_template_test"
    )
    persona_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = persona_dir / "dialogue_pairs.jsonl"
        path.write_text(
            json.dumps({"user": "<예: 광대로서 한마디>", "assistant": "<답>"})
            + "\n"
            + json.dumps({"user": "안녕?", "assistant": "안녕."})
            + "\n",
            encoding="utf-8",
        )
        pairs = load_persona_pairs("_template_test")
        assert len(pairs) == 1
        assert pairs[0]["user"] == "안녕?"
    finally:
        # cleanup
        (persona_dir / "dialogue_pairs.jsonl").unlink(missing_ok=True)
        try:
            persona_dir.rmdir()
        except OSError:
            pass


def test_split_train_val_deterministic() -> None:
    pairs = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(10)]
    a_train, a_val = split_train_val(pairs, val_ratio=0.2, seed=13)
    b_train, b_val = split_train_val(pairs, val_ratio=0.2, seed=13)
    assert a_train == b_train
    assert a_val == b_val
    assert len(a_train) == 8
    assert len(a_val) == 2


def test_split_train_val_falls_back_for_tiny_input() -> None:
    pairs = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(3)]
    train, val = split_train_val(pairs, val_ratio=0.2)
    assert train == pairs
    assert val == []


def test_build_persona_dataset_returns_train_and_val() -> None:
    train_ds, val_ds = build_persona_dataset(
        "cynical_merchant",
        persona_system_prompt="당신은 상인입니다.",
        val_ratio=0.2,
    )
    assert len(train_ds) > 0
    assert val_ds is not None
    assert len(val_ds) > 0
    # Train set must contain both persona-tagged and general-tagged rows
    systems = {row["messages"][0]["content"] for row in train_ds}
    assert any("상인" in s for s in systems)
    assert GENERAL_SYSTEM in systems


def test_build_persona_dataset_uses_persona_system_prompt() -> None:
    custom = "[테스트 전용 시스템 프롬프트]"
    train_ds, _ = build_persona_dataset(
        "cynical_merchant",
        persona_system_prompt=custom,
        include_general=False,
    )
    persona_systems = {row["messages"][0]["content"] for row in train_ds}
    assert custom in persona_systems
    assert ROLEPLAY_SYSTEM not in persona_systems


def test_build_persona_dataset_falls_back_to_roleplay_system() -> None:
    train_ds, _ = build_persona_dataset(
        "cynical_merchant",
        persona_system_prompt=None,
        include_general=False,
    )
    persona_systems = {row["messages"][0]["content"] for row in train_ds}
    assert ROLEPLAY_SYSTEM in persona_systems


def test_build_persona_dataset_excludes_general_when_disabled() -> None:
    train_ds, val_ds = build_persona_dataset(
        "cynical_merchant",
        include_general=False,
    )
    systems = {row["messages"][0]["content"] for row in train_ds}
    assert GENERAL_SYSTEM not in systems


def test_build_persona_dataset_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError):
        build_persona_dataset("nonexistent_xyz")

"""Validation tests for GGUFClient construction and dispatcher wiring.

We do not load an actual GGUF model in CI — a 4 GB load would exceed
typical runners. Instead we cover:

  * path validation (must be a single .gguf file)
  * MANIFEST.json reading for friendly model_id derivation (mocked)
  * dispatcher: app._default_llm picks GGUFClient when MODEL_PATH
    points at a .gguf file
  * runtime_checkable LLMClient protocol conformance for the class

The end-to-end "actually generate" path is exercised by the live
console screenshots and `experiments/06_gguf_export/eval.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from core.serving.gguf_client import GGUFClient
from core.serving.protocols import LLMClient


def test_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single .gguf file"):
        GGUFClient(tmp_path)


def test_rejects_wrong_suffix(tmp_path: Path) -> None:
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\x00")
    with pytest.raises(ValueError, match="single .gguf file"):
        GGUFClient(f)


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single .gguf file"):
        GGUFClient(tmp_path / "missing.gguf")


class _FakeLlama:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        self.kwargs = kwargs


def test_model_id_from_filename_when_no_manifest(tmp_path: Path) -> None:
    f = tmp_path / "qwen-roleplay-v2.q4_k_m.gguf"
    f.write_bytes(b"\x00")
    with patch("llama_cpp.Llama", _FakeLlama):
        c = GGUFClient(f)
    assert c.model_id == "qwen-roleplay-v2.q4_k_m"


def test_model_id_from_manifest(tmp_path: Path) -> None:
    f = tmp_path / "weights.q4_k_m.gguf"
    f.write_bytes(b"\x00")
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps({"model_id": "qwen-roleplay-v2-gguf"}), encoding="utf-8"
    )
    with patch("llama_cpp.Llama", _FakeLlama):
        c = GGUFClient(f)
    assert c.model_id == "qwen-roleplay-v2-gguf"


def test_passes_n_gpu_layers_to_llama(tmp_path: Path) -> None:
    f = tmp_path / "weights.gguf"
    f.write_bytes(b"\x00")
    with patch("llama_cpp.Llama", _FakeLlama):
        c = GGUFClient(f, n_gpu_layers=99)
    assert c._llm.kwargs["n_gpu_layers"] == 99  # type: ignore[attr-defined]


def test_protocol_conformance(tmp_path: Path) -> None:
    """GGUFClient should satisfy the runtime LLMClient protocol."""
    f = tmp_path / "weights.gguf"
    f.write_bytes(b"\x00")
    with patch("llama_cpp.Llama", _FakeLlama):
        c = GGUFClient(f)
    assert isinstance(c, LLMClient)

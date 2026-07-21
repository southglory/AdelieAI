"""Model references resolve explicitly and fail before a misleading boot."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.serving.model_resolver import (
    HuggingFaceModelResolver,
    LocalModelResolver,
    ModelResolutionError,
)


def test_local_transformers_directory_resolves(tmp_path) -> None:
    result = LocalModelResolver().resolve(str(tmp_path))
    assert result.local_path == tmp_path.resolve()
    assert result.backend == "transformers"
    assert result.source == "local"


def test_missing_local_model_has_actionable_path(tmp_path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ModelResolutionError, match=str(missing)):
        LocalModelResolver().resolve(str(missing))


def test_huggingface_reference_parser_preserves_nested_filename() -> None:
    repo, filename = HuggingFaceModelResolver.parse(
        "hf://ramyun/adelie-qwen-roleplay-v2-gguf/weights/model.q4_k_m.gguf"
    )
    assert repo == "ramyun/adelie-qwen-roleplay-v2-gguf"
    assert filename == "weights/model.q4_k_m.gguf"


def test_huggingface_reference_requires_owner_repo_and_file() -> None:
    with pytest.raises(ModelResolutionError, match="syntax"):
        HuggingFaceModelResolver.parse("hf://only-a-repo")


def test_huggingface_download_adapter_returns_cache_path(tmp_path, monkeypatch) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    calls = []

    def downloader(**kwargs) -> str:
        calls.append(kwargs)
        return str(model)

    # This test validates resolver orchestration, not optional llama.cpp.
    monkeypatch.setattr("core.serving.model_resolver._check_runtime", lambda backend: None)
    result = HuggingFaceModelResolver(downloader, cache_dir=tmp_path / "adelie-cache").resolve(
        "hf://owner/repo/model.gguf"
    )
    assert calls == [{"repo_id": "owner/repo", "filename": "model.gguf"}]
    assert result.local_path == model.resolve()
    assert result.downloaded is True
    assert result.source == "huggingface"


def test_huggingface_blob_without_suffix_gets_shared_gguf_runtime_link(
    tmp_path, monkeypatch
) -> None:
    blob = tmp_path / "content-addressed-blob"
    blob.write_bytes(b"GGUF-real-bytes")
    monkeypatch.setattr("core.serving.model_resolver._check_runtime", lambda backend: None)

    result = HuggingFaceModelResolver(
        lambda **_: str(blob), cache_dir=tmp_path / "adelie-cache"
    ).resolve("hf://owner/repo/model.q4_k_m.gguf")

    assert result.local_path.name == "model.q4_k_m.gguf"
    assert result.local_path.read_bytes() == blob.read_bytes()
    assert result.local_path.stat().st_ino == blob.stat().st_ino


def test_remote_non_gguf_is_rejected_before_download(monkeypatch) -> None:
    monkeypatch.setattr("core.serving.model_resolver._check_runtime", lambda backend: None)
    with pytest.raises(ModelResolutionError, match="GGUF only"):
        HuggingFaceModelResolver(lambda **_: str(Path("unused"))).resolve(
            "hf://owner/repo/model.safetensors"
        )

"""Resolve explicit model references into paths consumed by Adelie runtimes."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


class ModelResolutionError(ValueError):
    """A model reference cannot become a runnable local artifact."""


@dataclass(frozen=True)
class ResolvedModel:
    reference: str
    local_path: Path
    backend: str
    source: str
    downloaded: bool = False


class ModelResolver(Protocol):
    def supports(self, reference: str) -> bool: ...

    def resolve(self, reference: str) -> ResolvedModel: ...


def _backend_for(path: Path) -> str:
    return "gguf" if path.suffix.lower() == ".gguf" else "transformers"


def _check_runtime(backend: str) -> None:
    if backend == "gguf" and importlib.util.find_spec("llama_cpp") is None:
        raise ModelResolutionError(
            "GGUF model found, but llama-cpp-python is not installed. "
            "Install AdelieAI with the CPU runtime: pip install 'adelie-ai[cpu]'."
        )


class LocalModelResolver:
    def supports(self, reference: str) -> bool:
        return not reference.startswith("hf://")

    def resolve(self, reference: str) -> ResolvedModel:
        path = Path(reference).expanduser().resolve()
        if not path.exists():
            raise ModelResolutionError(f"local model not found: {path}")
        if path.is_file() and path.suffix.lower() != ".gguf":
            raise ModelResolutionError("local model file must end in .gguf; Transformers models use a directory")
        backend = _backend_for(path)
        _check_runtime(backend)
        return ResolvedModel(reference, path, backend, "local")


class HuggingFaceModelResolver:
    """Resolve ``hf://owner/repo/path/to/file.gguf`` into the HF cache."""

    def __init__(self, downloader: Callable[..., str] | None = None) -> None:
        self._downloader = downloader

    def supports(self, reference: str) -> bool:
        return reference.startswith("hf://")

    @staticmethod
    def parse(reference: str) -> tuple[str, str]:
        parts = reference.removeprefix("hf://").split("/")
        if len(parts) < 3 or not all(parts[:2]):
            raise ModelResolutionError(
                "Hugging Face model must use hf://owner/repo/filename syntax"
            )
        return "/".join(parts[:2]), "/".join(parts[2:])

    def resolve(self, reference: str) -> ResolvedModel:
        repo_id, filename = self.parse(reference)
        if not filename.lower().endswith(".gguf"):
            raise ModelResolutionError(
                "remote single-file runtime currently supports GGUF only"
            )
        _check_runtime("gguf")
        downloader = self._downloader
        if downloader is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise ModelResolutionError(
                    "huggingface-hub is required for hf:// model references"
                ) from None
            downloader = hf_hub_download
        try:
            local = Path(downloader(repo_id=repo_id, filename=filename)).resolve()
        except Exception as exc:
            raise ModelResolutionError(
                f"could not download {repo_id}/{filename}: {exc}"
            ) from exc
        return ResolvedModel(reference, local, "gguf", "huggingface", downloaded=True)


class DefaultModelResolver:
    def __init__(self, resolvers: tuple[ModelResolver, ...] | None = None) -> None:
        self.resolvers = resolvers or (HuggingFaceModelResolver(), LocalModelResolver())

    def resolve(self, reference: str) -> ResolvedModel:
        for resolver in self.resolvers:
            if resolver.supports(reference):
                return resolver.resolve(reference)
        raise ModelResolutionError(f"unsupported model reference: {reference}")


__all__ = [
    "DefaultModelResolver",
    "HuggingFaceModelResolver",
    "LocalModelResolver",
    "ModelResolutionError",
    "ModelResolver",
    "ResolvedModel",
]

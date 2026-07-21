"""Resolve explicit model references into paths consumed by Adelie runtimes."""

from __future__ import annotations

import importlib.util
import os
import shutil
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

    def __init__(
        self,
        downloader: Callable[..., str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._downloader = downloader
        self.cache_dir = Path(
            cache_dir
            or os.environ.get("ADELIE_MODEL_CACHE", Path.home() / ".cache" / "adelie" / "models")
        )

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
            downloaded = Path(downloader(repo_id=repo_id, filename=filename)).resolve()
        except Exception as exc:
            raise ModelResolutionError(
                f"could not download {repo_id}/{filename}: {exc}"
            ) from exc
        local = self._materialize_runtime_path(repo_id, filename, downloaded)
        return ResolvedModel(reference, local, "gguf", "huggingface", downloaded=True)

    def _materialize_runtime_path(
        self, repo_id: str, filename: str, downloaded: Path
    ) -> Path:
        """Give content-addressed HF blobs a stable runtime suffix without copying."""
        if downloaded.suffix.lower() == ".gguf":
            return downloaded
        if not downloaded.is_file():
            raise ModelResolutionError(f"downloaded model is missing: {downloaded}")

        repo_slug = repo_id.replace("/", "--")
        destination = (self.cache_dir / repo_slug / filename).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.stat().st_size == downloaded.stat().st_size:
                return destination
            destination.unlink()

        temporary = destination.with_name(destination.name + ".tmp")
        temporary.unlink(missing_ok=True)
        try:
            os.link(downloaded, temporary)
        except OSError:
            try:
                temporary.symlink_to(downloaded)
            except OSError:
                shutil.copy2(downloaded, temporary)
        temporary.replace(destination)
        return destination


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

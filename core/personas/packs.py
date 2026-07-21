"""Portable persona-pack loading and Character Card import.

This module owns the boundary between untrusted, portable character files and
the small :class:`Persona` object consumed by the runtime.  Importers normalize
external formats into a directory ending in ``.adelie``; the loader validates
that directory before the registry can expose it.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import shutil
import struct
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from core.personas.registry import Persona


DEFAULT_PACKS_DIR = Path("packs")
MAX_IMPORT_BYTES = 10 * 1024 * 1024
PERSONA_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


class PackValidationError(ValueError):
    """A portable persona file is unsafe, malformed, or incomplete."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    corpus_path: str = "rag_corpus/"
    chunk_size: int = Field(default=512, ge=64, le=8192)
    chunk_overlap: int = Field(default=64, ge=0, le=2048)
    retrieval: str = "hybrid"
    top_k: int = Field(default=4, ge=1, le=50)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    profile: str = "prompt"
    model: str | None = None


class PersonaPackManifest(BaseModel):
    """Versioned public metadata for a runnable persona pack."""

    model_config = ConfigDict(extra="allow")

    spec_version: str = "0.3"
    persona_id: str
    display_name: str
    description: str = ""
    emoji: str = "🎭"
    language: str = "ko"
    license: str = "unspecified"
    system_prompt: str = "system_prompt.md"
    base_model: str | dict[str, Any] = "Qwen/Qwen2.5-7B-Instruct"
    adapter: dict[str, Any] | None = None
    rag: RagConfig = Field(default_factory=RagConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    target_tier: int = Field(default=1, ge=1, le=5)
    industry: str = "general"
    source: dict[str, Any] = Field(default_factory=dict)

    @field_validator("persona_id")
    @classmethod
    def validate_persona_id(cls, value: str) -> str:
        if not PERSONA_ID_RE.fullmatch(value):
            raise ValueError("use 2-64 lowercase letters, digits, '_' or '-'")
        return value

    @field_validator("system_prompt")
    @classmethod
    def validate_relative_prompt_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("must be a relative path inside the pack")
        return value


@dataclass(frozen=True)
class LoadedPersonaPack:
    root: Path
    manifest: PersonaPackManifest
    persona: Persona
    source_format: str
    warnings: tuple[str, ...] = ()


@runtime_checkable
class PersonaImporter(Protocol):
    """Normalize one portable source format into an Adelie pack directory."""

    source_format: str

    def supports(self, filename: str, payload: bytes) -> bool: ...

    def import_into(self, filename: str, payload: bytes, packs_dir: Path) -> LoadedPersonaPack: ...


def _base_model_hint(value: str | dict[str, Any]) -> str:
    if isinstance(value, str):
        return value
    return str(value.get("id") or value.get("path") or "Qwen/Qwen2.5-7B-Instruct")


def _adapter_hint(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    raw = value.get("id") or value.get("path")
    return str(raw) if raw else None


def load_persona_pack(path: str | Path) -> LoadedPersonaPack:
    """Validate and load an unpacked ``*.adelie`` directory."""
    root = Path(path).resolve()
    issues: list[str] = []
    if not root.is_dir():
        raise PackValidationError(["pack must be an unpacked directory"])
    if root.suffix != ".adelie":
        issues.append("pack directory name must end in .adelie")

    manifest_path = root / "MANIFEST.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PackValidationError(["MANIFEST.json is missing"]) from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackValidationError([f"MANIFEST.json is invalid: {exc}"]) from None

    try:
        manifest = PersonaPackManifest.model_validate(raw)
    except ValidationError as exc:
        raise PackValidationError(
            [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        ) from None

    prompt_path = (root / manifest.system_prompt).resolve()
    if root not in prompt_path.parents:
        issues.append("system_prompt resolves outside the pack")
        prompt = ""
    else:
        try:
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            issues.append(f"{manifest.system_prompt} is missing")
            prompt = ""
        except UnicodeDecodeError:
            issues.append(f"{manifest.system_prompt} must be UTF-8")
            prompt = ""
    if not prompt:
        issues.append("system prompt must not be empty")

    if manifest.rag.enabled:
        corpus = (root / manifest.rag.corpus_path).resolve()
        if root not in corpus.parents or not corpus.is_dir():
            issues.append("enabled RAG corpus is missing or outside the pack")
        elif not any(p.is_file() for p in corpus.rglob("*")):
            issues.append("enabled RAG corpus contains no files")

    if issues:
        raise PackValidationError(issues)

    persona = Persona(
        persona_id=manifest.persona_id,
        display_name=manifest.display_name,
        description=manifest.description,
        emoji=manifest.emoji,
        system_prompt=prompt,
        base_model_hint=_base_model_hint(manifest.base_model),
        adapter_hint=_adapter_hint(manifest.adapter),
        rag_enabled=manifest.rag.enabled,
        target_tier=manifest.target_tier,
        industry=manifest.industry,
        pack_path=str(root),
        source_format=str(manifest.source.get("format") or "adelie"),
    )
    return LoadedPersonaPack(
        root=root,
        manifest=manifest,
        persona=persona,
        source_format=str(manifest.source.get("format") or "adelie"),
    )


def discover_persona_packs(packs_dir: str | Path = DEFAULT_PACKS_DIR) -> tuple[LoadedPersonaPack, ...]:
    """Load valid packs without allowing one broken user file to break boot."""
    root = Path(packs_dir)
    if not root.is_dir():
        return ()
    loaded: list[LoadedPersonaPack] = []
    for candidate in sorted(root.glob("*.adelie")):
        if not candidate.is_dir():
            continue
        try:
            loaded.append(load_persona_pack(candidate))
        except PackValidationError:
            continue
    return tuple(loaded)


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if len(slug) < 2:
        digest = __import__("hashlib").sha256(value.encode("utf-8")).hexdigest()[:10]
        slug = f"persona_{digest}"
    return slug[:64]


def _character_card_json(payload: bytes) -> dict[str, Any]:
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackValidationError([f"character card JSON is invalid: {exc}"]) from None
    if not isinstance(raw, dict):
        raise PackValidationError(["character card must be a JSON object"])
    data = raw.get("data", raw)
    if not isinstance(data, dict):
        raise PackValidationError(["character card data must be an object"])
    return data


def _png_character_payload(payload: bytes) -> bytes:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise PackValidationError(["file is not a PNG character card"])
    offset = 8
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        if data_end + 4 > len(payload):
            break
        data = payload[data_start:data_end]
        if chunk_type == b"tEXt" and b"\x00" in data:
            keyword, text = data.split(b"\x00", 1)
            if keyword == b"chara":
                try:
                    return base64.b64decode(text, validate=True)
                except (binascii.Error, ValueError):
                    raise PackValidationError(["PNG chara metadata is not valid base64"]) from None
        offset = data_end + 4
    raise PackValidationError(["PNG has no Character Card 'chara' metadata"])


def _prompt_from_card(data: dict[str, Any]) -> str:
    parts = []
    for label, key in (
        ("Character", "description"),
        ("Personality", "personality"),
        ("Scenario", "scenario"),
        ("First message", "first_mes"),
        ("Example dialogue", "mes_example"),
        ("Post-history instructions", "post_history_instructions"),
    ):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"[{label}]\n{value.strip()}")
    if not parts:
        raise PackValidationError(["character card has no description or prompt fields"])
    return "\n\n".join(parts)


class CharacterCardImporter:
    source_format = "character-card-v2"

    def supports(self, filename: str, payload: bytes) -> bool:
        suffix = Path(filename).suffix.lower()
        return suffix in {".json", ".png"} or payload.startswith(b"\x89PNG")

    def import_into(self, filename: str, payload: bytes, packs_dir: Path) -> LoadedPersonaPack:
        if len(payload) > MAX_IMPORT_BYTES:
            raise PackValidationError([f"import exceeds {MAX_IMPORT_BYTES // (1024 * 1024)} MB limit"])
        card_payload = _png_character_payload(payload) if payload.startswith(b"\x89PNG") else payload
        data = _character_card_json(card_payload)
        name = str(data.get("name") or "Imported character").strip()
        persona_id = _slugify(name)
        destination = packs_dir / f"{persona_id}.adelie"
        if destination.exists():
            raise PackValidationError([f"persona already installed: {persona_id}"])

        prompt = _prompt_from_card(data)
        description = str(data.get("description") or data.get("personality") or "").strip()
        manifest = PersonaPackManifest(
            persona_id=persona_id,
            display_name=name,
            description=description[:500],
            base_model="active-runtime",
            source={"format": self.source_format, "filename": Path(filename).name},
        )
        packs_dir.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
        try:
            (destination / "MANIFEST.json").write_text(
                json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (destination / "system_prompt.md").write_text(prompt + "\n", encoding="utf-8")
            return load_persona_pack(destination)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise


class AdelieArchiveImporter:
    source_format = "adelie-archive"

    def supports(self, filename: str, payload: bytes) -> bool:
        return Path(filename).suffix.lower() in {".adelie", ".zip"} and zipfile.is_zipfile(
            __import__("io").BytesIO(payload)
        )

    def import_into(self, filename: str, payload: bytes, packs_dir: Path) -> LoadedPersonaPack:
        if len(payload) > MAX_IMPORT_BYTES:
            raise PackValidationError([f"archive exceeds {MAX_IMPORT_BYTES // (1024 * 1024)} MB control-plane limit"])
        from io import BytesIO

        packs_dir.mkdir(parents=True, exist_ok=True)
        staging = packs_dir / ".import-staging"
        normalized = packs_dir / ".import-normalized.adelie"
        if staging.exists():
            shutil.rmtree(staging)
        if normalized.exists():
            shutil.rmtree(normalized)
        staging.mkdir()
        try:
            with zipfile.ZipFile(BytesIO(payload)) as archive:
                for info in archive.infolist():
                    member = Path(info.filename)
                    if member.is_absolute() or ".." in member.parts:
                        raise PackValidationError([f"unsafe archive member: {info.filename}"])
                    target = (staging / member).resolve()
                    if staging.resolve() not in target.parents and target != staging.resolve():
                        raise PackValidationError([f"unsafe archive member: {info.filename}"])
                archive.extractall(staging)
            roots = [p.parent for p in staging.rglob("MANIFEST.json")]
            if len(roots) != 1:
                raise PackValidationError(["archive must contain exactly one MANIFEST.json"])
            pack_root = roots[0]
            if pack_root.suffix != ".adelie":
                shutil.move(str(pack_root), normalized)
                pack_root = normalized
            loaded = load_persona_pack(pack_root)
            destination = packs_dir / f"{loaded.manifest.persona_id}.adelie"
            if destination.exists():
                raise PackValidationError([f"persona already installed: {loaded.manifest.persona_id}"])
            shutil.move(str(pack_root), destination)
            return load_persona_pack(destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(normalized, ignore_errors=True)


class PersonaImportService:
    def __init__(self, packs_dir: str | Path = DEFAULT_PACKS_DIR) -> None:
        self.packs_dir = Path(packs_dir)
        self.importers: tuple[PersonaImporter, ...] = (
            AdelieArchiveImporter(),
            CharacterCardImporter(),
        )

    def install(self, filename: str, payload: bytes) -> LoadedPersonaPack:
        for importer in self.importers:
            if importer.supports(filename, payload):
                return importer.import_into(filename, payload, self.packs_dir)
        raise PackValidationError(["supported formats: Character Card V2 JSON/PNG or zipped .adelie"])


__all__ = [
    "AdelieArchiveImporter",
    "CharacterCardImporter",
    "DEFAULT_PACKS_DIR",
    "LoadedPersonaPack",
    "PackValidationError",
    "PersonaImportService",
    "PersonaImporter",
    "PersonaPackManifest",
    "discover_persona_packs",
    "load_persona_pack",
]

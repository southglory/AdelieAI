# ADR 0001: Separate portable persona metadata from model weights

Status: Accepted

## Context

Character Card files are normally kilobytes while local GGUF models are several
gigabytes. Requiring every persona pack to duplicate its model would make import,
validation, sharing, and multi-persona use impractical. AdelieAI also needs to
accept untrusted PNG, JSON, and ZIP control files without treating them as model
artifacts.

## Decision

An `.adelie` pack is the portable **control plane** for one persona: identity,
system prompt, optional RAG corpus, provenance, and runtime requirements. Model
weights may be embedded as a declared variant or resolved separately by a
runtime adapter and shared cache. Character Card import produces a prompt-only
pack that is immediately runnable with the active AdelieAI backend.

Importers normalize external formats into `.adelie`; the runtime consumes only
validated packs. Archive paths, prompt paths, sizes, and required files are
validated before installation.

## Consequences

- A character can be imported and inspected without a multi-gigabyte transfer.
- Multiple personas can share one GGUF or base model.
- “Self-contained” now has two explicit meanings: portable control-plane pack,
  or offline runtime bundle with an embedded model variant.
- Runtime resolution and checksums become a separate adapter boundary for Wave 2.

## Rollback or supersession

The loader remains versioned by `spec_version`. A future spec may require fully
embedded weights by introducing a new runtime profile; existing prompt-only
packs remain loadable or can be migrated by an importer.

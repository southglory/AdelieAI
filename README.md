<div align="center">
  <img src="assets/adelie.png" width="200" alt="AdelieAI Adelie penguin mascot"/>

# AdelieAI

**A persona engine that ships small enough to deploy.**
*Self-hosted. OSS. Batteries included.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![tests](https://img.shields.io/badge/tests-147%20passing-brightgreen.svg)](#testing)
[![Persona Pack v0.1](https://img.shields.io/badge/persona%20pack-v0.1-blueviolet.svg)](docs/PERSONA_PACK.md)

</div>

---

## What you get

A pipeline that takes you from **persona idea → deployable character**:

1. Prepare 60–120 dialogue pairs in your character's voice
2. Train a LoRA adapter on a Qwen 7B base (~80 seconds on a single 3090)
3. Compare the new character against base + previous versions (LLM-as-judge)
4. Pack everything into a `.adelie` persona pack — one self-contained artifact
5. *(v0.2)* Quantize to AWQ / GGUF — the same persona ships at 1/4 the size
6. Drop into a game NPC, Discord bot, customer-service worker, or CLI chat

Each `.adelie` pack is a single character with consistent voice, optional RAG-grounded knowledge, and a reproducible training recipe.

## Why a *persona engine*?

Most LLM toolkits give you "an assistant" — generic, hedged, breaks character. Game NPCs, brand voices, virtual companions, vertical-domain workers all need the *opposite*: a model that stays in character across long interactions and runs on hardware the user actually has.

AdelieAI ships:

- **Hybrid RAG** for grounding personas in lore / docs / knowledge bases (BM25 + dense + RRF + cross-encoder rerank)
- **LangGraph 4-node agent** so personas reason in steps (planner → retriever → reasoner → reporter)
- **TRL + PEFT LoRA training** with reproducibility manifest (`recipe.md` + `MANIFEST.json`)
- **Adapter comparison harness** with LLM-as-judge scoring
- **From-scratch nanoGPT** for the curious — same architecture family as Qwen2 (RMSNorm + RoPE + SwiGLU)
- **HTMX + Jinja2 console** so you can drive everything from a browser

## Live console

![sessions list with role-play personas](docs/screenshots/01_sessions.png)

Three Korean role-play personas (penguin / fish / knight) loaded as sessions. Top-right `llm:` shows the active model id — base alone, or `base+persona-id` when a persona pack is mounted.

| Frame | What it shows |
|---|---|
| [`01_sessions.png`](docs/screenshots/01_sessions.png) | Sessions list with three loaded personas |
| [`02_run_config.png`](docs/screenshots/02_run_config.png) | Per-session sampling controls + retrieval `k` |
| [`03_answer.png`](docs/screenshots/03_answer.png) | Streamed answer + token strip |
| [`04_docs_unavailable.png`](docs/screenshots/04_docs_unavailable.png) | Graceful fallback when RAG is unavailable |
| [`05_health.png`](docs/screenshots/05_health.png) | `/health` JSON output |
| [`06_swagger.png`](docs/screenshots/06_swagger.png) | Swagger UI at `/docs` |

> Regenerate any time with `scripts/capture_screenshots.py` — a Playwright walker over the live console.

## Persona pack format

A `.adelie` persona pack is the unit of distribution:

```
penguin_relaxed.adelie/
├── MANIFEST.json
├── adapter.safetensors
├── system_prompt.md
├── rag_corpus/
└── recipe.md
```

Full spec: [`docs/PERSONA_PACK.md`](docs/PERSONA_PACK.md). Roadmap to v0.2 adds `merged.awq.safetensors` and `merged.q4_k_m.gguf` so the same persona ships across GPU server / vLLM cluster / end-user CPU.

## Architecture in one table

| Layer | Components |
|---|---|
| **LLM serving** | `transformers` + LoRA adapter auto-loader, SSE token streaming, sampling presets |
| **RAG pipeline** | Recursive splitter · multilingual-e5 (KO+EN) · ChromaDB · BM25 · RRF fusion · bge-reranker-v2-m3 cross-encoder |
| **Agent loop** | LangGraph 4-node graph (planner → retriever → reasoner → reporter) |
| **Sessions** | Pydantic state machine · event sourcing · SQLAlchemy (SQLite default, Postgres swap) · IDOR guard |
| **Evaluation** | LLM-as-judge faithfulness / relevance / citation coverage; head-to-head adapter comparison |
| **Console UI** | HTMX + Jinja2 — single process, no JS framework |
| **Training** | TRL `SFTTrainer` LoRA, plus a pure-PyTorch nanoGPT for from-scratch experiments |
| **Logging** | Structured JSON + per-request id propagation |
| **Tests** | 147 unit + Playwright E2E walker |

## Design principles

1. **Asset ownership.** Every model lives under `models/{upstream,ours}/<id>/MANIFEST.json` listing source URL, revision sha, license, and exact `update_command`. HF Hub is a download channel, not a runtime dependency.
2. **Protocol-first.** `LLMClient`, `Retriever`, `SessionStore`, `Reranker`, `Embedder`, `VectorStore`, `BM25Index`, `Chunker` are all `typing.Protocol` — implementations are interchangeable.
3. **Zero API spend.** No call sites for Anthropic, OpenAI, or any hosted vendor. All inference is local.
4. **Apache-2.0 OSS preferred.** Qwen2.5 family · multilingual-e5 · bge-reranker. Mixed licenses are documented in MANIFEST and labelled in the model registry.
5. **Shipping size matters.** A persona is not "done" until it ships at deployable size. The v0.2 quantization track is first-class, not an afterthought.

## Install

```bash
git clone https://github.com/southglory/AdelieAI
cd AdelieAI
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,train]"

# Torch with CUDA (e.g. RTX 3090)
.venv/Scripts/pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124

# Pull the default models (each takes a few minutes)
.venv/Scripts/python -m huggingface_hub snapshot_download \
    Qwen/Qwen2.5-7B-Instruct \
    --local-dir models/upstream/Qwen2.5-7B-Instruct
.venv/Scripts/python -m huggingface_hub snapshot_download \
    intfloat/multilingual-e5-small \
    --local-dir models/upstream/multilingual-e5-small
.venv/Scripts/python -m huggingface_hub snapshot_download \
    BAAI/bge-reranker-v2-m3 \
    --local-dir models/upstream/bge-reranker-v2-m3
```

## Run

```bash
PYTHONUTF8=1 .venv/Scripts/uvicorn core.api.app:app --port 8770
```

Open `http://localhost:8770/web/`.

`/health` returns:
```json
{
  "status": "ok",
  "llm": "Qwen/Qwen2.5-7B-Instruct",
  "embedder": "intfloat/multilingual-e5-small",
  "reranker": "BAAI/bge-reranker-v2-m3",
  "retriever": "HybridRetriever",
  "store": "SqlSessionStore"
}
```

When a persona pack is mounted, `llm` becomes `base+persona-id`.

## Train a persona

```bash
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/train_lora_roleplay.py \
    --dataset mixed --epochs 4 \
    --output models/ours/qwen-roleplay-v2
```

Outputs `MANIFEST.json` + `recipe.md` + adapter weights (`~150 MB`, gitignored). Mount at runtime:

```bash
LORA_PATH=models/ours/qwen-roleplay-v2 \
PYTHONUTF8=1 .venv/Scripts/uvicorn core.api.app:app --port 8770
```

[`docs/TRAINING.md`](docs/TRAINING.md) — when to LoRA vs prompt vs full fine-tune, dataset rules, hyperparameter rationale (`r=16`, `α=32`, `lr=2e-4`, 4 epochs), v1 → v2 lessons, known traps.

## Compare personas

```bash
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/compare_adapters.py \
    --adapter v1=models/ours/qwen-roleplay-v1 \
    --adapter v2=models/ours/qwen-roleplay-v2
```

Writes `docs/compare/{ts}.json` (full text + scores) and `docs/compare/{ts}.md` (table + per-prompt outputs). Default judge is the base model itself; production setups should plug in a stronger external judge.

## From-scratch transformer

Pure-PyTorch decoder-only transformer (`core/training/models/nano_gpt.py`, ~250 lines, no `transformers` dependency at the model layer). RMSNorm + RoPE + SwiGLU — same architecture family as Qwen2 so LoRA-tuned and from-scratch results compare like-for-like.

```bash
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/train_nano_gpt.py \
    --output models/ours/nano-gpt-v0 \
    --steps 1500
```

A 69M model trains end-to-end in ~5 minutes on an RTX 3090.

## Roadmap

| version | adds |
|---|---|
| **v0.1** (current) | Persona pack format spec, LoRA training, hybrid RAG, LangGraph agent, comparison harness |
| **v0.2** | AWQ + GGUF quantization track — same persona, 1/4 the deployable size |
| **v0.3** | Distillation track (7B teacher → 1.5B student) — mobile-class personas |
| **v0.4** | vLLM serving — multiple personas concurrent on one GPU |
| **v0.5** | Tool-use personas — function calling per persona |
| **v0.6** | Multi-persona orchestration — N personas cooperating on a single quest |

## Testing

```bash
# 147 unit tests
.venv/Scripts/python -m pytest tests -q

# End-to-end Playwright walk
.venv/Scripts/python -m pytest tests/e2e -v --browser chromium -o addopts=
```

## Mascot

Adelie penguin — small, sturdy, plays on the ice without making a fuss. The engine, in spirit: focused, self-reliant, plays in its own pond.

## Contributing

Apache 2.0. Small PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the first-issue list.

## Sibling project

[`differentia-llm`](https://github.com/southglory/differentia-llm) — the private incubator AdelieAI was extracted from. Multi-agent orchestration experiments, mission notes, live training journals.

---

*made with cold flippers* 🐧

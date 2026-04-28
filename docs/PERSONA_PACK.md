# Persona Pack Format Specification

A *persona pack* is a self-contained directory (or `.zip`) bundling everything needed to run a single character with consistent voice and grounded knowledge.

> Spec status: **v0.2 draft**. v0.1 layout is stable. v0.2 adds the GGUF q4_k_m variant — *implemented and shipping*. AWQ variant remains roadmap (parked behind a Windows triton blocker — re-opens on Linux/WSL).

## Layout (v0.1)

```
{persona_id}.adelie/
├── MANIFEST.json
├── adapter.safetensors          # LoRA weights (fp16, ~150 MB for 7B base)
├── system_prompt.md             # Character voice + behavior contract
├── rag_corpus/                  # Optional knowledge source for RAG grounding
│   └── *.md
└── recipe.md                    # Human-readable training reproduction note
```

## MANIFEST.json schema (v0.1)

```json
{
  "spec_version": "0.1",
  "persona_id": "penguin_relaxed",
  "display_name": "놀고 있는 펭귄",
  "description": "An off-duty Adelie penguin. Casual register, observational humor, refuses to break character.",
  "language": "ko",
  "license": "apache-2.0",

  "base_model": {
    "id": "Qwen/Qwen2.5-7B-Instruct",
    "revision": "main",
    "min_quant": "fp16"
  },

  "adapter": {
    "kind": "lora",
    "path": "adapter.safetensors",
    "rank": 16,
    "alpha": 32,
    "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    "trainable_params": 40370176
  },

  "system_prompt": "system_prompt.md",

  "rag": {
    "enabled": false,
    "corpus_path": "rag_corpus/",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "retrieval": "hybrid",
    "rrf_k": 60,
    "top_k": 4
  },

  "training_recipe": "recipe.md",
  "produced_by": "differentia-llm core.training.trainer.train_lora",
  "produced_at": "2026-04-15T10:30:00Z"
}
```

## Loading a persona

```python
from adelie import Persona

p = Persona.load("packs/penguin_relaxed.adelie")

async for token in p.stream("이 동물원은 어떻게 생각해?"):
    print(token, end="", flush=True)
```

The SDK loads `base_model`, applies `adapter`, and injects `system_prompt`. If `rag.enabled` is true, queries are augmented with hybrid retrieval over `rag_corpus/` before generation.

## Validation rules

A pack is valid if:

- `MANIFEST.json` parses and conforms to the schema above
- `spec_version` is supported by the loader
- `adapter.path` exists and is a `safetensors` file
- `system_prompt.md` exists and is non-empty
- If `rag.enabled` is true, `rag_corpus/` exists with at least one indexable document
- Base model declared in `base_model.id` is locally available (or downloadable)

The CLI ships a validator: `python -m adelie.persona validate path/to/persona.adelie/`.

## v0.2 — quantization variants

v0.2 adds optional quantized artifacts side-by-side with the fp16 source:

```
{persona_id}.adelie/
├── MANIFEST.json                # gains "variants" section
├── adapter.safetensors          # source (training adapter, fp16)
├── merged.q4_k_m.gguf           # llama.cpp GGUF variant — IMPLEMENTED (~4.4 GB for Qwen2.5-7B base)
├── merged.awq.safetensors       # AWQ 4-bit merged base+adapter — ROADMAP (Linux/WSL)
├── ...
```

`MANIFEST.json` `variants` section:
```json
"variants": [
  {"runtime": "transformers-fp16", "path": "adapter.safetensors", "size_mb": 154},
  {"runtime": "llama.cpp-q4_k_m",  "path": "merged.q4_k_m.gguf",     "size_mb": 4467, "compression_ratio": 3.25},
  {"runtime": "vllm-awq",          "path": "merged.awq.safetensors", "size_mb": 2120, "status": "roadmap"}
]
```

This lets a deployer pick the variant matching their runtime: GPU server (fp16+adapter), end-user CPU (GGUF, *available now*), or vLLM at scale (AWQ, *coming after the Windows triton blocker*).

### v0.2 — what's actually shipping

A reference q4_k_m artifact for `qwen-roleplay-v2` lives at:

```
models/ours/qwen-roleplay-v2-gguf/
├── MANIFEST.json
├── recipe.md                                 # full reproduction
└── qwen-roleplay-v2.q4_k_m.gguf             # 4.36 GB single file
```

Mounted via:
```bash
MODEL_PATH=models/ours/qwen-roleplay-v2-gguf/qwen-roleplay-v2.q4_k_m.gguf \
  uvicorn core.api.app:app --port 8770
```

`core/api/app.py::_default_llm` dispatches to `GGUFClient` when `MODEL_PATH` ends in `.gguf`. The persona gallery / chat thread UX is identical to the FP16 path; only the underlying weights are smaller.

The recipe (merge → GGUF FP16 → q4_k_m) lives in the `differentia-llm` sibling repo at `experiments/05_awq_quantize/merge.py` and `experiments/06_gguf_export/run.py` — both reproducible with one command each, see [`models/ours/qwen-roleplay-v2-gguf/recipe.md`](../models/ours/qwen-roleplay-v2-gguf/recipe.md).

## Roadmap — v0.3 (distillation)

A persona's *behavioral fingerprint* may be transferred from a 7B teacher to a 1.5B student. v0.3 treats distilled checkpoints as a third variant kind:

```json
{"runtime": "transformers-fp16-distilled", "path": "distilled-1b5.safetensors", "base_replacement": true, "size_mb": 3000}
```

Where `base_replacement: true` indicates the distilled checkpoint replaces — rather than adapts — the original base.

## Why this format

1. **One pack = one character = one deploy artifact.** No assembling files at deploy time.
2. **Provenance complete.** `recipe.md` + `produced_by` + `revision` make every persona reproducible end-to-end.
3. **Runtime-pluggable.** v0.2 variants ship the same persona to GPU servers, vLLM clusters, and end-user CPUs without re-training.
4. **Schema versioned.** `spec_version` lets the loader degrade gracefully across format evolutions.

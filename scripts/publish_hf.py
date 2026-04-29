"""Publish AdelieAI's trained adapter + GGUF to the HuggingFace Hub.

Reads the token from $HF_TOKEN (or `.env` via python-dotenv if installed).
Creates two repos under the user's namespace:
  {user}/adelie-qwen-roleplay-v2-lora    — LoRA adapter (~165 MB)
  {user}/adelie-qwen-roleplay-v2-gguf    — GGUF q4_k_m (~4.4 GB)

Each repo gets a custom README.md (model card) generated here from
project facts; the local `MANIFEST.json` and `recipe.md` are uploaded
alongside for full provenance.

Usage:
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 scripts/publish_hf.py
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 scripts/publish_hf.py --dry-run
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 scripts/publish_hf.py --only lora
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional — env can be set directly

from huggingface_hub import HfApi, create_repo, upload_folder, upload_file


REPO_NAME_LORA = "adelie-qwen-roleplay-v2-lora"
REPO_NAME_GGUF = "adelie-qwen-roleplay-v2-gguf"

# Where the trained weights actually live (differentia-llm sibling repo).
WEIGHTS_ROOT = Path(
    r"C:/Users/coolb/Documents/GithubLinkedProjects/differentia-llm/models/ours"
)


LORA_CARD = """\
---
license: other
license_name: tongyi-qianwen
license_link: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-7B-Instruct
base_model_relation: adapter
language:
  - ko
  - en
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - korean
  - roleplay
  - persona
  - qwen2.5
---

# AdelieAI · Qwen-Roleplay v2 (LoRA adapter)

Korean role-play persona LoRA fine-tuned on top of `Qwen/Qwen2.5-7B-Instruct`.
Part of the [AdelieAI](https://github.com/southglory/AdelieAI) persona engine.

## What it does

Tilts the base model toward a **Korean role-play register** while
preserving general assistant capability. Trained on a *mixed* dataset
(60 role-play pairs + 60 general pairs) to avoid catastrophic forgetting
of general knowledge — see Pitfalls below.

Three personas use this single shared adapter, differentiated by system
prompt:

- 🐧 `penguin_relaxed` — Adelie penguin, casual observational humor
- 🐟 `fish_swimmer`    — fish drifting through open water
- ⚔️ `knight_brave`   — sworn knight, formal-speech, faces dragons head-on

Verticals (cynical_merchant / cold_detective / ancient_dragon) also mount
this same adapter — style differences are driven by their system prompts
+ optional grounding (RAG / KG).

## Training recipe

| | |
|---|---|
| Base | `Qwen/Qwen2.5-7B-Instruct` (bf16) |
| Adapter rank | r=16, α=32 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer | AdamW, lr=2e-4, cosine schedule |
| Epochs | 4 (sweet spot — 5 over-fits) |
| Batch | per_device=2 × grad_accum=4 (effective 8) |
| Memory tricks | `bf16=True` · `gradient_checkpointing=True` |
| Hardware | single RTX 3090 (24 GB) |
| Wall-clock | ~25–30 min |

Full reproducibility: [`recipe.md`](./recipe.md) + [`MANIFEST.json`](./MANIFEST.json) in this repo.

## How to use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="bfloat16",
    device_map="cuda",
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "ramyun/adelie-qwen-roleplay-v2-lora")

system = "당신은 판타지 세계의 냉소적인 잡화상 주인입니다. 짧고 무뚝뚝한 어조로 답하세요."
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": "할인 좀 안 돼요?"},
]
inputs = tok.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
out = model.generate(inputs, max_new_tokens=128, temperature=0.7, top_p=0.9)
print(tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))
```

Or mount via the AdelieAI engine:

```bash
huggingface-cli download ramyun/adelie-qwen-roleplay-v2-lora \\
    --local-dir models/ours/qwen-roleplay-v2

LORA_PATH=models/ours/qwen-roleplay-v2 \\
PYTHONUTF8=1 .venv/Scripts/uvicorn core.api.app:app --port 8770
```

## Evaluation

Behavioral test suite (see [AdelieAI/docs/eval/](https://github.com/southglory/AdelieAI/tree/main/docs/eval)) — 25 prompts per persona, must_contain / must_not_contain substring grader:

| persona | round 1 | after prompt strengthening |
|---|---|---|
| 💰 cynical_merchant | 76% | **96%** |
| 🔍 cold_detective | 80% | **88%** |
| 🐉 ancient_dragon | 84% | **96%** |

Round-by-round audit trail under `docs/eval/iterations/` in the AdelieAI repo.

## Pitfalls (what we learned)

1. **60 pairs is *not* enough** to beat the v2 mixed baseline. We tried — see [`docs/MILESTONES.md` `[training/lora] (1st cycle)`](https://github.com/southglory/AdelieAI/blob/main/docs/MILESTONES.md). Plan for 200+ pairs OR mix in general-domain pairs at ≥1:1 ratio (this adapter does the latter).
2. **Single-register data → catastrophic forgetting**. v1 trained on role-play pairs only — general Python/RAG questions devolved into character style. v2's mixed dataset is the regularizer.
3. **Hybrid EN+KO system prompts** beat pure Korean for grounding-heavy personas (T3+). English rules block + Korean style anchors keeps both signals strong.

## License

This adapter inherits the **Tongyi Qianwen License v1** from the base model
(`Qwen/Qwen2.5-7B-Instruct`). Commercial use is permitted under
that license's terms (notably: < 100M monthly active users and a
notice retention requirement). The training recipe and code (in
[AdelieAI](https://github.com/southglory/AdelieAI)) are Apache-2.0.

## Citation

```bibtex
@software{adelieai_2026,
  title  = {AdelieAI: a persona engine for small-deployment LLMs},
  author = {ramyun},
  year   = {2026},
  url    = {https://github.com/southglory/AdelieAI}
}
```
"""


GGUF_CARD = """\
---
license: other
license_name: tongyi-qianwen
license_link: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-7B-Instruct
base_model_relation: quantized
language:
  - ko
  - en
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - q4_k_m
  - llama.cpp
  - korean
  - roleplay
  - persona
  - qwen2.5
---

# AdelieAI · Qwen-Roleplay v2 (GGUF q4_k_m)

CPU-friendly **q4_k_m quantization** of the merged
`Qwen2.5-7B-Instruct + adelie-qwen-roleplay-v2-lora` checkpoint.
Ships in a single 4.4 GB file — runs on a laptop without a GPU.

> Companion to the LoRA adapter at
> [`ramyun/adelie-qwen-roleplay-v2-lora`](https://huggingface.co/ramyun/adelie-qwen-roleplay-v2-lora).
> The LoRA repo is for FP16 + GPU mounting; this GGUF is for laptops /
> end-users without CUDA.

## How to use

### llama.cpp / llama-cpp-python

```bash
pip install llama-cpp-python --only-binary=:all: \\
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

huggingface-cli download ramyun/adelie-qwen-roleplay-v2-gguf \\
    qwen-roleplay-v2.q4_k_m.gguf --local-dir ./models
```

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./models/qwen-roleplay-v2.q4_k_m.gguf",
    n_ctx=4096,
    n_threads=8,
)
out = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "당신은 판타지 세계의 냉소적인 잡화상 주인입니다."},
        {"role": "user", "content": "할인 좀 안 돼요?"},
    ],
    temperature=0.7,
)
print(out["choices"][0]["message"]["content"])
```

### Or mount via the AdelieAI engine

```bash
huggingface-cli download ramyun/adelie-qwen-roleplay-v2-gguf \\
    qwen-roleplay-v2.q4_k_m.gguf \\
    --local-dir models/ours/qwen-roleplay-v2-gguf

MODEL_PATH=models/ours/qwen-roleplay-v2-gguf/qwen-roleplay-v2.q4_k_m.gguf \\
PYTHONUTF8=1 .venv/Scripts/uvicorn core.api.app:app --port 8770
```

## Quantization details

| | |
|---|---|
| Source | merged FP16 (`Qwen2.5-7B-Instruct + adelie-qwen-roleplay-v2-lora`) |
| Method | llama.cpp `convert_hf_to_gguf.py` → `quantize` to `q4_k_m` |
| Original size | 14.5 GB (FP16) |
| **Quantized size** | **4.4 GB** (3.25× compression) |
| Runtime VRAM | 0 (pure CPU) |
| Runtime RAM | ~5 GB |

Style preservation verified: 4/5 persona prompts + 5/5 general prompts pass under greedy decoding (see [`recipe.md`](./recipe.md)).

## Recommended sampling

`temperature=0.7` and `top_p=0.9`. Greedy decoding (`temperature=0`) sometimes diverges on q4_k_m — known quantization artifact, mitigated by sampling.

## License

Inherits the **Tongyi Qianwen License v1** from the base model
(`Qwen/Qwen2.5-7B-Instruct`).

## Citation

```bibtex
@software{adelieai_2026,
  title  = {AdelieAI: a persona engine for small-deployment LLMs},
  author = {ramyun},
  year   = {2026},
  url    = {https://github.com/southglory/AdelieAI}
}
```
"""


def push_lora(api: HfApi, user: str, dry_run: bool) -> str:
    repo_id = f"{user}/{REPO_NAME_LORA}"
    src = WEIGHTS_ROOT / "qwen-roleplay-v2"
    if not src.exists():
        raise FileNotFoundError(f"LoRA source missing: {src}")

    if dry_run:
        print(f"[dry-run] would create {repo_id} (model)")
        print(f"[dry-run] would upload from {src} (excluding checkpoint-*)")
        return repo_id

    create_repo(repo_id, repo_type="model", private=False, exist_ok=True, token=api.token)

    # Write the model card to a tmp location and upload as README.md
    card_path = src / "_HF_README_TMP.md"
    card_path.write_text(LORA_CARD, encoding="utf-8")
    upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=api.token,
        commit_message="model card",
    )
    card_path.unlink(missing_ok=True)

    # Upload everything in src except checkpoints + the merged tmp readme
    upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        repo_type="model",
        token=api.token,
        commit_message="initial release: adapter + tokenizer + recipe",
        ignore_patterns=[
            "checkpoint-*",
            "_HF_README_TMP.md",
            "README.md",  # we already pushed our custom card
            "training_args.bin",  # internal trainer state, not useful
        ],
    )
    print(f"  ✅ pushed {repo_id}")
    return repo_id


def push_gguf(api: HfApi, user: str, dry_run: bool) -> str:
    repo_id = f"{user}/{REPO_NAME_GGUF}"
    src = WEIGHTS_ROOT / "qwen-roleplay-v2-gguf"
    if not src.exists():
        raise FileNotFoundError(f"GGUF source missing: {src}")

    if dry_run:
        print(f"[dry-run] would create {repo_id} (model)")
        print(f"[dry-run] would upload {src}/qwen-roleplay-v2.q4_k_m.gguf (~4.4 GB) + MANIFEST.json")
        return repo_id

    create_repo(repo_id, repo_type="model", private=False, exist_ok=True, token=api.token)

    # Custom README
    card_path = src / "_HF_README_TMP.md"
    card_path.write_text(GGUF_CARD, encoding="utf-8")
    upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=api.token,
        commit_message="model card",
    )
    card_path.unlink(missing_ok=True)

    # Upload the GGUF + manifest
    upload_folder(
        folder_path=str(src),
        repo_id=repo_id,
        repo_type="model",
        token=api.token,
        commit_message="initial release: q4_k_m GGUF (~4.4 GB)",
        ignore_patterns=["_HF_README_TMP.md"],
    )
    print(f"  ✅ pushed {repo_id}")
    return repo_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="describe what would be pushed without doing it")
    ap.add_argument("--only", choices=["lora", "gguf"], help="push only one")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not set. Put it in .env or export it.", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    me = api.whoami()
    user = me["name"]
    print(f"🤗 logged in as: {user}")

    if args.only != "gguf":
        print("\n=== LoRA adapter ===")
        push_lora(api, user, args.dry_run)

    if args.only != "lora":
        print("\n=== GGUF q4_k_m ===")
        push_gguf(api, user, args.dry_run)

    if not args.dry_run:
        print("\n🎉 done. Visit:")
        print(f"  https://huggingface.co/{user}/{REPO_NAME_LORA}")
        print(f"  https://huggingface.co/{user}/{REPO_NAME_GGUF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

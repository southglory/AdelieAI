# Training methodology

This is the *why and how* of our fine-tuning. A per-run snapshot is auto-emitted to `models/ours/{name}/recipe.md`; this document is the methodology layer above those snapshots.

## 0. Decision tree — prompt vs LoRA vs full fine-tune

| What's the problem? | Solution |
|---|---|
| "Different answer in one specific scenario." | System prompt + a few in-context examples |
| "Consistent *style* / persona across all answers." | **LoRA** |
| "Update the model's knowledge (e.g. inject a domain corpus)." | Full fine-tune **or** LoRA + RAG |
| "Base model can't do the underlying task." | Bigger base model |

Why LoRA is the sweet spot:
- 7B trains on a single 24 GB GPU.
- Adapter file 50–150 MB → ship many verticals side by side.
- Base weights are untouched — license stays clean.
- Comparable runs are easy: same base + adapter A vs adapter B.

## 1. Dataset design rules

Aligned with `mission/02_llm-tuning-training.md`:

1. **Two registers, one model — split by system prompt.** When you want one model that does both character role-play and normal Q&A, use the system prompt as the register selector. We ship `ROLEPLAY_SYSTEM` and `GENERAL_SYSTEM`.
2. **Roughly 50/50 mix.** v1 (role-play only) overfit; v2 corrected it with an even split. As a rule: 60% on the register you want to inject, 40% on the register you want to preserve.
3. **Short and varied.** 60–200 chars per pair. Diversify characters, domains, scenarios. Don't repeat patterns.
4. **No banned phrases in the data.** "AI", "인공지능", "상상해보면" must not appear in character answers — the test `tests/test_training.py::test_dataset_excludes_meta_phrases` enforces this.
5. **Language consistency.** We force Korean-only answers via a CJK character ratio test, so stray English tokens don't slip in.

The data lives inline as a Python list in `core/training/dataset.py` — 60 + 60 = 120 pairs. Inline (not external JSON) because PR diffs show every change and the pytest format checks are trivial.

## 2. Hyperparameter choices

`core/training/lora_config.py`:

```python
LoraConfig(
    r=16,                  # rank — 8–32 typical; 16 is the standard for 7B.
    lora_alpha=32,         # alpha = 2r is the canonical strength setting.
    lora_dropout=0.05,     # tiny dataset (60–200 pairs) — needs some regularisation.
    bias="none",           # don't train biases — saves memory.
    task_type="CAUSAL_LM",
    target_modules=[       # all 7 Qwen2 projection blocks:
        "q_proj","k_proj","v_proj","o_proj",  # attention
        "gate_proj","up_proj","down_proj",     # MLP
    ],
)
```

→ Qwen2-7B: **trainable ≈ 40M (0.53 %)**.

`SFTConfig`:
- `num_train_epochs=4` — 60–120 pairs at 4 epochs lands `mean_token_accuracy ≥ 0.90`. 5 epochs starts overfitting (we saw an English token leak on v1).
- `per_device_batch_size=2 × grad_accum=4` → effective batch 8.
- `learning_rate=2e-4` — PEFT defaults; works well for LoRA.
- `warmup_ratio=0.05` — 5 % warmup steps.
- `bf16=True` — safe on a 3090 (avoids fp16 NaNs).
- `gradient_checkpointing=True` + `use_reentrant=False` — fits 7B + adapters in 24 GB.
- `max_length=1024` — enough for short character answers.

## 3. v1 → v2 evolution

**v1 — instructive failure**
- Data: 60 ROLEPLAY pairs only.
- 5 epochs. Loss 2.95 → 0.34.
- ✅ Character voice landed precisely.
- ❌ Generic questions came back in 1st-person character. ("As an AI I'd say… *as a fish*…")
- ❌ One English-token leak ("warehouse").

Lesson: **see only one register at training time → break the other register at inference time.** Obvious in retrospect; necessary to actually trip on it.

**v2**
- Data: 60 ROLEPLAY + 60 GENERAL = 120 pairs.
- 4 epochs (5 → 4 to dampen overfit). Loss 3.05 → 0.39.
- ✅ Character voice still on point.
- ✅ Generic questions land in our dataset's exact phrasing ("Python의 비동기 웹 프레임워크 … Swagger 자동 문서화").
- ✅ English leak gone.

## 4. Comparison harness

`core/eval/compare.py` + `scripts/compare_adapters.py`:

```bash
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/compare_adapters.py \
    --adapter v1=models/ours/qwen-roleplay-v1 \
    --adapter v2=models/ours/qwen-roleplay-v2
```

Outputs `docs/compare/{ts}.json` + `{ts}.md`:
- Every prompt × every candidate run + answer recorded.
- LLM-as-judge `answer_relevance` score per (prompt, adapter).
- Markdown table + per-prompt response bodies.
- Stdout summary: `mean answer_relevance per label`.

The judge defaults to the base model (self-evaluation). Production should plug in a stronger external judge.

## 5. Provenance

Every training run produces, side-by-side with the adapter weights, a `MANIFEST.json` and `recipe.md`:

`models/ours/{name}/MANIFEST.json`:
```json
{
  "model_id": "qwen-roleplay-v2",
  "base_model": "models/upstream/Qwen2.5-7B-Instruct",
  "kind": "lora-adapter",
  "license": "apache-2.0",
  "trainable_params": 40370176,
  "total_params": 7655986688,
  "num_epochs": 4,
  "n_pairs": 120,
  "elapsed_seconds": 121.3,
  "final_loss": 0.95,
  "produced_by": "core.training.trainer.train_lora",
  "update_policy": "diverged"
}
```

`recipe.md` is the human-readable reproduction note. `adapter_model.safetensors` (~150 MB) is gitignored — users regenerate locally.

## 6. Adding a new LoRA — workflow

1. **Append data pairs** to `core/training/dataset.py` (inline list — diff-friendly).
2. **Tests** — `pytest tests/test_training.py -q` checks Korean-only, no banned phrases, no duplicates.
3. **Train**:
    ```bash
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
        scripts/train_lora_roleplay.py \
        --dataset mixed --epochs 4 \
        --output models/ours/qwen-mydomain-v1
    ```
4. **Compare**: `scripts/compare_adapters.py --adapter mydomain=models/ours/qwen-mydomain-v1`
5. **Run**: `LORA_PATH=models/ours/qwen-mydomain-v1 uvicorn ...`

## 7. Known traps

- **Windows + Python 3.13:** `from trl import SFTTrainer` after `peft + transformers` segfaults. `core/training/trainer.py` forces `import datasets, trl` at module import to lock the safe order. The same trick is in `tests/conftest.py` for collection.
- **`PYTHONUTF8` is mandatory:** Korean Windows defaults to cp949, which fails to decode some of trl's Jinja templates. Use `PYTHONUTF8=1 -X utf8`.
- **24 GB VRAM ceiling:** 7B + adapter + optimizer state + KV cache. `gradient_checkpointing=True` keeps batch 2 / grad_accum 4 inside the budget. To train 13B you need QLoRA (4-bit base).

## 8. Next experiments

- **v3** — grow data by 50 %; reduce LoRA `r` to 8 to dampen overfit further.
- **DPO** — preferred / rejected pairs through TRL's `DPOTrainer`.
- **Domain vertical** — `qwen-erp-advisor-v1`: 70 ERP-finance pairs + 30 general.
- **More from-scratch** — see `core/training/models/nano_gpt.py` for the existing 69M baseline; the next obvious step is curriculum learning on a larger Korean corpus.

Each experiment runs through the same `compare_adapters.py` harness — every change defended with a number.

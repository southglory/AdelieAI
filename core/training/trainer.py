"""SFT-LoRA trainer wrapping TRL's SFTTrainer.

Usage:
    from core.training.trainer import train_lora
    train_lora(
        base_model_path="models/upstream/Qwen2.5-7B-Instruct",
        output_dir="models/ours/qwen-roleplay-v1",
        num_epochs=3,
    )

Outputs an adapter — the base weights are not modified, just an
LoRA delta saved alongside.
"""

import json
import time
from pathlib import Path

# Important: importing `datasets` and `trl` BEFORE `torch`/`peft`/
# `transformers` avoids a Windows segfault triggered by initialising
# pyarrow + torch in the wrong order. Touch them at module load.
import datasets as _datasets  # noqa: F401
import trl as _trl  # noqa: F401

from core.logging import get_logger
from core.training.dataset import (
    GENERAL_PAIRS,
    ROLEPLAY_PAIRS,
    build_mixed_dataset,
    build_persona_dataset,
    build_roleplay_dataset,
    dataset_stats,
    load_persona_pairs,
)
from core.training.lora_config import default_lora_config

log = get_logger("differentia.training")


def train_lora(
    *,
    base_model_path: str | Path,
    output_dir: str | Path,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 2,
    grad_accum: int = 4,
    max_seq_length: int = 1024,
    warmup_ratio: float = 0.05,
    save_recipe: bool = True,
    dataset_kind: str = "roleplay",
    persona_id: str | None = None,
    persona_system_prompt: str | None = None,
    val_ratio: float = 0.2,
) -> dict:
    """Run the full LoRA fine-tune. Returns a stats dict suitable for
    logging or for emitting into the output MANIFEST.

    dataset_kind:
      - "roleplay": ROLEPLAY_PAIRS only (v1 default)
      - "mixed":    ROLEPLAY_PAIRS + GENERAL_PAIRS (v2 — preserves
                    generalisation by also showing non-character
                    Q&A under the general system prompt)
      - "persona":  Load `personas/{persona_id}/dialogue_pairs.jsonl`
                    + GENERAL_PAIRS, wrap with the persona's own
                    system prompt. Step 6.1 — per-persona LoRA.

    val_ratio:
      - When > 0 and `dataset_kind == "persona"`, the dataset is split
        80/20 (default) and val loss is logged each epoch via the
        SFTTrainer's eval_dataset hook. Set to 0 to skip evaluation.
    """
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    base_model_path = Path(base_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "lora_train_start",
        extra={
            "base_model": str(base_model_path),
            "output_dir": str(output_dir),
            "num_epochs": num_epochs,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = default_lora_config()
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()

    val_ds = None
    if dataset_kind == "persona":
        if not persona_id:
            raise ValueError("persona_id required when dataset_kind='persona'")
        train_ds, val_ds = build_persona_dataset(
            persona_id,
            persona_system_prompt=persona_system_prompt,
            val_ratio=val_ratio,
        )
        persona_pairs = load_persona_pairs(persona_id)
        stats = {
            "n_pairs": len(persona_pairs) + len(GENERAL_PAIRS),
            "n_persona": len(persona_pairs),
            "n_general": len(GENERAL_PAIRS),
            "persona_id": persona_id,
            "val_ratio": val_ratio,
            "val_size": len(val_ds) if val_ds is not None else 0,
        }
    elif dataset_kind == "mixed":
        train_ds = build_mixed_dataset()
        stats = {
            "n_pairs": len(ROLEPLAY_PAIRS) + len(GENERAL_PAIRS),
            "n_roleplay": len(ROLEPLAY_PAIRS),
            "n_general": len(GENERAL_PAIRS),
        }
    else:
        train_ds = build_roleplay_dataset()
        stats = dataset_stats()

    sft_cfg_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        max_length=max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )
    if val_ds is not None and len(val_ds) > 0:
        sft_cfg_kwargs.update(
            eval_strategy="epoch",
            per_device_eval_batch_size=per_device_batch_size,
            # Step 6.1.A 의 함정: 4 epoch 까지 가면 small-data persona LoRA 가
            # 과적합 (val_loss U-shape). val_loss 최저 epoch 의 weight 를 최종으로.
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=num_epochs,
        )
    sft_cfg = SFTConfig(**sft_cfg_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    if val_ds is not None and len(val_ds) > 0:
        trainer_kwargs["eval_dataset"] = val_ds
    trainer = SFTTrainer(**trainer_kwargs)

    t0 = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - t0

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "base_model": str(base_model_path),
        "output_dir": str(output_dir),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
        "num_epochs": num_epochs,
        "lr": learning_rate,
        "per_device_batch_size": per_device_batch_size,
        "grad_accum": grad_accum,
        "max_seq_length": max_seq_length,
        "n_pairs": stats["n_pairs"],
        "dataset_kind": dataset_kind,
        "elapsed_seconds": round(elapsed, 1),
        "final_loss": float(train_result.training_loss),
    }
    # Per-persona val metrics (Step 6.1) — captured from the trainer's
    # log_history if eval_strategy was active.
    if val_ds is not None and len(val_ds) > 0:
        eval_logs = [
            log_row for log_row in trainer.state.log_history
            if "eval_loss" in log_row
        ]
        if eval_logs:
            summary["val_loss_per_epoch"] = [
                round(float(row["eval_loss"]), 4) for row in eval_logs
            ]
            summary["val_loss_final"] = summary["val_loss_per_epoch"][-1]
    if "persona_id" in stats:
        summary["persona_id"] = stats["persona_id"]
    log.info("lora_train_done", extra=summary)

    if save_recipe:
        (output_dir / "recipe.md").write_text(
            _render_recipe(summary), encoding="utf-8"
        )
        (output_dir / "MANIFEST.json").write_text(
            json.dumps(_render_manifest(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return summary


def _render_recipe(summary: dict) -> str:
    return (
        "# Training Recipe\n\n"
        "이 adapter는 differentia-llm `core.training.trainer.train_lora()`로 산출되었다.\n\n"
        "## 입력\n\n"
        f"- base model: `{summary['base_model']}`\n"
        f"- 데이터셋: `core/training/dataset.py` ROLEPLAY_PAIRS ({summary['n_pairs']}건)\n\n"
        "## 하이퍼파라미터\n\n"
        f"- num_epochs: {summary['num_epochs']}\n"
        f"- learning_rate: {summary['lr']}\n"
        f"- per_device_batch_size: {summary['per_device_batch_size']}\n"
        f"- gradient_accumulation_steps: {summary['grad_accum']}\n"
        f"- max_seq_length: {summary['max_seq_length']}\n"
        f"- LoRA r=16, alpha=32, dropout=0.05, "
        f"target_modules=q/k/v/o/gate/up/down_proj\n\n"
        "## 결과\n\n"
        f"- trainable params: {summary['trainable_params']:,} / {summary['total_params']:,}"
        f" ({summary['trainable_pct']}%)\n"
        f"- final training loss: {summary['final_loss']:.4f}\n"
        f"- wall clock: {summary['elapsed_seconds']}s\n\n"
        "## 재현\n\n"
        "```python\n"
        "from core.training.trainer import train_lora\n"
        f"train_lora(base_model_path={summary['base_model']!r},\n"
        f"           output_dir={summary['output_dir']!r},\n"
        f"           num_epochs={summary['num_epochs']})\n"
        "```\n"
    )


def _render_manifest(summary: dict) -> dict:
    return {
        "model_id": Path(summary["output_dir"]).name,
        "base_model": summary["base_model"],
        "kind": "lora-adapter",
        "license": "apache-2.0",
        "trainable_params": summary["trainable_params"],
        "total_params": summary["total_params"],
        "num_epochs": summary["num_epochs"],
        "n_pairs": summary["n_pairs"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "final_loss": summary["final_loss"],
        "produced_by": "differentia-llm core.training.trainer.train_lora",
        "update_policy": "diverged",
        "note": "우리가 만든 adapter. base 모델 weights는 변경 없음.",
    }

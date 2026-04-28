"""CLI: fine-tune Qwen2.5-7B with LoRA on the role-play dataset.

Usage:
    .venv/Scripts/python scripts/train_lora_roleplay.py
    .venv/Scripts/python scripts/train_lora_roleplay.py --epochs 5 --output models/ours/qwen-roleplay-v2
"""

import argparse
from pathlib import Path

from core.training.trainer import train_lora


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="models/upstream/Qwen2.5-7B-Instruct",
        help="path to base model",
    )
    parser.add_argument(
        "--output",
        default="models/ours/qwen-roleplay-v1",
        help="adapter output directory",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq", type=int, default=1024)
    parser.add_argument(
        "--dataset",
        choices=["roleplay", "mixed", "persona"],
        default="roleplay",
        help=(
            "roleplay = role-play pairs only (v1); "
            "mixed = role-play + general (v2); "
            "persona = personas/{id}/dialogue_pairs.jsonl + general (v3, Step 6.1)"
        ),
    )
    parser.add_argument(
        "--persona",
        default=None,
        help=(
            "Persona id when --dataset=persona. Loads "
            "personas/{persona}/dialogue_pairs.jsonl and uses the "
            "persona's own system prompt (from registry)."
        ),
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Held-out fraction for per-epoch eval. Set 0 to skip.",
    )
    args = parser.parse_args()

    base = Path(args.base)
    if not base.exists():
        raise SystemExit(f"base model not found: {base}")

    persona_system_prompt = None
    if args.dataset == "persona":
        if not args.persona:
            raise SystemExit("--persona <id> is required when --dataset=persona")
        # Pull the persona's own system prompt from the registry so the
        # LoRA learns *that* persona's voice, not generic role-play.
        from core.personas.registry import get_persona
        p = get_persona(args.persona)
        if p is None:
            raise SystemExit(
                f"persona '{args.persona}' not in registry. "
                f"Add to core/personas/registry.py first."
            )
        persona_system_prompt = p.system_prompt

    summary = train_lora(
        base_model_path=base,
        output_dir=args.output,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq,
        dataset_kind=args.dataset,
        persona_id=args.persona if args.dataset == "persona" else None,
        persona_system_prompt=persona_system_prompt,
        val_ratio=args.val_ratio,
    )
    print()
    print("=== Training summary ===")
    for k, v in summary.items():
        print(f"  {k:24s} = {v}")


if __name__ == "__main__":
    main()

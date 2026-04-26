"""CLI: train a small NanoGPT from random init on our role-play +
general dataset.

This is the mission/02_llm-tuning-training "from scratch" milestone:
no transformers / accelerate / TRL — just PyTorch + the Qwen tokenizer
(reused so the vocab is sane on Korean text).

Usage:
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 \\
        scripts/train_nano_gpt.py \\
        --tokenizer models/upstream/Qwen2.5-0.5B-Instruct \\
        --output models/ours/nano-gpt-v0 \\
        --steps 1500
"""

import argparse
from pathlib import Path

# trl/datasets aren't used here, but we touch them early for safety
# in case the user later imports core.training.* alongside this script.
import datasets as _datasets  # noqa: F401

from core.training.dataset import (
    GENERAL_PAIRS,
    GENERAL_SYSTEM,
    ROLEPLAY_PAIRS,
    ROLEPLAY_SYSTEM,
)
from core.training.models.nano_gpt import NanoGPTConfig
from core.training.nano_gpt_trainer import (
    encode_pairs_with_template,
    train_nano_gpt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        default="models/upstream/Qwen2.5-0.5B-Instruct",
        help="path to a HF tokenizer directory (we reuse Qwen's)",
    )
    parser.add_argument(
        "--output",
        default="models/ours/nano-gpt-v0",
        help="output directory for model.pt + MANIFEST + recipe",
    )
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block-size", type=int, default=384)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=384)
    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise SystemExit(f"tokenizer not found: {tokenizer_path}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs: list[tuple[str, str, str]] = []
    for p in ROLEPLAY_PAIRS:
        pairs.append((ROLEPLAY_SYSTEM, p["user"], p["assistant"]))
    for p in GENERAL_PAIRS:
        pairs.append((GENERAL_SYSTEM, p["user"], p["assistant"]))
    print(f"corpus: {len(pairs)} conversations")

    sequences = encode_pairs_with_template(pairs, tokenizer)
    total_tokens = sum(len(s) for s in sequences)
    print(f"tokenised: {total_tokens:,} tokens across {len(sequences)} convs")

    cfg = NanoGPTConfig(
        # Qwen2 has special tokens beyond vocab_size (e.g. <|im_start|>);
        # len(tokenizer) is the safe upper bound for embedding sizing.
        vocab_size=len(tokenizer),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    print(
        f"architecture: vocab={cfg.vocab_size}  block={cfg.block_size}  "
        f"L={cfg.n_layer} H={cfg.n_head} D={cfg.n_embd}"
    )

    summary = train_nano_gpt(
        sequences=sequences,
        config=cfg,
        output_dir=args.output,
        eos_token_id=int(tokenizer.eos_token_id),
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print()
    print("=== summary ===")
    for k, v in summary.items():
        print(f"  {k:24s} = {v}")


if __name__ == "__main__":
    main()

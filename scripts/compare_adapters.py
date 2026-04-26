"""CLI: head-to-head adapter comparison.

  base                                  → no LoRA
  v1  ↔ models/ours/qwen-roleplay-v1    → roleplay-only
  v2  ↔ models/ours/qwen-roleplay-v2    → roleplay + general

The base loads once and acts as the LLM-as-judge for the other
candidates as well; pass --judge-base to use a separate base model
as a neutral judge instead.

Output:
  docs/compare/{ts}.json
  docs/compare/{ts}.md
"""

import argparse
import asyncio
import time
from pathlib import Path

# Force these imports first to avoid the Windows trl/transformers
# segfault we documented in core/training/trainer.py.
import datasets as _datasets  # noqa: F401
import trl as _trl  # noqa: F401

from core.eval.compare import (
    DEFAULT_PROMPTS,
    compare_adapters,
    save_report,
)
from core.serving.transformers_client import TransformersClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="models/upstream/Qwen2.5-7B-Instruct",
        help="path to base model",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help='label=path  e.g. --adapter v1=models/ours/qwen-roleplay-v1',
    )
    parser.add_argument(
        "--out-dir",
        default="docs/compare",
        help="output directory for json + md",
    )
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        raise SystemExit(f"base model not found: {base_path}")

    candidates: list[tuple[str, TransformersClient]] = []
    print(f"loading base ← {base_path}")
    base_client = TransformersClient(str(base_path))
    candidates.append(("base", base_client))

    for spec in args.adapter:
        if "=" not in spec:
            raise SystemExit(f"--adapter spec must be label=path, got: {spec}")
        label, path = spec.split("=", 1)
        path_obj = Path(path)
        if not path_obj.exists():
            raise SystemExit(f"adapter not found: {path_obj}")
        print(f"loading {label} ← base + {path_obj}")
        client = TransformersClient(str(base_path), lora_path=str(path_obj))
        candidates.append((label, client))

    print(f"\nrunning {len(DEFAULT_PROMPTS)} prompts × {len(candidates)} models")
    report = asyncio.run(
        compare_adapters(
            candidates=candidates,
            prompts=DEFAULT_PROMPTS,
            judge=base_client,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    )

    out_dir = Path(args.out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = out_dir / f"{ts}.json"
    md_path = out_dir / f"{ts}.md"
    save_report(report, json_path=json_path, markdown_path=md_path)
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")

    # quick stdout summary
    print("\n=== mean answer_relevance per label ===")
    by_label: dict[str, list[float]] = {}
    for r in report.runs:
        if r.relevance is not None:
            by_label.setdefault(r.label, []).append(r.relevance)
    for label, scores in by_label.items():
        avg = sum(scores) / len(scores)
        print(f"  {label:24s}  {avg:.3f}  (n={len(scores)})")


if __name__ == "__main__":
    main()

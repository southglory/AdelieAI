"""CLI: one round of EvalGardener loop.

Runs the persona's eval, then produces a tactical + strategic
analysis report saved as Markdown.

The agent (Claude) reads the report and proposes concrete edits
(synonym additions, new prompts, axis pivot) to the user. User
approves → manual YAML edit → next iteration.

Usage:
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 \\
        scripts/eval_iterate.py --persona cynical_merchant
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def _build_clients(model_path: str | None, lora_path: str | None):
    from core.serving.stub_client import StubLLMClient
    from core.tools import ToolRegistry
    from core.tools.evidence_search import EvidenceSearch

    if model_path and Path(model_path).is_file() and Path(model_path).suffix == ".gguf":
        from core.serving.gguf_client import GGUFClient
        llm = GGUFClient(Path(model_path))
    elif model_path and Path(model_path).exists() and (
        any(Path(model_path).glob("*.safetensors"))
        or any(Path(model_path).glob("model-*.safetensors"))
    ):
        from core.serving.transformers_client import TransformersClient
        llm = TransformersClient(model_path, lora_path=lora_path)
    else:
        llm = StubLLMClient()

    tools = ToolRegistry()
    tools.register(EvidenceSearch())

    try:
        from core.retrieval.graph_retriever_rdflib import RdflibGraphRetriever
        kg = RdflibGraphRetriever()
    except Exception:
        from core.retrieval.graph_retriever_stub import RdfGraphRetriever
        kg = RdfGraphRetriever()

    return llm, kg, tools


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--persona", required=True)
    p.add_argument(
        "--out",
        default="docs/eval/iterations",
        help="output dir for the iteration report",
    )
    p.add_argument("--label", default="", help="optional label appended to filename")
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    p.add_argument("--lora-path", default=os.environ.get("LORA_PATH"))
    args = p.parse_args()

    from core.eval.iteration import (
        build_iteration_report,
        render_iteration_md,
    )
    from core.eval.persona_eval import load_eval_spec, run_persona_eval

    print(f"[eval_iterate] persona={args.persona}")
    print(f"[eval_iterate] loading LLM…")
    llm, kg, tools = _build_clients(args.model_path, args.lora_path)
    print(f"[eval_iterate] llm={type(llm).__name__} model_id={getattr(llm, 'model_id', '?')}")

    spec = load_eval_spec(args.persona)
    eval_report = asyncio.run(run_persona_eval(
        args.persona,
        llm=llm,
        graph_retriever=kg,
        tool_registry=tools,
    ))

    ts = time.strftime("%Y%m%d_%H%M%S")
    report = build_iteration_report(
        persona_id=args.persona,
        current_eval=eval_report,
        spec=spec,
        timestamp=ts,
    )

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    label_slug = ("_" + args.label) if args.label else ""
    md_path = out_dir / f"{args.persona}_{ts}{label_slug}.md"
    md_path.write_text(render_iteration_md(report), encoding="utf-8")

    s = report.strategic

    # Console summary
    print()
    print(f"=== iteration · {args.persona} ===")
    print(f"  pass_rate              : {eval_report.pass_rate * 100:.0f}%")
    print(f"  iterations 누적         : {s.n_iterations}")
    print(f"  variance_band (last 3) : ±{s.variance_band * 100:.1f}%")
    print(f"  last_gain              : {s.last_gain * 100:+.1f}%")
    print(f"  plateaued              : {'YES' if s.plateaued else 'no'}")
    print(f"  decision_ready         : {'YES' if s.decision_ready else 'no'}")
    print(f"  axis 추천              : {s.axis_recommendation}")
    print(f"  → {s.axis_rationale}")
    print()
    if report.tactical:
        print(f"  tactical 제안 {len(report.tactical)}건:")
        for t in report.tactical[:6]:
            target = t.target_category or "(global)"
            if t.target_prompt_id:
                target += f"/{t.target_prompt_id}"
            print(f"    · {t.kind} — {target}")
    print()
    print(f"  보고서: {md_path}")


if __name__ == "__main__":
    main()

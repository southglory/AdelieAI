"""CLI: run a persona's behavioral eval suite against the live LLM.

Usage:
    PYTHONUTF8=1 .venv/Scripts/python -X utf8 \\
        scripts/eval_persona.py \\
        --persona ancient_dragon \\
        --out docs/eval/runs/

Loads `personas/{persona_id}/eval_prompts.yaml`, generates against the
active LLM (env: MODEL_PATH / LORA_PATH same as uvicorn), grades, and
saves a Markdown report.
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
    """Return (llm, graph_retriever, tool_registry) — production-parity stack."""
    from core.serving.stub_client import StubLLMClient
    from core.tools import ToolRegistry
    from core.tools.evidence_search import EvidenceSearch

    if model_path and Path(model_path).is_file() and Path(model_path).suffix == ".gguf":
        from core.serving.gguf_client import GGUFClient
        llm = GGUFClient(Path(model_path))
    elif model_path and Path(model_path).exists() and any(
        Path(model_path).glob("*.safetensors")
    ) or any(Path(model_path).glob("model-*.safetensors")):
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
    p.add_argument("--persona", required=True, help="persona_id (e.g. cynical_merchant)")
    p.add_argument(
        "--out",
        default="docs/eval/runs",
        help="output directory for the Markdown report",
    )
    p.add_argument(
        "--label",
        default="",
        help="adapter / build label appended to the report title (e.g. 'qwen-roleplay-v2')",
    )
    p.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH"),
        help="overrides MODEL_PATH env. directory with safetensors or a .gguf file",
    )
    p.add_argument(
        "--lora-path",
        default=os.environ.get("LORA_PATH"),
        help="overrides LORA_PATH env. ignored for GGUF paths",
    )
    args = p.parse_args()

    from core.eval.persona_eval import render_report_md, run_persona_eval

    llm, kg, tools = _build_clients(args.model_path, args.lora_path)
    print(f"[eval_persona] llm={type(llm).__name__} model_id={getattr(llm, 'model_id', '?')}")

    report = asyncio.run(run_persona_eval(
        args.persona,
        llm=llm,
        graph_retriever=kg,
        tool_registry=tools,
    ))

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    label_slug = args.label.replace("/", "_").replace(":", "_") if args.label else getattr(llm, "model_id", "default").replace("/", "_")
    md_path = out_dir / f"{args.persona}_{label_slug}_{ts}.md"
    md_path.write_text(render_report_md(report, adapter_label=args.label or label_slug), encoding="utf-8")

    # Console summary
    print()
    print(f"=== {args.persona} eval ===")
    print(f"  pass_rate         : {report.pass_rate * 100:.0f}% ({sum(1 for r in report.results if r.outcome == 'pass')}/{report.n_prompts})")
    print(f"  banned_violations : {report.banned_violations_total}")
    print(f"  cjk_ratio_avg     : {report.cjk_ratio_avg}")
    print(f"  cjk_han_count     : {report.cjk_han_count_total}")
    for cat, rate in sorted(report.pass_by_category.items()):
        print(f"  · {cat:25s} : {rate * 100:.0f}%")
    print()
    print(f"  report: {md_path}")


if __name__ == "__main__":
    main()

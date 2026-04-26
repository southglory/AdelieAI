# Contributing to AdelieAI

One more footprint on the ice — welcome.

## First-time guide

1. **Open an issue first.** Substantial changes go through an issue so we can agree on intent before code.
2. **Tests required.** New features ship with new tests. `pytest tests -q` must stay green (147/147).
3. **Provenance is mandatory.** Anything pulled from outside (a model, a snippet, a corpus) needs a `MANIFEST.json` or a `# PROVENANCE` comment naming source URL, revision, and license.
4. **Apache-2.0 compatible only.** We don't accept code or model weights with incompatible licenses.

## Code style

- Python 3.11+
- Type hints everywhere (`typing.Protocol` over abstract base classes).
- Comment the *why*, not the what — code already says what.
- Comments may be Korean if it helps you reason; identifiers stay English.

## Good first issues

- **Dataset diversification (LoRA v3):** add code-Q&A and ERP-domain pairs to the existing 60+60 set.
- **Korean reranker comparison:** bge-reranker-v2-m3 vs jina-multilingual-reranker — drive `scripts/compare_adapters.py` against an evaluation suite.
- **AMD ROCm / Apple MPS:** verify the stack on non-NVIDIA GPUs.
- **Eval metrics:** add BLEU/ROUGE alongside the LLM-as-judge metrics; first-class RAGAS integration as an alternative scorer.
- **Postgres swap verification:** end-to-end test pointing `DATABASE_URL` at Postgres + pgvector instead of SQLite + Chroma.
- **WebSocket streaming:** alternative transport to the current SSE.

## PR checklist

- [ ] `pytest tests -q` passes
- [ ] `pytest tests/e2e -v --browser chromium -o addopts=` passes (if you touched the web UI)
- [ ] New external assets carry a MANIFEST or PROVENANCE
- [ ] README / USAGE updated when behavior changed

---

Build small, verify, share. 🐧

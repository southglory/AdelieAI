# AdelieAI: a persona engine that ships small enough to deploy

> **TL;DR** — AdelieAI is an open-source persona engine that turns 60–120 dialogue
> pairs into a deployable LoRA + GGUF character that fits on a laptop. Two
> trained checkpoints went live on HuggingFace today, alongside a self-improving
> evaluation loop and a one-click DPO data harvester baked into the chat UI.
>
> - 🐧 Code: [github.com/southglory/AdelieAI](https://github.com/southglory/AdelieAI)
> - 🤗 GPU LoRA: [`ramyun/adelie-qwen-roleplay-v2-lora`](https://huggingface.co/ramyun/adelie-qwen-roleplay-v2-lora)
> - 🤗 CPU GGUF: [`ramyun/adelie-qwen-roleplay-v2-gguf`](https://huggingface.co/ramyun/adelie-qwen-roleplay-v2-gguf)
> - License: Apache 2.0 (code) · Tongyi Qianwen v1 (model weights, inherited from Qwen2.5)

## The problem with "an assistant"

Most LLM toolkits treat the model as a single helpful assistant. That's
the wrong abstraction for game NPCs, brand personas, vertical-domain
workers, or virtual companions — all of which need the *opposite*:
a model that **stays in character across long interactions** and runs on
hardware the user actually has.

A grumpy fantasy merchant who suddenly hedges with "*I cannot make moral
judgments*" breaks the experience. A noir detective that drops into
generic-assistant mode mid-deduction kills the scene. The base model's
"helpful, hedged, neutral" defaults are precisely what you don't want.

So the question becomes: how do you keep a small model **in character**
for hundreds of turns, ship it at a deployable size, and avoid burning
$0.0X / token on a hosted API for what is fundamentally a *personality
layer*?

That's the problem AdelieAI tries to solve.

## What's in the box

A pipeline that takes you from **persona idea → deployable character**:

1. Prepare 60–120 dialogue pairs in your character's style
2. Train a LoRA adapter on a Qwen 7B base (~25 min on a single 3090)
3. Compare against base + previous versions with an LLM-as-judge harness
4. Pack everything into a `.adelie` persona pack (one self-contained artifact)
5. Quantize to GGUF q4_k_m → 4.4 GB single file, runs on a CPU laptop
6. Drop into a game NPC, Discord bot, customer-service worker, or CLI chat

Each `.adelie` pack is a single character with consistent style, optional
RAG-grounded knowledge, and a fully reproducible training recipe.

## Three modes in the console

The web console exposes three top-level modes — they share infrastructure
where it makes sense, but answer different questions:

| Mode | Question it answers |
|---|---|
| **Persona** (`/web/personas`) | "What does this character say?" — open-ended chat with rating per turn |
| **Demo** (`/demo/{gaming,legal,knowledge}`) | "What does this character look like in its native habitat?" — same chat backend, dressed up with a per-vertical UI skin |
| **Session** (`/web/sessions`) | "Given a research goal, what's the synthesized answer?" — one-shot agentic run with planner → retriever → reasoner → reporter chain |

The demos are intentionally *visually distinct* — same engine, three
industry-shaped faces:

- 💰 **`/demo/gaming`** — JRPG shop scene, gold counter, blunt merchant tone (T2)
- 🔍 **`/demo/legal`** — noir detective office, cork board, evidence chips, `evidence_search` tool active (T3)
- 🐉 **`/demo/knowledge`** — ancient archive, inline-SVG knowledge graph, real `rdflib` + OWL-RL forward chaining over a Turtle corpus, side-panel SPARQL (T4)

The point isn't "we made a chat UI" — it's that **the same engine**,
through tier ladders, scales from a stateless system-prompt persona (T1)
to a multi-agent quest with KG reasoning (T5), without rewriting the
core. The tier you need depends on the use case; the engine doesn't
charge you for tiers you don't use.

## The lessons that actually shaped the codebase

This part is what doesn't usually fit in a launch post — but it's why
AdelieAI looks the way it does today, and why the README and
`docs/MILESTONES.md` are structured the way they are.

### "60 pairs per persona is enough" — it isn't

We tried it. A per-persona LoRA on 60 dialogue pairs **underperformed**
the shared `qwen-roleplay-v2` mixed baseline (60 role-play + 60 general
pairs combined). 80% behavioral test pass rate vs 90% for the shared
baseline. The fix wasn't "train harder" — it was *don't train this
adapter at all yet, push system-prompt strengthening first*.

Three personas + three rounds of system-prompt work later:

| Persona | Round 1 (baseline) | After prompt strengthening |
|---|---|---|
| 🐉 ancient_dragon | 84% | **96%** |
| 💰 cynical_merchant | 76% | **96%** (over multiple rounds) |
| 🔍 cold_detective | 80% | **88%** |

Saved ~10 GB of useless adapters and a few hours of GPU time. The
takeaway is in
[`docs/MILESTONES.md`](https://github.com/southglory/AdelieAI/blob/main/docs/MILESTONES.md)
under `[training/lora] (1st cycle)`.

### Hybrid English-rules + Korean-anchors prompts

Pure-Korean system prompts diluted KG context for the dragon persona —
`kg_grounding` tests fell to 40%. Switching to **English rule blocks**
(more weight in Qwen2.5's instruction tuning) **+ Korean style anchors**
(in-character voice samples) recovered to 80%, then 96%. Counter-
intuitive if you assume "the persona speaks Korean, so the prompt
should be Korean too" — actually the *rules* and the *style* are
two different signals and benefit from being in two different
languages.

### EvalGardener — agent-in-the-loop self-improving eval

Behavioral test suites have a chronic problem: they measure model
quality, but the *suites themselves* drift. New failure modes appear
that aren't covered. False positives accumulate. Variance bands swallow
real progress.

We built **EvalGardener**, a 5-phase loop where an agent (Claude, in
our case) sits between measurement and the next training round:

```
Measure → Tactical analysis → Strategic analysis → Generate → Eval-Re
```

Each round produces a markdown audit trail under
`docs/eval/iterations/`, plus an *axis recommendation* for the next
round (test-pool expansion / prompt strengthening / LoRA training /
DPO / base swap), driven by a small decision matrix.

The honest framing: **we couldn't find this exact pattern in the
literature we surveyed**. That doesn't mean it's novel in any rigorous
sense — only that we wrote it down with citations to the components we
borrowed from (DSPy, EvalPlus, TextGrad, Constitutional AI, Self-Refine,
Active Learning Survey, LMSys Arena Hard, Sleeper Agents). Full
references in
[`docs/eval/methods/iteration_loop.md`](https://github.com/southglory/AdelieAI/blob/main/docs/eval/methods/iteration_loop.md).

### Rating UX for DPO data, not 5-star reviews

We initially shipped a 5-tier star rating widget. Then noticed the
RLHF community converged on a different shape: **3-tier + dismiss**
(`good · fine · bad · dismiss`). Re-read the design choice:

- Real DPO training needs **chosen vs rejected** — binary preference
- 5-star → 4-vs-5 noise; the middle bucket carries no preference signal
- Click fatigue: 5 buttons feel like a survey, 4 buttons feel like a reaction
- **`dismiss` is a separate axis** — "this turn was just chatter, don't put it in the training pool" — and the 5-star UI had no way to say that

Refactored within the same day to 3-tier + dismiss. The
`docs/MILESTONES.md` timeline records this as a **`(2nd cycle)` return
to the same area** — the kind of design correction that's usually lost
to history but is exactly what makes the codebase
self-explanatory months later.

The DPO data is harvested by `scripts/export_dpo.py` — turns where the
same prompt got both a *good* and a *bad* rating produce
`(chosen, rejected)` JSONL pairs ready for a TRL `DPOTrainer` run.

### Stub-mode honesty

When you visit AdelieAI without weights downloaded, the demo pages
*still work* — `StubLLMClient` ships persona-aware canned replies. But
during this work we discovered the stub had a subtle bug: the same
prompt repeated would produce the *same* canned reply, which broke
the DPO test (which requires divergent replies for divergent ratings).

We tried two patches first, then realized the **root issue** was
architectural: the stub was trying to mimic an LLM sampler, which it
fundamentally isn't. Created a separate `ScriptedLLMClient` for tests
that need exact reply control, kept the stub honest about being a
"best-effort dev preview." Three rounds of refactoring on this single
issue, all logged in MILESTONES under `[serving/stub]`.

## What's on HuggingFace today

Two flavors of the same character, picked by hardware:

| | Size | Hardware | Repo |
|---|---|---|---|
| **LoRA + FP16** | ~165 MB | GPU (≥14 GB VRAM) | [`ramyun/adelie-qwen-roleplay-v2-lora`](https://huggingface.co/ramyun/adelie-qwen-roleplay-v2-lora) |
| **GGUF q4_k_m** | ~4.4 GB | CPU laptop | [`ramyun/adelie-qwen-roleplay-v2-gguf`](https://huggingface.co/ramyun/adelie-qwen-roleplay-v2-gguf) |

Both inherit Qwen2.5's [Tongyi Qianwen License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE) (commercial use permitted under the terms there).

```bash
# GPU path
huggingface-cli download ramyun/adelie-qwen-roleplay-v2-lora \
    --local-dir models/ours/qwen-roleplay-v2

# CPU path
huggingface-cli download ramyun/adelie-qwen-roleplay-v2-gguf \
    --local-dir models/ours/qwen-roleplay-v2-gguf
```

Mount and run:

```bash
LORA_PATH=models/ours/qwen-roleplay-v2 \
PYTHONUTF8=1 .venv/Scripts/uvicorn core.api.app:app --port 8770
```

Open `http://localhost:8770/web/personas` and pick a character.

## What's next (the roadmap)

| Version | Adds |
|---|---|
| v0.1 | Persona pack format, LoRA training, hybrid RAG, LangGraph agent |
| v0.1.5 | Persona gallery + multi-turn chat with per-turn telemetry |
| v0.2 (current) | GGUF q4_k_m quantization (3.25× compression, no GPU needed) |
| **v0.3** | Distillation track (7B teacher → 1.5B student) — mobile-class personas |
| **v0.4** | DPO trainer (consume the rating data we're now collecting) |
| **v0.5** | vLLM serving — multiple personas concurrent on one GPU |
| **v0.6** | Multi-persona orchestration — N personas cooperating on a single quest |

The DPO trainer (v0.4) is gated by *enough rated chat data accumulating*
— which is exactly what the rating widget exists to gather. If you try
the engine and rate a few hundred turns honestly, that data pool grows.

## The Adelie penguin, why

> Small, sturdy, plays on the ice without making a fuss.

The mascot isn't decorative — it's the engine's spirit. Focused,
self-reliant, plays in its own pond. No API keys, no hosted dependencies
at runtime, no billing surprise on month two. It runs on what you own.

If that pitch resonates, **clone the repo, pull the GGUF, send a few
chats**, and tell us what breaks first.

---

*Comments / PRs welcome:*
- 🐧 [github.com/southglory/AdelieAI](https://github.com/southglory/AdelieAI)
- 🤗 [`ramyun/adelie-qwen-roleplay-v2-lora`](https://huggingface.co/ramyun/adelie-qwen-roleplay-v2-lora) · [`ramyun/adelie-qwen-roleplay-v2-gguf`](https://huggingface.co/ramyun/adelie-qwen-roleplay-v2-gguf)
- License: Apache 2.0 (code) · Tongyi Qianwen v1 (weights)

*— made with cold flippers* 🐧

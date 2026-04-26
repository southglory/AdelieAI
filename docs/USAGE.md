# AdelieAI — End-to-end usage

A guided tour through the console UI: every feature exercised once. See the README *Live console* section + `docs/screenshots/` for the screenshot pack (`scripts/capture_screenshots.py` regenerates).

## 0. Boot

```bash
PYTHONUTF8=1 .venv/Scripts/uvicorn core.api.app:app --port 8770
```

The following environment variables are auto-detected. Missing models trigger graceful fallbacks (stub LLM, no embedder, etc.) — startup never fails because a path doesn't exist.

| Env | Default | Effect |
|---|---|---|
| `MODEL_PATH` | `models/upstream/Qwen2.5-7B-Instruct` | Main LLM |
| `LORA_PATH` | (unset) | Optional LoRA adapter loaded on top of MODEL_PATH |
| `EMBEDDING_MODEL_PATH` | `models/upstream/multilingual-e5-small` | RAG embedder |
| `RERANKER_MODEL_PATH` | `models/upstream/bge-reranker-v2-m3` | Cross-encoder reranker |
| `DATABASE_URL` | `sqlite+aiosqlite:///data/sessions.db` | Sessions + events store |
| `DOCS_DATABASE_URL` | `sqlite+aiosqlite:///data/docs.db` | Document + chunks store |
| `CHROMA_DIR` | `data/chroma` | Vector index |

`/health` shows what got loaded:
```json
{"status":"ok","version":"0.1.0",
 "llm":"Qwen/Qwen2.5-7B-Instruct",
 "embedder":"intfloat/multilingual-e5-small",
 "reranker":"BAAI/bge-reranker-v2-m3",
 "retriever":"HybridRetriever",
 "store":"SqlSessionStore"}
```

## 1. User identity

`http://localhost:8770/web/` redirects to `/web/sessions`. The top nav has a `User:` field — fill in any string and click **switch**. The value is persisted in an httpOnly cookie (the same pattern the engine documents under "private-llm-envelope": session token theft mitigation).

The user id flows into every subsequent request as the IDOR guard parameter — `SessionStore.get(session_id, user_id)` returns `None` when another user's id is supplied, never a 200.

## 2. Document ingest

`/web/docs` exposes an **Ingest** form (collapsible). Paste plain text or a markdown blob; submit.

Internal pipeline (`IngestService.ingest`):
1. `RecursiveTextSplitter` — `\n\n` → `\n` → `. ` → ` ` → char fallback. `chunk_size=1000`, `chunk_overlap=120`.
2. `embed_passages([chunk_text…])` — e5 receives the `passage:` prefix automatically.
3. `ChromaVectorStore.upsert` — HNSW cosine.
4. `SqlDocumentStore.add` — Postgres swap on a URL change.
5. `InMemoryBM25.add` — same chunks indexed for sparse retrieval.

A row is added immediately (`hx-swap=afterbegin`).

## 3. Search (retriever, called directly)

The same docs page has a **Search** form for natural-language queries.

Internal pipeline (`HybridRetriever.retrieve`):
1. `embed_query("…")` (e5 `query:` prefix)
2. `ChromaVectorStore.search` (top 20 candidates)
3. `InMemoryBM25.search` (top 20 candidates)
4. **Reciprocal Rank Fusion** with `rrf_k=60` — `score(c) = Σ 1 / (60 + rank_r)`
5. (Optional) `CrossEncoderReranker.rerank` re-scores `(query, chunk)` pairs with bge-reranker-v2-m3
6. `DocumentStore.get_chunks` hydrates final chunk metadata

The response carries `method: "dense" | "bm25" | "hybrid" | "reranked"` so the frontend (and any client) sees which path produced the ranking.

## 4. Document detail

Click a document id to view its content + the chunk list — useful for verifying the chunker boundaries.

## 5. Create a session

`/web/sessions` has a **Create** form. The session enters `state = PENDING`. State machine has five-way defenses:
- Legal transitions: `PENDING → RUNNING`, `RUNNING → {COMPLETED, FAILED, CANCELLED}`, `PENDING → CANCELLED`.
- Three terminal states are sinks — no further transitions.
- Event sourcing: every transition + LLM call + retrieval result is appended to the `agent_events` table.

## 6. Run config + execute

Click into a session to see the **Run config** form. Six fields:

| Field | Default | Meaning |
|---|---|---|
| system | (empty) | System prompt (Qwen chat template) |
| max_new_tokens | 256 | Generation cap |
| temperature | 0.7 | Sampling temperature (0 = greedy) |
| top_p | 0.9 | Nucleus sampling |
| top_k | 50 | Top-k sampling |
| **retrieval_k** | 0 | 0 = retrieval off; ≥1 = top-k chunks injected as context |

Two run buttons:

- **▶ generate (streaming)** — single-shot. With `retrieval_k > 0`: one retrieval call → one LLM call. Tokens stream over SSE.
- **🤖 agentic (LangGraph)** — four-node graph:
  1. **planner** (LLM, JSON output: `skip_retrieval`, `search_queries`, `rationale`)
  2. **retriever** (Hybrid + Rerank) — runs each `search_query`, dedups, top-k merges
  3. **reasoner** (LLM with the retrieved context block)
  4. **reporter** — appends a Sources footer

## 7. Streaming output

While streaming, the UI shows `first token N ms` (TTFT). Typical numbers on a 3090 + Qwen-7B (bf16): 300–700 ms first-token, ~70 ms per subsequent token.

## 8. Result + audit log

After completion the page reloads with the **Answer** card and the **Events** timeline below.

### Answer card
- Body: the LLM-generated text (no markdown rendering — shown verbatim).
- Footer: the actual sampling parameters used + system prompt — full audit.

### Events timeline
Auto-refreshing (2-second polling). Each row:
- `time` — UTC HH:MM:SS
- `type` — `state_transition` / `retrieval` / `llm_call` / `final` / `error` / `tool_call`
- `from → to` — only populated on state transitions
- `tokens (in/out)` · `latency` — populated on LLM calls

In agentic mode each event also carries `payload.node` so `plan` / `retrieve` / `reason` / `report` are distinguishable.

## 9. Persistence

Restart `uvicorn` — everything survives:
- Sessions + events → SQLite (`data/sessions.db`)
- Documents + chunks → SQLite (`data/docs.db`)
- Embeddings → ChromaDB persistent client (`data/chroma/`)
- BM25 → in-memory; rebuilt on startup via `IngestService.warm_bm25_from_doc_store`

## 10. Observability (uvicorn stdout)

```json
{"ts":"2026-04-25T...","level":"INFO","logger":"differentia.http","msg":"request","request_id":"abc123def","method":"POST","path":"/api/v1/agents/sessions","status":201,"duration_ms":42}
{"ts":"2026-04-25T...","level":"INFO","logger":"differentia.agent.agentic","msg":"session_started","request_id":"abc123def","session_id":"...","mode":"agentic","retrieval_k":3}
{"ts":"2026-04-25T...","level":"INFO","logger":"differentia.agent.nodes","msg":"planner_done","request_id":"abc123def","queries":2,"skip_retrieval":false}
```

Send `X-Request-Id` from upstream and it propagates through every log line in the pipeline; otherwise a 12-char hex id is generated. Both flows return the id in the response header for end-to-end tracing.

## 11. JSON API (CLI / programmatic clients)

```bash
# Create a session
curl -X POST http://localhost:8770/api/v1/agents/sessions \
  -H "X-User-Id: alice" -H "Content-Type: application/json" \
  -d '{"goal":"...","model_spec":"qwen-7b"}'

# Synchronous run
curl -X POST http://localhost:8770/api/v1/agents/sessions/{id}/run \
  -H "X-User-Id: alice" -H "Content-Type: application/json" \
  -d '{"max_new_tokens":256,"temperature":0.7,"retrieval_k":3}'

# Streaming run
curl -N -X POST http://localhost:8770/api/v1/agents/sessions/{id}/run/stream \
  -H "X-User-Id: alice" -H "Content-Type: application/json" \
  -d '{"retrieval_k":3}'

# Agentic (LangGraph 4-node)
curl -X POST "http://localhost:8770/api/v1/agents/sessions/{id}/run/agentic?retrieval_k=5" \
  -H "X-User-Id: alice"

# Events timeline
curl http://localhost:8770/api/v1/agents/sessions/{id}/events \
  -H "X-User-Id: alice"

# Ingest a document
curl -X POST http://localhost:8770/api/v1/docs \
  -H "X-User-Id: alice" -H "Content-Type: application/json" \
  -d '{"title":"...","source":"...","content":"..."}'

# Search
curl -X POST http://localhost:8770/api/v1/docs/search \
  -H "X-User-Id: alice" -H "Content-Type: application/json" \
  -d '{"query":"...","k":5}'
```

Auto-generated Swagger lives at `http://localhost:8770/docs`.

## 12. Test reproduction

```bash
# 147 unit tests
.venv/Scripts/python -m pytest tests -q

# E2E Playwright walk (~20 s). -o addopts= unsets the unit-mode "-p no:playwright" flag.
.venv/Scripts/python -m pytest tests/e2e -v --browser chromium -o addopts=

# Drive a live server (real Qwen-7B loaded) instead of spawning one
E2E_BASE_URL=http://localhost:8770 .venv/Scripts/python -m pytest tests/e2e -v \
    --browser chromium -o addopts=
```

Screenshot set (`docs/screenshots/01_sessions.png` ~ `06_swagger.png`) is regenerated by `scripts/capture_screenshots.py` rather than the e2e suite, so it works in stub-only mode without needing real model weights.

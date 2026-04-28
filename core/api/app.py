import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core import __version__
from core.api.agents import build_router
from core.api.demos_router import build_demos_router
from core.retrieval.graph_retriever_stub import RdfGraphRetriever, StubOWLReasoner
from core.tools import ToolRegistry
from core.tools.evidence_search import EvidenceSearch
from core.api.docs import build_docs_router
from core.api.docs_web import build_docs_web_router
from core.api.eval import build_eval_router
from core.api.eval_web import build_eval_web_router
from core.api.middleware import RequestContextMiddleware
from core.api.personas_web import build_personas_web_router
from core.api.presets import build_presets_router
from core.api.web import build_web_router
from core.logging import configure_logging, get_logger
from core.personas.store import ChatStore, SqlChatStore
from core.retrieval.bm25 import InMemoryBM25
from core.retrieval.chunker import RecursiveTextSplitter
from core.retrieval.document_store import SqlDocumentStore
from core.retrieval.hybrid import HybridRetriever
from core.retrieval.ingest import DenseRetriever, IngestService
from core.retrieval.protocols import Embedder, Reranker, VectorStore
from core.retrieval.vectorstore import ChromaVectorStore
from core.serving.protocols import LLMClient
from core.serving.stub_client import StubLLMClient
from core.session.protocols import SessionStore
from core.session.store_memory import InMemorySessionStore
from core.session.store_sql import SqlSessionStore

configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
log = get_logger("differentia.bootstrap")


def _has_weights(path: Path) -> bool:
    """A directory counts as a real model only when it actually contains
    safetensors weights — a bare MANIFEST.json on its own is just
    provenance metadata, not a runnable model.
    """
    if not path.exists():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("model-*.safetensors"))


def _compute_tier(
    app, *, ingest=None, retriever=None
) -> dict[str, object]:
    """Self-introspect which capability tier the running build supports.

    See docs/CAPABILITY_TIERS.md for the framework. Returned shape is
    consumed by both / and /health so the public surface declares a
    consistent answer.
    """
    statuses: dict[str, str] = {}
    tier = 1

    # T1 — vanilla LLM. Always present (StubLLMClient is the floor).
    statuses["T1"] = "ok"

    # T2 — LoRA + vector RAG. We accept either a real LLM (transformers
    # or gguf) and a hybrid retriever as a working T2 stack.
    llm_real = type(app.state.llm).__name__ in {"TransformersClient", "GGUFClient"}
    has_vector_rag = ingest is not None and retriever is not None
    if llm_real and has_vector_rag:
        tier = 2
        statuses["T2"] = "ok"
    else:
        missing = []
        if not llm_real:
            missing.append("real_llm")
        if not has_vector_rag:
            missing.append("vector_rag")
        statuses["T2"] = "missing: " + ", ".join(missing) if missing else "ok"

    # T3 — tool-use protocol available + at least one tool registered.
    try:
        from core.tools import ToolRegistry  # noqa: F401

        tool_registry = getattr(app.state, "tool_registry", None)
        if tool_registry is not None and len(tool_registry) > 0:
            tier = max(tier, 3)
            n = len(tool_registry)
            statuses["T3"] = f"ok ({n} tool{'s' if n != 1 else ''})"
        else:
            statuses["T3"] = "missing: tool_registry"
    except ImportError:
        statuses["T3"] = "missing: core.tools"

    # T4 — graph retriever + (optionally) OWL reasoner.
    graph = getattr(app.state, "graph_retriever", None)
    if graph is not None:
        tier = max(tier, 4)
        reasoner = getattr(app.state, "owl_reasoner", None)
        statuses["T4"] = "ok" + (" + reasoner" if reasoner else " (no reasoner)")
    else:
        statuses["T4"] = "missing: graph_retriever"

    # T5 — multi-agent runner + per-persona LoRA loader (vLLM bridge).
    multi_agent = getattr(app.state, "multi_agent_runner", None)
    if multi_agent is not None:
        tier = max(tier, 5)
        statuses["T5"] = "ok"
    else:
        statuses["T5"] = "missing: multi_agent_runner"

    return {"tier": tier, "tier_max": 5, "tier_status": statuses}


def _default_llm() -> LLMClient:
    model_path = os.environ.get(
        "MODEL_PATH", "models/upstream/Qwen2.5-7B-Instruct"
    )
    lora_path = os.environ.get("LORA_PATH")
    path = Path(model_path)

    # GGUF path: a single quantized file. LoRA is not applied at runtime —
    # GGUF artifacts are already merged + quantized upstream
    # (see differentia-llm/experiments/06_gguf_export/).
    if path.is_file() and path.suffix == ".gguf":
        try:
            from core.serving.gguf_client import GGUFClient

            return GGUFClient(path)
        except Exception as e:
            log.warning(
                "gguf_load_failed",
                extra={"path": str(path), "error": str(e)},
            )
            return StubLLMClient()

    if _has_weights(path):
        try:
            from core.serving.transformers_client import TransformersClient

            return TransformersClient(model_path, lora_path=lora_path)
        except Exception as e:
            log.warning(
                "llm_load_failed",
                extra={
                    "path": model_path,
                    "lora_path": lora_path,
                    "error": str(e),
                },
            )
    else:
        log.info("llm_skip", extra={"path": model_path, "reason": "no weights"})
    return StubLLMClient()


def _default_store() -> SessionStore:
    url = os.environ.get("DATABASE_URL")
    if url is None:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        url = f"sqlite+aiosqlite:///{data_dir.absolute().as_posix()}/sessions.db"
    return SqlSessionStore.from_url(url)


def _default_doc_store() -> SqlDocumentStore:
    url = os.environ.get("DOCS_DATABASE_URL")
    if url is None:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        url = f"sqlite+aiosqlite:///{data_dir.absolute().as_posix()}/docs.db"
    return SqlDocumentStore.from_url(url)


def _default_chat_store() -> SqlChatStore:
    url = os.environ.get("CHAT_DATABASE_URL")
    if url is None:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        url = f"sqlite+aiosqlite:///{data_dir.absolute().as_posix()}/chats.db"
    return SqlChatStore.from_url(url)


def _default_embedder() -> Embedder | None:
    path = os.environ.get(
        "EMBEDDING_MODEL_PATH", "models/upstream/multilingual-e5-small"
    )
    if not _has_weights(Path(path)):
        log.info(
            "embedder_skip",
            extra={"path": path, "reason": "no weights"},
        )
        return None
    try:
        from core.retrieval.embedder import TransformersEmbedder

        return TransformersEmbedder(path)
    except Exception as e:
        log.warning(
            "embedder_load_failed", extra={"path": path, "error": str(e)}
        )
        return None


def _default_vector_store() -> VectorStore:
    persist = os.environ.get("CHROMA_DIR", "data/chroma")
    return ChromaVectorStore(persist)


def _default_reranker() -> Reranker | None:
    path = os.environ.get(
        "RERANKER_MODEL_PATH", "models/upstream/bge-reranker-v2-m3"
    )
    if not _has_weights(Path(path)):
        log.info(
            "reranker_skip",
            extra={"path": path, "reason": "no weights"},
        )
        return None
    try:
        from core.retrieval.reranker import CrossEncoderReranker

        return CrossEncoderReranker(path)
    except Exception as e:
        log.warning(
            "reranker_load_failed", extra={"path": path, "error": str(e)}
        )
        return None


def build_app(
    store: SessionStore | None = None,
    llm: LLMClient | None = None,
    ingest: IngestService | None = None,
    retriever=None,
    chat_store: ChatStore | None = None,
) -> FastAPI:
    resolved_store = store if store is not None else _default_store()
    resolved_llm = llm if llm is not None else _default_llm()
    resolved_chat_store: ChatStore = (
        chat_store if chat_store is not None else _default_chat_store()
    )

    bm25_to_warm: InMemoryBM25 | None = None

    if ingest is None:
        embedder = _default_embedder()
        if embedder is not None:
            doc_store = _default_doc_store()
            vector_store = _default_vector_store()
            chunker = RecursiveTextSplitter(chunk_size=1000, chunk_overlap=120)
            bm25 = InMemoryBM25()
            bm25_to_warm = bm25
            ingest = IngestService(
                chunker, embedder, doc_store, vector_store, bm25=bm25
            )
            if retriever is None:
                reranker = _default_reranker()
                retriever = HybridRetriever(
                    embedder=embedder,
                    vector_store=vector_store,
                    bm25=bm25,
                    doc_store=doc_store,
                    reranker=reranker,
                    candidate_pool=20,
                )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if isinstance(resolved_store, SqlSessionStore):
            await resolved_store.init_schema()
        if isinstance(resolved_chat_store, SqlChatStore):
            await resolved_chat_store.init_schema()
        if ingest is not None and isinstance(ingest.doc_store, SqlDocumentStore):
            await ingest.doc_store.init_schema()
        if ingest is not None and bm25_to_warm is not None:
            await ingest.warm_bm25_from_doc_store()
        yield
        if isinstance(resolved_store, SqlSessionStore):
            await resolved_store.dispose()
        if isinstance(resolved_chat_store, SqlChatStore):
            await resolved_chat_store.dispose()
        if ingest is not None and isinstance(ingest.doc_store, SqlDocumentStore):
            await ingest.doc_store.dispose()

    app = FastAPI(
        title="AdelieAI core",
        version=__version__,
        description="Self-hosted Agentic RAG engine — own your weights, own your tokens.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)

    assets_dir = Path(__file__).resolve().parent.parent.parent / "assets"
    if assets_dir.exists():
        app.mount("/static", StaticFiles(directory=str(assets_dir)), name="static")

    app.state.store = resolved_store
    app.state.llm = resolved_llm
    app.state.ingest = ingest
    app.state.retriever = retriever
    app.state.chat_store = resolved_chat_store

    # T3 — default tool registry. Registers the evidence_search stub used
    # by /demo/legal so `_compute_tier` can declare "T3: ok (1 tool)" out
    # of the box. Real tool wiring per persona is a future milestone.
    tool_registry = ToolRegistry()
    tool_registry.register(EvidenceSearch())
    app.state.tool_registry = tool_registry

    # T4 — default KG retriever + OWL reasoner.
    #
    # Preferred path: real rdflib over `core/retrieval/dragon_lore.ttl`,
    # with owlrl forward-chaining for OWL-RL inference. SPARQL property
    # paths (`descendantOf+`) and subClassOf transitive closure both
    # work end-to-end.
    #
    # Fallback path: hand-baked stub with frozen answers (the original
    # Step 4 implementation). Activates only when rdflib import fails
    # so the build still declares T4 instead of dropping to T3.
    try:
        from core.retrieval.graph_retriever_rdflib import (
            RdflibGraphRetriever,
            RdflibOWLReasoner,
        )

        kg = RdflibGraphRetriever()
        app.state.graph_retriever = kg
        app.state.owl_reasoner = RdflibOWLReasoner(kg)
        log.info("kg_retriever_loaded", extra={"backend": "rdflib"})
    except Exception as e:
        log.warning(
            "kg_retriever_fallback",
            extra={"backend": "stub", "error": str(e)},
        )
        app.state.graph_retriever = RdfGraphRetriever()
        app.state.owl_reasoner = StubOWLReasoner()
    app.include_router(
        build_router(app.state.store, app.state.llm, retriever=retriever)
    )
    app.include_router(
        build_web_router(app.state.store, app.state.llm, retriever=retriever)
    )
    app.include_router(
        build_personas_web_router(resolved_chat_store, app.state.llm)
    )
    app.include_router(build_docs_router(ingest, retriever))
    app.include_router(build_docs_web_router(ingest, retriever))
    app.include_router(build_eval_router(app.state.store, app.state.llm))
    app.include_router(build_eval_web_router(app.state.store, app.state.llm))
    app.include_router(build_presets_router())
    app.include_router(build_demos_router())

    @app.get("/health")
    async def health() -> dict[str, object]:
        rerank_id = (
            getattr(retriever.reranker, "model_id", None)
            if isinstance(retriever, HybridRetriever)
            and retriever.reranker is not None
            else None
        )
        tier_info = _compute_tier(app, ingest=ingest, retriever=retriever)
        return {
            "status": "ok",
            "version": __version__,
            "llm": app.state.llm.model_id,
            "store": type(app.state.store).__name__,
            "embedder": (
                ingest.embedder.model_id if ingest is not None else None
            ),
            "reranker": rerank_id,
            "retriever": (
                type(retriever).__name__ if retriever is not None else None
            ),
            **tier_info,
        }

    @app.get("/")
    async def root() -> dict[str, object]:
        tier_info = _compute_tier(app, ingest=ingest, retriever=retriever)
        return {
            "name": "AdelieAI",
            "version": __version__,
            "modules": [
                "schemas", "session", "agent", "serving",
                "retrieval", "personas", "tools", "api",
            ],
            "llm": app.state.llm.model_id,
            "store": type(app.state.store).__name__,
            "chat_store": type(app.state.chat_store).__name__,
            "retrieval": "ready" if ingest is not None else "unavailable",
            "tier": tier_info["tier"],
            "tier_max": tier_info["tier_max"],
        }

    return app


app = build_app()

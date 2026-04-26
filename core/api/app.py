import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core import __version__
from core.api.agents import build_router
from core.api.docs import build_docs_router
from core.api.docs_web import build_docs_web_router
from core.api.eval import build_eval_router
from core.api.eval_web import build_eval_web_router
from core.api.middleware import RequestContextMiddleware
from core.api.presets import build_presets_router
from core.api.web import build_web_router
from core.logging import configure_logging, get_logger
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


def _default_llm() -> LLMClient:
    model_path = os.environ.get(
        "MODEL_PATH", "models/upstream/Qwen2.5-7B-Instruct"
    )
    lora_path = os.environ.get("LORA_PATH")
    if _has_weights(Path(model_path)):
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
) -> FastAPI:
    resolved_store = store if store is not None else _default_store()
    resolved_llm = llm if llm is not None else _default_llm()

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
        if ingest is not None and isinstance(ingest.doc_store, SqlDocumentStore):
            await ingest.doc_store.init_schema()
        if ingest is not None and bm25_to_warm is not None:
            await ingest.warm_bm25_from_doc_store()
        yield
        if isinstance(resolved_store, SqlSessionStore):
            await resolved_store.dispose()
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
    app.include_router(
        build_router(app.state.store, app.state.llm, retriever=retriever)
    )
    app.include_router(
        build_web_router(app.state.store, app.state.llm, retriever=retriever)
    )
    app.include_router(build_docs_router(ingest, retriever))
    app.include_router(build_docs_web_router(ingest, retriever))
    app.include_router(build_eval_router(app.state.store, app.state.llm))
    app.include_router(build_eval_web_router(app.state.store, app.state.llm))
    app.include_router(build_presets_router())

    @app.get("/health")
    async def health() -> dict[str, object]:
        rerank_id = (
            getattr(retriever.reranker, "model_id", None)
            if isinstance(retriever, HybridRetriever)
            and retriever.reranker is not None
            else None
        )
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
        }

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "name": "differentia-llm core",
            "version": __version__,
            "modules": ["schemas", "session", "agent", "serving", "retrieval", "api"],
            "llm": app.state.llm.model_id,
            "store": type(app.state.store).__name__,
            "retrieval": "ready" if ingest is not None else "unavailable",
        }

    return app


app = build_app()

"""End-to-end Playwright flow that walks the whole UI and saves a
screenshot at every step under docs/screenshots/. Run:

    .venv/Scripts/python -m pytest tests/e2e/test_flow.py -v

Uses a session-scoped uvicorn fixture (stub LLM, real e5 embedder,
no reranker) so the flow is deterministic and quick. To run against
a live server with the real LLM, set E2E_BASE_URL=http://localhost:8766
"""
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

DOC_TITLE = "Engine overview"
DOC_CONTENT = (
    "differentia-llm 엔진은 다섯 가지 모듈로 구성된다.\n\n"
    "1. schemas — AgentSession, AgentEvent, Chunk, RetrievedContext, EvalResult.\n\n"
    "2. session — 세션 상태 머신과 이벤트 소싱. PENDING → RUNNING → COMPLETED 전이.\n\n"
    "3. retrieval — RecursiveTextSplitter, multilingual-e5 임베딩, ChromaDB 벡터 스토어, "
    "BM25 + Reciprocal Rank Fusion, bge-reranker-v2-m3 cross-encoder.\n\n"
    "4. agent — LangGraph 4-노드 (planner → retriever → reasoner → reporter).\n\n"
    "5. serving — Transformers/Stub LLMClient. Local Qwen2.5 모델 사용."
)


@pytest.fixture(scope="session")
def shots(screenshots: Path) -> Path:
    return screenshots


def _shot(page: Page, shots: Path, name: str) -> None:
    page.screenshot(path=str(shots / f"{name}.png"), full_page=True)


def test_e2e_full_flow(page: Page, base_url: str, shots: Path) -> None:
    page.set_viewport_size({"width": 1280, "height": 900})

    # 1. landing — sessions list
    page.goto(f"{base_url}/web/sessions")
    expect(page.locator("h2")).to_contain_text("Agent Sessions")
    _shot(page, shots, "01_sessions_empty")

    # 2. set user identity via cookie form
    page.locator("input[name='user_id']").fill("e2e-tester")
    page.locator("nav form button[type='submit']").click()
    page.wait_for_url(f"{base_url}/web/sessions")
    _shot(page, shots, "02_user_set")

    # 3. ingest a document
    page.goto(f"{base_url}/web/docs")
    expect(page.locator("h2")).to_contain_text("Documents")
    _shot(page, shots, "03_docs_empty")

    page.locator("summary", has_text="Ingest").click()
    page.locator("input[name='title']").fill(DOC_TITLE)
    page.locator("textarea[name='content']").fill(DOC_CONTENT)
    _shot(page, shots, "04_docs_ingest_form")

    page.locator("form[hx-post='/web/docs'] button[type='submit']").click()
    expect(page.locator("#doc-list")).to_contain_text(DOC_TITLE, timeout=15_000)
    _shot(page, shots, "05_docs_ingested")

    # 4. search the retriever directly
    page.locator("summary", has_text="Search").click()
    page.locator("input[name='query']").fill("LangGraph 4 노드")
    page.locator("form[hx-post='/web/docs/search'] button[type='submit']").click()
    expect(page.locator("#search-results table")).to_be_visible(timeout=15_000)
    _shot(page, shots, "06_docs_search_results")

    # 5. open the doc detail page
    page.locator("#doc-list a").first.click()
    expect(page.locator("h2")).to_contain_text("Document")
    _shot(page, shots, "07_doc_detail")

    # 6. create a session
    page.goto(f"{base_url}/web/sessions")
    page.locator("input[name='goal']").fill(
        "differentia-llm 엔진의 retrieval 모듈에는 어떤 컴포넌트가 있나?"
    )
    page.locator(".create-form button[type='submit']").click()
    expect(page.locator("#session-list .row").first).to_be_visible(timeout=15_000)
    _shot(page, shots, "08_session_created")

    # 7. open the session detail page → run config form
    page.locator("#session-list a").first.click()
    expect(page.locator("h2")).to_contain_text("Session")
    page.locator("input[name='retrieval_k']").fill("3")
    _shot(page, shots, "09_session_detail_run_config")

    # 8. trigger streaming generate, wait for completion
    with page.expect_response(
        lambda r: r.url.endswith("/run/stream"), timeout=120_000
    ) as resp_info:
        page.locator("#generate-btn").click()
    resp_info.value
    expect(page.locator("#stream-status")).to_contain_text("done", timeout=120_000)
    _shot(page, shots, "10_stream_done")

    # 9. wait for the page to reload with the answer card + events
    page.wait_for_load_state("networkidle")
    expect(page.locator(".answer-card")).to_be_visible(timeout=30_000)
    _shot(page, shots, "11_session_with_answer")

    # 10. health endpoint as JSON snapshot
    page.goto(f"{base_url}/health")
    _shot(page, shots, "12_health")

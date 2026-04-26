import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

E2E_BASE_URL = os.environ.get("E2E_BASE_URL")
SCREENSHOT_DIR = Path(__file__).parent.parent.parent / "docs" / "screenshots"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def base_url() -> str:
    """If E2E_BASE_URL is set, point at that running server. Otherwise
    spawn a fresh uvicorn with a stub LLM (fast + deterministic).
    """
    if E2E_BASE_URL:
        return E2E_BASE_URL.rstrip("/")

    port = _free_port()
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["MODEL_PATH"] = os.environ.get(
        "E2E_MODEL_PATH", str(repo / "no-such-path-forces-stub-llm")
    )
    env["EMBEDDING_MODEL_PATH"] = os.environ.get(
        "E2E_EMBEDDING_MODEL_PATH",
        str(repo / "models" / "upstream" / "multilingual-e5-small"),
    )
    env["RERANKER_MODEL_PATH"] = os.environ.get(
        "E2E_RERANKER_MODEL_PATH",
        str(repo / "no-such-path-disables-reranker"),
    )
    env["DATABASE_URL"] = f"sqlite+aiosqlite:///{(repo / 'data' / 'e2e_sessions.db').as_posix()}"
    env["DOCS_DATABASE_URL"] = f"sqlite+aiosqlite:///{(repo / 'data' / 'e2e_docs.db').as_posix()}"
    env["CHROMA_DIR"] = str(repo / "data" / "e2e_chroma")
    (repo / "data" / "e2e_sessions.db").unlink(missing_ok=True)
    (repo / "data" / "e2e_docs.db").unlink(missing_ok=True)
    import shutil

    shutil.rmtree(repo / "data" / "e2e_chroma", ignore_errors=True)

    cmd = [
        sys.executable, "-m", "uvicorn", "core.api.app:app",
        "--port", str(port), "--log-level", "warning",
    ]
    proc = subprocess.Popen(cmd, cwd=str(repo), env=env)
    url = f"http://127.0.0.1:{port}"

    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError("uvicorn did not start in 90s")

    yield url

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def screenshots() -> Path:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    return SCREENSHOT_DIR

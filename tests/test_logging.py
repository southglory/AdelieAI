import io
import json
import logging

import pytest
from fastapi.testclient import TestClient

from core.api.app import build_app
from core.logging import (
    JsonFormatter,
    configure_logging,
    get_logger,
    request_id_var,
)
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore


@pytest.fixture
def captured_logs() -> list[str]:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    prior_handlers = root.handlers[:]
    prior_level = root.level
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    yield buf
    root.handlers = prior_handlers
    root.setLevel(prior_level)


def _records(buf: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buf.getvalue().strip().splitlines() if line]


def test_json_formatter_basic_fields(captured_logs: io.StringIO) -> None:
    log = get_logger("test")
    log.info("hello", extra={"foo": "bar"})
    records = _records(captured_logs)
    assert len(records) == 1
    r = records[0]
    assert r["msg"] == "hello"
    assert r["level"] == "INFO"
    assert r["foo"] == "bar"
    assert r["request_id"] == "-"


def test_request_id_propagates(captured_logs: io.StringIO) -> None:
    log = get_logger("test")
    token = request_id_var.set("abc123")
    try:
        log.info("scoped")
    finally:
        request_id_var.reset(token)
    r = _records(captured_logs)[0]
    assert r["request_id"] == "abc123"


def test_middleware_assigns_request_id(captured_logs: io.StringIO) -> None:
    app = build_app(store=InMemorySessionStore(), llm=StubLLMClient())
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert "x-request-id" in {k.lower() for k in r.headers.keys()}
    rid = r.headers["x-request-id"]
    records = _records(captured_logs)
    request_records = [x for x in records if x.get("msg") == "request"]
    assert any(
        x["request_id"] == rid and x["status"] == 200 and x["path"] == "/health"
        for x in request_records
    )


def test_middleware_propagates_external_request_id(
    captured_logs: io.StringIO,
) -> None:
    app = build_app(store=InMemorySessionStore(), llm=StubLLMClient())
    client = TestClient(app)
    r = client.get("/health", headers={"X-Request-Id": "trace-xyz"})
    assert r.headers["x-request-id"] == "trace-xyz"


def test_session_logs_emitted(captured_logs: io.StringIO) -> None:
    app = build_app(store=InMemorySessionStore(), llm=StubLLMClient())
    client = TestClient(app)
    headers = {"X-User-Id": "alice"}
    sid = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "g", "model_spec": "m"},
        headers=headers,
    ).json()["id"]
    client.post(f"/api/v1/agents/sessions/{sid}/run", headers=headers)

    records = _records(captured_logs)
    started = [x for x in records if x.get("msg") == "session_started"]
    completed = [x for x in records if x.get("msg") == "session_completed"]
    assert started
    assert completed
    assert started[0]["session_id"] == sid
    assert completed[0]["session_id"] == sid
    assert completed[0]["model_id"] == "stub-deterministic-1"

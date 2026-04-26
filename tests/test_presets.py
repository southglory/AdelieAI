import pytest
from fastapi.testclient import TestClient

from core.agent.presets import PRESETS, Preset, get_preset, list_presets
from core.api.app import build_app
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore


def test_built_in_preset_names() -> None:
    names = {p.name for p in list_presets()}
    assert {"default", "role-play", "factual", "concise", "code", "rag-strict"} <= names


def test_role_play_preset_has_anti_meta_instructions() -> None:
    p = get_preset("role-play")
    assert isinstance(p, Preset)
    assert p.system is not None
    assert "AI" in p.system
    assert "메타" in p.system or "디스클레이머" in p.system
    assert p.temperature >= 0.8
    assert "한국어" in p.system


def test_rag_strict_enables_retrieval() -> None:
    p = get_preset("rag-strict")
    assert p is not None
    assert p.retrieval_k > 0
    assert "[1]" in p.system or "인용" in p.system


def test_get_preset_unknown_returns_none() -> None:
    assert get_preset("totally-made-up") is None


def test_preset_objects_are_frozen() -> None:
    p = list_presets()[0]
    with pytest.raises(Exception):
        p.temperature = 999  # type: ignore[misc]


@pytest.fixture
def client() -> TestClient:
    app = build_app(store=InMemorySessionStore(), llm=StubLLMClient())
    return TestClient(app)


def test_list_presets_endpoint(client: TestClient) -> None:
    r = client.get("/api/v1/presets")
    assert r.status_code == 200
    data = r.json()
    names = {p["name"] for p in data}
    assert "role-play" in names
    assert "factual" in names


def test_get_single_preset_endpoint(client: TestClient) -> None:
    r = client.get("/api/v1/presets/role-play")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "role-play"
    assert body["temperature"] >= 0.8
    assert body["system"] is not None


def test_unknown_preset_404(client: TestClient) -> None:
    r = client.get("/api/v1/presets/garbage")
    assert r.status_code == 404

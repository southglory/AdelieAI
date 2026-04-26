import pytest
from fastapi.testclient import TestClient

from core.eval.heuristics import citation_coverage, retrieval_recall_at_k
from core.eval.judges import _parse_score, judge_answer_relevance, judge_faithfulness
from core.eval.runner import evaluate_session
from core.api.app import build_app
from core.serving.protocols import GenerationParams, GenerationResult
from core.session.events import build_event
from core.schemas.agent import EventType, SessionState
from core.serving.stub_client import StubLLMClient
from core.session.store_memory import InMemorySessionStore


# --- _parse_score ---


def test_parse_score_extracts_first_float() -> None:
    assert _parse_score("0.83") == 0.83
    assert _parse_score("점수: 0.5") == 0.5
    assert _parse_score("answer is 1.0") == 1.0


def test_parse_score_clamps_to_one() -> None:
    assert _parse_score("1.5") == 1.0
    assert _parse_score("100") == 1.0


def test_parse_score_handles_x_out_of_10_scale() -> None:
    assert _parse_score("8") == 0.8
    assert _parse_score("score 7") == 0.7


def test_parse_score_returns_none_on_no_number() -> None:
    assert _parse_score("not a number") is None


# --- citation_coverage ---


def test_citation_coverage_full() -> None:
    score, details = citation_coverage("위 사실은 [1], [2], [3] 참조.", retrieved_count=3)
    assert score == 1.0
    assert details["in_range"] == [1, 2, 3]
    assert not details["out_of_range"]


def test_citation_coverage_partial() -> None:
    score, _ = citation_coverage("[1] 만 있다", retrieved_count=4)
    assert score == 0.25


def test_citation_coverage_out_of_range_flagged() -> None:
    score, details = citation_coverage("[1], [99]", retrieved_count=3)
    assert score == pytest.approx(1 / 3)
    assert details["out_of_range"] == [99]


def test_citation_coverage_no_retrieval() -> None:
    score, details = citation_coverage("text [1]", retrieved_count=0)
    assert score == 0.0
    assert details["out_of_range"] == [1]


# --- retrieval_recall_at_k ---


def test_recall_at_k() -> None:
    score, details = retrieval_recall_at_k(["a", "b", "c"], ["b", "d"])
    assert score == 0.5
    assert details["hits"] == 1


def test_recall_no_ground_truth() -> None:
    score, details = retrieval_recall_at_k(["a"], [])
    assert score == 0.0
    assert "no ground truth" in details["reason"]


# --- LLM-as-judge with scripted client ---


class ScriptedJudge:
    """Deterministic judge — returns a fixed score string per call."""

    model_id = "scripted-judge"

    def __init__(self, scores: list[str]) -> None:
        self.scores = list(scores)

    async def generate(self, prompt: str, params: GenerationParams | None = None):
        text = self.scores.pop(0)
        return GenerationResult(
            text=text,
            tokens_in=10, tokens_out=4, latency_ms=1,
            model_id=self.model_id, params=params or GenerationParams(),
        )

    async def astream(self, prompt: str, params: GenerationParams | None = None):
        from core.serving.protocols import StreamEvent
        result = await self.generate(prompt, params)
        yield StreamEvent(type="chunk", text=result.text)
        yield StreamEvent(type="done", tokens_in=10, tokens_out=4, latency_ms=1)


async def test_judge_faithfulness_with_scripted_score() -> None:
    judge = ScriptedJudge(["0.85"])
    score, details = await judge_faithfulness(
        judge,
        question="q?",
        answer="a.",
        contexts=["context that supports the answer"],
    )
    assert score == 0.85
    assert details["evaluator_model"] == "scripted-judge"


async def test_judge_faithfulness_no_contexts_returns_none() -> None:
    judge = ScriptedJudge([])
    score, details = await judge_faithfulness(
        judge, question="q?", answer="a.", contexts=[]
    )
    assert score is None
    assert details["reason"] == "no contexts"


async def test_judge_answer_relevance() -> None:
    judge = ScriptedJudge(["0.7"])
    score, _ = await judge_answer_relevance(
        judge, question="how do I X?", answer="here is how you X..."
    )
    assert score == 0.7


# --- evaluate_session orchestrator ---


async def test_evaluate_session_combines_all_metrics() -> None:
    store = InMemorySessionStore()
    judge = ScriptedJudge(["0.9", "0.8"])
    session = await store.create("alice", "what is X?", "test-llm")

    await store.append_event(
        build_event(
            session_id=session.id,
            event_type=EventType.RETRIEVAL,
            payload={
                "method": "hybrid",
                "results": [
                    {"chunk_id": "c1", "doc_id": "d1", "preview": "context one ..."},
                    {"chunk_id": "c2", "doc_id": "d1", "preview": "context two ..."},
                ],
            },
        )
    )
    await store.append_event(
        build_event(
            session_id=session.id,
            event_type=EventType.FINAL,
            payload={"answer": "X is the thing per [1] and [2]."},
        )
    )
    await store.transition(session.id, "alice", SessionState.RUNNING)
    await store.transition(session.id, "alice", SessionState.COMPLETED)

    result = await evaluate_session(
        store=store, llm=judge, session_id=session.id, user_id="alice"
    )
    assert result.faithfulness == 0.9
    assert result.answer_relevance == 0.8
    assert result.citation_coverage == 1.0
    assert result.evaluator_model_id == "scripted-judge"
    metric_names = {m.name for m in result.metrics}
    assert {"faithfulness", "answer_relevance", "citation_coverage"} <= metric_names


async def test_evaluate_session_with_no_retrieval_skips_faithfulness() -> None:
    store = InMemorySessionStore()
    judge = ScriptedJudge(["0.6"])  # only relevance asked
    session = await store.create("alice", "g?", "t")
    await store.append_event(
        build_event(
            session_id=session.id,
            event_type=EventType.FINAL,
            payload={"answer": "no docs answer"},
        )
    )

    result = await evaluate_session(
        store=store, llm=judge, session_id=session.id, user_id="alice"
    )
    assert result.faithfulness is None
    assert result.answer_relevance == 0.6


# --- API ---


@pytest.fixture
def client() -> TestClient:
    judge = ScriptedJudge(["0.85", "0.75"] * 5)
    app = build_app(store=InMemorySessionStore(), llm=judge)
    return TestClient(app)


def test_eval_endpoint_full_cycle(client: TestClient) -> None:
    headers = {"X-User-Id": "alice"}
    sid = client.post(
        "/api/v1/agents/sessions",
        json={"goal": "what?", "model_spec": "t"},
        headers=headers,
    ).json()["id"]
    client.post(f"/api/v1/agents/sessions/{sid}/run", headers=headers)

    r = client.post(f"/api/v1/eval/{sid}", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["answer_relevance"] is not None


def test_eval_endpoint_404(client: TestClient) -> None:
    r = client.post("/api/v1/eval/no-such-session", headers={"X-User-Id": "alice"})
    assert r.status_code == 404


def test_eval_unauthorized(client: TestClient) -> None:
    r = client.post("/api/v1/eval/abc")
    assert r.status_code == 401


def test_web_evaluate_renders_eval_partial(client: TestClient) -> None:
    create = client.post(
        "/web/sessions", data={"goal": "g", "model_spec": "t"}
    )
    sid = create.text.split('id="session-')[1].split('"')[0]
    client.post(f"/web/sessions/{sid}/run")
    r = client.post(f"/web/sessions/{sid}/evaluate")
    assert r.status_code == 200
    assert "Eval scores" in r.text
    assert "answer_relevance" in r.text
    assert "citation_coverage" in r.text

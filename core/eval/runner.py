from core.eval.heuristics import citation_coverage, retrieval_recall_at_k
from core.eval.judges import judge_answer_relevance, judge_faithfulness
from core.logging import get_logger
from core.schemas.agent import EventType
from core.schemas.eval import EvalResult, MetricScore
from core.serving.protocols import LLMClient
from core.session.protocols import SessionStore
from core.session.store_memory import SessionNotFound

log = get_logger("differentia.eval")


def _extract_question_answer_contexts(events: list) -> tuple[str, str, list[str]]:
    """Pull (goal, final answer, list of retrieved-chunk previews) out
    of an existing session's event log. The agent runner already
    persisted everything we need.
    """
    final = next(
        (e for e in events if e.event_type == EventType.FINAL), None
    )
    answer = (final.payload.get("answer") if final else "") or ""

    contexts: list[str] = []
    retrieval_events = [
        e for e in events if e.event_type == EventType.RETRIEVAL
    ]
    for ev in retrieval_events:
        for hit in ev.payload.get("results", []) or []:
            preview = hit.get("preview")
            if isinstance(preview, str) and preview:
                contexts.append(preview)
        for hit in ev.payload.get("merged", []) or []:
            preview = hit.get("preview")
            if isinstance(preview, str) and preview:
                contexts.append(preview)
    return answer, contexts


async def evaluate_session(
    *,
    store: SessionStore,
    llm: LLMClient,
    session_id: str,
    user_id: str,
    ground_truth_chunk_ids: list[str] | None = None,
) -> EvalResult:
    """Score a completed session against the standard RAG metric set.

    LLM-as-judge for faithfulness + answer_relevance (uses the same
    LLMClient — same instance can act as judge; for production you'd
    inject a stronger / different model). Heuristic for
    citation_coverage. Optional retrieval_recall@k when ground truth
    chunk ids are supplied by the caller.
    """
    session = await store.get(session_id, user_id)
    if session is None:
        raise SessionNotFound(session_id)

    events = await store.events(session_id, user_id)
    answer, contexts = _extract_question_answer_contexts(events)
    question = session.goal

    log.info(
        "eval_started",
        extra={
            "session_id": session_id,
            "user_id": user_id,
            "n_contexts": len(contexts),
        },
    )

    metrics: list[MetricScore] = []

    faith_score, faith_details = await judge_faithfulness(
        llm, question=question, answer=answer, contexts=contexts
    )
    if faith_score is not None:
        metrics.append(
            MetricScore(
                name="faithfulness", score=faith_score, details=faith_details
            )
        )

    rel_score, rel_details = await judge_answer_relevance(
        llm, question=question, answer=answer
    )
    if rel_score is not None:
        metrics.append(
            MetricScore(
                name="answer_relevance", score=rel_score, details=rel_details
            )
        )

    cite_score, cite_details = citation_coverage(answer, len(contexts))
    metrics.append(
        MetricScore(
            name="citation_coverage", score=cite_score, details=cite_details
        )
    )

    recall_score: float | None = None
    if ground_truth_chunk_ids:
        retrieved_ids: list[str] = []
        for ev in events:
            if ev.event_type == EventType.RETRIEVAL:
                for hit in ev.payload.get("results", []) or []:
                    if isinstance(hit.get("chunk_id"), str):
                        retrieved_ids.append(hit["chunk_id"])
                for hit in ev.payload.get("merged", []) or []:
                    if isinstance(hit.get("chunk_id"), str):
                        retrieved_ids.append(hit["chunk_id"])
        recall_score, recall_details = retrieval_recall_at_k(
            retrieved_ids, ground_truth_chunk_ids
        )
        metrics.append(
            MetricScore(
                name="retrieval_recall_at_k",
                score=recall_score,
                details=recall_details,
            )
        )

    log.info(
        "eval_done",
        extra={
            "session_id": session_id,
            "faithfulness": faith_score,
            "answer_relevance": rel_score,
            "citation_coverage": cite_score,
            "retrieval_recall_at_k": recall_score,
        },
    )

    return EvalResult(
        session_id=session_id,
        faithfulness=faith_score,
        answer_relevance=rel_score,
        citation_coverage=cite_score,
        retrieval_recall_at_k=recall_score,
        metrics=metrics,
        evaluator_model_id=getattr(llm, "model_id", None),
    )

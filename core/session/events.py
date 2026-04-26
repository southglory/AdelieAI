import uuid
from datetime import datetime, timezone

from core.schemas.agent import AgentEvent, EventType, SessionState


def build_event(
    *,
    session_id: str,
    event_type: EventType,
    from_state: SessionState | None = None,
    to_state: SessionState | None = None,
    payload: dict | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: int = 0,
) -> AgentEvent:
    return AgentEvent(
        id=str(uuid.uuid4()),
        session_id=session_id,
        event_type=event_type,
        from_state=from_state,
        to_state=to_state,
        payload=payload or {},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        occurred_at=datetime.now(timezone.utc),
    )

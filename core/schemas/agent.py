from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class SessionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    STATE_TRANSITION = "state_transition"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    LLM_CALL = "llm_call"
    ERROR = "error"
    FINAL = "final"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentSession(BaseModel):
    model_config = ConfigDict(frozen=False, validate_assignment=True)

    id: str
    user_id: str
    goal: str
    state: SessionState = SessionState.PENDING
    model_spec: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None


class AgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    session_id: str
    event_type: EventType
    from_state: SessionState | None = None
    to_state: SessionState | None = None
    payload: dict = Field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    occurred_at: datetime = Field(default_factory=_utcnow)

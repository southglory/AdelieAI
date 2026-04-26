from typing import Protocol, runtime_checkable

from core.schemas.agent import AgentEvent, AgentSession, SessionState


@runtime_checkable
class SessionStore(Protocol):
    async def create(self, user_id: str, goal: str, model_spec: str) -> AgentSession: ...

    async def get(self, session_id: str, user_id: str) -> AgentSession | None: ...

    async def list_sessions(
        self, user_id: str, limit: int = 50
    ) -> list[AgentSession]: ...

    async def transition(
        self, session_id: str, user_id: str, to: SessionState
    ) -> AgentSession: ...

    async def append_event(self, event: AgentEvent) -> None: ...

    async def events(self, session_id: str, user_id: str) -> list[AgentEvent]: ...

    async def soft_delete(self, session_id: str, user_id: str) -> AgentSession: ...

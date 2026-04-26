import asyncio
import uuid
from datetime import datetime, timezone

from core.schemas.agent import AgentEvent, AgentSession, EventType, SessionState
from core.session.events import build_event
from core.session.state_machine import is_terminal, validate_transition


class SessionNotFound(Exception):
    pass


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._events: dict[str, list[AgentEvent]] = {}
        self._lock = asyncio.Lock()

    async def create(self, user_id: str, goal: str, model_spec: str) -> AgentSession:
        async with self._lock:
            session = AgentSession(
                id=str(uuid.uuid4()),
                user_id=user_id,
                goal=goal,
                model_spec=model_spec,
            )
            self._sessions[session.id] = session
            self._events[session.id] = []
            return session

    async def get(self, session_id: str, user_id: str) -> AgentSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.user_id != user_id:
                return None
            return session

    async def list_sessions(
        self, user_id: str, limit: int = 50
    ) -> list[AgentSession]:
        async with self._lock:
            owned = [s for s in self._sessions.values() if s.user_id == user_id]
            owned.sort(key=lambda s: s.created_at, reverse=True)
            return owned[:limit]

    async def transition(
        self, session_id: str, user_id: str, to: SessionState
    ) -> AgentSession:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.user_id != user_id:
                raise SessionNotFound(session_id)
            validate_transition(session.state, to)
            from_state = session.state
            session.state = to
            session.updated_at = datetime.now(timezone.utc)
            if is_terminal(to):
                session.completed_at = session.updated_at
            self._events[session_id].append(
                build_event(
                    session_id=session_id,
                    event_type=EventType.STATE_TRANSITION,
                    from_state=from_state,
                    to_state=to,
                )
            )
            return session

    async def append_event(self, event: AgentEvent) -> None:
        async with self._lock:
            if event.session_id not in self._events:
                raise SessionNotFound(event.session_id)
            self._events[event.session_id].append(event)

    async def events(self, session_id: str, user_id: str) -> list[AgentEvent]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.user_id != user_id:
                return []
            return list(self._events[session_id])

    async def soft_delete(self, session_id: str, user_id: str) -> AgentSession:
        return await self.transition(session_id, user_id, SessionState.CANCELLED)

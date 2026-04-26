import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from core.schemas.agent import AgentEvent, AgentSession, EventType, SessionState
from core.session.events import build_event
from core.session.state_machine import is_terminal, validate_transition
from core.session.store_memory import SessionNotFound


class Base(DeclarativeBase):
    pass


class _SessionRow(Base):
    __tablename__ = "agent_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    goal: Mapped[str] = mapped_column(String)
    state: Mapped[str] = mapped_column(String(16))
    model_spec: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def to_pydantic(self) -> AgentSession:
        return AgentSession(
            id=self.id,
            user_id=self.user_id,
            goal=self.goal,
            state=SessionState(self.state),
            model_spec=self.model_spec,
            created_at=_aware(self.created_at),
            updated_at=_aware(self.updated_at),
            completed_at=_aware(self.completed_at) if self.completed_at else None,
        )


class _EventRow(Base):
    __tablename__ = "agent_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("agent_sessions.id"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(32))
    from_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    to_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    def to_pydantic(self) -> AgentEvent:
        return AgentEvent(
            id=self.id,
            session_id=self.session_id,
            event_type=EventType(self.event_type),
            from_state=SessionState(self.from_state) if self.from_state else None,
            to_state=SessionState(self.to_state) if self.to_state else None,
            payload=self.payload or {},
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            latency_ms=self.latency_ms,
            occurred_at=_aware(self.occurred_at),
        )


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SqlSessionStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessionmaker = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

    @classmethod
    def from_url(cls, url: str, **engine_kwargs: Any) -> "SqlSessionStore":
        engine = create_async_engine(url, **engine_kwargs)
        return cls(engine)

    async def init_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def create(
        self, user_id: str, goal: str, model_spec: str
    ) -> AgentSession:
        now = datetime.now(timezone.utc)
        row = _SessionRow(
            id=str(uuid.uuid4()),
            user_id=user_id,
            goal=goal,
            state=SessionState.PENDING.value,
            model_spec=model_spec,
            created_at=now,
            updated_at=now,
        )
        async with self._sessionmaker() as s:
            s.add(row)
            await s.commit()
            return row.to_pydantic()

    async def get(self, session_id: str, user_id: str) -> AgentSession | None:
        async with self._sessionmaker() as s:
            row = await s.get(_SessionRow, session_id)
            if row is None or row.user_id != user_id:
                return None
            return row.to_pydantic()

    async def list_sessions(
        self, user_id: str, limit: int = 50
    ) -> list[AgentSession]:
        async with self._sessionmaker() as s:
            stmt = (
                select(_SessionRow)
                .where(_SessionRow.user_id == user_id)
                .order_by(_SessionRow.created_at.desc())
                .limit(limit)
            )
            rows = (await s.execute(stmt)).scalars().all()
            return [r.to_pydantic() for r in rows]

    async def transition(
        self, session_id: str, user_id: str, to: SessionState
    ) -> AgentSession:
        async with self._sessionmaker() as s:
            row = await s.get(_SessionRow, session_id)
            if row is None or row.user_id != user_id:
                raise SessionNotFound(session_id)
            from_state = SessionState(row.state)
            validate_transition(from_state, to)
            now = datetime.now(timezone.utc)
            row.state = to.value
            row.updated_at = now
            if is_terminal(to):
                row.completed_at = now

            ev = build_event(
                session_id=session_id,
                event_type=EventType.STATE_TRANSITION,
                from_state=from_state,
                to_state=to,
            )
            s.add(_event_to_row(ev))
            await s.commit()
            return row.to_pydantic()

    async def append_event(self, event: AgentEvent) -> None:
        async with self._sessionmaker() as s:
            session_row = await s.get(_SessionRow, event.session_id)
            if session_row is None:
                raise SessionNotFound(event.session_id)
            s.add(_event_to_row(event))
            await s.commit()

    async def events(
        self, session_id: str, user_id: str
    ) -> list[AgentEvent]:
        async with self._sessionmaker() as s:
            session_row = await s.get(_SessionRow, session_id)
            if session_row is None or session_row.user_id != user_id:
                return []
            stmt = (
                select(_EventRow)
                .where(_EventRow.session_id == session_id)
                .order_by(_EventRow.occurred_at.asc())
            )
            rows = (await s.execute(stmt)).scalars().all()
            return [r.to_pydantic() for r in rows]

    async def soft_delete(
        self, session_id: str, user_id: str
    ) -> AgentSession:
        return await self.transition(session_id, user_id, SessionState.CANCELLED)


def _event_to_row(ev: AgentEvent) -> _EventRow:
    payload = ev.payload
    try:
        json.dumps(payload)
    except TypeError:
        payload = {k: str(v) for k, v in payload.items()}
    return _EventRow(
        id=ev.id,
        session_id=ev.session_id,
        event_type=ev.event_type.value,
        from_state=ev.from_state.value if ev.from_state else None,
        to_state=ev.to_state.value if ev.to_state else None,
        payload=payload,
        tokens_in=ev.tokens_in,
        tokens_out=ev.tokens_out,
        latency_ms=ev.latency_ms,
        occurred_at=ev.occurred_at,
    )

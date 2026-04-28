"""Chat-turn persistence — one row per user/assistant message in a
persona conversation, plus per-turn telemetry (tokens + latency).

Two implementations:
- ``InMemoryChatStore``: tests, ephemeral demos.
- ``SqlChatStore``: SQLite by default, swap via DATABASE_URL.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Protocol, runtime_checkable

from sqlalchemy import DateTime, Integer, String, delete, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatTurn:
    id: int | None
    persona_id: str
    user_id: str
    role: Role
    content: str
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int | None
    created_at: datetime
    # 5-tier user rating on assistant turns. 1 = bad, 5 = excellent. None
    # = not rated yet. User-only on assistant rows; user rows stay None.
    # Used to harvest chosen/rejected pairs for DPO (Step 6.2).
    rating: int | None = None


@runtime_checkable
class ChatStore(Protocol):
    async def list_turns(
        self, persona_id: str, user_id: str
    ) -> list[ChatTurn]: ...

    async def append(self, turn: ChatTurn) -> ChatTurn: ...

    async def reset(self, persona_id: str, user_id: str) -> int: ...

    async def turn_count(self, persona_id: str, user_id: str) -> int: ...

    async def rate(
        self, turn_id: int, rating: int | None
    ) -> ChatTurn | None: ...


class InMemoryChatStore:
    """Process-local store. Resets when the app restarts."""

    def __init__(self) -> None:
        self._turns: list[ChatTurn] = []
        self._next_id = 1

    async def list_turns(
        self, persona_id: str, user_id: str
    ) -> list[ChatTurn]:
        return [
            t
            for t in self._turns
            if t.persona_id == persona_id and t.user_id == user_id
        ]

    async def append(self, turn: ChatTurn) -> ChatTurn:
        stored = ChatTurn(
            id=self._next_id,
            persona_id=turn.persona_id,
            user_id=turn.user_id,
            role=turn.role,
            content=turn.content,
            tokens_in=turn.tokens_in,
            tokens_out=turn.tokens_out,
            latency_ms=turn.latency_ms,
            created_at=turn.created_at,
            rating=turn.rating,
        )
        self._turns.append(stored)
        self._next_id += 1
        return stored

    async def reset(self, persona_id: str, user_id: str) -> int:
        before = len(self._turns)
        self._turns = [
            t
            for t in self._turns
            if not (t.persona_id == persona_id and t.user_id == user_id)
        ]
        return before - len(self._turns)

    async def turn_count(self, persona_id: str, user_id: str) -> int:
        return len(await self.list_turns(persona_id, user_id))

    async def rate(
        self, turn_id: int, rating: int | None
    ) -> ChatTurn | None:
        for i, t in enumerate(self._turns):
            if t.id == turn_id:
                updated = ChatTurn(
                    id=t.id,
                    persona_id=t.persona_id,
                    user_id=t.user_id,
                    role=t.role,
                    content=t.content,
                    tokens_in=t.tokens_in,
                    tokens_out=t.tokens_out,
                    latency_ms=t.latency_ms,
                    created_at=t.created_at,
                    rating=rating,
                )
                self._turns[i] = updated
                return updated
        return None


class _Base(DeclarativeBase):
    pass


class _ChatTurnRow(_Base):
    __tablename__ = "chat_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    persona_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(String)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def to_dataclass(self) -> ChatTurn:
        return ChatTurn(
            id=self.id,
            persona_id=self.persona_id,
            user_id=self.user_id,
            role=self.role,  # type: ignore[arg-type]
            content=self.content,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            latency_ms=self.latency_ms,
            created_at=_aware(self.created_at),
            rating=self.rating,
        )


def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class SqlChatStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessionmaker = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

    @classmethod
    def from_url(cls, url: str) -> "SqlChatStore":
        return cls(create_async_engine(url, future=True))

    async def init_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
            # Defensive migration — Step 6.2 added `rating` column. Existing
            # databases created before that don't have the column, so add
            # it idempotently. Works on SQLite (default) and Postgres.
            await conn.run_sync(self._ensure_rating_column)

    @staticmethod
    def _ensure_rating_column(sync_conn) -> None:  # type: ignore[no-untyped-def]
        from sqlalchemy import inspect, text

        inspector = inspect(sync_conn)
        cols = {c["name"] for c in inspector.get_columns("chat_turns")}
        if "rating" not in cols:
            sync_conn.execute(
                text("ALTER TABLE chat_turns ADD COLUMN rating INTEGER")
            )

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def list_turns(
        self, persona_id: str, user_id: str
    ) -> list[ChatTurn]:
        async with self._sessionmaker() as session:
            stmt = (
                select(_ChatTurnRow)
                .where(
                    _ChatTurnRow.persona_id == persona_id,
                    _ChatTurnRow.user_id == user_id,
                )
                .order_by(_ChatTurnRow.id.asc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [r.to_dataclass() for r in rows]

    async def append(self, turn: ChatTurn) -> ChatTurn:
        async with self._sessionmaker() as session:
            row = _ChatTurnRow(
                persona_id=turn.persona_id,
                user_id=turn.user_id,
                role=turn.role,
                content=turn.content,
                tokens_in=turn.tokens_in,
                tokens_out=turn.tokens_out,
                latency_ms=turn.latency_ms,
                created_at=_aware(turn.created_at),
                rating=turn.rating,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row.to_dataclass()

    async def reset(self, persona_id: str, user_id: str) -> int:
        async with self._sessionmaker() as session:
            stmt = delete(_ChatTurnRow).where(
                _ChatTurnRow.persona_id == persona_id,
                _ChatTurnRow.user_id == user_id,
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount or 0

    async def turn_count(self, persona_id: str, user_id: str) -> int:
        async with self._sessionmaker() as session:
            stmt = select(_ChatTurnRow.id).where(
                _ChatTurnRow.persona_id == persona_id,
                _ChatTurnRow.user_id == user_id,
            )
            rows = (await session.execute(stmt)).scalars().all()
            return len(rows)

    async def rate(
        self, turn_id: int, rating: int | None
    ) -> ChatTurn | None:
        async with self._sessionmaker() as session:
            row = await session.get(_ChatTurnRow, turn_id)
            if row is None:
                return None
            row.rating = rating
            await session.commit()
            await session.refresh(row)
            return row.to_dataclass()

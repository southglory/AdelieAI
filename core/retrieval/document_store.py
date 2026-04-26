import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from core.schemas.retrieval import Chunk, Document


class Base(DeclarativeBase):
    pass


class _DocRow(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    source: Mapped[str] = mapped_column(String(512))
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class _ChunkRow(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class SqlDocumentStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessionmaker = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

    @classmethod
    def from_url(cls, url: str, **engine_kwargs: Any) -> "SqlDocumentStore":
        return cls(create_async_engine(url, **engine_kwargs))

    async def init_schema(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self._engine.dispose()

    async def add(self, doc: Document, chunks: list[Chunk]) -> None:
        async with self._sessionmaker() as s:
            s.add(
                _DocRow(
                    id=doc.id,
                    title=doc.title,
                    source=doc.source,
                    content=doc.content,
                    metadata_json=doc.metadata,
                    created_at=doc.created_at,
                )
            )
            for c in chunks:
                s.add(
                    _ChunkRow(
                        id=c.id,
                        doc_id=c.doc_id,
                        position=c.position,
                        text=c.text,
                        metadata_json=c.metadata,
                    )
                )
            await s.commit()

    async def get(self, doc_id: str) -> Document | None:
        async with self._sessionmaker() as s:
            row = await s.get(_DocRow, doc_id)
            if row is None:
                return None
            return Document(
                id=row.id,
                title=row.title,
                source=row.source,
                content=row.content,
                metadata=row.metadata_json or {},
                created_at=_aware(row.created_at),
            )

    async def list_docs(self, limit: int = 100) -> list[Document]:
        async with self._sessionmaker() as s:
            stmt = select(_DocRow).order_by(_DocRow.created_at.desc()).limit(limit)
            rows = (await s.execute(stmt)).scalars().all()
            return [
                Document(
                    id=r.id,
                    title=r.title,
                    source=r.source,
                    content=r.content,
                    metadata=r.metadata_json or {},
                    created_at=_aware(r.created_at),
                )
                for r in rows
            ]

    async def list_chunks(self, doc_id: str) -> list[Chunk]:
        async with self._sessionmaker() as s:
            stmt = (
                select(_ChunkRow)
                .where(_ChunkRow.doc_id == doc_id)
                .order_by(_ChunkRow.position.asc())
            )
            rows = (await s.execute(stmt)).scalars().all()
            return [_row_to_chunk(r) for r in rows]

    async def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        async with self._sessionmaker() as s:
            stmt = select(_ChunkRow).where(_ChunkRow.id.in_(chunk_ids))
            rows = (await s.execute(stmt)).scalars().all()
            return [_row_to_chunk(r) for r in rows]

    async def all_chunks(self) -> list[Chunk]:
        async with self._sessionmaker() as s:
            stmt = select(_ChunkRow).order_by(_ChunkRow.doc_id, _ChunkRow.position)
            rows = (await s.execute(stmt)).scalars().all()
            return [_row_to_chunk(r) for r in rows]

    async def delete(self, doc_id: str) -> None:
        async with self._sessionmaker() as s:
            doc = await s.get(_DocRow, doc_id)
            if doc is None:
                return
            stmt_chunks = select(_ChunkRow).where(_ChunkRow.doc_id == doc_id)
            for c in (await s.execute(stmt_chunks)).scalars().all():
                await s.delete(c)
            await s.delete(doc)
            await s.commit()


def _row_to_chunk(r: _ChunkRow) -> Chunk:
    return Chunk(
        id=r.id,
        doc_id=r.doc_id,
        position=r.position,
        text=r.text,
        metadata=r.metadata_json or {},
    )


def build_document(
    *,
    title: str,
    source: str,
    content: str,
    metadata: dict | None = None,
) -> Document:
    return Document(
        id=str(uuid.uuid4()),
        title=title,
        source=source,
        content=content,
        metadata=metadata or {},
        created_at=datetime.now(timezone.utc),
    )

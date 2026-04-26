import pytest

from core.schemas.agent import EventType, SessionState
from core.session.events import build_event
from core.session.state_machine import InvalidTransition
from core.session.store_memory import SessionNotFound
from core.session.store_sql import SqlSessionStore


@pytest.fixture
async def store() -> SqlSessionStore:
    s = SqlSessionStore.from_url("sqlite+aiosqlite:///:memory:")
    await s.init_schema()
    yield s
    await s.dispose()


async def test_create_get_roundtrip(store: SqlSessionStore) -> None:
    session = await store.create("alice", "summarize", "qwen-7b")
    fetched = await store.get(session.id, "alice")
    assert fetched is not None
    assert fetched.id == session.id
    assert fetched.goal == "summarize"
    assert fetched.state == SessionState.PENDING


async def test_state_machine_persisted(store: SqlSessionStore) -> None:
    session = await store.create("alice", "g", "m")
    running = await store.transition(session.id, "alice", SessionState.RUNNING)
    assert running.state == SessionState.RUNNING

    completed = await store.transition(session.id, "alice", SessionState.COMPLETED)
    assert completed.state == SessionState.COMPLETED
    assert completed.completed_at is not None

    events = await store.events(session.id, "alice")
    assert len(events) == 2
    assert events[0].event_type == EventType.STATE_TRANSITION
    assert events[0].from_state == SessionState.PENDING
    assert events[0].to_state == SessionState.RUNNING


async def test_idor_on_get_and_transition(store: SqlSessionStore) -> None:
    session = await store.create("alice", "g", "m")
    assert await store.get(session.id, "bob") is None
    with pytest.raises(SessionNotFound):
        await store.transition(session.id, "bob", SessionState.RUNNING)


async def test_invalid_transition_raises(store: SqlSessionStore) -> None:
    session = await store.create("alice", "g", "m")
    with pytest.raises(InvalidTransition):
        await store.transition(session.id, "alice", SessionState.COMPLETED)


async def test_list_sessions_user_scoped_and_ordered(
    store: SqlSessionStore,
) -> None:
    a1 = await store.create("alice", "first", "m")
    a2 = await store.create("alice", "second", "m")
    await store.create("bob", "bob's", "m")

    alice_sessions = await store.list_sessions("alice")
    assert len(alice_sessions) == 2
    assert alice_sessions[0].id == a2.id
    assert alice_sessions[1].id == a1.id


async def test_append_and_read_custom_event(store: SqlSessionStore) -> None:
    session = await store.create("alice", "g", "m")
    ev = build_event(
        session_id=session.id,
        event_type=EventType.LLM_CALL,
        payload={"model_id": "stub", "preview": "hello"},
        tokens_in=10,
        tokens_out=20,
        latency_ms=42,
    )
    await store.append_event(ev)

    events = await store.events(session.id, "alice")
    assert len(events) == 1
    assert events[0].event_type == EventType.LLM_CALL
    assert events[0].payload["model_id"] == "stub"
    assert events[0].tokens_in == 10


async def test_persistence_across_engine_restart(tmp_path) -> None:
    db_path = tmp_path / "sessions.db"
    url = f"sqlite+aiosqlite:///{db_path.as_posix()}"

    store1 = SqlSessionStore.from_url(url)
    await store1.init_schema()
    session = await store1.create("alice", "persistent", "m")
    await store1.transition(session.id, "alice", SessionState.RUNNING)
    await store1.dispose()

    store2 = SqlSessionStore.from_url(url)
    await store2.init_schema()
    fetched = await store2.get(session.id, "alice")
    assert fetched is not None
    assert fetched.state == SessionState.RUNNING

    events = await store2.events(session.id, "alice")
    assert len(events) == 1
    await store2.dispose()

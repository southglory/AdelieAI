import pytest

from core.schemas.agent import EventType, SessionState
from core.session.events import build_event
from core.session.state_machine import InvalidTransition, validate_transition
from core.session.store_memory import InMemorySessionStore, SessionNotFound


@pytest.fixture
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


async def test_session_lifecycle_happy_path(store: InMemorySessionStore) -> None:
    session = await store.create("alice", "summarize Q3", "claude-opus-4-7")
    assert session.state == SessionState.PENDING

    running = await store.transition(session.id, "alice", SessionState.RUNNING)
    assert running.state == SessionState.RUNNING

    completed = await store.transition(session.id, "alice", SessionState.COMPLETED)
    assert completed.state == SessionState.COMPLETED
    assert completed.completed_at is not None

    events = await store.events(session.id, "alice")
    assert len(events) == 2
    assert events[0].from_state == SessionState.PENDING
    assert events[0].to_state == SessionState.RUNNING
    assert events[1].to_state == SessionState.COMPLETED


async def test_invalid_transition_rejected(store: InMemorySessionStore) -> None:
    session = await store.create("alice", "g", "m")
    with pytest.raises(InvalidTransition):
        await store.transition(session.id, "alice", SessionState.COMPLETED)


def test_terminal_states_block_further_transitions() -> None:
    with pytest.raises(InvalidTransition):
        validate_transition(SessionState.COMPLETED, SessionState.RUNNING)
    with pytest.raises(InvalidTransition):
        validate_transition(SessionState.CANCELLED, SessionState.RUNNING)


async def test_idor_isolation(store: InMemorySessionStore) -> None:
    session = await store.create("alice", "g", "m")
    assert await store.get(session.id, "bob") is None
    with pytest.raises(SessionNotFound):
        await store.transition(session.id, "bob", SessionState.RUNNING)


async def test_soft_delete_terminal(store: InMemorySessionStore) -> None:
    session = await store.create("alice", "g", "m")
    cancelled = await store.soft_delete(session.id, "alice")
    assert cancelled.state == SessionState.CANCELLED
    assert cancelled.completed_at is not None


async def test_append_custom_event(store: InMemorySessionStore) -> None:
    session = await store.create("alice", "g", "m")
    event = build_event(
        session_id=session.id,
        event_type=EventType.LLM_CALL,
        payload={"prompt": "hi"},
        tokens_in=10,
        tokens_out=20,
        latency_ms=42,
    )
    await store.append_event(event)
    events = await store.events(session.id, "alice")
    assert len(events) == 1
    assert events[0].event_type == EventType.LLM_CALL
    assert events[0].tokens_in == 10

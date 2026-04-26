from core.schemas.agent import SessionState

ALLOWED_TRANSITIONS: dict[SessionState, frozenset[SessionState]] = {
    SessionState.PENDING: frozenset({SessionState.RUNNING, SessionState.CANCELLED}),
    SessionState.RUNNING: frozenset(
        {SessionState.COMPLETED, SessionState.FAILED, SessionState.CANCELLED}
    ),
    SessionState.COMPLETED: frozenset(),
    SessionState.FAILED: frozenset(),
    SessionState.CANCELLED: frozenset(),
}

TERMINAL_STATES: frozenset[SessionState] = frozenset(
    {SessionState.COMPLETED, SessionState.FAILED, SessionState.CANCELLED}
)


class InvalidTransition(Exception):
    def __init__(self, from_state: SessionState, to_state: SessionState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Cannot transition from {from_state.value} to {to_state.value}"
        )


def validate_transition(from_state: SessionState, to_state: SessionState) -> None:
    if to_state not in ALLOWED_TRANSITIONS[from_state]:
        raise InvalidTransition(from_state, to_state)


def is_terminal(state: SessionState) -> bool:
    return state in TERMINAL_STATES

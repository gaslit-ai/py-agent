"""Library-level errors."""
from __future__ import annotations

from .types import ExecutionState, ExecutionStatus, StepStatus


class PlanValidationError(ValueError):
    """Raised when a Plan fails structural validation (cycles, unknown deps, etc.)."""


class ExecutionError(RuntimeError):
    """Raised by `raise_if_failed` when an execution ended in a non-completed state."""

    def __init__(self, state: ExecutionState) -> None:
        self.state = state
        failed = [
            (sid, res.error.message if res.error else "no error info")
            for sid, res in state.steps.items()
            if res.status == StepStatus.FAILED
        ]
        msg = f"execution {state.execution_id} ended as {state.status}"
        if failed:
            details = "; ".join(f"{sid}: {err}" for sid, err in failed)
            msg = f"{msg} — failed steps: {details}"
        super().__init__(msg)


def raise_if_failed(state: ExecutionState) -> ExecutionState:
    """Raise ExecutionError if `state` is not COMPLETED. Returns state unchanged otherwise."""
    if state.status != ExecutionStatus.COMPLETED:
        raise ExecutionError(state)
    return state

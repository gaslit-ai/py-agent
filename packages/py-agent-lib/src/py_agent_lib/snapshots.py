"""Snapshot helpers — Pydantic v2 already gives us a clean JSON round-trip.

These exist so callers don't have to know about the underlying model.
"""
from __future__ import annotations

from .types import ExecutionState


def serialize_execution_state(state: ExecutionState) -> str:
    """Return a JSON string snapshot of the given execution state."""
    return state.model_dump_json()


def deserialize_execution_state(snapshot: str | bytes) -> ExecutionState:
    """Parse a JSON snapshot back into an ExecutionState."""
    return ExecutionState.model_validate_json(snapshot)

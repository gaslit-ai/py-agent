"""Core domain types: Step, StepResult, ExecutionState, status enums."""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return current UTC time, aware."""
    return datetime.now(UTC)


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BlockedReason(StrEnum):
    DEPENDENCY_FAILED = "dependency_failed"
    DEPENDENCY_CANCELLED = "dependency_cancelled"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    FAIL_FAST = "fail_fast"


class ErrorInfo(BaseModel):
    """Structured error info attached to failed step results."""

    model_config = ConfigDict(frozen=True)

    message: str
    type: str | None = None
    traceback: str | None = None


class Step(BaseModel):
    """An immutable unit of work in a Plan."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    deps: tuple[str, ...] = ()
    payload: Any = None


class StepResult(BaseModel):
    """Mutable result for a step. Handlers return these; the executor mutates state."""

    step_id: str
    status: StepStatus = StepStatus.PENDING
    output: Any = None
    error: ErrorInfo | None = None
    blocked_reason: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ExecutionState(BaseModel):
    """Snapshot of a run. Round-trips losslessly via Pydantic."""

    execution_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    steps: dict[str, StepResult] = Field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None

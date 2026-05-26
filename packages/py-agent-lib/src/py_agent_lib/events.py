"""Execution events. Pydantic discriminated union — parsing routes by `type` field."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .types import ErrorInfo, ExecutionStatus


class _EventBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_id: str
    ts: datetime


class ExecutionStarted(_EventBase):
    type: Literal["execution_started"] = "execution_started"
    started_at: datetime


class ExecutionCompleted(_EventBase):
    type: Literal["execution_completed"] = "execution_completed"
    status: ExecutionStatus
    finished_at: datetime


class ExecutionCancelled(_EventBase):
    type: Literal["execution_cancelled"] = "execution_cancelled"
    finished_at: datetime


class StepScheduled(_EventBase):
    type: Literal["step_scheduled"] = "step_scheduled"
    step_id: str


class StepStarted(_EventBase):
    type: Literal["step_started"] = "step_started"
    step_id: str


class StepCompleted(_EventBase):
    type: Literal["step_completed"] = "step_completed"
    step_id: str


class StepFailed(_EventBase):
    type: Literal["step_failed"] = "step_failed"
    step_id: str
    error: ErrorInfo


class StepBlocked(_EventBase):
    type: Literal["step_blocked"] = "step_blocked"
    step_id: str
    blocked_reason: str | None = None


class StepCancelled(_EventBase):
    type: Literal["step_cancelled"] = "step_cancelled"
    step_id: str


ExecutionEvent: TypeAlias = Annotated[
    ExecutionStarted
    | ExecutionCompleted
    | ExecutionCancelled
    | StepScheduled
    | StepStarted
    | StepCompleted
    | StepFailed
    | StepBlocked
    | StepCancelled,
    Field(discriminator="type"),
]


ExecutionEventAdapter: TypeAdapter[ExecutionEvent] = TypeAdapter(ExecutionEvent)
"""Use for parsing untrusted event payloads — `ExecutionEventAdapter.validate_python(data)`."""

"""Per-step input/output telemetry, separate from the executor's lifecycle events.

The executor already emits `step_started` / `step_completed` / `step_failed`
events with timing metadata. This module captures the **data** that went through
each step — the LLM prompts, the parsed outputs, the merge inputs — into a
recorder you can later drain to a file or merge into events at serialization time.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..context import StepContext
from ..executor import StepHandler
from ..types import ErrorInfo, Step, StepResult


class StepTelemetry(BaseModel):
    """Captured input/output for one step. All fields except `step_id` and `action` are optional.

    The recorder merges multiple updates for the same `step_id`: non-None fields
    from later records override earlier ones, but None values do not clobber
    previously-set fields.
    """

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    step_id: str
    action: str
    label: str | None = None
    input: Any = None
    output: Any = None
    error: ErrorInfo | None = None


InputCapture = Callable[[Step, StepContext], Any]


class TelemetryRecorder:
    """Async-safe (single-event-loop) dict of `step_id -> StepTelemetry`.

    Records merge by step_id. Use `.get()` for a single step or `.all()` for a
    snapshot of everything captured so far.
    """

    def __init__(self) -> None:
        self._records: dict[str, StepTelemetry] = {}

    def record(self, telemetry: StepTelemetry) -> StepTelemetry:
        existing = self._records.get(telemetry.step_id)
        if existing is None:
            self._records[telemetry.step_id] = telemetry
            return telemetry
        # Merge: take non-None fields from `telemetry`, keep existing for None ones.
        # Identity fields (step_id, action) are preserved from the original record.
        # NOTE: use getattr (not model_dump) so nested Pydantic models stay as
        # instances rather than being flattened to dicts.
        update: dict[str, Any] = {}
        for field in StepTelemetry.model_fields:
            if field in ("step_id", "action"):
                continue
            value = getattr(telemetry, field)
            if value is not None:
                update[field] = value
        merged = existing.model_copy(update=update)
        self._records[telemetry.step_id] = merged
        return merged

    def get(self, step_id: str) -> StepTelemetry | None:
        return self._records.get(step_id)

    def all(self) -> dict[str, StepTelemetry]:
        """Snapshot copy of all records keyed by step_id."""
        return dict(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, step_id: object) -> bool:
        return isinstance(step_id, str) and step_id in self._records


def with_telemetry(
    handler: StepHandler,
    *,
    recorder: TelemetryRecorder,
    label: str | None = None,
    capture_input: InputCapture | None = None,
) -> StepHandler:
    """Wrap a step handler to record its input/output/error to a TelemetryRecorder.

    Args:
        handler: the underlying step handler.
        recorder: where to record telemetry.
        label: optional NER-style label tag attached to every record from this handler.
        capture_input: callable returning what to record as `input`. Defaults to
            `step.payload`.

    Notes:
        - The input record is emitted before the handler runs.
        - The output record is emitted after a successful call.
        - If the handler raises, an error record is emitted and the exception is
          re-raised so the executor still sees it.
    """

    async def wrapped(step: Step, ctx: StepContext) -> StepResult:
        recorder.record(
            StepTelemetry(
                step_id=step.id,
                action=step.action,
                label=label,
                input=(capture_input(step, ctx) if capture_input else step.payload),
            )
        )
        try:
            result = await handler(step, ctx)
        except Exception as e:
            recorder.record(
                StepTelemetry(
                    step_id=step.id,
                    action=step.action,
                    label=label,
                    error=ErrorInfo(message=str(e), type=type(e).__name__),
                )
            )
            raise
        recorder.record(
            StepTelemetry(
                step_id=step.id,
                action=step.action,
                label=label,
                output=result.output,
                error=result.error,
            )
        )
        return result

    return wrapped


__all__ = [
    "InputCapture",
    "StepTelemetry",
    "TelemetryRecorder",
    "with_telemetry",
]

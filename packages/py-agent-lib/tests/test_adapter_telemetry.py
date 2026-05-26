"""Tests for adapters.telemetry — per-step input/output capture."""
from __future__ import annotations

from py_agent_lib import PlanBuilder, StepContext, StepResult, StepStatus
from py_agent_lib.adapters.telemetry import (
    StepTelemetry,
    TelemetryRecorder,
    with_telemetry,
)
from py_agent_lib.executor import DagExecutor
from py_agent_lib.types import Step


def test_recorder_stores_initial_record() -> None:
    rec = TelemetryRecorder()
    rec.record(StepTelemetry(step_id="a", action="extract", input="hello"))
    assert rec.get("a") is not None
    assert rec.get("a").input == "hello"


def test_recorder_merges_non_none_fields() -> None:
    rec = TelemetryRecorder()
    rec.record(StepTelemetry(step_id="a", action="extract", input="hello"))
    rec.record(StepTelemetry(step_id="a", action="extract", output={"k": 1}))
    merged = rec.get("a")
    assert merged is not None
    assert merged.input == "hello"  # preserved
    assert merged.output == {"k": 1}  # newly set


def test_recorder_overrides_with_newer_values() -> None:
    rec = TelemetryRecorder()
    rec.record(StepTelemetry(step_id="a", action="extract", input="v1"))
    rec.record(StepTelemetry(step_id="a", action="extract", input="v2"))
    assert rec.get("a").input == "v2"


def test_recorder_does_not_override_with_none() -> None:
    """A later record with input=None must not wipe a previously-set input."""
    rec = TelemetryRecorder()
    rec.record(StepTelemetry(step_id="a", action="extract", input="set"))
    rec.record(StepTelemetry(step_id="a", action="extract", output="later"))
    rec_final = rec.get("a")
    assert rec_final.input == "set"
    assert rec_final.output == "later"


def test_recorder_all_returns_snapshot_copy() -> None:
    rec = TelemetryRecorder()
    rec.record(StepTelemetry(step_id="a", action="extract"))
    snapshot = rec.all()
    snapshot["a"] = "tampered"  # type: ignore[assignment]
    assert isinstance(rec.get("a"), StepTelemetry)


def test_recorder_contains_and_len() -> None:
    rec = TelemetryRecorder()
    assert len(rec) == 0
    assert "a" not in rec
    rec.record(StepTelemetry(step_id="a", action="x"))
    assert len(rec) == 1
    assert "a" in rec


async def test_with_telemetry_records_input_and_output() -> None:
    rec = TelemetryRecorder()

    async def handler(step: Step, ctx: StepContext) -> StepResult:
        return StepResult(step_id=step.id, status=StepStatus.COMPLETED, output={"n": 42})

    wrapped = with_telemetry(handler, recorder=rec, label="person_name")

    builder = PlanBuilder()
    builder.add_step(id="x", action="extract", payload="some input text")
    plan = builder.build()

    state = await DagExecutor().execute(plan, {"extract": wrapped})
    assert state.steps["x"].status == StepStatus.COMPLETED

    tel = rec.get("x")
    assert tel is not None
    assert tel.label == "person_name"
    assert tel.input == "some input text"  # default capture: step.payload
    assert tel.output == {"n": 42}
    assert tel.error is None


async def test_with_telemetry_uses_custom_capture_input() -> None:
    rec = TelemetryRecorder()

    async def handler(step: Step, ctx: StepContext) -> StepResult:
        return StepResult(step_id=step.id, status=StepStatus.COMPLETED)

    def capture(step: Step, ctx: StepContext):
        return {"step_id": step.id, "context_kind": "test"}

    wrapped = with_telemetry(handler, recorder=rec, capture_input=capture)

    builder = PlanBuilder()
    builder.add_step(id="x", action="extract", payload="ignored")
    plan = builder.build()

    await DagExecutor().execute(plan, {"extract": wrapped})
    tel = rec.get("x")
    assert tel.input == {"step_id": "x", "context_kind": "test"}


async def test_with_telemetry_records_error_when_handler_raises() -> None:
    rec = TelemetryRecorder()

    async def handler(step: Step, ctx: StepContext) -> StepResult:
        raise RuntimeError("kaboom")

    wrapped = with_telemetry(handler, recorder=rec)

    builder = PlanBuilder()
    builder.add_step(id="x", action="extract", payload="in")
    plan = builder.build()

    state = await DagExecutor().execute(plan, {"extract": wrapped})
    assert state.steps["x"].status == StepStatus.FAILED

    tel = rec.get("x")
    assert tel.input == "in"
    assert tel.output is None
    assert tel.error is not None
    assert tel.error.message == "kaboom"
    assert tel.error.type == "RuntimeError"


def test_step_telemetry_accepts_arbitrary_types() -> None:
    """input/output can hold non-Pydantic values like raw strings or dicts."""

    class Anything:
        def __init__(self, x):
            self.x = x

    tel = StepTelemetry(step_id="x", action="a", input=Anything(1), output={"k": [1, 2, 3]})
    assert isinstance(tel.input, Anything)
    assert tel.output["k"] == [1, 2, 3]

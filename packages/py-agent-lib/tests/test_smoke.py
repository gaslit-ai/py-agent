"""End-to-end smoke test exercising plan + executor + context + events."""
from __future__ import annotations

import asyncio

import pytest

from py_agent_lib import (
    DagExecutor,
    ExecuteOptions,
    ExecutionStatus,
    PlanBuilder,
    PlanValidationError,
    StepContext,
    StepResult,
    StepStatus,
    deserialize_execution_state,
    raise_if_failed,
    serialize_execution_state,
    with_retry,
    with_timeout,
)
from py_agent_lib.types import Step


async def test_fanout_merge_dag() -> None:
    """Two parallel extracts + a merge depending on both."""
    builder = PlanBuilder()
    builder.add_step(id="extract:a", action="extract", payload="a")
    builder.add_step(id="extract:b", action="extract", payload="b")
    builder.add_step(
        id="merge", action="merge", deps=["extract:a", "extract:b"]
    )
    plan = builder.build()

    async def extract(step: Step, ctx: StepContext) -> StepResult:
        return StepResult(
            step_id=step.id, status=StepStatus.COMPLETED, output=step.payload
        )

    async def merge(step: Step, ctx: StepContext) -> StepResult:
        outputs = ctx.get_dependency_outputs()
        return StepResult(
            step_id=step.id,
            status=StepStatus.COMPLETED,
            output={"merged": sorted(outputs.values())},
        )

    state = await DagExecutor().execute(
        plan, {"extract": extract, "merge": merge}
    )
    assert state.status == ExecutionStatus.COMPLETED
    assert state.steps["merge"].output == {"merged": ["a", "b"]}


async def test_event_stream_emits_lifecycle() -> None:
    builder = PlanBuilder()
    builder.add_step(id="x", action="work")
    plan = builder.build()

    async def work(step: Step, ctx: StepContext) -> StepResult:
        return StepResult(step_id=step.id, status=StepStatus.COMPLETED)

    events = []
    async for event in DagExecutor().stream_execute(plan, {"work": work}):
        events.append(event.type)

    assert "execution_started" in events
    assert "step_scheduled" in events
    assert "step_started" in events
    assert "step_completed" in events
    assert events[-1] == "execution_completed"


async def test_handler_exception_becomes_failed_step() -> None:
    builder = PlanBuilder()
    builder.add_step(id="x", action="bad")
    plan = builder.build()

    async def bad(step: Step, ctx: StepContext) -> StepResult:
        raise ValueError("nope")

    state = await DagExecutor().execute(plan, {"bad": bad})
    assert state.status == ExecutionStatus.FAILED
    assert state.steps["x"].status == StepStatus.FAILED
    assert state.steps["x"].error is not None
    assert "nope" in state.steps["x"].error.message
    assert state.steps["x"].error.type == "ValueError"


async def test_fail_fast_blocks_dependents() -> None:
    builder = PlanBuilder()
    builder.add_step(id="a", action="bad")
    builder.add_step(id="b", action="ok", deps=["a"])
    plan = builder.build()

    async def bad(step: Step, ctx: StepContext) -> StepResult:
        raise RuntimeError("boom")

    async def ok(step: Step, ctx: StepContext) -> StepResult:
        return StepResult(step_id=step.id, status=StepStatus.COMPLETED)

    state = await DagExecutor(fail_fast=True).execute(
        plan, {"bad": bad, "ok": ok}
    )
    assert state.steps["a"].status == StepStatus.FAILED
    assert state.steps["b"].status == StepStatus.BLOCKED
    assert state.steps["b"].blocked_reason == "dependency_failed"


async def test_cancellation_marks_pending_cancelled() -> None:
    builder = PlanBuilder()
    builder.add_step(id="slow", action="slow")
    builder.add_step(id="never", action="never", deps=["slow"])
    plan = builder.build()

    cancel = asyncio.Event()

    async def slow(step: Step, ctx: StepContext) -> StepResult:
        # Wait until cancelled, then return cancelled.
        for _ in range(50):
            if ctx.is_cancelled():
                return StepResult(step_id=step.id, status=StepStatus.CANCELLED)
            await asyncio.sleep(0.01)
        return StepResult(step_id=step.id, status=StepStatus.COMPLETED)

    async def never(step: Step, ctx: StepContext) -> StepResult:
        return StepResult(step_id=step.id, status=StepStatus.COMPLETED)

    async def trigger() -> None:
        await asyncio.sleep(0.05)
        cancel.set()

    trigger_task = asyncio.create_task(trigger())
    executor = DagExecutor(cancel_running_on_fail_fast=True)
    state = await executor.execute(
        plan, {"slow": slow, "never": never}, ExecuteOptions(cancel_event=cancel)
    )
    assert state.status == ExecutionStatus.CANCELLED
    assert state.steps["never"].status == StepStatus.CANCELLED
    await trigger_task


async def test_snapshot_resume_skips_completed() -> None:
    builder = PlanBuilder()
    builder.add_step(id="a", action="ok")
    builder.add_step(id="b", action="ok", deps=["a"])
    plan = builder.build()

    call_count = {"n": 0}

    async def ok(step: Step, ctx: StepContext) -> StepResult:
        call_count["n"] += 1
        return StepResult(step_id=step.id, status=StepStatus.COMPLETED, output=step.id)

    state1 = await DagExecutor().execute(plan, {"ok": ok})
    snapshot = serialize_execution_state(state1)
    restored = deserialize_execution_state(snapshot)
    call_count["n"] = 0  # reset

    state2 = await DagExecutor().execute(
        plan, {"ok": ok}, ExecuteOptions(resume_from=restored)
    )
    assert state2.status == ExecutionStatus.COMPLETED
    # Both steps were completed in snapshot, so nothing should re-run.
    assert call_count["n"] == 0


def test_plan_validation_catches_cycles() -> None:
    builder = PlanBuilder()
    builder.add_step(id="a", action="x", deps=["b"])
    builder.add_step(id="b", action="x", deps=["a"])
    with pytest.raises(PlanValidationError, match="cycle"):
        builder.build()


def test_plan_validation_catches_unknown_deps() -> None:
    builder = PlanBuilder()
    builder.add_step(id="a", action="x", deps=["ghost"])
    with pytest.raises(PlanValidationError, match="unknown step"):
        builder.build()


def test_plan_validation_catches_self_dep() -> None:
    builder = PlanBuilder()
    builder.add_step(id="a", action="x", deps=["a"])
    with pytest.raises(PlanValidationError, match="cannot depend on itself"):
        builder.build()


async def test_with_timeout_fails_long_handler() -> None:
    builder = PlanBuilder()
    builder.add_step(id="x", action="slow")
    plan = builder.build()

    async def slow(step: Step, ctx: StepContext) -> StepResult:
        await asyncio.sleep(1.0)
        return StepResult(step_id=step.id, status=StepStatus.COMPLETED)

    state = await DagExecutor().execute(plan, {"slow": with_timeout(slow, 0.05)})
    assert state.steps["x"].status == StepStatus.FAILED
    assert "timeout" in (state.steps["x"].error.message if state.steps["x"].error else "")


async def test_with_retry_retries_on_exception() -> None:
    builder = PlanBuilder()
    builder.add_step(id="x", action="flaky")
    plan = builder.build()

    attempts = {"n": 0}

    async def flaky(step: Step, ctx: StepContext) -> StepResult:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("transient")
        return StepResult(step_id=step.id, status=StepStatus.COMPLETED)

    state = await DagExecutor().execute(
        plan, {"flaky": with_retry(flaky, retries=3, base_delay=0.001)}
    )
    assert state.steps["x"].status == StepStatus.COMPLETED
    assert attempts["n"] == 3


def test_raise_if_failed_raises_on_non_completed() -> None:
    from py_agent_lib import ExecutionError, ExecutionState

    state = ExecutionState(execution_id="x", status=ExecutionStatus.FAILED)
    with pytest.raises(ExecutionError):
        raise_if_failed(state)

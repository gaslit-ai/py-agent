"""DagExecutor: scheduler over `graphlib.TopologicalSorter` + `asyncio.TaskGroup`.

Invariants:
- Only PENDING steps with all deps COMPLETED are scheduled.
- `max_parallel_steps` caps global concurrency; `action_concurrency` caps per-action.
- `fail_fast` stops scheduling new steps after the first FAILED step.
- `cancel_running_on_fail_fast=True` also sets a shared cancel event for cooperative aborts.
- External cancellation (`cancel_event` in options) propagates to handlers via `ctx.cancel_event`.
- On run end, remaining PENDING steps become CANCELLED (external cancel) or BLOCKED (deps failed/etc.).
- Resume: COMPLETED steps stay completed; all others reset to PENDING.
"""
from __future__ import annotations

import asyncio
import traceback
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import Any, Protocol

from pydantic import ValidationError

from .context import StepContext
from .events import (
    ExecutionCancelled,
    ExecutionCompleted,
    ExecutionEvent,
    ExecutionStarted,
    StepBlocked,
    StepCancelled,
    StepCompleted,
    StepFailed,
    StepScheduled,
    StepStarted,
)
from .plan import Plan
from .types import (
    BlockedReason,
    ErrorInfo,
    ExecutionState,
    ExecutionStatus,
    Step,
    StepResult,
    StepStatus,
    utc_now,
)


class StepHandler(Protocol):
    """Async callable for an action. Return a StepResult."""

    async def __call__(self, step: Step, ctx: StepContext) -> StepResult: ...


EventHandler = Callable[[ExecutionEvent], Awaitable[None] | None]


@dataclass(slots=True, kw_only=True)
class ExecuteOptions:
    on_event: EventHandler | None = None
    cancel_event: asyncio.Event | None = None
    execution_id: str | None = None
    resume_from: ExecutionState | None = None


@dataclass(slots=True)
class _SchedState:
    fail_fast_triggered: bool = False
    running_count: int = 0
    running_by_action: dict[str, int] = field(default_factory=dict)


class DagExecutor:
    def __init__(
        self,
        *,
        max_parallel_steps: int = 8,
        fail_fast: bool = False,
        cancel_running_on_fail_fast: bool = False,
        action_concurrency: Mapping[str, int] | None = None,
    ) -> None:
        if max_parallel_steps <= 0:
            raise ValueError("max_parallel_steps must be > 0")
        if action_concurrency:
            for action, limit in action_concurrency.items():
                if limit <= 0:
                    raise ValueError(
                        f"action_concurrency limit for {action!r} must be > 0"
                    )
        self.max_parallel_steps = max_parallel_steps
        self.fail_fast = fail_fast
        self.cancel_running_on_fail_fast = cancel_running_on_fail_fast
        self.action_concurrency: dict[str, int] = dict(action_concurrency or {})

    async def execute(
        self,
        plan: Plan,
        handlers: Mapping[str, StepHandler],
        options: ExecuteOptions | None = None,
    ) -> ExecutionState:
        opts = options or ExecuteOptions()
        execution_id = (
            opts.execution_id
            or (opts.resume_from.execution_id if opts.resume_from else None)
            or str(uuid.uuid4())
        )
        external_cancel = opts.cancel_event or asyncio.Event()
        handler_cancel = asyncio.Event()  # propagated to handlers
        emit = _make_emitter(execution_id, opts.on_event)

        state = _init_state(execution_id, plan, opts.resume_from)
        await emit(
            ExecutionStarted(
                execution_id=execution_id,
                ts=state.started_at or utc_now(),
                started_at=state.started_at or utc_now(),
            )
        )

        # Mirror cancel_event into handler_cancel.
        cancel_propagator = asyncio.create_task(
            _propagate_cancel(external_cancel, handler_cancel)
        )

        try:
            await self._run_scheduler(
                plan=plan,
                handlers=handlers,
                state=state,
                emit=emit,
                external_cancel=external_cancel,
                handler_cancel=handler_cancel,
            )
        finally:
            cancel_propagator.cancel()

        # External cancellation: pending → cancelled.
        if external_cancel.is_set():
            for sid, res in state.steps.items():
                if res.status == StepStatus.PENDING:
                    res.status = StepStatus.CANCELLED
                    res.finished_at = res.finished_at or utc_now()
                    await emit(
                        StepCancelled(execution_id=execution_id, ts=utc_now(), step_id=sid)
                    )
            state.status = ExecutionStatus.CANCELLED
            state.finished_at = utc_now()
            await emit(
                ExecutionCancelled(
                    execution_id=execution_id,
                    ts=utc_now(),
                    finished_at=state.finished_at,
                )
            )
            return state

        # Remaining pending → blocked (deps failed/cancelled/blocked, or fail_fast triggered).
        for sid, res in state.steps.items():
            if res.status == StepStatus.PENDING:
                reason = _derive_blocked_reason(plan, state, sid, was_fail_fast=self._was_fail_fast)
                res.status = StepStatus.BLOCKED
                res.blocked_reason = reason
                res.finished_at = res.finished_at or utc_now()
                await emit(
                    StepBlocked(
                        execution_id=execution_id,
                        ts=utc_now(),
                        step_id=sid,
                        blocked_reason=reason,
                    )
                )

        all_completed = all(r.status == StepStatus.COMPLETED for r in state.steps.values())
        state.status = ExecutionStatus.COMPLETED if all_completed else ExecutionStatus.FAILED
        state.finished_at = utc_now()
        await emit(
            ExecutionCompleted(
                execution_id=execution_id,
                ts=utc_now(),
                status=state.status,
                finished_at=state.finished_at,
            )
        )
        return state

    async def stream_execute(
        self,
        plan: Plan,
        handlers: Mapping[str, StepHandler],
        options: ExecuteOptions | None = None,
    ) -> AsyncIterator[ExecutionEvent]:
        """Async-iterate execution events. The run completes when the iterator is exhausted."""
        opts = options or ExecuteOptions()
        queue: asyncio.Queue[ExecutionEvent | None] = asyncio.Queue()

        original = opts.on_event

        async def _capture(event: ExecutionEvent) -> None:
            await queue.put(event)
            if original is not None:
                result = original(event)
                if asyncio.iscoroutine(result):
                    await result

        merged = ExecuteOptions(
            on_event=_capture,
            cancel_event=opts.cancel_event,
            execution_id=opts.execution_id,
            resume_from=opts.resume_from,
        )

        async def _runner() -> None:
            try:
                await self.execute(plan, handlers, merged)
            finally:
                await queue.put(None)

        task = asyncio.create_task(_runner())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # ------- internals -------

    _was_fail_fast: bool = False  # type: ignore[misc]

    async def _run_scheduler(
        self,
        *,
        plan: Plan,
        handlers: Mapping[str, StepHandler],
        state: ExecutionState,
        emit: Callable[[ExecutionEvent], Awaitable[None]],
        external_cancel: asyncio.Event,
        handler_cancel: asyncio.Event,
    ) -> None:
        # Completed-on-resume steps are NOT added to the sorter — they're
        # implicitly satisfied as dependencies. graphlib doesn't expose a way to
        # mark a node "done" without first having `get_ready()` return it, so
        # this is the cleanest way to model already-completed work.
        ts: TopologicalSorter[str] = TopologicalSorter()
        for step in plan:
            if state.steps[step.id].status == StepStatus.COMPLETED:
                continue
            remaining_deps = tuple(
                d for d in step.deps
                if state.steps[d].status != StepStatus.COMPLETED
            )
            ts.add(step.id, *remaining_deps)
        ts.prepare()

        global_sem = asyncio.Semaphore(self.max_parallel_steps)
        action_sems: dict[str, asyncio.Semaphore] = {
            action: asyncio.Semaphore(limit)
            for action, limit in self.action_concurrency.items()
        }
        completion_queue: asyncio.Queue[str] = asyncio.Queue()
        sched = _SchedState()
        self._was_fail_fast = False

        try:
            async with asyncio.TaskGroup() as tg:
                while True:
                    if external_cancel.is_set():
                        return  # post-processing in caller marks pending→cancelled

                    if not (self.fail_fast and sched.fail_fast_triggered):
                        for sid in ts.get_ready():
                            step = plan.get_step(sid)
                            res = state.steps[sid]
                            if res.status != StepStatus.PENDING:
                                # Already completed via resume; mark done and skip.
                                ts.done(sid)
                                continue
                            await emit(
                                StepScheduled(
                                    execution_id=state.execution_id,
                                    ts=utc_now(),
                                    step_id=sid,
                                )
                            )
                            sched.running_count += 1
                            sched.running_by_action[step.action] = (
                                sched.running_by_action.get(step.action, 0) + 1
                            )
                            tg.create_task(
                                self._run_step(
                                    step=step,
                                    state=state,
                                    plan=plan,
                                    handlers=handlers,
                                    emit=emit,
                                    handler_cancel=handler_cancel,
                                    global_sem=global_sem,
                                    action_sem=action_sems.get(step.action),
                                    completion_queue=completion_queue,
                                )
                            )

                    if sched.running_count == 0:
                        return  # nothing left to do

                    sid = await completion_queue.get()
                    sched.running_count -= 1
                    step = plan.get_step(sid)
                    sched.running_by_action[step.action] -= 1
                    if sched.running_by_action[step.action] == 0:
                        del sched.running_by_action[step.action]
                    ts.done(sid)

                    if (
                        self.fail_fast
                        and state.steps[sid].status == StepStatus.FAILED
                        and not sched.fail_fast_triggered
                    ):
                        sched.fail_fast_triggered = True
                        self._was_fail_fast = True
                        if self.cancel_running_on_fail_fast:
                            handler_cancel.set()
        except* asyncio.CancelledError:
            # External cancellation propagated; let post-processing handle it.
            pass

    async def _run_step(
        self,
        *,
        step: Step,
        state: ExecutionState,
        plan: Plan,
        handlers: Mapping[str, StepHandler],
        emit: Callable[[ExecutionEvent], Awaitable[None]],
        handler_cancel: asyncio.Event,
        global_sem: asyncio.Semaphore,
        action_sem: asyncio.Semaphore | None,
        completion_queue: asyncio.Queue[str],
    ) -> None:
        res = state.steps[step.id]
        try:
            async with _acquire(global_sem, action_sem):
                # Re-check cancellation after acquiring slots.
                if handler_cancel.is_set():
                    res.status = StepStatus.CANCELLED
                    res.finished_at = utc_now()
                    await emit(
                        StepCancelled(
                            execution_id=state.execution_id,
                            ts=utc_now(),
                            step_id=step.id,
                        )
                    )
                    return

                res.status = StepStatus.RUNNING
                res.started_at = utc_now()
                await emit(
                    StepStarted(
                        execution_id=state.execution_id,
                        ts=utc_now(),
                        step_id=step.id,
                    )
                )

                handler = handlers.get(step.action)
                if handler is None:
                    err = ErrorInfo(
                        message=f"no handler for action {step.action!r}",
                        type="MissingHandler",
                    )
                    await _fail(res, err, step.id, state.execution_id, emit)
                    return

                ctx = StepContext(
                    execution_id=state.execution_id,
                    step_id=step.id,
                    plan=plan,
                    results=state.steps,
                    cancel_event=handler_cancel,
                )

                try:
                    raw = await handler(step, ctx)
                    result = (
                        raw if isinstance(raw, StepResult) else StepResult.model_validate(raw)
                    )
                except asyncio.CancelledError:
                    res.status = StepStatus.CANCELLED
                    res.finished_at = utc_now()
                    await emit(
                        StepCancelled(
                            execution_id=state.execution_id,
                            ts=utc_now(),
                            step_id=step.id,
                        )
                    )
                    return
                except ValidationError as e:
                    await _fail(
                        res,
                        ErrorInfo(
                            message=f"handler returned invalid StepResult: {e}",
                            type="ValidationError",
                        ),
                        step.id,
                        state.execution_id,
                        emit,
                    )
                    return
                except Exception as e:
                    await _fail(
                        res,
                        ErrorInfo(
                            message=str(e),
                            type=type(e).__name__,
                            traceback="".join(
                                traceback.format_exception(type(e), e, e.__traceback__)
                            ),
                        ),
                        step.id,
                        state.execution_id,
                        emit,
                    )
                    return

                # Reconcile handler-reported status with executor state.
                status = result.status
                if status in (StepStatus.PENDING, StepStatus.RUNNING):
                    status = StepStatus.FAILED if result.error else StepStatus.COMPLETED
                elif status == StepStatus.COMPLETED and result.error:
                    status = StepStatus.FAILED

                res.status = status
                res.output = result.output
                res.error = result.error
                res.blocked_reason = (
                    result.blocked_reason if status == StepStatus.BLOCKED else None
                )
                res.finished_at = result.finished_at or utc_now()

                if status == StepStatus.COMPLETED:
                    await emit(
                        StepCompleted(
                            execution_id=state.execution_id,
                            ts=utc_now(),
                            step_id=step.id,
                        )
                    )
                elif status == StepStatus.FAILED:
                    await emit(
                        StepFailed(
                            execution_id=state.execution_id,
                            ts=utc_now(),
                            step_id=step.id,
                            error=res.error
                            or ErrorInfo(message="step failed", type="Unknown"),
                        )
                    )
                elif status == StepStatus.CANCELLED:
                    await emit(
                        StepCancelled(
                            execution_id=state.execution_id,
                            ts=utc_now(),
                            step_id=step.id,
                        )
                    )
                elif status == StepStatus.BLOCKED:
                    await emit(
                        StepBlocked(
                            execution_id=state.execution_id,
                            ts=utc_now(),
                            step_id=step.id,
                            blocked_reason=res.blocked_reason,
                        )
                    )
        finally:
            await completion_queue.put(step.id)


# ----- module-level helpers -----


def _make_emitter(
    execution_id: str, on_event: EventHandler | None
) -> Callable[[ExecutionEvent], Awaitable[None]]:
    async def emit(event: ExecutionEvent) -> None:
        if on_event is None:
            return
        result = on_event(event)
        if asyncio.iscoroutine(result):
            await result

    return emit


def _init_state(
    execution_id: str, plan: Plan, resume_from: ExecutionState | None
) -> ExecutionState:
    state = ExecutionState(
        execution_id=execution_id,
        status=ExecutionStatus.RUNNING,
        started_at=utc_now(),
    )
    if resume_from is not None:
        unknown = set(resume_from.steps) - set(plan.steps)
        if unknown:
            raise ValueError(
                f"snapshot includes unknown step(s): {sorted(unknown)}"
            )
        for step in plan:
            resumed = resume_from.steps.get(step.id)
            if resumed and resumed.status == StepStatus.COMPLETED:
                state.steps[step.id] = resumed.model_copy(deep=True)
            else:
                state.steps[step.id] = StepResult(step_id=step.id)
    else:
        for step in plan:
            state.steps[step.id] = StepResult(step_id=step.id)
    return state


async def _fail(
    res: StepResult,
    err: ErrorInfo,
    step_id: str,
    execution_id: str,
    emit: Callable[[ExecutionEvent], Awaitable[None]],
) -> None:
    res.status = StepStatus.FAILED
    res.error = err
    res.finished_at = utc_now()
    await emit(
        StepFailed(execution_id=execution_id, ts=utc_now(), step_id=step_id, error=err)
    )


@asynccontextmanager
async def _acquire(
    global_sem: asyncio.Semaphore, action_sem: asyncio.Semaphore | None
) -> AsyncIterator[None]:
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(global_sem)
        if action_sem is not None:
            await stack.enter_async_context(action_sem)
        yield


async def _propagate_cancel(src: asyncio.Event, dst: asyncio.Event) -> None:
    try:
        await src.wait()
        dst.set()
    except asyncio.CancelledError:
        pass


def _derive_blocked_reason(
    plan: Plan, state: ExecutionState, step_id: str, *, was_fail_fast: bool
) -> str | None:
    step = plan.get_step(step_id)
    has_failed = has_cancelled = has_blocked = False
    for dep in step.deps:
        dep_res = state.steps.get(dep)
        if dep_res is None:
            continue
        if dep_res.status == StepStatus.FAILED:
            has_failed = True
        elif dep_res.status == StepStatus.CANCELLED:
            has_cancelled = True
        elif dep_res.status == StepStatus.BLOCKED:
            has_blocked = True
    if has_failed:
        return BlockedReason.DEPENDENCY_FAILED.value
    if has_cancelled:
        return BlockedReason.DEPENDENCY_CANCELLED.value
    if has_blocked:
        return BlockedReason.DEPENDENCY_BLOCKED.value
    if was_fail_fast:
        return BlockedReason.FAIL_FAST.value
    return None


# Optional explicit `Any` re-export for handlers that need to type StepResult.output broadly.
__all__ = [
    "Any",
    "DagExecutor",
    "EventHandler",
    "ExecuteOptions",
    "StepHandler",
]

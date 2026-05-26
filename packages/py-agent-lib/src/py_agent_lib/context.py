"""StepContext: per-step view into the run — plan, results, cancellation."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .types import Step, StepResult, StepStatus

if TYPE_CHECKING:
    from .plan import Plan


class StepContext:
    """Passed to step handlers. Read-only views over plan and execution state."""

    __slots__ = ("_cancel_event", "_execution_id", "_plan", "_results", "_step_id")

    def __init__(
        self,
        *,
        execution_id: str,
        step_id: str,
        plan: Plan,
        results: Mapping[str, StepResult],
        cancel_event: asyncio.Event,
    ) -> None:
        self._execution_id = execution_id
        self._step_id = step_id
        self._plan = plan
        self._results = results
        self._cancel_event = cancel_event

    @property
    def execution_id(self) -> str:
        return self._execution_id

    @property
    def step_id(self) -> str:
        return self._step_id

    @property
    def cancel_event(self) -> asyncio.Event:
        """asyncio.Event handlers should poll for cooperative cancellation."""
        return self._cancel_event

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def get_step(self, step_id: str) -> Step:
        return self._plan.get_step(step_id)

    def get_result(self, step_id: str) -> StepResult | None:
        """Snapshot copy of a step result (or None if not started yet)."""
        res = self._results.get(step_id)
        return res.model_copy(deep=True) if res else None

    def require_result(self, step_id: str) -> StepResult:
        res = self.get_result(step_id)
        if res is None:
            raise LookupError(f"missing result for step {step_id!r}")
        return res

    def get_dependency_results(self) -> dict[str, StepResult]:
        """Completed-only dependency results, keyed by dep id."""
        step = self._plan.get_step(self._step_id)
        out: dict[str, StepResult] = {}
        for dep in step.deps:
            res = self._results.get(dep)
            if res is not None and res.status == StepStatus.COMPLETED:
                out[dep] = res.model_copy(deep=True)
        return out

    def get_dependency_outputs(self) -> dict[str, Any]:
        """Just the `.output` field of completed deps."""
        return {dep: res.output for dep, res in self.get_dependency_results().items()}

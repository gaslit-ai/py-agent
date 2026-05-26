"""High-level run_plan helper — execute + aggregate Usage + compute cost in one call."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict

from .executor import DagExecutor, ExecuteOptions, StepHandler
from .plan import Plan
from .types import ExecutionState, StepResult, StepStatus
from .usage import CostBreakdown, PricingRule, Usage, aggregate_usage, estimate_cost


class TaskOutput(BaseModel):
    """Optional wrapper handlers can use to attach Usage to their step output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: Any = None
    usage: Usage | None = None


UsageExtractor = Callable[[StepResult], Usage | None]


def default_usage_extractor(result: StepResult) -> Usage | None:
    """Pull Usage from `result.output` if it's a TaskOutput; otherwise return None."""
    if isinstance(result.output, TaskOutput):
        return result.output.usage
    if isinstance(result.output, dict) and "usage" in result.output:
        u = result.output["usage"]
        if isinstance(u, Usage):
            return u
        if isinstance(u, dict):
            try:
                return Usage.model_validate(u)
            except Exception:
                return None
    return None


class RunResult(NamedTuple):
    state: ExecutionState
    usage: list[Usage]
    costs: CostBreakdown | None


async def run_plan(
    plan: Plan,
    handlers: Mapping[str, StepHandler],
    *,
    executor: DagExecutor | None = None,
    options: ExecuteOptions | None = None,
    pricing_rules: list[PricingRule] | None = None,
    usage_extractor: UsageExtractor = default_usage_extractor,
) -> RunResult:
    """Run a plan, then aggregate Usage from completed steps and (optionally) compute cost."""
    exec_ = executor or DagExecutor()
    state = await exec_.execute(plan, handlers, options)

    raw_usages: list[Usage] = []
    for res in state.steps.values():
        if res.status != StepStatus.COMPLETED:
            continue
        u = usage_extractor(res)
        if u is not None:
            raw_usages.append(u)

    aggregated = aggregate_usage(raw_usages)
    costs = estimate_cost(aggregated, pricing_rules) if pricing_rules else None
    return RunResult(state=state, usage=aggregated, costs=costs)

"""py-agent-lib — domain-agnostic DAG scheduler/executor for step-based workflows."""
from __future__ import annotations

from .context import StepContext
from .errors import ExecutionError, PlanValidationError, raise_if_failed
from .events import (
    ExecutionCancelled,
    ExecutionCompleted,
    ExecutionEvent,
    ExecutionEventAdapter,
    ExecutionStarted,
    StepBlocked,
    StepCancelled,
    StepCompleted,
    StepFailed,
    StepScheduled,
    StepStarted,
)
from .executor import DagExecutor, EventHandler, ExecuteOptions, StepHandler
from .observers import (
    BufferedObserver,
    ExecutionObserver,
    ObserverFailurePolicy,
    OverflowPolicy,
    compose_observers,
    compose_observers_parallel,
)
from .plan import Plan, PlanBuilder
from .policies import with_retry, with_timeout
from .runner import RunResult, TaskOutput, UsageExtractor, default_usage_extractor, run_plan
from .snapshots import deserialize_execution_state, serialize_execution_state
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
from .usage import (
    CostBreakdown,
    CostLine,
    PricingRule,
    Usage,
    aggregate_usage,
    estimate_cost,
)

__version__ = "0.1.0"

__all__ = [
    # types
    "BlockedReason",
    "ErrorInfo",
    "ExecutionState",
    "ExecutionStatus",
    "Step",
    "StepResult",
    "StepStatus",
    "utc_now",
    # plan
    "Plan",
    "PlanBuilder",
    # context
    "StepContext",
    # executor
    "DagExecutor",
    "EventHandler",
    "ExecuteOptions",
    "StepHandler",
    # events
    "ExecutionCancelled",
    "ExecutionCompleted",
    "ExecutionEvent",
    "ExecutionEventAdapter",
    "ExecutionStarted",
    "StepBlocked",
    "StepCancelled",
    "StepCompleted",
    "StepFailed",
    "StepScheduled",
    "StepStarted",
    # errors
    "ExecutionError",
    "PlanValidationError",
    "raise_if_failed",
    # observers
    "BufferedObserver",
    "ExecutionObserver",
    "ObserverFailurePolicy",
    "OverflowPolicy",
    "compose_observers",
    "compose_observers_parallel",
    # policies
    "with_retry",
    "with_timeout",
    # snapshots
    "deserialize_execution_state",
    "serialize_execution_state",
    # usage / pricing
    "CostBreakdown",
    "CostLine",
    "PricingRule",
    "Usage",
    "aggregate_usage",
    "estimate_cost",
    # runner
    "RunResult",
    "TaskOutput",
    "UsageExtractor",
    "default_usage_extractor",
    "run_plan",
]

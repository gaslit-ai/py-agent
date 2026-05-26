"""OpenTelemetry observer — maps step events to OTel spans.

Each `step_started` opens a span; the matching terminal event (completed/failed/
cancelled) closes it with the appropriate status. Blocked steps emit a single
short-lived span so they're visible in traces.

This module is import-safe without `opentelemetry-api` installed; the import
error is deferred to constructor time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...events import ExecutionEvent

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


class OtelObserver:
    """Bridge execution events to OpenTelemetry spans.

    Args:
        tracer: an OTel Tracer. If None, uses the global tracer for `py_agent_lib`.

    Span layout:
        - One span per step, named `py-agent-lib.step.{step_id}`
        - Attributes: `execution_id`, `step_id`, plus `blocked_reason` / `error.type` when set
        - Status: OK on completed, ERROR on failed/cancelled
    """

    _SPAN_PREFIX = "py-agent-lib.step"

    def __init__(self, tracer: Tracer | None = None) -> None:
        try:
            from opentelemetry import trace
        except ImportError as e:
            raise ImportError(
                "OtelObserver requires opentelemetry-api; "
                "install with `pip install py-agent-lib[otel]`"
            ) from e
        self._trace_mod = trace
        self._tracer: Any = tracer or trace.get_tracer("py_agent_lib")
        self._spans: dict[tuple[str, str], Span] = {}

    def on_event(self, event: ExecutionEvent) -> None:
        match event.type:
            case "step_started":
                span = self._tracer.start_span(
                    name=f"{self._SPAN_PREFIX}.{event.step_id}",
                    attributes={
                        "execution_id": event.execution_id,
                        "step_id": event.step_id,
                    },
                )
                self._spans[(event.execution_id, event.step_id)] = span

            case "step_completed":
                span = self._spans.pop((event.execution_id, event.step_id), None)
                if span is not None:
                    from opentelemetry.trace import Status, StatusCode

                    span.set_status(Status(StatusCode.OK))
                    span.end()

            case "step_failed":
                span = self._spans.pop((event.execution_id, event.step_id), None)
                if span is not None:
                    from opentelemetry.trace import Status, StatusCode

                    span.set_attribute("error.type", event.error.type or "Unknown")
                    span.set_attribute("error.message", event.error.message)
                    span.set_status(Status(StatusCode.ERROR, event.error.message))
                    span.end()

            case "step_cancelled":
                span = self._spans.pop((event.execution_id, event.step_id), None)
                if span is not None:
                    from opentelemetry.trace import Status, StatusCode

                    span.set_status(Status(StatusCode.ERROR, "cancelled"))
                    span.end()

            case "step_blocked":
                # Blocked steps never received a step_started; record a brief span.
                span = self._tracer.start_span(
                    name=f"{self._SPAN_PREFIX}.{event.step_id}",
                    attributes={
                        "execution_id": event.execution_id,
                        "step_id": event.step_id,
                        "blocked_reason": event.blocked_reason or "unknown",
                    },
                )
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, "blocked"))
                span.end()

            case _:
                # execution_started / execution_completed / step_scheduled — not bridged.
                pass


__all__ = ["OtelObserver"]

"""Tests for adapters.observers — JsonlObserver and OtelObserver."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from py_agent_lib.adapters.observers import JsonlObserver
from py_agent_lib.adapters.observers.jsonl import default_serializer
from py_agent_lib.events import (
    ExecutionCompleted,
    ExecutionStarted,
    StepCompleted,
    StepFailed,
    StepStarted,
)
from py_agent_lib.types import ErrorInfo, ExecutionStatus


def _now() -> datetime:
    return datetime(2026, 5, 26, tzinfo=UTC)


def test_jsonl_writes_one_line_per_event(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    obs = JsonlObserver(path)
    obs.on_event(ExecutionStarted(execution_id="e1", ts=_now(), started_at=_now()))
    obs.on_event(StepStarted(execution_id="e1", ts=_now(), step_id="a"))
    obs.on_event(StepCompleted(execution_id="e1", ts=_now(), step_id="a"))
    obs.on_event(
        ExecutionCompleted(
            execution_id="e1",
            ts=_now(),
            status=ExecutionStatus.COMPLETED,
            finished_at=_now(),
        )
    )

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 4
    types = [json.loads(line)["type"] for line in lines]
    assert types == [
        "execution_started",
        "step_started",
        "step_completed",
        "execution_completed",
    ]


def test_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "events.ndjson"
    obs = JsonlObserver(path)
    obs.on_event(ExecutionStarted(execution_id="e1", ts=_now(), started_at=_now()))
    assert path.exists()


def test_jsonl_appends_across_observer_instances(tmp_path: Path) -> None:
    """Two observers writing to the same path should both append, not clobber."""
    path = tmp_path / "events.ndjson"
    JsonlObserver(path).on_event(ExecutionStarted(execution_id="e1", ts=_now(), started_at=_now()))
    JsonlObserver(path).on_event(StepStarted(execution_id="e1", ts=_now(), step_id="a"))
    assert len(path.read_text(encoding="utf-8").strip().split("\n")) == 2


def test_jsonl_custom_serializer_enriches_events(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"

    def enrich(event):
        d = event.model_dump(mode="json")
        d["extra"] = {"run_input_text": "Hello world."}
        return json.dumps(d)

    obs = JsonlObserver(path, serialize=enrich)
    obs.on_event(StepStarted(execution_id="e1", ts=_now(), step_id="x"))

    line = path.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert parsed["extra"]["run_input_text"] == "Hello world."
    assert parsed["type"] == "step_started"


def test_jsonl_failed_event_includes_error(tmp_path: Path) -> None:
    path = tmp_path / "events.ndjson"
    obs = JsonlObserver(path)
    obs.on_event(
        StepFailed(
            execution_id="e1",
            ts=_now(),
            step_id="x",
            error=ErrorInfo(message="boom", type="RuntimeError"),
        )
    )
    parsed = json.loads(path.read_text(encoding="utf-8").strip())
    assert parsed["error"]["message"] == "boom"
    assert parsed["error"]["type"] == "RuntimeError"


def test_default_serializer_round_trips() -> None:
    """A default-serialized event should re-parse via the discriminated union adapter."""
    from py_agent_lib.events import ExecutionEventAdapter

    event = StepCompleted(execution_id="e1", ts=_now(), step_id="x")
    line = default_serializer(event)
    parsed = ExecutionEventAdapter.validate_python(json.loads(line))
    assert parsed.type == "step_completed"
    assert parsed.step_id == "x"  # type: ignore[attr-defined]


# ----- OtelObserver -----


def test_otel_observer_creates_span_per_step() -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from py_agent_lib.adapters.observers import OtelObserver  # lazy via __getattr__

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    obs = OtelObserver(tracer=tracer)
    obs.on_event(StepStarted(execution_id="e1", ts=_now(), step_id="a"))
    obs.on_event(StepCompleted(execution_id="e1", ts=_now(), step_id="a"))

    obs.on_event(StepStarted(execution_id="e1", ts=_now(), step_id="b"))
    obs.on_event(
        StepFailed(
            execution_id="e1",
            ts=_now(),
            step_id="b",
            error=ErrorInfo(message="boom", type="ValueError"),
        )
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    names = {s.name for s in spans}
    assert names == {"py-agent-lib.step.a", "py-agent-lib.step.b"}

    by_step = {s.attributes["step_id"]: s for s in spans}  # type: ignore[index]
    assert by_step["a"].status.status_code.name == "OK"
    assert by_step["b"].status.status_code.name == "ERROR"
    assert by_step["b"].attributes["error.type"] == "ValueError"  # type: ignore[index]


def test_otel_observer_emits_span_for_blocked_step() -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from py_agent_lib.adapters.observers import OtelObserver
    from py_agent_lib.events import StepBlocked

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    obs = OtelObserver(tracer=tracer)
    obs.on_event(
        StepBlocked(
            execution_id="e1",
            ts=_now(),
            step_id="b",
            blocked_reason="dependency_failed",
        )
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "py-agent-lib.step.b"
    assert spans[0].attributes["blocked_reason"] == "dependency_failed"  # type: ignore[index]
    assert spans[0].status.status_code.name == "ERROR"

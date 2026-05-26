"""Concrete observer implementations.

- `JsonlObserver` ships with the core (no extra deps)
- `OtelObserver` requires `pip install py-agent-lib[otel]`
"""
from __future__ import annotations

from .jsonl import JsonlObserver, Serializer, default_serializer

__all__ = ["JsonlObserver", "Serializer", "default_serializer"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy-load OtelObserver so importing this package doesn't require opentelemetry."""
    if name == "OtelObserver":
        from .otel import OtelObserver

        return OtelObserver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

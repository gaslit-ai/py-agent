"""NDJSON file observer — one event per line."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ...events import ExecutionEvent

Serializer = Callable[[ExecutionEvent], str]


def default_serializer(event: ExecutionEvent) -> str:
    """Plain Pydantic JSON dump. Pass a custom serializer to enrich events at write time."""
    return event.model_dump_json()


class JsonlObserver:
    """Append-only NDJSON observer. Opens the file per write (crash-safe, no buffering).

    Args:
        path: file path. Parent directories are created automatically.
        serialize: custom serializer; defaults to plain Pydantic JSON. Use this to
            enrich events with extra fields (e.g. attach per-step telemetry that
            was captured elsewhere).
    """

    def __init__(self, path: str | Path, *, serialize: Serializer = default_serializer) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._serialize = serialize

    @property
    def path(self) -> Path:
        return self._path

    def on_event(self, event: ExecutionEvent) -> None:
        line = self._serialize(event)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")


__all__ = ["JsonlObserver", "Serializer", "default_serializer"]

"""Generic NDJSON read/write for Pydantic records.

Default training-data format: one Pydantic model per line. spaCy/HuggingFace
exporters live in a separate subpackage so this module stays dependency-free.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_records_jsonl(
    records: Iterable[BaseModel],
    path: str | Path,
    *,
    append: bool = False,
) -> int:
    """Write Pydantic `records` as NDJSON. Returns the number of records written.

    Creates parent directories as needed. Overwrites by default; pass `append=True`
    to add to an existing file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with target.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")
            count += 1
    return count


def read_records_jsonl(
    path: str | Path,
    *,
    model: type[T],
) -> Iterator[T]:
    """Stream Pydantic records from NDJSON. Skips blank lines.

    Raises pydantic.ValidationError on malformed records — callers can wrap with
    try/except per-line if they want lenient parsing.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield model.model_validate_json(line)


__all__ = ["read_records_jsonl", "write_records_jsonl"]

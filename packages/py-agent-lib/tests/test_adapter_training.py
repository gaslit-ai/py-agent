"""Tests for adapters.training — generic Pydantic NDJSON read/write."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError

from py_agent_lib.adapters.training import read_records_jsonl, write_records_jsonl


class _Record(BaseModel):
    name: str
    score: float = Field(ge=0, le=1)
    tags: list[str] = Field(default_factory=list)


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    records = [
        _Record(name="alpha", score=0.9, tags=["a", "b"]),
        _Record(name="beta", score=0.5),
    ]
    written = write_records_jsonl(records, path)
    assert written == 2

    read_back = list(read_records_jsonl(path, model=_Record))
    assert read_back == records


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "out.jsonl"
    write_records_jsonl([_Record(name="x", score=0.0)], path)
    assert path.exists()


def test_write_overwrites_by_default(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    write_records_jsonl([_Record(name="a", score=0.1)], path)
    write_records_jsonl([_Record(name="b", score=0.2)], path)
    records = list(read_records_jsonl(path, model=_Record))
    assert len(records) == 1
    assert records[0].name == "b"


def test_write_append_mode(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    write_records_jsonl([_Record(name="a", score=0.1)], path)
    write_records_jsonl([_Record(name="b", score=0.2)], path, append=True)
    records = list(read_records_jsonl(path, model=_Record))
    assert [r.name for r in records] == ["a", "b"]


def test_write_empty_iterable(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    written = write_records_jsonl([], path)
    assert written == 0
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "with_blanks.jsonl"
    path.write_text(
        '\n  \n{"name":"a","score":0.1}\n\n{"name":"b","score":0.2}\n   \n',
        encoding="utf-8",
    )
    records = list(read_records_jsonl(path, model=_Record))
    assert [r.name for r in records] == ["a", "b"]


def test_read_validation_error_on_bad_record(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"name":"a","score":0.5}\n{"name":"b","score":99.0}\n',  # score out of range
        encoding="utf-8",
    )
    it = read_records_jsonl(path, model=_Record)
    first = next(it)
    assert first.name == "a"
    with pytest.raises(ValidationError):
        next(it)


def test_records_iterable_can_be_a_generator(tmp_path: Path) -> None:
    path = tmp_path / "gen.jsonl"

    def gen():
        for i in range(3):
            yield _Record(name=f"r{i}", score=i / 10)

    written = write_records_jsonl(gen(), path)
    assert written == 3
    assert len(list(read_records_jsonl(path, model=_Record))) == 3

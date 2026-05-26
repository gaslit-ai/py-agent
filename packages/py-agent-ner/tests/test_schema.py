"""Tests for `single_label_schema` — the per-label Pydantic v2 response model."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from py_agent_ner import LabelExtraction, single_label_schema


def test_pins_label_to_literal_single_value() -> None:
    """Pydantic v2 emits `Literal["x"]` as JSON Schema `{"const": "x"}` (single allowed value).

    Some backends look for `enum`; the OpenAI/Ollama Pydantic→schema pipelines
    handle both. Either form constrains the LLM to that one value.
    """
    Schema = single_label_schema("person_name")
    label_schema = Schema.model_json_schema()["properties"]["label"]
    accepted = label_schema.get("const") or label_schema.get("enum")
    assert accepted == "person_name" or accepted == ["person_name"]


def test_accepts_only_the_pinned_label() -> None:
    Schema = single_label_schema("person_name")
    Schema(label="person_name", matches=["Alice"], confidence=0.9)
    with pytest.raises(ValidationError):
        Schema(label="other", matches=["Alice"], confidence=0.9)


def test_returns_a_subclass_of_label_extraction() -> None:
    Schema = single_label_schema("x")
    assert issubclass(Schema, LabelExtraction)
    inst = Schema(label="x", matches=[], confidence=0.5)
    assert isinstance(inst, LabelExtraction)


def test_is_cached() -> None:
    a = single_label_schema("x")
    b = single_label_schema("x")
    assert a is b
    c = single_label_schema("y")
    assert c is not a


def test_class_name_includes_label() -> None:
    Schema = single_label_schema("contact_handle")
    assert "contact_handle" in Schema.__name__


def test_rejects_empty_label_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        single_label_schema("")


def test_inherits_descriptions_from_label_extraction() -> None:
    """`matches` and `confidence` descriptions come from the base class; `label` gets a new one."""
    Schema = single_label_schema("x")
    schema_json = Schema.model_json_schema()
    for field in ("label", "matches", "confidence"):
        assert "description" in schema_json["properties"][field], field
    # the label description should mention the pinned value
    assert "'x'" in schema_json["properties"]["label"]["description"]


def test_allows_empty_matches() -> None:
    """A valid response can have no matches — label not present."""
    Schema = single_label_schema("x")
    inst = Schema(label="x", confidence=0.9)
    assert inst.matches == []


def test_round_trips_via_pydantic_validate_json() -> None:
    """The LLM gives us JSON; we parse it back via the scoped schema."""
    Schema = single_label_schema("person_name")
    raw = '{"label": "person_name", "matches": ["Alice", "Bob"], "confidence": 0.85}'
    inst = Schema.model_validate_json(raw)
    assert inst.matches == ["Alice", "Bob"]
    assert inst.confidence == 0.85

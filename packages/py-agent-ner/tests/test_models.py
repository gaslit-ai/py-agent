"""Tests for the Pydantic v2 shapes."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from py_agent_ner import (
    EntitySpan,
    LabelExtraction,
    LabelPrompt,
    TaggedEntity,
    TrainingRecord,
)

# ---- LabelPrompt ----


def test_label_prompt_basic() -> None:
    lp = LabelPrompt(label="person_name", instructions="Full names.")
    assert lp.label == "person_name"
    assert lp.instructions == "Full names."


def test_label_prompt_rejects_empty_label() -> None:
    with pytest.raises(ValidationError):
        LabelPrompt(label="", instructions="x")


def test_label_prompt_rejects_empty_instructions() -> None:
    with pytest.raises(ValidationError):
        LabelPrompt(label="x", instructions="")


def test_label_prompt_is_frozen() -> None:
    lp = LabelPrompt(label="x", instructions="y")
    with pytest.raises(ValidationError, match="frozen"):
        lp.label = "z"  # type: ignore[misc]


# ---- LabelExtraction ----


def test_label_extraction_basic() -> None:
    le = LabelExtraction(label="person_name", matches=["Alice", "Bob"], confidence=0.9)
    assert le.matches == ["Alice", "Bob"]


def test_label_extraction_empty_matches_allowed() -> None:
    """Empty matches = label not present. No sentinel."""
    le = LabelExtraction(label="x", matches=[], confidence=0.5)
    assert le.matches == []


def test_label_extraction_matches_defaults_to_empty() -> None:
    le = LabelExtraction(label="x", confidence=0.5)
    assert le.matches == []


def test_label_extraction_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        LabelExtraction(label="x", matches=["m"], confidence=1.5)
    with pytest.raises(ValidationError):
        LabelExtraction(label="x", matches=["m"], confidence=-0.1)


def test_label_extraction_treats_none_as_literal_match() -> None:
    """'none' is a valid match value — no magic sentinel."""
    le = LabelExtraction(label="negative_response", matches=["none", "no"], confidence=0.7)
    assert le.matches == ["none", "no"]


def test_label_extraction_label_can_be_named_none() -> None:
    """Nothing reserves the name 'none' for a label."""
    le = LabelExtraction(label="none", matches=["nothing"], confidence=0.5)
    assert le.label == "none"


# ---- EntitySpan ----


def test_entity_span_basic() -> None:
    span = EntitySpan(text="Alice", start=0, end=5)
    assert span.start == 0
    assert span.end == 5


def test_entity_span_rejects_end_le_start() -> None:
    with pytest.raises(ValidationError, match="end must be greater than start"):
        EntitySpan(text="x", start=5, end=5)


def test_entity_span_rejects_negative_start() -> None:
    with pytest.raises(ValidationError):
        EntitySpan(text="x", start=-1, end=3)


# ---- TaggedEntity ----


def test_tagged_entity_defaults_to_empty_spans() -> None:
    ent = TaggedEntity(label="x", confidence=0.5)
    assert ent.spans == []


# ---- TrainingRecord ----


def test_training_record_jsonl_round_trip() -> None:
    rec = TrainingRecord(
        input_text="Alice met Bob.",
        entities={
            "person_name": TaggedEntity(
                label="person_name",
                confidence=0.9,
                spans=[
                    EntitySpan(text="Alice", start=0, end=5),
                    EntitySpan(text="Bob", start=10, end=13),
                ],
            ),
        },
    )
    serialized = rec.model_dump_json()
    parsed = TrainingRecord.model_validate_json(serialized)
    assert parsed == rec


def test_models_are_frozen() -> None:
    span = EntitySpan(text="x", start=0, end=1)
    with pytest.raises(ValidationError, match="frozen"):
        span.start = 99  # type: ignore[misc]


def test_label_extraction_schema_includes_descriptions() -> None:
    """The descriptions on LabelExtraction become the LLM's schema guidance."""
    schema = LabelExtraction.model_json_schema()
    for field in ("label", "matches", "confidence"):
        assert "description" in schema["properties"][field], field

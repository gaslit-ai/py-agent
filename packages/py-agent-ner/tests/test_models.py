"""Tests for the canonical Pydantic shapes."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from py_agent_ner import (
    EntitySpan,
    ExtractedEntity,
    Extraction,
    TaggedEntity,
    TrainingRecord,
)

# ---- ExtractedEntity ----


def test_extracted_entity_minimal() -> None:
    e = ExtractedEntity(quote="Alice", label="person_name", confidence=0.9)
    assert e.quote == "Alice"
    assert e.label == "person_name"


def test_extracted_entity_rejects_empty_quote() -> None:
    with pytest.raises(ValidationError):
        ExtractedEntity(quote="", label="x", confidence=0.5)


def test_extracted_entity_rejects_empty_label() -> None:
    with pytest.raises(ValidationError):
        ExtractedEntity(quote="x", label="", confidence=0.5)


def test_extracted_entity_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        ExtractedEntity(quote="x", label="y", confidence=1.5)


def test_extracted_entity_quote_can_be_literal_none() -> None:
    """No magic sentinel — 'none' is a valid quote (e.g., for sentiment labels)."""
    e = ExtractedEntity(quote="none", label="negative_response", confidence=0.8)
    assert e.quote == "none"


def test_extracted_entity_label_can_be_named_none() -> None:
    """No reserved label names."""
    e = ExtractedEntity(quote="nothing", label="none", confidence=0.5)
    assert e.label == "none"


# ---- Extraction (LLM's full response) ----


def test_extraction_defaults_to_empty_entities() -> None:
    x = Extraction()
    assert x.entities == []


def test_extraction_accepts_list_of_entities() -> None:
    x = Extraction(
        entities=[
            ExtractedEntity(quote="Alice", label="person_name", confidence=0.9),
            ExtractedEntity(quote="Bob", label="person_name", confidence=0.8),
        ]
    )
    assert len(x.entities) == 2


# ---- EntitySpan ----


def test_entity_span_rejects_end_le_start() -> None:
    with pytest.raises(ValidationError, match="end must be greater than start"):
        EntitySpan(text="x", start=5, end=5)


def test_entity_span_offsets_check() -> None:
    span = EntitySpan(text="Alice", start=0, end=5)
    assert span.start == 0
    assert span.end == 5
    assert span.confidence is None  # default


def test_entity_span_with_confidence() -> None:
    span = EntitySpan(text="x", start=0, end=1, confidence=0.7)
    assert span.confidence == 0.7


def test_entity_span_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        EntitySpan(text="x", start=0, end=1, confidence=1.5)


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
                    EntitySpan(text="Alice", start=0, end=5, confidence=0.95),
                    EntitySpan(text="Bob", start=10, end=13, confidence=0.85),
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

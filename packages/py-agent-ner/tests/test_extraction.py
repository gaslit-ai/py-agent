"""Tests for the extraction→record primitives.

Also covers the canonical pattern users follow in their own code: build a
Pydantic response model inline with `create_model` + `Literal[*labels]`. There
is no helper for that — it's stock Pydantic, demonstrated here.
"""
from __future__ import annotations

from typing import Literal

import pytest
from pydantic import Field, ValidationError, create_model

from py_agent_ner import (
    EntitySpan,
    ExtractedEntity,
    Extraction,
    TaggedEntity,
    entities_to_spans,
    group_extraction_by_label,
    to_training_record,
)

# ---- inline response-model construction (the canonical user pattern) ----


def _make_scoped_extraction(labels: tuple[str, ...]) -> type[Extraction]:
    """Build the response model the way `examples/quickstart/run.py` does."""
    ScopedEntity = create_model(
        "ScopedEntity",
        __base__=ExtractedEntity,
        label=(
            Literal[labels],  # type: ignore[valid-type]
            Field(description="Entity type — must be one of the configured labels."),
        ),
    )
    return create_model(
        "ScopedExtraction",
        __base__=Extraction,
        entities=(
            list[ScopedEntity],
            Field(default_factory=list, description="Every extracted entity."),
        ),
    )


def test_inline_scoped_model_constrains_label_to_enum() -> None:
    Schema = _make_scoped_extraction(("person_name", "location_reference"))
    schema_json = Schema.model_json_schema()
    defs = schema_json.get("$defs") or schema_json.get("definitions") or {}
    entity_schema = (
        next(iter(defs.values())) if defs else schema_json["properties"]["entities"]["items"]
    )
    assert set(entity_schema["properties"]["label"]["enum"]) == {
        "person_name",
        "location_reference",
    }


def test_inline_scoped_model_rejects_out_of_set_label_at_runtime() -> None:
    Schema = _make_scoped_extraction(("person_name",))
    Schema(entities=[{"quote": "Alice", "label": "person_name", "confidence": 0.9}])
    with pytest.raises(ValidationError):
        Schema(entities=[{"quote": "Alice", "label": "BOGUS", "confidence": 0.9}])


def test_inline_scoped_model_carries_descriptions_into_json_schema() -> None:
    """Sanity: user-supplied descriptions land in the JSON schema sent to the LLM."""
    Schema = _make_scoped_extraction(("a", "b"))
    schema_json = Schema.model_json_schema()
    # entities-level description (user-supplied)
    assert "Every extracted entity." in schema_json["properties"]["entities"]["description"]
    # label-level description (user-supplied via Field on the override)
    defs = schema_json.get("$defs") or schema_json.get("definitions") or {}
    entity_schema = (
        next(iter(defs.values())) if defs else schema_json["properties"]["entities"]["items"]
    )
    assert "must be one of" in entity_schema["properties"]["label"]["description"].lower()
    # quote/confidence inherit their descriptions from the base class.
    assert "description" in entity_schema["properties"]["quote"]
    assert "description" in entity_schema["properties"]["confidence"]


# ---- group_extraction_by_label ----


def test_group_by_label_includes_empty_lists_for_absent_labels() -> None:
    extraction = Extraction(
        entities=[ExtractedEntity(quote="Alice", label="person_name", confidence=0.9)]
    )
    grouped = group_extraction_by_label(extraction, ("person_name", "location_reference"))
    assert grouped["person_name"][0].quote == "Alice"
    assert grouped["location_reference"] == []


def test_group_by_label_captures_off_list_labels_for_inspection() -> None:
    extraction = Extraction(
        entities=[ExtractedEntity(quote="x", label="off_list", confidence=0.5)]
    )
    grouped = group_extraction_by_label(extraction, ("person_name",))
    assert grouped["person_name"] == []
    assert grouped["off_list"][0].label == "off_list"


# ---- entities_to_spans ----


def test_entities_to_spans_attaches_confidence_per_span() -> None:
    text = "Alice met Bob."
    entities = [
        ExtractedEntity(quote="Alice", label="person_name", confidence=0.92),
        ExtractedEntity(quote="Bob", label="person_name", confidence=0.81),
    ]
    spans = entities_to_spans(entities, text)
    confs = {s.text: s.confidence for s in spans}
    assert confs["Alice"] == pytest.approx(0.92)
    assert confs["Bob"] == pytest.approx(0.81)


def test_entities_to_spans_strict_skips_near_misses() -> None:
    text = "Jordan Lee was here."
    entities = [ExtractedEntity(quote="Jordan  Lee", label="x", confidence=0.9)]
    assert entities_to_spans(entities, text) == []


def test_entities_to_spans_fuzzy_recovers_near_miss() -> None:
    text = "Jordan Lee was here."
    entities = [ExtractedEntity(quote="Jordan  Lee", label="x", confidence=0.9)]
    spans = entities_to_spans(entities, text, fuzzy=True)
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == spans[0].text


def test_entities_to_spans_empty_input() -> None:
    assert entities_to_spans([], "any text") == []


def test_user_can_compose_max_confidence_aggregation() -> None:
    """Docs-by-example: compose your own aggregation using the primitives."""
    text = "Alice met Bob."
    extraction = Extraction(
        entities=[
            ExtractedEntity(quote="Alice", label="person_name", confidence=0.5),
            ExtractedEntity(quote="Bob", label="person_name", confidence=0.9),
        ]
    )
    grouped = group_extraction_by_label(extraction, ("person_name",))
    items = grouped["person_name"]
    spans = entities_to_spans(items, text)
    custom = TaggedEntity(
        label="person_name",
        confidence=max(e.confidence for e in items),
        spans=spans,
    )
    assert custom.confidence == pytest.approx(0.9)


# ---- to_training_record (the opinionated convenience) ----


LABELS = ("person_name", "location_reference", "time_reference")


def test_to_training_record_groups_by_label_and_finds_spans() -> None:
    text = "Jordan Lee at City Library tomorrow."
    extraction = Extraction(
        entities=[
            ExtractedEntity(quote="Jordan Lee", label="person_name", confidence=0.92),
            ExtractedEntity(quote="City Library", label="location_reference", confidence=0.88),
            ExtractedEntity(quote="tomorrow", label="time_reference", confidence=0.95),
        ]
    )
    record = to_training_record(text, extraction, LABELS)
    assert record.input_text == text
    assert set(record.entities) == set(LABELS)
    person = record.entities["person_name"]
    assert person.confidence == pytest.approx(0.92)
    assert person.spans == [EntitySpan(text="Jordan Lee", start=0, end=10, confidence=0.92)]


def test_to_training_record_includes_absent_labels_with_empty_spans() -> None:
    text = "Just text."
    record = to_training_record(text, Extraction(entities=[]), LABELS)
    assert set(record.entities) == set(LABELS)
    assert all(ent.spans == [] for ent in record.entities.values())
    assert all(ent.confidence == 0.0 for ent in record.entities.values())


def test_to_training_record_preserves_label_order() -> None:
    text = "Alice met Bob."
    extraction = Extraction(
        entities=[ExtractedEntity(quote="Alice", label="person_name", confidence=0.9)]
    )
    ordered = ("time_reference", "location_reference", "person_name")
    record = to_training_record(text, extraction, ordered)
    assert list(record.entities) == list(ordered)


def test_to_training_record_attaches_per_quote_confidence() -> None:
    text = "Alice met Bob, then Carol."
    extraction = Extraction(
        entities=[
            ExtractedEntity(quote="Alice", label="person_name", confidence=1.0),
            ExtractedEntity(quote="Bob", label="person_name", confidence=0.9),
            ExtractedEntity(quote="Carol", label="person_name", confidence=0.8),
        ]
    )
    record = to_training_record(text, extraction, ("person_name",))
    person = record.entities["person_name"]
    confs = {s.text: s.confidence for s in person.spans}
    assert confs == pytest.approx({"Alice": 1.0, "Bob": 0.9, "Carol": 0.8})
    assert person.confidence == pytest.approx((1.0 + 0.9 + 0.8) / 3)


def test_to_training_record_handles_literal_none_quote() -> None:
    """'none' is a legitimate quote — no sentinel filtering anywhere."""
    text = "I have none of those."
    extraction = Extraction(
        entities=[ExtractedEntity(quote="none", label="time_reference", confidence=0.5)]
    )
    record = to_training_record(text, extraction, ("time_reference",))
    spans = record.entities["time_reference"].spans
    assert len(spans) == 1
    assert spans[0].text == "none"


def test_to_training_record_fuzzy_recovers_near_miss() -> None:
    text = "Jordan Lee was here."
    extraction = Extraction(
        entities=[ExtractedEntity(quote="Jordan  Lee", label="person_name", confidence=0.9)]
    )
    record = to_training_record(text, extraction, ("person_name",), fuzzy=True)
    spans = record.entities["person_name"].spans
    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == spans[0].text


def test_to_training_record_strict_drops_near_miss() -> None:
    text = "Jordan Lee was here."
    extraction = Extraction(
        entities=[ExtractedEntity(quote="Jordan  Lee", label="person_name", confidence=0.9)]
    )
    record = to_training_record(text, extraction, ("person_name",))
    assert record.entities["person_name"].spans == []

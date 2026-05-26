"""Canonical Pydantic v2 shapes for NER training-data generation.

Every field carries a `description` — for the LLM-facing shapes
(`ExtractedEntity`, `Extraction`) these descriptions are emitted into the JSON
schema sent to the model and guide it on what to fill in. For the trainer-
facing shapes (`EntitySpan`, `TaggedEntity`, `TrainingRecord`) they document
the contract for downstream consumers.

Override anything by subclassing: descriptions, validation, defaults — all
inherited Pydantic mechanics work normally.

Layers, in the order they're produced during a run:

1. `ExtractedEntity` — one `(quote, label, confidence)` tuple the LLM emits.
2. `Extraction`      — the LLM's full response: a list of ExtractedEntities.
3. `EntitySpan`      — one character-indexed span inside the input text.
4. `TaggedEntity`    — one label with confidence + the spans we found.
5. `TrainingRecord`  — one input text + all tagged entities (JSONL row shape).

Absence convention:
    A label that's not represented in `Extraction.entities` is "not present in
    this input." There is NO sentinel string — `"none"`, `"None"`, etc. are
    valid quote values treated as literal substrings.
"""
from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExtractedEntity(BaseModel):
    """One (quote, label, confidence) tuple from a multi-label extraction call."""

    model_config = ConfigDict(frozen=True)

    quote: str = Field(
        min_length=1,
        description=(
            "Exact verbatim substring from the input text. Must be character-for-character "
            "identical: same casing, same whitespace, same punctuation. Do not paraphrase, "
            "summarize, or normalize."
        ),
    )
    label: str = Field(
        min_length=1,
        description=(
            "Entity type for this quote. Must be one of the labels defined in the system "
            "prompt — the schema enforces this as an enum, so out-of-set values are rejected "
            "before they reach the application."
        ),
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "How certain you are that this quote is an instance of this label. 0.0 means "
            "uncertain; 1.0 means very confident. Use intermediate values to express partial "
            "confidence."
        ),
    )


class Extraction(BaseModel):
    """The LLM's full response — every entity extracted from one input text."""

    model_config = ConfigDict(frozen=True)

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description=(
            "Every entity instance present in the input text. Emit one entry per occurrence "
            "— if the same quote appears twice, return it twice. Return an empty list when "
            "no entities of any listed type are present."
        ),
    )


class EntitySpan(BaseModel):
    """One occurrence of an entity, with character offsets into the input text."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        min_length=1,
        description="The literal substring matched at this position.",
    )
    start: int = Field(
        ge=0,
        description="Inclusive character offset into the input text (0-indexed).",
    )
    end: int = Field(
        ge=1,
        description="Exclusive character offset. `text[start:end]` equals this span's `text`.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Per-occurrence confidence carried over from the LLM's extraction. None when "
            "the span was located by fuzzy matching with no exact quote correspondence."
        ),
    )

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class TaggedEntity(BaseModel):
    """All spans found for a single label, plus the label's overall confidence."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(
        min_length=1,
        description="The entity type for this group of spans.",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "Aggregated confidence across the spans (the default helper uses the mean of "
            "per-span confidences; 0.0 when no spans were found)."
        ),
    )
    spans: list[EntitySpan] = Field(
        default_factory=list,
        description="Every location where this label was found, character-indexed.",
    )


class TrainingRecord(BaseModel):
    """One row of training data: the input text + all extracted entities keyed by label."""

    model_config = ConfigDict(frozen=True)

    input_text: str = Field(
        min_length=1,
        description="The original text the LLM was given.",
    )
    entities: dict[str, TaggedEntity] = Field(
        default_factory=dict,
        description=(
            "All extracted entities, keyed by label name. Every requested label appears "
            "here; absent labels have an empty `spans` list and confidence 0.0."
        ),
    )

"""Pydantic v2 shapes for NER training-data generation.

Five shapes in two groups. Each field is documented; the descriptions on
`LabelExtraction` and its fields are emitted into the JSON schema sent to the
LLM and become its guidance.

LLM-facing (what the LLM sees and fills in — one call per label):

    LabelPrompt        # input: a label name + its instructions
    LabelExtraction    # output: {label, matches, confidence} for ONE label

Trainer-facing (what you assemble after the calls return):

    EntitySpan         # one occurrence of a quote, with character offsets
    TaggedEntity       # one label's confidence + all its spans
    TrainingRecord     # one input text + the per-label TaggedEntities

Architecture: one LLM call per (text, label). Each call's response_model is a
subclass of `LabelExtraction` with `label` pinned to `Literal["<that_label>"]`
— a single-value enum. The `matches` list is the substrings the LLM found for
THAT one label. You assemble `TrainingRecord` in your own code from N
`LabelExtraction` instances.

Absence convention: empty `matches` means "label not present in this text."
No sentinel strings.
"""
from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class LabelPrompt(BaseModel):
    """A label name plus its extraction instructions. Input only; never sent to the LLM as JSON."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(
        min_length=1,
        description="Short label identifier — snake_case is conventional.",
    )
    instructions: str = Field(
        min_length=1,
        description="Natural-language description of what to extract for this label.",
    )


class LabelExtraction(BaseModel):
    """The LLM's response for ONE per-label call.

    The pipeline pins `label` to a `Literal["<label_name>"]` subclass per call,
    so the schema enforces label-as-enum at the JSON-schema level. The
    `matches` list is the verbatim substrings the LLM found for that one label.
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(
        min_length=1,
        description=(
            "The entity type this call is extracting. The schema pins this to "
            "a single-value enum, so the only valid value is the label this call "
            "is for."
        ),
    )
    matches: list[str] = Field(
        default_factory=list,
        description=(
            "Verbatim substrings of the input text that are instances of this "
            "label. Empty list = label not present. One entry per occurrence — "
            "if the same quote appears twice, return it twice. No paraphrasing, "
            "no normalization, no casing or punctuation changes."
        ),
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description=(
            "How confident you are in this extraction overall, 0.0-1.0. Use "
            "intermediate values to express partial confidence. If `matches` is "
            "empty, this is your confidence that the label is not present."
        ),
    )


class EntitySpan(BaseModel):
    """One occurrence of an entity, with character offsets into the input text."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, description="The literal substring at this position.")
    start: int = Field(ge=0, description="Inclusive character offset (0-indexed).")
    end: int = Field(
        ge=1, description="Exclusive character offset. `input_text[start:end]` equals `text`."
    )

    @model_validator(mode="after")
    def _end_after_start(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class TaggedEntity(BaseModel):
    """All spans found for a single label, plus the label-level confidence."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(min_length=1, description="The entity type.")
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence carried over from the per-label LabelExtraction.",
    )
    spans: list[EntitySpan] = Field(
        default_factory=list,
        description="Every location where this label was found, character-indexed.",
    )


class TrainingRecord(BaseModel):
    """One row of training data: input text + all extracted entities keyed by label."""

    model_config = ConfigDict(frozen=True)

    input_text: str = Field(min_length=1, description="The original text the LLM was given.")
    entities: dict[str, TaggedEntity] = Field(
        default_factory=dict,
        description=(
            "All entities keyed by label name. Conventionally every label asked "
            "for appears here, with an empty `spans` list when the label was "
            "absent — but the data model doesn't enforce that."
        ),
    )

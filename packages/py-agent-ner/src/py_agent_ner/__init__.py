"""py-agent-ner — a small toolkit for building NER labelers.

This package is a **set of primitives**, not a runner. You own the loop, the
LLM client, the IO, the response model. py-agent-ner gives you the parts that
are genuinely worth sharing:

    # Pydantic v2 base shapes — subclass them in your code to fit your domain
    LabelPrompt          # label + instructions (input)
    ExtractedEntity      # quote + label + confidence (LLM output unit)
    Extraction           # list[ExtractedEntity] (LLM full response)
    EntitySpan           # one character-indexed span
    TaggedEntity         # label + confidence + spans (per-label group)
    TrainingRecord       # input_text + entities (JSONL row shape)

    # Prompt building — bring your own template or use the default
    DEFAULT_BASE_TEMPLATE
    build_system_prompt(label_prompts, base_template=None) -> str
    label_prompts_from_dict({label: instructions}) -> list[LabelPrompt]
    load_label_prompts(directory)                  -> list[LabelPrompt]
    labels_of(prompts)                             -> list[str]

    # Extraction → record (compose these or roll your own)
    group_extraction_by_label(extraction, labels) -> dict[label, [entities]]
    entities_to_spans(entities, text, fuzzy=False) -> [EntitySpan]
    to_training_record(text, extraction, labels, fuzzy=False) -> TrainingRecord

    # BIO TSV for HuggingFace / spaCy / Flair
    simple_tokenize, record_to_bio, jsonl_to_bio, records_to_bio,
    write_records_jsonl

The response model the LLM fills in is plain Pydantic — subclass
`ExtractedEntity` and `Extraction`, pin `label` with
`Literal[*your_labels]`, and override `Field(description=...)` to tune the
guidance the LLM sees. See `examples/quickstart/run.py` for the pattern.
"""
from __future__ import annotations

from .bio import (
    jsonl_to_bio,
    record_to_bio,
    records_to_bio,
    simple_tokenize,
    write_records_jsonl,
)
from .extraction import (
    entities_to_spans,
    group_extraction_by_label,
    to_training_record,
)
from .models import (
    EntitySpan,
    ExtractedEntity,
    Extraction,
    TaggedEntity,
    TrainingRecord,
)
from .prompts import (
    DEFAULT_BASE_TEMPLATE,
    LabelPrompt,
    build_system_prompt,
    label_prompts_from_dict,
    labels_of,
    load_label_prompts,
)

__version__ = "0.1.0"

__all__ = [
    # models
    "EntitySpan",
    "ExtractedEntity",
    "Extraction",
    "TaggedEntity",
    "TrainingRecord",
    # prompts
    "DEFAULT_BASE_TEMPLATE",
    "LabelPrompt",
    "build_system_prompt",
    "label_prompts_from_dict",
    "labels_of",
    "load_label_prompts",
    # extraction → record primitives
    "entities_to_spans",
    "group_extraction_by_label",
    "to_training_record",
    # BIO + JSONL
    "jsonl_to_bio",
    "record_to_bio",
    "records_to_bio",
    "simple_tokenize",
    "write_records_jsonl",
]

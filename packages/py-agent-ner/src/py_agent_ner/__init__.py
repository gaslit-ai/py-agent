"""py-agent-ner — small clear primitives for NER training-data generation.

Architecture (matches ts-agent-lib): **one LLM call per label, per text.**
Each call has a single-value `Literal[<label>]` enum on its schema and returns
`LabelExtraction(label, matches, confidence)` — the substrings it found for
that one label. You fan out N calls per text (asyncio.gather or DagExecutor),
then assemble a `TrainingRecord` from the N per-label responses in your own code.

This package provides:

    # The shapes
    LabelPrompt        # input: label + instructions
    LabelExtraction    # output of ONE call: label + matches[] + confidence
    EntitySpan         # one located occurrence with character offsets
    TaggedEntity       # one label's spans + confidence (assembled after calls)
    TrainingRecord     # one input text + all TaggedEntities (JSONL row)

    # Prompt rendering (single-label — one prompt per call)
    DEFAULT_BASE_TEMPLATE
    build_system_prompt(label_prompt, all_labels=None, ...) -> str
    label_prompts_from_dict({label: instructions}) -> list[LabelPrompt]
    load_label_prompts(directory)                  -> list[LabelPrompt]
    labels_of(prompts)                             -> list[str]

    # Schema builder (5-line wrapper around `create_model` + Literal; cached)
    single_label_schema(label_name) -> type[LabelExtraction]

    # BIO conversion for downstream training
    simple_tokenize, record_to_bio, jsonl_to_bio, records_to_bio,
    write_records_jsonl

What this package does NOT do:
  - Loop over your inputs (you do it: `for text in your_texts: ...`)
  - Fan out the per-label calls (you do it: `await asyncio.gather(...)`)
  - Assemble the TrainingRecord (you do it: build a `dict[label, TaggedEntity]`
    by calling `find_spans` on each `LabelExtraction.matches`)

The point is that the per-text loop, the fan-out, and the merge are all
five lines of code each — you can read what's happening directly in your
`run.py` instead of through a wrapper. See `examples/quickstart/run.py`.
"""
from __future__ import annotations

from .bio import (
    jsonl_to_bio,
    record_to_bio,
    records_to_bio,
    simple_tokenize,
    write_records_jsonl,
)
from .models import (
    EntitySpan,
    LabelExtraction,
    LabelPrompt,
    TaggedEntity,
    TrainingRecord,
)
from .prompts import (
    DEFAULT_BASE_TEMPLATE,
    build_system_prompt,
    label_prompts_from_dict,
    labels_of,
    load_label_prompts,
)
from .schema import single_label_schema

__version__ = "0.1.0"

__all__ = [
    # models
    "EntitySpan",
    "LabelExtraction",
    "LabelPrompt",
    "TaggedEntity",
    "TrainingRecord",
    # prompts
    "DEFAULT_BASE_TEMPLATE",
    "build_system_prompt",
    "label_prompts_from_dict",
    "labels_of",
    "load_label_prompts",
    # schema
    "single_label_schema",
    # BIO + JSONL
    "jsonl_to_bio",
    "record_to_bio",
    "records_to_bio",
    "simple_tokenize",
    "write_records_jsonl",
]

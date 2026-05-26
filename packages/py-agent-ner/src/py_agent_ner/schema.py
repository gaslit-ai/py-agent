"""`single_label_schema` — Pydantic v2 subclass of `LabelExtraction` with a single-value Literal enum on `label`.

This is the only schema-side helper in py-agent-ner. It exists because the
five-line equivalent is needed in every run.py and benefits from being cached
across the per-text fan-out. Nothing more is hidden.

Equivalent code, if you'd rather write it yourself:

    from typing import Literal
    from pydantic import Field, create_model
    from py_agent_ner import LabelExtraction

    Schema = create_model(
        "Extract_person_name",
        __base__=LabelExtraction,
        label=(
            Literal["person_name"],
            Field(description="Must be the string 'person_name'."),
        ),
    )

The JSON schema this emits has `"label": {"enum": ["person_name"], "type": "string", ...}`
— a single-value enum that the LLM backend (Ollama `format=…`, OpenAI structured
output, instructor's strict mode) rejects any other value for.
"""
from __future__ import annotations

from functools import cache
from typing import Literal

from pydantic import Field, create_model

from .models import LabelExtraction


@cache
def single_label_schema(label_name: str) -> type[LabelExtraction]:
    """Return a Pydantic v2 subclass of `LabelExtraction` with `label: Literal[label_name]`.

    Cached, so repeated calls with the same `label_name` return the same class
    object (which matters when you fan out the same label across many texts —
    Pydantic doesn't have to rebuild the schema each time).

    Args:
        label_name: the one label this response model accepts.

    Returns:
        `type[LabelExtraction]`. Pass it as `response_model=` to your LLM client.
    """
    if not label_name:
        raise ValueError("label_name must be non-empty")
    return create_model(
        f"Extract_{label_name}",
        __base__=LabelExtraction,
        label=(
            Literal[label_name],
            Field(description=f"Must be the string {label_name!r}."),
        ),
    )


__all__ = ["single_label_schema"]

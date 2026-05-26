"""Extraction → record conversion primitives.

Three small functions. You compose them yourself; nothing is hidden.

- `group_extraction_by_label(extraction, labels)` — `dict[label_name, [entities]]`,
  with empty lists for any requested label the LLM didn't return.

- `entities_to_spans(entities, text, fuzzy=...)` — `list[EntitySpan]` with the
  per-entity confidence carried onto each span. Uses
  `py_agent_lib.adapters.spans.find_spans` for the actual substring search.

- `to_training_record(text, extraction, labels, fuzzy=...)` — opinionated
  composition that uses mean per-entity confidence as the label confidence.
  Reimplement using the two primitives above if you want different semantics.

For the response model the LLM fills in, just use stock Pydantic — see the
example in `examples/quickstart/run.py`. Subclass `ExtractedEntity` and
`Extraction`, pin `label` with `Literal[*your_labels]`, override descriptions
with `Field(description=...)`. No builder function needed.
"""
from __future__ import annotations

from collections.abc import Iterable

from py_agent_lib.adapters.spans import find_spans

from .models import EntitySpan, ExtractedEntity, Extraction, TaggedEntity, TrainingRecord


def group_extraction_by_label(
    extraction: Extraction,
    labels: Iterable[str],
) -> dict[str, list[ExtractedEntity]]:
    """Group an Extraction's entities by label.

    Every label in `labels` gets a key in the output (possibly an empty list).
    Labels in the extraction but not in `labels` are still included — if the
    LLM somehow returned an off-list label (shouldn't happen when your
    response_model pins `label` to a `Literal`), it ends up in the dict so you
    can detect it.
    """
    out: dict[str, list[ExtractedEntity]] = {name: [] for name in labels}
    for ent in extraction.entities:
        out.setdefault(ent.label, []).append(ent)
    return out


def entities_to_spans(
    entities: Iterable[ExtractedEntity],
    text: str,
    *,
    fuzzy: bool = False,
) -> list[EntitySpan]:
    """Locate each entity's `quote` in `text`. Per-entity confidence is attached to its span.

    Strict by default; opt into rapidfuzz fallback with `fuzzy=True` to recover
    near-exact substrings (whitespace drift, etc.).
    """
    ent_list = list(entities)
    quotes = [e.quote for e in ent_list]
    spans_raw = find_spans(quotes, text, fuzzy=fuzzy)
    conf_by_quote = {e.quote: e.confidence for e in ent_list}
    return [
        EntitySpan(
            text=s.text,
            start=s.start,
            end=s.end,
            confidence=conf_by_quote.get(s.text),
        )
        for s in spans_raw
    ]


def to_training_record(
    text: str,
    extraction: Extraction,
    labels: Iterable[str],
    *,
    fuzzy: bool = False,
) -> TrainingRecord:
    """Convenience: group → find spans → build `TrainingRecord` with mean confidence.

    Opinionated: label-level confidence is the arithmetic mean of per-entity
    confidences (0.0 when no entities for that label). If you want max, min,
    weighted, or anything else, compose with `group_extraction_by_label` and
    `entities_to_spans` directly.

    Args:
        text: the original input text.
        extraction: the LLM's response (`Extraction` or any subclass).
        labels: the expected label set, in the order you want them in the
            output. Every label gets a `TaggedEntity`, even when absent.
        fuzzy: rapidfuzz fallback for non-exact quotes.
    """
    label_list = list(labels)
    by_label = group_extraction_by_label(extraction, label_list)

    entities: dict[str, TaggedEntity] = {}
    for name in label_list:
        items = by_label[name]
        spans = entities_to_spans(items, text, fuzzy=fuzzy)
        avg_conf = sum(e.confidence for e in items) / len(items) if items else 0.0
        entities[name] = TaggedEntity(label=name, confidence=avg_conf, spans=spans)

    return TrainingRecord(input_text=text, entities=entities)


__all__ = [
    "entities_to_spans",
    "group_extraction_by_label",
    "to_training_record",
]

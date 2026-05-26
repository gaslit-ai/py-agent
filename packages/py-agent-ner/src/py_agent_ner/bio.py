"""JSONL → CoNLL BIO TSV conversion for downstream NER training.

The canonical NER training format is BIO-tagged tokens (one `token\\ttag` per
line, blank line between records). HuggingFace Transformers, spaCy, Flair,
GLiNER all consume it directly.

The tokenizer here is intentionally simple — Unicode word-characters and
isolated punctuation. Your training script should re-tokenize with its target
model's tokenizer (BERT WordPiece, XLM-R SentencePiece, etc.) and re-align
labels, which is the standard pattern in HuggingFace's token-classification
tutorial.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from py_agent_lib.adapters.training import read_records_jsonl, write_records_jsonl

from .models import TrainingRecord

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Word + punctuation tokenization preserving character offsets.

    Returns `(tokens, char_spans)` where `char_spans[i] = (start, end)` so that
    `text[start:end] == tokens[i]`.
    """
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in _TOKEN_RE.finditer(text):
        tokens.append(match.group())
        spans.append((match.start(), match.end()))
    return tokens, spans


def record_to_bio(record: TrainingRecord) -> list[tuple[str, str]]:
    """Tokenize a TrainingRecord and assign BIO tags from its character spans.

    Returns `[(token, tag), ...]` where tag is `O`, `B-<label>`, or `I-<label>`.
    """
    tokens, token_spans = simple_tokenize(record.input_text)
    tags = ["O"] * len(tokens)

    for label, entity in record.entities.items():
        for span in entity.spans:
            # Find every token whose character range overlaps the entity span.
            overlapping: list[int] = []
            for i, (t_start, t_end) in enumerate(token_spans):
                if t_start >= span.end:
                    break
                if t_end <= span.start:
                    continue
                overlapping.append(i)

            if not overlapping:
                continue
            tags[overlapping[0]] = f"B-{label}"
            for idx in overlapping[1:]:
                tags[idx] = f"I-{label}"

    return list(zip(tokens, tags, strict=True))


def jsonl_to_bio(input_path: str | Path, output_path: str | Path) -> int:
    """Stream-convert a JSONL of `TrainingRecord` to a CoNLL-style BIO TSV.

    Returns the number of records converted. Creates parent directories as needed.
    """
    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dst.open("w", encoding="utf-8") as f:
        for record in read_records_jsonl(src, model=TrainingRecord):
            for token, tag in record_to_bio(record):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")
            count += 1
    return count


def records_to_bio(records: Iterable[TrainingRecord], output_path: str | Path) -> int:
    """Like `jsonl_to_bio` but takes in-memory records instead of a JSONL file."""
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dst.open("w", encoding="utf-8") as f:
        for record in records:
            for token, tag in record_to_bio(record):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")
            count += 1
    return count


# Re-export so users can write training_data.jsonl from a list of TrainingRecord without
# importing from py_agent_lib directly.
__all__ = [
    "jsonl_to_bio",
    "record_to_bio",
    "records_to_bio",
    "simple_tokenize",
    "write_records_jsonl",
]

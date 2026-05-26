"""Training-data writers.

The default JSONL format is shipped here. spaCy `DocBin`, HuggingFace `datasets`,
and other downstream formats live in a separate subpackage so this submodule
stays dependency-free.
"""
from __future__ import annotations

from .jsonl import read_records_jsonl, write_records_jsonl

__all__ = ["read_records_jsonl", "write_records_jsonl"]

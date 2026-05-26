"""Find character-indexed spans of candidate strings within an input text.

Two strategies:
- **strict**: exact substring matches via `re.finditer(re.escape(cand), text)`
- **fuzzy**:  rapidfuzz `partial_ratio_alignment` fallback when strict yields nothing

The fuzzy path is opt-in (`fuzzy=True`) and only runs for candidates whose strict
match set is empty, so well-behaved LLM outputs cost nothing.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class Span(NamedTuple):
    text: str
    start: int
    end: int


def _require_rapidfuzz() -> None:
    try:
        import rapidfuzz  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "rapidfuzz is required for fuzzy spans; "
            "install with `pip install py-agent-lib[fuzzy]`"
        ) from e


def find_spans(
    candidates: list[str],
    text: str,
    *,
    fuzzy: bool = False,
    fuzzy_threshold: float = 90.0,
) -> list[Span]:
    """Return deduped character-indexed spans of `candidates` within `text`.

    Args:
        candidates: candidate substrings to locate. Empty strings are skipped;
            otherwise every string is treated as a literal substring to find.
            (Pass an empty `candidates` list to signal "label not present" —
            this function doesn't recognize any sentinel value.)
        text: the haystack.
        fuzzy: if True and a candidate has no strict matches, fall back to
            rapidfuzz `partial_ratio_alignment` and accept the best alignment
            when its score is at or above `fuzzy_threshold`.
        fuzzy_threshold: 0-100 score threshold for fuzzy acceptance.

    Returns:
        List of Spans, deduplicated by (start, end). Order is by first-seen.
    """
    spans: list[Span] = []
    seen: set[tuple[int, int]] = set()

    for cand in candidates:
        if not cand:
            continue

        strict_found = False
        for match in re.finditer(re.escape(cand), text):
            key = (match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            spans.append(Span(text=match.group(), start=match.start(), end=match.end()))
            strict_found = True

        if strict_found or not fuzzy:
            continue

        _require_rapidfuzz()
        from rapidfuzz.fuzz import partial_ratio_alignment

        alignment = partial_ratio_alignment(cand, text)
        if alignment is None:
            continue
        if alignment.score < fuzzy_threshold:
            continue
        start, end = alignment.dest_start, alignment.dest_end
        if end <= start:
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        spans.append(Span(text=text[start:end], start=start, end=end))

    return spans


__all__ = ["Span", "find_spans"]

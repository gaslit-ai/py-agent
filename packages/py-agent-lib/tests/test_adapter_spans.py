"""Tests for adapters.spans — strict and fuzzy span finding."""
from __future__ import annotations

import pytest

from py_agent_lib.adapters.spans import Span, find_spans


def test_strict_finds_exact_substrings() -> None:
    text = "Jordan Lee at City Library tomorrow 4:30 PM @coach_sam."
    spans = find_spans(["Jordan Lee", "City Library", "@coach_sam"], text)
    by_text = {s.text: s for s in spans}
    assert by_text["Jordan Lee"].start == 0
    assert by_text["Jordan Lee"].end == 10
    assert text[by_text["City Library"].start : by_text["City Library"].end] == "City Library"
    assert by_text["@coach_sam"].text == "@coach_sam"


def test_strict_returns_multiple_occurrences() -> None:
    text = "cat dog cat"
    spans = find_spans(["cat"], text)
    assert len(spans) == 2
    assert spans[0] == Span("cat", 0, 3)
    assert spans[1] == Span("cat", 8, 11)


def test_strict_dedupes_repeated_candidates() -> None:
    text = "abc def"
    spans = find_spans(["abc", "abc", "def"], text)
    assert len(spans) == 2  # not 3


def test_strict_treats_none_as_literal_substring() -> None:
    """`find_spans` recognizes no sentinel — 'none' is a normal candidate to find."""
    text = "I have none and you have None."
    spans = find_spans(["none", "None"], text)
    assert len(spans) == 2
    by_text = {s.text for s in spans}
    assert by_text == {"none", "None"}


def test_strict_no_special_handling_for_none() -> None:
    """The string 'none' as a candidate finds its literal occurrence; nothing is filtered."""
    text = "Hello none world"
    spans = find_spans(["none"], text)
    assert len(spans) == 1
    assert spans[0].text == "none"
    assert text[spans[0].start : spans[0].end] == "none"


def test_strict_skips_empty_strings() -> None:
    spans = find_spans(["", "hello"], "hello world")
    assert len(spans) == 1
    assert spans[0].text == "hello"


def test_strict_handles_regex_metacharacters() -> None:
    """Candidates containing regex special chars should not break the matcher."""
    text = "Price is $19.99 (USD)."
    spans = find_spans(["$19.99", "(USD)"], text)
    assert len(spans) == 2
    assert text[spans[0].start : spans[0].end] in {"$19.99", "(USD)"}


def test_strict_no_match_returns_empty() -> None:
    spans = find_spans(["missing"], "totally different text")
    assert spans == []


def test_fuzzy_disabled_by_default() -> None:
    """If `fuzzy=False`, near-miss candidates are dropped."""
    spans = find_spans(["Jordan  Lee"], "Jordan Lee was here.")  # double space
    assert spans == []


def test_fuzzy_finds_near_miss() -> None:
    pytest.importorskip("rapidfuzz")
    text = "Jordan Lee was here yesterday."
    spans = find_spans(
        ["Jordan  Lee"],  # extra space — would fail strict
        text,
        fuzzy=True,
        fuzzy_threshold=80.0,
    )
    assert len(spans) == 1
    assert spans[0].start >= 0
    # The recovered span should match the actual text in the haystack.
    assert text[spans[0].start : spans[0].end] == spans[0].text


def test_fuzzy_below_threshold_returns_empty() -> None:
    pytest.importorskip("rapidfuzz")
    text = "Jordan Lee was here."
    spans = find_spans(
        ["totally unrelated string"],
        text,
        fuzzy=True,
        fuzzy_threshold=95.0,
    )
    assert spans == []


def test_fuzzy_prefers_strict_when_available() -> None:
    """When strict matches exist, fuzzy fallback is skipped."""
    pytest.importorskip("rapidfuzz")
    text = "Jordan Lee Jordan Lee"
    spans = find_spans(["Jordan Lee"], text, fuzzy=True)
    assert len(spans) == 2  # both strict matches; fuzzy not invoked
    assert all(s.text == "Jordan Lee" for s in spans)

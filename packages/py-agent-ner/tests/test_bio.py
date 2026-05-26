"""Tests for JSONL → CoNLL BIO conversion."""
from __future__ import annotations

from pathlib import Path

from py_agent_ner import (
    EntitySpan,
    TaggedEntity,
    TrainingRecord,
    jsonl_to_bio,
    record_to_bio,
    records_to_bio,
    simple_tokenize,
    write_records_jsonl,
)


def test_simple_tokenize_splits_on_words_and_punctuation() -> None:
    tokens, spans = simple_tokenize("Alice met Bob.")
    assert tokens == ["Alice", "met", "Bob", "."]
    assert spans == [(0, 5), (6, 9), (10, 13), (13, 14)]
    for tok, (s, e) in zip(tokens, spans, strict=True):
        assert "Alice met Bob."[s:e] == tok


def test_simple_tokenize_handles_extra_whitespace() -> None:
    tokens, _ = simple_tokenize("Jordan  Lee  here")
    assert tokens == ["Jordan", "Lee", "here"]


def test_simple_tokenize_handles_handles_and_numbers() -> None:
    tokens, _ = simple_tokenize("@coach_sam said 4:30 PM")
    assert tokens == ["@", "coach_sam", "said", "4", ":", "30", "PM"]


def test_record_to_bio_basic_fan_out() -> None:
    record = TrainingRecord(
        input_text="Jordan Lee at City Library tomorrow.",
        entities={
            "person_name": TaggedEntity(
                label="person_name",
                confidence=0.9,
                spans=[EntitySpan(text="Jordan Lee", start=0, end=10)],
            ),
            "location_reference": TaggedEntity(
                label="location_reference",
                confidence=0.9,
                spans=[EntitySpan(text="City Library", start=14, end=26)],
            ),
            "time_reference": TaggedEntity(
                label="time_reference",
                confidence=0.9,
                spans=[EntitySpan(text="tomorrow", start=27, end=35)],
            ),
        },
    )

    assert record_to_bio(record) == [
        ("Jordan", "B-person_name"),
        ("Lee", "I-person_name"),
        ("at", "O"),
        ("City", "B-location_reference"),
        ("Library", "I-location_reference"),
        ("tomorrow", "B-time_reference"),
        (".", "O"),
    ]


def test_record_to_bio_multiple_spans_each_get_their_own_b_tag() -> None:
    record = TrainingRecord(
        input_text="Alice met Bob and Carol.",
        entities={
            "person_name": TaggedEntity(
                label="person_name",
                confidence=0.9,
                spans=[
                    EntitySpan(text="Alice", start=0, end=5),
                    EntitySpan(text="Bob", start=10, end=13),
                    EntitySpan(text="Carol", start=18, end=23),
                ],
            ),
        },
    )

    tagged = record_to_bio(record)
    bs = [i for i, (_, tag) in enumerate(tagged) if tag.startswith("B-")]
    assert len(bs) == 3  # three separate entities of the same label


def test_record_to_bio_no_entities_all_o() -> None:
    record = TrainingRecord(input_text="No entities here.", entities={})
    tagged = record_to_bio(record)
    assert all(tag == "O" for _, tag in tagged)


def test_record_to_bio_handles_multi_token_entity_with_punctuation() -> None:
    record = TrainingRecord(
        input_text="The handle @coach_sam is in.",
        entities={
            "contact_handle": TaggedEntity(
                label="contact_handle",
                confidence=0.95,
                spans=[EntitySpan(text="@coach_sam", start=11, end=21)],
            ),
        },
    )
    tagged = record_to_bio(record)
    tags_by_token = dict(tagged)
    assert tags_by_token["@"] == "B-contact_handle"
    assert tags_by_token["coach_sam"] == "I-contact_handle"
    assert tags_by_token["is"] == "O"


def test_jsonl_to_bio_writes_blank_line_between_records(tmp_path: Path) -> None:
    records = [
        TrainingRecord(
            input_text="Alice met Bob.",
            entities={
                "person_name": TaggedEntity(
                    label="person_name",
                    confidence=0.9,
                    spans=[
                        EntitySpan(text="Alice", start=0, end=5),
                        EntitySpan(text="Bob", start=10, end=13),
                    ],
                ),
            },
        ),
        TrainingRecord(input_text="Hi there.", entities={}),
    ]
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.bio.tsv"
    write_records_jsonl(records, src)

    n = jsonl_to_bio(src, dst)
    assert n == 2

    content = dst.read_text(encoding="utf-8")
    blocks = [block for block in content.split("\n\n") if block]
    assert len(blocks) == 2
    first_block_lines = blocks[0].split("\n")
    assert first_block_lines[0] == "Alice\tB-person_name"
    assert first_block_lines[2] == "Bob\tB-person_name"


def test_records_to_bio_in_memory(tmp_path: Path) -> None:
    """records_to_bio should write directly without a JSONL intermediate."""
    records = [
        TrainingRecord(
            input_text="Alice.",
            entities={
                "person_name": TaggedEntity(
                    label="person_name",
                    confidence=0.9,
                    spans=[EntitySpan(text="Alice", start=0, end=5)],
                ),
            },
        ),
    ]
    dst = tmp_path / "out.bio.tsv"
    n = records_to_bio(records, dst)
    assert n == 1
    assert "Alice\tB-person_name" in dst.read_text(encoding="utf-8")


def test_jsonl_to_bio_creates_parent_dirs(tmp_path: Path) -> None:
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "nested" / "deeper" / "out.bio.tsv"
    write_records_jsonl([TrainingRecord(input_text="x", entities={})], src)
    jsonl_to_bio(src, dst)
    assert dst.exists()

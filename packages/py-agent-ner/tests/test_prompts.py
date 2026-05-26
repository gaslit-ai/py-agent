"""Tests for label-prompt loading and combined system-prompt rendering."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from py_agent_ner import (
    DEFAULT_BASE_TEMPLATE,
    LabelPrompt,
    build_system_prompt,
    label_prompts_from_dict,
    labels_of,
    load_label_prompts,
)


def test_label_prompts_from_dict_basic() -> None:
    prompts = label_prompts_from_dict(
        {"person_name": "Full names.", "time_reference": "Times."}
    )
    assert labels_of(prompts) == ["person_name", "time_reference"]
    assert prompts[0].instructions == "Full names."


def test_label_prompts_from_dict_strips_whitespace() -> None:
    prompts = label_prompts_from_dict({"x": "  do x  \n"})
    assert prompts[0].instructions == "do x"


def test_label_prompts_from_dict_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        label_prompts_from_dict({})


def test_label_prompts_from_dict_rejects_blank_instructions() -> None:
    with pytest.raises(ValueError, match="empty instructions"):
        label_prompts_from_dict({"x": "   \n"})


def test_load_label_prompts_from_dir(tmp_path: Path) -> None:
    (tmp_path / "person_name.jinja").write_text("Names.", encoding="utf-8")
    (tmp_path / "time_reference.jinja").write_text("Times.", encoding="utf-8")
    prompts = load_label_prompts(tmp_path)
    # sorted by filename
    assert labels_of(prompts) == ["person_name", "time_reference"]
    assert prompts[0].instructions == "Names."


def test_load_label_prompts_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_label_prompts(tmp_path / "nope")


def test_load_label_prompts_rejects_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no '\.jinja' files"):
        load_label_prompts(tmp_path)


def test_load_label_prompts_rejects_blank_file(tmp_path: Path) -> None:
    (tmp_path / "x.jinja").write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_label_prompts(tmp_path)


def test_label_prompt_is_frozen() -> None:
    lp = LabelPrompt(label="x", instructions="y")
    with pytest.raises(ValidationError, match="frozen"):
        lp.label = "changed"  # type: ignore[misc]


def test_build_system_prompt_lists_all_labels() -> None:
    prompts = label_prompts_from_dict(
        {"person_name": "Names.", "location_reference": "Places."}
    )
    rendered = build_system_prompt(prompts)
    assert "person_name" in rendered
    assert "Names." in rendered
    assert "location_reference" in rendered
    assert "Places." in rendered


def test_build_system_prompt_default_template_has_no_jinja_residue() -> None:
    prompts = label_prompts_from_dict({"x": "do x"})
    rendered = build_system_prompt(prompts, base_template=DEFAULT_BASE_TEMPLATE)
    assert "{{" not in rendered
    assert "{%" not in rendered
    assert "%}" not in rendered


def test_build_system_prompt_custom_template() -> None:
    base = "Labels: {% for lp in label_prompts %}{{ lp.label }} {% endfor %}"
    prompts = label_prompts_from_dict({"a": "do a", "b": "do b"})
    assert build_system_prompt(prompts, base_template=base) == "Labels: a b "


def test_build_system_prompt_rejects_empty_prompts() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_system_prompt([])

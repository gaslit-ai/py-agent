"""Tests for label loading + single-label prompt rendering."""
from __future__ import annotations

from pathlib import Path

import pytest

from py_agent_ner import (
    DEFAULT_BASE_TEMPLATE,
    LabelPrompt,
    build_system_prompt,
    label_prompts_from_dict,
    labels_of,
    load_label_prompts,
)

# ---- label_prompts_from_dict ----


def test_label_prompts_from_dict_basic() -> None:
    prompts = label_prompts_from_dict(
        {"person_name": "Full names.", "time_reference": "Times."}
    )
    assert labels_of(prompts) == ["person_name", "time_reference"]
    assert prompts[0].instructions == "Full names."


def test_label_prompts_from_dict_strips_whitespace() -> None:
    prompts = label_prompts_from_dict({"x": "  do x  \n"})
    assert prompts[0].instructions == "do x"


def test_label_prompts_from_dict_rejects_empty_mapping() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        label_prompts_from_dict({})


def test_label_prompts_from_dict_rejects_blank_instructions() -> None:
    with pytest.raises(ValueError, match="empty instructions"):
        label_prompts_from_dict({"x": "   \n"})


# ---- load_label_prompts ----


def test_load_label_prompts_from_dir(tmp_path: Path) -> None:
    (tmp_path / "person_name.jinja").write_text("Names.", encoding="utf-8")
    (tmp_path / "time_reference.jinja").write_text("Times.", encoding="utf-8")
    prompts = load_label_prompts(tmp_path)
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


# ---- build_system_prompt ----


def test_build_system_prompt_renders_single_label() -> None:
    lp = LabelPrompt(label="person_name", instructions="Full names.")
    rendered = build_system_prompt(lp)
    assert "person_name" in rendered
    assert "Full names." in rendered
    assert "person_name entity extraction" in rendered  # default expert_role


def test_build_system_prompt_lists_all_labels_for_context() -> None:
    """`all_labels` should appear in the rendered prompt so the LLM sees siblings."""
    lp = LabelPrompt(label="x", instructions="do x")
    rendered = build_system_prompt(lp, all_labels=["x", "y", "z"])
    assert "x" in rendered
    assert "- y" in rendered
    assert "- z" in rendered


def test_build_system_prompt_all_labels_defaults_to_current_only() -> None:
    """When `all_labels` is not given, the template just lists the current label."""
    lp = LabelPrompt(label="x", instructions="do x")
    rendered = build_system_prompt(lp)
    # contains the current label in the allowed_labels block
    assert "- x" in rendered


def test_build_system_prompt_custom_expert_role() -> None:
    lp = LabelPrompt(label="x", instructions="do x")
    rendered = build_system_prompt(lp, expert_role="strict classifier of x")
    assert "You are a strict classifier of x classifier" in rendered


def test_build_system_prompt_custom_template() -> None:
    lp = LabelPrompt(label="abc", instructions="do abc")
    base = "Label={{ label }} | Instr={{ label_instructions }}"
    assert build_system_prompt(lp, base_template=base) == "Label=abc | Instr=do abc"


def test_default_template_renders_cleanly() -> None:
    """Sanity: the shipped template has no leftover Jinja markup after rendering."""
    lp = LabelPrompt(label="x", instructions="do x")
    rendered = build_system_prompt(lp, all_labels=["x"], base_template=DEFAULT_BASE_TEMPLATE)
    assert "{{" not in rendered
    assert "{%" not in rendered
    assert "%}" not in rendered


def test_default_template_emits_json_quoted_label() -> None:
    """Rule 3 uses `{{ label | tojson }}` so the LLM sees the literal JSON string."""
    lp = LabelPrompt(label="contact_handle", instructions="do it")
    rendered = build_system_prompt(lp)
    assert '"contact_handle"' in rendered

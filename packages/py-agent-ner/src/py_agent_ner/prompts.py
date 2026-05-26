"""Label-prompt loading + combined multi-label system-prompt rendering.

These are NER-specific concepts and live here, NOT in py-agent-lib (which is
domain-agnostic and never imports Jinja2).

Two-part flow:
    1. Define each label's extraction instructions via either:
       - `label_prompts_from_dict({"person_name": "Full names...", ...})`
       - `load_label_prompts("./labels/")`   (one file per label)
       Both return `list[LabelPrompt]` — just (label, instructions) pairs.

    2. Render those into a single multi-label system prompt via
       `build_system_prompt(label_prompts, base_template=...)`. That string is
       what the pipeline sends to the LLM.

The pipeline calls (2) for you. (1) is also what you pass to
`extract_training_data(..., labels=...)`.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class LabelPrompt(BaseModel):
    """One label and the extraction instructions that describe it."""

    model_config = ConfigDict(frozen=True)

    label: str = Field(
        min_length=1,
        description="Short label identifier (snake_case is conventional).",
    )
    instructions: str = Field(
        min_length=1,
        description="Natural-language description of what to extract for this label.",
    )


def label_prompts_from_dict(labels: Mapping[str, str]) -> list[LabelPrompt]:
    """Build a list of `LabelPrompt` from a `{label: instructions}` mapping.

    Raises:
        ValueError: if `labels` is empty, or any value is blank after `.strip()`.
    """
    if not labels:
        raise ValueError("labels must not be empty")
    out: list[LabelPrompt] = []
    for label, raw in labels.items():
        instructions = raw.strip()
        if not instructions:
            raise ValueError(f"label {label!r} has empty instructions")
        out.append(LabelPrompt(label=label, instructions=instructions))
    return out


def load_label_prompts(
    labels_dir: str | Path,
    *,
    extension: str = ".jinja",
) -> list[LabelPrompt]:
    """Read every `<extension>` file in `labels_dir`. Filename = label, contents = instructions.

    Files are sorted by name for deterministic ordering. The file content is
    treated as a plain string; no per-label Jinja rendering happens here.

    Raises:
        FileNotFoundError: if the directory doesn't exist or has no matching files.
        ValueError:        if any file is blank after `.strip()`.
    """
    directory = Path(labels_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"labels_dir does not exist: {directory}")

    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == extension)
    if not files:
        raise FileNotFoundError(f"no {extension!r} files in {directory}")

    out: list[LabelPrompt] = []
    for path in files:
        instructions = path.read_text(encoding="utf-8").strip()
        if not instructions:
            raise ValueError(f"label prompt file is empty: {path}")
        out.append(LabelPrompt(label=path.stem, instructions=instructions))
    return out


def labels_of(prompts: Iterable[LabelPrompt]) -> list[str]:
    """Convenience: extract the ordered label names from a sequence of LabelPrompt."""
    return [p.label for p in prompts]


DEFAULT_BASE_TEMPLATE = """\
You are an entity extraction system.

Extract every instance of the following entity types from the input text:

{% for lp in label_prompts -%}
- **{{ lp.label }}**: {{ lp.instructions }}
{% endfor %}

Rules:
1) Return ONLY exact, verbatim substrings from the input text as `quote`.
2) The `label` field MUST be one of the labels listed above (it's a schema-enforced enum).
3) Emit one entry per occurrence — if the same quote appears twice, return it twice.
4) If no entities of any listed type are present, return an empty `entities` list.
5) Do not infer, normalize, paraphrase, or change casing or punctuation.
6) `confidence` is 0.0-1.0 per item.
"""


def _require_jinja() -> None:
    try:
        import jinja2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "jinja2 is required; install with `pip install py-agent-ner`"
        ) from e


def build_system_prompt(
    label_prompts: Iterable[LabelPrompt],
    *,
    base_template: str | None = None,
) -> str:
    """Render the multi-label system prompt for one extraction call.

    Args:
        label_prompts: every label this run will extract.
        base_template: Jinja2 source. Defaults to `DEFAULT_BASE_TEMPLATE`.

    Available Jinja variables:
        - `label_prompts`: iterable of `LabelPrompt` (each has `.label`, `.instructions`)
    """
    _require_jinja()
    import jinja2

    prompts = list(label_prompts)
    if not prompts:
        raise ValueError("label_prompts must not be empty")

    env = jinja2.Environment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )
    template = env.from_string(
        base_template if base_template is not None else DEFAULT_BASE_TEMPLATE
    )
    return template.render(label_prompts=prompts)


__all__ = [
    "DEFAULT_BASE_TEMPLATE",
    "LabelPrompt",
    "build_system_prompt",
    "label_prompts_from_dict",
    "labels_of",
    "load_label_prompts",
]

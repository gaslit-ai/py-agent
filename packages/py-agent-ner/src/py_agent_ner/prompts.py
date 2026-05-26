"""Label-prompt loading + single-label system-prompt rendering.

`label_prompts_from_dict` and `load_label_prompts` parse the (label,
instructions) pairs. `build_system_prompt` renders ONE LabelPrompt into the
system prompt for ONE LLM call — the per-label call you fan out.

Why single-label rendering: the architecture is one LLM call per label, each
with its own focused system prompt that says "extract THIS one entity type."
The `all_labels` kwarg lets the template optionally list every label in the
run as context — useful for disambiguation, but the model is still asked for
exactly one label per call.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from .models import LabelPrompt


def label_prompts_from_dict(labels: Mapping[str, str]) -> list[LabelPrompt]:
    """Build `[LabelPrompt]` from a `{label: instructions}` mapping.

    Raises:
        ValueError: if `labels` is empty or any value is blank after `.strip()`.
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

    Files are sorted by name for deterministic ordering.

    Raises:
        FileNotFoundError: the directory doesn't exist or has no matching files.
        ValueError: any file is blank after `.strip()`.
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
You are a {{ expert_role }} classifier.

You label ONE entity type at a time.

Entity label for this run: {{ label }}

Label-specific instructions:
{{ label_instructions }}

Rules:
1) Return exact, verbatim substrings from the input text in `matches`.
2) If the label is not present, return an empty `matches` array.
3) The `label` field MUST be the string {{ label | tojson }} — the JSON schema enforces this as a single-value enum.
4) `confidence` is a number 0.0-1.0.
5) Do not infer, normalize, paraphrase, or change casing or punctuation.

All labels in this run (context only — focus on {{ label }}):
{% for allowed_label in allowed_labels -%}
- {{ allowed_label }}
{% endfor %}
"""


def _require_jinja() -> None:
    try:
        import jinja2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "jinja2 is required; install with `pip install py-agent-ner`"
        ) from e


def build_system_prompt(
    label_prompt: LabelPrompt,
    *,
    all_labels: Iterable[str] | None = None,
    expert_role: str | None = None,
    base_template: str | None = None,
) -> str:
    """Render the system prompt for ONE per-label LLM call.

    Args:
        label_prompt: the single label this call is for.
        all_labels: optional list of every label in this run. Surfaced to the
            template as `allowed_labels` for context; the model is still asked
            for only `label_prompt.label`. Defaults to `[label_prompt.label]`.
        expert_role: value for `{{ expert_role }}` in the template. Defaults
            to `"<label> entity extraction"`.
        base_template: Jinja2 source. Defaults to `DEFAULT_BASE_TEMPLATE`.

    Available Jinja variables (in `DEFAULT_BASE_TEMPLATE`):
        - `label`              — `label_prompt.label`
        - `label_instructions` — `label_prompt.instructions`
        - `allowed_labels`     — `all_labels` (or `[label_prompt.label]`)
        - `expert_role`        — the value above
    """
    _require_jinja()
    import jinja2

    env = jinja2.Environment(
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )
    template = env.from_string(
        base_template if base_template is not None else DEFAULT_BASE_TEMPLATE
    )
    return template.render(
        label=label_prompt.label,
        label_instructions=label_prompt.instructions,
        allowed_labels=list(all_labels) if all_labels is not None else [label_prompt.label],
        expert_role=expert_role or f"{label_prompt.label} entity extraction",
    )


__all__ = [
    "DEFAULT_BASE_TEMPLATE",
    "build_system_prompt",
    "label_prompts_from_dict",
    "labels_of",
    "load_label_prompts",
]

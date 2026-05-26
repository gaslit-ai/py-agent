"""Quickstart: generate NER training data from a few texts using local gemma4.

Every piece is visible. You change labels by editing files under `labels/`.
You change the system-prompt template by editing `prompts/base_system.jinja`.
You change the **structured-output schema** — including the field
descriptions the LLM sees — by editing the `ScopedEntity` and `ScopedExtraction`
classes defined directly in `main()` below.

Run from this directory:

    python run.py

Writes here:
  - training_data.jsonl     — one TrainingRecord per line
  - training_data.bio.tsv   — CoNLL BIO TSV ready for HuggingFace / spaCy
"""
import asyncio
from pathlib import Path
from typing import Literal

from py_agent_lib.adapters.llm import from_ollama
from py_agent_ner import (
    ExtractedEntity,
    Extraction,
    build_system_prompt,
    jsonl_to_bio,
    labels_of,
    load_label_prompts,
    to_training_record,
    write_records_jsonl,
)
from pydantic import Field, create_model

TEXTS: list[str] = [
    "Jordan Lee at City Library tomorrow 4:30 PM @coach_sam.",
    "Meeting with Dr. Alex Chen at 2pm in the Oakland office.",
    "Tweet @nina_w about the conference next Friday morning.",
]


async def main() -> None:
    here = Path(__file__).parent

    # ── 1. Load labels — one `.jinja` file per label under labels/ ────────────
    label_prompts = load_label_prompts(here / "labels")
    label_names = tuple(labels_of(label_prompts))

    # ── 2. Compile the system prompt ──────────────────────────────────────────
    base_template = (here / "prompts" / "base_system.jinja").read_text(encoding="utf-8")
    system_prompt = build_system_prompt(label_prompts, base_template=base_template)

    # ── 3. Build the structured-output response model ─────────────────────────
    # This is plain Pydantic. You see every field, every Literal value, every
    # description the LLM gets from the JSON schema. Edit anything here to
    # change what the LLM is asked for — no helper function is hiding it.
    ScopedEntity = create_model(
        "ScopedEntity",
        __base__=ExtractedEntity,
        # `Literal[label_names]` pins `label` to your loaded labels at the
        # JSON-schema level — out-of-set values are rejected by the LLM backend.
        label=(
            Literal[label_names],  # type: ignore[valid-type]
            Field(description=(
                "Entity type for this quote. MUST be one of the labels listed "
                "in the system prompt — the schema enforces this as an enum."
            )),
        ),
        # Optional: tighten the canonical descriptions further if you want.
        # quote=(str, Field(min_length=1, description="…")),
        # confidence=(float, Field(ge=0, le=1, description="…")),
    )
    ScopedExtraction = create_model(
        "ScopedExtraction",
        __base__=Extraction,
        entities=(
            list[ScopedEntity],
            Field(default_factory=list, description=(
                "Every entity instance present in the input text. Emit one entry "
                "per occurrence — duplicates if a quote appears twice. Empty list "
                "if no entities of any listed type are present."
            )),
        ),
    )

    # ── 4. Pick an LLM ────────────────────────────────────────────────────────
    llm = from_ollama("gemma4", think=False)

    # ── 5. Your loop (sequential here — swap for asyncio.gather for parallel) ─
    records = []
    for text in TEXTS:
        extraction = await llm.extract(
            system=system_prompt,
            user=text,
            response_model=ScopedExtraction,
        )
        records.append(to_training_record(text, extraction, label_names, fuzzy=True))

    # ── 6. Write outputs ──────────────────────────────────────────────────────
    output = here / "training_data.jsonl"
    bio_output = here / "training_data.bio.tsv"
    write_records_jsonl(records, output)
    n_bio = jsonl_to_bio(output, bio_output)

    print(f"wrote {len(records)} record(s) -> {output}")
    print(f"BIO TSV ({n_bio} record(s))    -> {bio_output}\n")
    for rec in records:
        print(f"  {rec.input_text}")
        for label, ent in rec.entities.items():
            spans = [f"{s.text!r}@[{s.start},{s.end})" for s in ent.spans]
            print(f"    {label} (conf={ent.confidence:.2f}): {', '.join(spans) or '(none)'}")
        print()


if __name__ == "__main__":
    asyncio.run(main())

"""Quickstart: NER training-data generation via per-label fan-out.

This mirrors the ts-agent-lib architecture. Every step is in this file — no
helper is hiding anything.

For each input text:
  1. Fan out N parallel LLM calls (one per label) via asyncio.gather.
  2. Each call uses a single-label JSON schema: `label` is `Literal["<that_label>"]`
     and the model returns the `matches[]` list of substrings for THAT label.
  3. After the gather, we have N `LabelExtraction`s for this text.
  4. We locate each match's character offsets with `find_spans` and assemble
     one `TrainingRecord` per text.

What you can change here:
  - Labels: add/edit/remove `.jinja` files under `labels/`. Filename = label
    name, contents = the extraction instructions.
  - The per-label system prompt: edit `prompts/base_system.jinja`.
  - The response model: see `single_label_schema` — or write your own subclass
    of `LabelExtraction` inline with `Literal["x"]`.
  - The LLM: swap `from_ollama` for `from_openai`, `from_anthropic`, or `from_litellm`.

Outputs (next to this script):
  - training_data.jsonl     — one TrainingRecord per line
  - training_data.bio.tsv   — CoNLL BIO TSV ready for HuggingFace / spaCy
"""
import asyncio
from pathlib import Path

from py_agent_lib.adapters.llm import from_ollama
from py_agent_lib.adapters.spans import find_spans
from py_agent_ner import (
    EntitySpan,
    LabelExtraction,
    LabelPrompt,
    TaggedEntity,
    TrainingRecord,
    build_system_prompt,
    jsonl_to_bio,
    labels_of,
    load_label_prompts,
    single_label_schema,
    write_records_jsonl,
)

TEXTS: list[str] = [
    "Jordan Lee at City Library tomorrow 4:30 PM @coach_sam.",
    "Meeting with Dr. Alex Chen at 2pm in the Oakland office.",
    "Tweet @nina_w about the conference next Friday morning.",
]


async def extract_one_label(
    *,
    llm,
    text: str,
    label_prompt: LabelPrompt,
    all_labels: list[str],
    base_template: str,
) -> LabelExtraction:
    """One LLM call: extract every instance of THIS ONE label in `text`.

    Visible from top to bottom:
      - The system prompt is rendered just for this label
      - The response model is the single-label-Literal subclass of LabelExtraction
      - The user message is just the text
    """
    system = build_system_prompt(
        label_prompt,
        all_labels=all_labels,
        base_template=base_template,
    )
    response_model = single_label_schema(label_prompt.label)
    return await llm.extract(
        system=system,
        user=text,
        response_model=response_model,
    )


def assemble_training_record(
    text: str,
    label_prompts: list[LabelPrompt],
    extractions: list[LabelExtraction],
    *,
    fuzzy: bool = True,
) -> TrainingRecord:
    """Locate each extraction's matches in `text` and assemble the TrainingRecord.

    Five lines of real work; everything else is the dict-building boilerplate.
    You could inline this in main() if you'd rather see it there.
    """
    entities: dict[str, TaggedEntity] = {}
    for lp, ext in zip(label_prompts, extractions, strict=True):
        spans_raw = find_spans(ext.matches, text, fuzzy=fuzzy)
        entities[lp.label] = TaggedEntity(
            label=lp.label,
            confidence=ext.confidence,
            spans=[EntitySpan(text=s.text, start=s.start, end=s.end) for s in spans_raw],
        )
    return TrainingRecord(input_text=text, entities=entities)


async def main() -> None:
    here = Path(__file__).parent

    # Load labels + the single-label prompt template once.
    label_prompts = load_label_prompts(here / "labels")
    all_labels = labels_of(label_prompts)
    base_template = (here / "prompts" / "base_system.jinja").read_text(encoding="utf-8")

    llm = from_ollama("gemma4", think=False)

    records: list[TrainingRecord] = []
    for text in TEXTS:
        # Fan-out: N parallel LLM calls (one per label) for THIS text.
        extractions: list[LabelExtraction] = await asyncio.gather(*[
            extract_one_label(
                llm=llm,
                text=text,
                label_prompt=lp,
                all_labels=all_labels,
                base_template=base_template,
            )
            for lp in label_prompts
        ])
        # Merge: assemble the TrainingRecord from N per-label extractions.
        records.append(assemble_training_record(text, label_prompts, extractions))

    # Write the JSONL training data and a CoNLL BIO TSV.
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

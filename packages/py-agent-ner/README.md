# py-agent-ner

NER training-data generation built on [py-agent-lib](https://github.com/gaslit-ai/py-agent/tree/main/packages/py-agent-lib).

**Architecture:** one LLM call per (text, label). Each call uses a single-value
`Literal["<that_label>"]` enum on its response schema; the model returns the
verbatim substrings it found for that label. You fan out N calls per text
(asyncio or `DagExecutor`) and assemble a `TrainingRecord` from the N
responses. The package gives you the shapes, the prompt template, the schema
builder, and the BIO converter — nothing else. The loop, the fan-out, and
the merge are in your code where you can see them.

## Install

```bash
pip install py-agent-ner
# or for local dev within this monorepo:
uv sync
```

`py-agent-ner` pulls in `py-agent-lib[llm,fuzzy]` (Pydantic + httpx +
instructor + rapidfuzz) plus Jinja2 — everything needed to run.

## What you write yourself

The loop, the fan-out, the assembly. The package never hides any of these.
The canonical pattern (from `examples/quickstart/run.py`):

```python
import asyncio
from py_agent_lib.adapters.llm import from_ollama
from py_agent_lib.adapters.spans import find_spans
from py_agent_ner import (
    EntitySpan, LabelExtraction, TaggedEntity, TrainingRecord,
    build_system_prompt, load_label_prompts, labels_of, single_label_schema,
)

label_prompts  = load_label_prompts("./labels")
all_labels     = labels_of(label_prompts)
base_template  = open("./prompts/base_system.jinja").read()
llm            = from_ollama("gemma4", think=False)

records = []
for text in your_texts:
    # Fan out: N LLM calls, one per label, in parallel
    extractions = await asyncio.gather(*[
        llm.extract(
            system=build_system_prompt(lp, all_labels=all_labels, base_template=base_template),
            user=text,
            response_model=single_label_schema(lp.label),
        )
        for lp in label_prompts
    ])
    # Merge: assemble the TrainingRecord
    entities = {
        lp.label: TaggedEntity(
            label=lp.label,
            confidence=ext.confidence,
            spans=[EntitySpan(text=s.text, start=s.start, end=s.end)
                   for s in find_spans(ext.matches, text, fuzzy=True)],
        )
        for lp, ext in zip(label_prompts, extractions, strict=True)
    }
    records.append(TrainingRecord(input_text=text, entities=entities))
```

That's the whole story. Add JSONL output and BIO conversion:

```python
from py_agent_ner import write_records_jsonl, jsonl_to_bio
write_records_jsonl(records, "training_data.jsonl")
jsonl_to_bio("training_data.jsonl", "training_data.bio.tsv")
```

## Public API

```python
from py_agent_ner import (
    # Pydantic v2 shapes (every field carries a description)
    LabelPrompt,          # input: label + instructions
    LabelExtraction,      # output of ONE call: label + matches + confidence
    EntitySpan,           # one located occurrence with character offsets
    TaggedEntity,         # one label's spans + confidence
    TrainingRecord,       # input text + per-label TaggedEntities (JSONL row)

    # Prompts — single-label rendering, matching the per-label fan-out
    DEFAULT_BASE_TEMPLATE,
    build_system_prompt,       # render ONE label's system prompt
    label_prompts_from_dict,   # {label: instructions} → [LabelPrompt]
    load_label_prompts,        # dir of .jinja → [LabelPrompt]
    labels_of,                 # convenience

    # Schema — 5-line helper around create_model + Literal[label]
    single_label_schema,       # name → type[LabelExtraction] with Literal[name]

    # BIO conversion (JSONL → CoNLL TSV for HuggingFace / spaCy / Flair)
    simple_tokenize,
    record_to_bio,
    jsonl_to_bio,
    records_to_bio,
    write_records_jsonl,
)
```

## What py-agent-ner intentionally does NOT provide

- **No `extract_training_data`-style runner.** You write the per-text loop.
- **No "merge" helper.** Building the `dict[label, TaggedEntity]` from N
  `LabelExtraction`s is five lines you write in your code.
- **No multi-label wrapper class.** The LLM returns `LabelExtraction` per
  call. There is no "Extraction containing entities for all labels" type.
- **No DAG plumbing inside the package.** Use `asyncio.gather` (cheapest),
  or wire your fan-out as steps on `py-agent-lib`'s `DagExecutor` if you
  want its observers / retries / cancellation / snapshots.

## Architecture

```
your run.py
    │   loops over texts
    │   fans out N LLM calls per text via asyncio.gather
    │   assembles TrainingRecord from N LabelExtractions
    │
    ▼
py-agent-ner
    models.py    LabelPrompt, LabelExtraction, EntitySpan, TaggedEntity, TrainingRecord
    prompts.py   build_system_prompt + DEFAULT_BASE_TEMPLATE + loaders
    schema.py    single_label_schema (Literal[label_name] subclass of LabelExtraction)
    bio.py       JSONL → CoNLL BIO TSV
    │
    ▼
py-agent-lib    (domain-agnostic — knows nothing about NER)
    adapters/llm        from_ollama, from_openai, from_anthropic, from_litellm
    adapters/spans      find_spans (strict + opt-in rapidfuzz)
    adapters/training   write_records_jsonl, read_records_jsonl (generic Pydantic JSONL)
    (also: DagExecutor, observers, snapshots, etc. — for users who want them)
```

## Training a model from the output

`jsonl_to_bio` converts the JSONL to BIO TSV — the canonical input for:

- **HuggingFace** — follow the [official token-classification tutorial](https://huggingface.co/docs/transformers/tasks/token_classification) verbatim.
- **spaCy** — `python -m spacy convert training_data.bio.tsv ./spacy-data/ --converter ner`
- **GLiNER, Flair, custom** — all consume BIO.

## Tests

```bash
uv run pytest packages/py-agent-ner/tests/
```

Tests don't need a live model — they use a `FakeLlm` that returns canned
`LabelExtraction` responses for the per-label calls.

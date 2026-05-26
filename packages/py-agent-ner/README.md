# py-agent-ner

NER training-data generation built on [py-agent-lib](https://github.com/gaslit-ai/py-agent/tree/main/packages/py-agent-lib).

Use an LLM to label text → write a JSONL with character-indexed spans →
convert to BIO TSV → fine-tune a small NER model.

## Install

```bash
pip install py-agent-ner
# or, for local dev:
pip install -e .
```

`py-agent-ner` pulls in `py-agent-lib[adapters]` (Pydantic + Jinja2 + instructor
+ httpx + rapidfuzz), so everything you need to run is one install away.

## Quickstart (no setup)

```python
import asyncio
from py_agent_lib.adapters.llm import from_ollama
from py_agent_ner import extract_training_data, jsonl_to_bio

LABELS = {
    "person_name":        "Full names. Ignore organizations and locations.",
    "location_reference": "Places, venues, addresses. Include 'at'/'in' if it's part of the phrase.",
    "time_reference":     "Times like 'tomorrow', '4:30 PM', 'next Monday'.",
}
TEXTS = ["Jordan Lee at City Library tomorrow 4:30 PM."]

async def main():
    llm = from_ollama("gemma4", think=False)
    records = await extract_training_data(
        texts=TEXTS,
        labels=LABELS,
        llm=llm,
        output_path="training_data.jsonl",
        fuzzy=True,
    )
    jsonl_to_bio("training_data.jsonl", "training_data.bio.tsv")
    return records

asyncio.run(main())
```

That's it. You get:
- `training_data.jsonl` — one Pydantic `TrainingRecord` per line, with character-indexed spans
- `training_data.bio.tsv` — CoNLL BIO format, ready to feed into HuggingFace token-classification or `spacy convert`

A more complete end-to-end demo lives at `examples/quickstart/`.

## Public API

```python
from py_agent_ner import (
    # Canonical Pydantic shapes
    LabelExtraction,    # the LLM's per-label response
    EntitySpan,         # one character-indexed span
    TaggedEntity,       # one label with confidence + spans
    TrainingRecord,     # one input text + all entities (JSONL row shape)

    # The fan-out-merge pipeline
    build_plan,           # build a Plan from a list of LabelPrompt
    make_handlers,        # build extract+merge handlers for one text
    extract_training_data,  # one-shot: run it, get records, write JSONL
    MERGE_STEP_ID,

    # BIO conversion for downstream training
    simple_tokenize,      # word + punctuation tokenizer with offsets
    record_to_bio,        # one TrainingRecord -> [(token, tag), ...]
    jsonl_to_bio,         # JSONL file -> CoNLL TSV file
    records_to_bio,       # in-memory records -> CoNLL TSV file

    # Prompts (re-exported from py-agent-lib for convenience)
    LabelPrompt,
    label_prompts_from_dict,
    load_label_prompts,
    labels_of,
    DEFAULT_BASE_TEMPLATE,
)
```

## Architecture

`py-agent-ner` is a thin opinionated layer over py-agent-lib:

```
your labeler
     │
     ▼
py-agent-ner ─── pipeline.py     (build_plan + make_handlers + extract_training_data)
     │           models.py       (LabelExtraction, EntitySpan, TaggedEntity, TrainingRecord)
     │           bio.py          (simple_tokenize, record_to_bio, jsonl_to_bio)
     │
     ▼
py-agent-lib ── adapters.llm     (Structured LLM clients: Ollama native, OpenAI, Anthropic, ...)
                adapters.prompts  (Jinja2 prompt loading + DEFAULT_BASE_TEMPLATE)
                adapters.spans    (strict + fuzzy span finding)
                adapters.training (JSONL read/write)
                DagExecutor       (fan-out scheduling, retries, observers, snapshots)
```

Everything in `py-agent-ner` is replaceable — if you want a different prompt
scaffold, write your own `LabelPrompt` builder. If you want a different span
strategy, use `find_spans` directly with your own `make_handlers`. If you want
a different output format, take the `list[TrainingRecord]` from
`extract_training_data` and write whatever you want.

## Training a model from the output

The JSONL output is the canonical intermediate format. To fine-tune a model:

### HuggingFace Transformers

```python
from py_agent_ner import jsonl_to_bio

# 1) Convert JSONL → BIO TSV
jsonl_to_bio("training_data.jsonl", "training_data.bio.tsv")

# 2) Load BIO TSV, tokenize with your target model's tokenizer, align labels
#    using `word_ids()`, then fine-tune with `Trainer` +
#    `DataCollatorForTokenClassification`. This is the official HuggingFace
#    token-classification tutorial verbatim:
#    https://huggingface.co/docs/transformers/tasks/token_classification
```

### spaCy

```bash
python -m spacy convert training_data.bio.tsv ./spacy-data/ --converter ner
python -m spacy init config config.cfg --lang en --pipeline ner
python -m spacy train config.cfg --paths.train ./spacy-data/training_data.bio.spacy
```

### GLiNER / Flair / custom

All consume BIO. Once `training_data.bio.tsv` exists, it's a solved problem.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

`FakeLlm` in `tests/test_pipeline.py` returns canned `LabelExtraction` responses,
so the suite runs in milliseconds without a live model.

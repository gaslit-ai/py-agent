# Quickstart

End-to-end NER training-data generation: per-label fan-out via local gemma4.

## Run

```bash
# from py-agent/examples/quickstart/
uv run python run.py
```

Outputs three files in this directory:

- `training_data.jsonl` — one `TrainingRecord` per line
- `training_data.bio.tsv` — CoNLL BIO TSV (ready for HuggingFace / spaCy)
- (no event log here — `run.py` doesn't wire observers; add one if you want)

## What this does

```
For each input text:
    asyncio.gather(
        extract_one_label(text, person_name),       # LLM call #1
        extract_one_label(text, location_reference),# LLM call #2
        extract_one_label(text, time_reference),    # LLM call #3
        extract_one_label(text, contact_handle),    # LLM call #4
    )                                               # N parallel calls
    assemble_training_record(text, extractions)     # locate spans + assemble
```

Each `extract_one_label` sends ONE LLM call with:
- A single-label system prompt (rendered from `prompts/base_system.jinja`)
- A response model where `label: Literal["<that_label>"]` — single-value enum
- The model returns `LabelExtraction(label, matches, confidence)` — its
  substring matches for that one label

After the gather, you have N `LabelExtraction`s. `assemble_training_record`
calls `find_spans` on each `matches` list and builds the `TrainingRecord`.

## Change anything

| To change... | Edit... |
|---|---|
| Which labels run | files under `labels/` (filename = label name, contents = instructions) |
| The single-label prompt scaffold | `prompts/base_system.jinja` |
| The Pydantic response model | the `single_label_schema` call in `run.py` (or replace it with your own inline `class Extract_X(LabelExtraction): label: Literal["x"] = Field(...)`) |
| Strict vs fuzzy span matching | `fuzzy=True` arg passed to `assemble_training_record` |
| LLM provider | swap `from_ollama` for `from_openai`, `from_anthropic`, or `from_litellm` (all from `py_agent_lib.adapters.llm`) |
| The merge logic | edit `assemble_training_record` in `run.py` — it's right there |

## After this runs — train a model

The `training_data.bio.tsv` is the canonical input format for:

- **HuggingFace Transformers** — see the official [token-classification tutorial](https://huggingface.co/docs/transformers/tasks/token_classification)
- **spaCy** — `python -m spacy convert training_data.bio.tsv ./spacy-data/ --converter ner`
- **GLiNER / Flair / custom** — all consume BIO

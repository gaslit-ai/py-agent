# Quickstart

Minimal NER labeler — 4 labels, 3 input texts, local gemma4 via Ollama.

## Run

```bash
# from py-agent-ner/examples/quickstart/
python run.py
```

Outputs three files in this directory:

- `training_data.jsonl` — one `TrainingRecord` per line (the JSONL training format)
- `training_data.bio.tsv` — same data as CoNLL BIO TSV (ready for HuggingFace/spaCy)
- `events.ndjson` — execution event log (per-step lifecycle, one event per line)

## Build your own labeler

This example is intentionally small. To label your own data:

1. Edit `LABELS` in `run.py` — names and one-line instructions per label
2. (Optional) Edit `prompts/base_system.jinja` to change the system-prompt scaffold
3. Swap the `from_ollama("gemma4", think=False)` line for whatever LLM you want:
   ```python
   from py_agent_lib.adapters.llm import from_openai, from_anthropic, from_litellm
   llm = from_openai("gpt-4.1-mini")
   ```
4. Feed in your own `TEXTS` (probably from a file)

The pipeline doesn't care about domain — same fan-out-merge shape regardless of
whether you're tagging entities, intents, sentiment, or whatever else fits the
"per-label structured extraction" mold.

## Alternative: labels from disk

If your label set grows, move instructions out of `LABELS` and into per-label
`.jinja` files:

```python
from py_agent_ner import load_label_prompts

label_prompts = load_label_prompts(
    here / "labels",
    base_template_path=here / "prompts" / "base_system.jinja",
)
records = await extract_training_data(texts=TEXTS, labels=label_prompts, llm=llm)
```

The `labels/` directory in this example shows the structure — one file per
label, file content = the instructions, filename (without `.jinja`) = the
label name.

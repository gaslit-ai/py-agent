"""Optional adapters that integrate py-agent-lib with the wider Python ecosystem.

py-agent-lib is domain-agnostic. These adapters are generic plumbing — there
is no NER, no labels, no prompts here. Build whatever you want on top.

Each submodule has its own optional dependency footprint — install only what
you actually need:

- `adapters.llm`              → `pip install py-agent-lib[llm]`
  Structured-output LLM clients (OpenAI/Ollama-native/Anthropic/LiteLLM via instructor).
- `adapters.spans`            → `pip install py-agent-lib[fuzzy]` for fuzzy mode
  Generic substring locator (strict by default, fuzzy fallback opt-in).
- `adapters.observers.jsonl`  → ships with the core
  Append-only NDJSON file observer for execution events.
- `adapters.observers.otel`   → `pip install py-agent-lib[otel]`
  Maps execution events to OpenTelemetry spans.
- `adapters.training.jsonl`   → ships with the core
  Generic Pydantic-model-per-line JSONL reader/writer.
- `adapters.telemetry`        → ships with the core
  Per-step input/output capture wrapper for handlers.
"""

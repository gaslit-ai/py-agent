"""Structured-output LLM adapters. See `base.StructuredLlm` for the protocol.

Two implementation strategies:

- `from_openai`, `from_anthropic`, `from_litellm` — via the `instructor` library
  with Pydantic-aware structured output and automatic retries on validation.
- `from_ollama` — direct call to Ollama's native `/api/chat` endpoint, using
  its native `format` parameter for model-level JSON-schema enforcement.
  Bypasses instructor entirely (more reliable for smaller local models).
"""
from __future__ import annotations

from .base import LlmError, StructuredLlm
from .instructor_impl import from_anthropic, from_litellm, from_openai
from .ollama_native import from_ollama

__all__ = [
    "LlmError",
    "StructuredLlm",
    "from_anthropic",
    "from_litellm",
    "from_ollama",
    "from_openai",
]

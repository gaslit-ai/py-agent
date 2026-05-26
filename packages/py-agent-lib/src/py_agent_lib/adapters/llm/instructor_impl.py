"""Concrete StructuredLlm implementations via the `instructor` library.

instructor provides Pydantic-aware structured output with automatic retries on
validation failure across OpenAI, Anthropic, Ollama (OpenAI-compat), and litellm.
This module is the only place that depends on it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from .base import LlmError

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseModel)


class _OpenAiStyleLlm:
    """For providers whose client exposes `client.chat.completions.create(...)`.

    Covers OpenAI, Azure OpenAI, and Ollama (via its OpenAI-compatible `/v1` endpoint).
    """

    def __init__(
        self,
        model: str,
        client: Any,
        default_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self._client = client
        self._defaults: dict[str, Any] = dict(default_kwargs or {})

    async def extract(
        self,
        *,
        response_model: type[T],
        system: str,
        user: str,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> T:
        merged: dict[str, Any] = {**self._defaults, **kwargs}
        try:
            return await self._client.chat.completions.create(
                model=self.model,
                response_model=response_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_retries=max_retries,
                **merged,
            )
        except Exception as e:
            raise LlmError(f"{type(self).__name__} call failed: {e}") from e


class _AnthropicStyleLlm:
    """For Anthropic, which uses `client.messages.create(...)` with a top-level `system`."""

    def __init__(
        self,
        model: str,
        client: Any,
        default_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self._client = client
        # Anthropic requires max_tokens; pick a reasonable default callers can override.
        self._defaults: dict[str, Any] = {"max_tokens": 4096, **(default_kwargs or {})}

    async def extract(
        self,
        *,
        response_model: type[T],
        system: str,
        user: str,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> T:
        merged: dict[str, Any] = {**self._defaults, **kwargs}
        try:
            return await self._client.messages.create(
                model=self.model,
                response_model=response_model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_retries=max_retries,
                **merged,
            )
        except Exception as e:
            raise LlmError(f"{type(self).__name__} call failed: {e}") from e


def _require(package: str, extras: str) -> None:
    try:
        __import__(package)
    except ImportError as e:
        raise ImportError(
            f"{package!r} is required for this adapter; "
            f"install with `pip install py-agent-lib[{extras}]`"
        ) from e


def from_openai(
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **defaults: Any,
) -> _OpenAiStyleLlm:
    """Create a structured-LLM client for OpenAI (or any OpenAI-compatible provider)."""
    _require("instructor", "llm")
    _require("openai", "llm")
    import instructor
    from openai import AsyncOpenAI

    client = instructor.from_openai(AsyncOpenAI(api_key=api_key, base_url=base_url))
    return _OpenAiStyleLlm(model, client, defaults)


def from_anthropic(
    model: str,
    *,
    api_key: str | None = None,
    **defaults: Any,
) -> _AnthropicStyleLlm:
    """Create a structured-LLM client for Anthropic models."""
    _require("instructor", "anthropic")
    _require("anthropic", "anthropic")
    import instructor
    from anthropic import AsyncAnthropic

    client = instructor.from_anthropic(AsyncAnthropic(api_key=api_key))
    return _AnthropicStyleLlm(model, client, defaults)


def from_litellm(model: str, **defaults: Any) -> _OpenAiStyleLlm:
    """Create a structured-LLM client backed by litellm's universal router.

    Pass a litellm-style model identifier, e.g. "gpt-4o", "claude-3-5-sonnet",
    "ollama/gemma4", "groq/llama-3.1-70b-versatile".
    """
    _require("instructor", "litellm")
    _require("litellm", "litellm")
    import instructor
    import litellm

    client = instructor.from_litellm(litellm.acompletion)
    return _OpenAiStyleLlm(model, client, defaults)


__all__ = [
    "from_anthropic",
    "from_litellm",
    "from_openai",
]

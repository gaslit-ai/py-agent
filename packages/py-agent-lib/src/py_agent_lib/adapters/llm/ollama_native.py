"""Native Ollama client — calls `/api/chat` directly with `format=<json_schema>`.

This bypasses instructor and the OpenAI-compat shim. Ollama enforces the JSON
schema at the model level via its native `format` parameter, which works
reliably across all supported models — including smaller ones (gemma4,
llama3.2) that mishandle the schema-injection-into-system-prompt pattern that
instructor's `Mode.JSON` uses against the `/v1` endpoint.

Since we don't go through instructor, we re-implement the "retry on Pydantic
validation failure" loop here. On failure we append the error back into the
chat history as a user turn so the model gets a chance to self-correct.
"""
from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .base import LlmError

T = TypeVar("T", bound=BaseModel)


# Ollama's /api/chat treats these as top-level request fields; everything
# else lives inside the `options` sub-object.
_TOP_LEVEL_FIELDS = frozenset({"think", "keep_alive", "stream", "tools"})


class _OllamaNativeLlm:
    """Async client for Ollama's native `/api/chat` with structured-output enforcement."""

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        default_kwargs: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self._host = host.rstrip("/")
        self._defaults: dict[str, Any] = dict(default_kwargs or {})
        self._timeout = timeout
        self._client: Any = None  # lazy httpx.AsyncClient

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Optional — Python will GC it otherwise."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def extract(
        self,
        *,
        response_model: type[T],
        system: str,
        user: str,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> T:
        schema = response_model.model_json_schema()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        attempts = max_retries + 1
        last_error: ValidationError | None = None

        for attempt in range(attempts):
            try:
                content = await self._call(messages, schema, kwargs)
            except Exception as e:
                import httpx

                if isinstance(e, httpx.HTTPError):
                    raise LlmError(f"ollama HTTP call failed: {e}") from e
                raise LlmError(f"ollama call failed: {e}") from e

            try:
                return response_model.model_validate_json(content)
            except ValidationError as e:
                last_error = e
                if attempt == attempts - 1:
                    break
                # Feed the failure back to the model for the next attempt.
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "That response failed JSON-schema validation:\n"
                            f"{e}\n\n"
                            "Return ONLY a JSON object matching the schema exactly. "
                            "Do not include schema metadata fields like 'properties', "
                            "'description', or 'type'."
                        ),
                    },
                ]

        raise LlmError(
            f"ollama response failed validation after {attempts} attempt(s): {last_error}"
        ) from last_error

    async def _call(
        self,
        messages: list[dict[str, Any]],
        format_schema: dict[str, Any],
        extra_kwargs: dict[str, Any],
    ) -> str:
        merged: dict[str, Any] = {**self._defaults, **extra_kwargs}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": format_schema,
        }
        options: dict[str, Any] = {}
        for key, value in merged.items():
            if key in _TOP_LEVEL_FIELDS:
                body[key] = value
            else:
                options[key] = value
        if options:
            body["options"] = options

        resp = await self._http().post(f"{self._host}/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


def from_ollama(
    model: str,
    *,
    host: str = "http://localhost:11434",
    timeout: float = 120.0,
    **defaults: Any,
) -> _OllamaNativeLlm:
    """Create a structured-LLM client backed by Ollama's native `/api/chat`.

    Uses Ollama's native `format` parameter (JSON-schema enforcement at the model
    level) rather than the OpenAI-compat shim. This works reliably across all
    Ollama-supported models, including smaller ones (gemma4, llama3.2) that
    mishandle instructor's schema-into-prompt pattern.

    Defaults you pass as keyword args after `host`/`timeout` are applied to every
    call. Known Ollama top-level fields (`think`, `keep_alive`, `stream`,
    `tools`) go at the request top level; everything else (temperature, top_p,
    num_ctx, repeat_penalty, ...) goes into the `options` sub-object per
    Ollama's API.

    Example:
        llm = from_ollama("gemma4", think=False, temperature=0.0)
    """
    try:
        import httpx  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "httpx is required for the Ollama adapter; "
            "install with `pip install py-agent-lib[llm]`"
        ) from e
    return _OllamaNativeLlm(model, host=host, default_kwargs=defaults, timeout=timeout)


__all__ = ["from_ollama"]

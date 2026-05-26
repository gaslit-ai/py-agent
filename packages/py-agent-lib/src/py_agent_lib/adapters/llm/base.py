"""StructuredLlm protocol — async call returning a validated Pydantic instance."""
from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LlmError(RuntimeError):
    """Raised by an adapter when an underlying LLM call fails after retries.

    The original exception is preserved as `__cause__`.
    """


@runtime_checkable
class StructuredLlm(Protocol):
    """Provider-agnostic async client returning a Pydantic-validated response.

    Implementations are expected to:
    - Send `system` and `user` as a chat-style two-turn prompt.
    - Constrain the model's response to `response_model`'s JSON schema.
    - Retry on Pydantic validation failure (commonly via instructor's `max_retries`).
    - Raise `LlmError` (or a subclass) on terminal failure.
    """

    model: str

    async def extract(
        self,
        *,
        response_model: type[T],
        system: str,
        user: str,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> T: ...


__all__ = ["LlmError", "StructuredLlm", "T"]

"""Retry + timeout wrappers for step handlers. Built on `tenacity` + `asyncio.timeout`."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .types import ErrorInfo, StepResult, StepStatus, utc_now

if TYPE_CHECKING:
    from .context import StepContext
    from .executor import StepHandler
    from .types import Step


def with_timeout(handler: StepHandler, seconds: float) -> StepHandler:
    """Wrap a handler so it fails with a TimeoutError after `seconds`."""

    async def wrapped(step: Step, ctx: StepContext) -> StepResult:
        try:
            async with asyncio.timeout(seconds):
                return await handler(step, ctx)
        except TimeoutError:
            return StepResult(
                step_id=step.id,
                status=StepStatus.FAILED,
                error=ErrorInfo(
                    message=f"step exceeded timeout of {seconds}s",
                    type="TimeoutError",
                ),
                finished_at=utc_now(),
            )

    return wrapped


def with_retry(
    handler: StepHandler,
    *,
    retries: int = 2,
    base_delay: float = 0.25,
    max_delay: float = 10.0,
    retry_on: type[BaseException] | tuple[type[BaseException], ...] = Exception,
) -> StepHandler:
    """Wrap a handler with exponential-backoff retry on raising exceptions.

    Notes:
    - A handler that *returns* a failed StepResult is NOT retried; only raised
      exceptions are retried. This mirrors the TS lib's contract.
    - `retries` is the number of retries after the first attempt — total attempts = retries + 1.
    """

    async def wrapped(step: Step, ctx: StepContext) -> StepResult:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(retries + 1),
            wait=wait_exponential_jitter(initial=base_delay, max=max_delay),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await handler(step, ctx)
        except RetryError as e:  # pragma: no cover — reraise=True bypasses this normally
            raise e
        raise RuntimeError("unreachable: retry loop exited without returning")  # safety

    return wrapped


__all__ = ["with_retry", "with_timeout"]

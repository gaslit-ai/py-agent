"""Observer composition + buffered observer. For OTel/structlog adapters live elsewhere."""
from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .events import ExecutionEvent

logger = logging.getLogger(__name__)


@runtime_checkable
class ExecutionObserver(Protocol):
    """Implementations receive every execution event. Sync or async."""

    def on_event(self, event: ExecutionEvent) -> None | Awaitable[None]: ...


ObserverErrorHandler = Callable[[ExecutionObserver, ExecutionEvent, BaseException], None]


class ObserverFailurePolicy(StrEnum):
    STRICT = "strict"  # raise on observer error (stops execution)
    LENIENT = "lenient"  # log and continue


def compose_observers(
    observers: Iterable[ExecutionObserver],
    *,
    failure_policy: ObserverFailurePolicy = ObserverFailurePolicy.LENIENT,
    on_error: ObserverErrorHandler | None = None,
) -> Callable[[ExecutionEvent], Awaitable[None]]:
    """Sequentially run observers. Use when ordering matters."""
    observer_list = list(observers)

    async def emit(event: ExecutionEvent) -> None:
        for obs in observer_list:
            try:
                result = obs.on_event(event)
                if asyncio.iscoroutine(result):
                    await result
            except BaseException as e:
                if on_error is not None:
                    on_error(obs, event, e)
                if failure_policy == ObserverFailurePolicy.STRICT:
                    raise
                logger.warning("observer %r failed on event %s: %s", obs, event.type, e)

    return emit


def compose_observers_parallel(
    observers: Iterable[ExecutionObserver],
    *,
    failure_policy: ObserverFailurePolicy = ObserverFailurePolicy.LENIENT,
    on_error: ObserverErrorHandler | None = None,
) -> Callable[[ExecutionEvent], Awaitable[None]]:
    """Fan out an event to all observers concurrently."""
    observer_list = list(observers)

    async def emit(event: ExecutionEvent) -> None:
        async def _one(obs: ExecutionObserver) -> None:
            try:
                result = obs.on_event(event)
                if asyncio.iscoroutine(result):
                    await result
            except BaseException as e:
                if on_error is not None:
                    on_error(obs, event, e)
                if failure_policy == ObserverFailurePolicy.STRICT:
                    raise
                logger.warning("observer %r failed on event %s: %s", obs, event.type, e)

        async with asyncio.TaskGroup() as tg:
            for obs in observer_list:
                tg.create_task(_one(obs))

    return emit


class OverflowPolicy(StrEnum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"


class BufferedObserver:
    """Base class for observers that should process events off the executor's hot path.

    Subclass and override `process()`. Call `start()` before use and `stop()` to drain.
    """

    def __init__(
        self,
        *,
        max_queue_size: int = 1000,
        overflow_policy: OverflowPolicy = OverflowPolicy.BLOCK,
    ) -> None:
        self._queue: asyncio.Queue[ExecutionEvent | None] = asyncio.Queue(max_queue_size)
        self._overflow_policy = overflow_policy
        self._worker: asyncio.Task[None] | None = None
        # Keep refs to fire-and-forget enqueue tasks so they aren't GC'd mid-await.
        self._pending_puts: set[asyncio.Task[None]] = set()

    @abstractmethod
    async def process(self, event: ExecutionEvent) -> None:
        """Subclass: handle one event. Slow work belongs here."""
        raise NotImplementedError

    async def start(self) -> None:
        if self._worker is not None:
            return
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is None:
            return
        await self._queue.put(None)
        await self._worker
        self._worker = None

    def on_event(self, event: ExecutionEvent) -> None:
        if self._overflow_policy == OverflowPolicy.BLOCK:
            task = asyncio.create_task(self._queue.put(event))
            self._pending_puts.add(task)
            task.add_done_callback(self._pending_puts.discard)
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            if self._overflow_policy == OverflowPolicy.DROP_NEWEST:
                logger.warning("observer queue full; dropping newest event %s", event.type)
                return
            # DROP_OLDEST
            try:
                _ = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                logger.warning("observer queue full; dropped oldest event")
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover — race
                logger.warning("observer queue still full after drop; losing event %s", event.type)

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            try:
                await self.process(event)
            except BaseException as e:
                logger.exception("buffered observer process() failed: %s", e)

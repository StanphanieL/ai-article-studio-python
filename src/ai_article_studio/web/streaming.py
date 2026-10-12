"""SSE responses that own the lifetime of their synchronous content iterator."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import anyio
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


class ClosingStreamingResponse(StreamingResponse):
    """Close synchronous content explicitly, including on client disconnect."""

    def __init__(
        self,
        content: Iterator[str],
        on_unstarted_close: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        self._content = content
        self._on_unstarted_close = on_unstarted_close
        self._started = False
        self._closed = False
        self._tracked_content = self._iterate_content()
        super().__init__(self._tracked_content, **kwargs)

    def _iterate_content(self) -> Iterator[str]:
        self._started = True
        while True:
            try:
                chunk = next(self._content)
            except StopIteration:
                return
            yield chunk

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(self._close_content)

    def _close_content(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._content, "close", None)
            if close is not None:
                close()
        finally:
            self._tracked_content.close()
            if not self._started and self._on_unstarted_close is not None:
                self._on_unstarted_close()

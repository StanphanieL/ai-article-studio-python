from __future__ import annotations

from collections.abc import Iterator
from threading import get_ident

import anyio
import pytest
from starlette.requests import ClientDisconnect
from starlette.types import Message, Scope

from ai_article_studio.web.streaming import ClosingStreamingResponse


def _scope(spec_version: str) -> Scope:
    return {"type": "http", "asgi": {"spec_version": spec_version}}


async def _drive_response(response: ClosingStreamingResponse, spec_version: str, disconnect_at: str | None) -> list[Message]:
    messages: list[Message] = []
    disconnected = anyio.Event()

    async def send(message: Message) -> None:
        messages.append(message)
        is_header = message["type"] == "http.response.start"
        body = message.get("body", b"")
        should_disconnect = (disconnect_at == "headers" and is_header) or (
            not is_header and disconnect_at is not None and f"event: {disconnect_at}".encode() in body
        )
        if should_disconnect:
            if spec_version == "2.4":
                raise OSError("client disconnected")
            disconnected.set()
            await anyio.sleep_forever()

    async def receive() -> Message:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    with anyio.fail_after(2):
        if spec_version == "2.4" and disconnect_at is not None:
            with pytest.raises(ClientDisconnect):
                await response(_scope(spec_version), receive, send)
        else:
            await response(_scope(spec_version), receive, send)
    return messages


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_disconnect_closes_original_content_before_response_returns(spec_version: str) -> None:
    closed_on: list[int] = []
    consumed_after_disconnect: list[bool] = []

    def content() -> Iterator[str]:
        try:
            yield "event: content\ndata: partial\n\n"
            consumed_after_disconnect.append(True)
            yield "event: done\ndata: saved\n\n"
        finally:
            closed_on.append(get_ident())

    original = content()
    response = ClosingStreamingResponse(original, media_type="text/event-stream")
    anyio.run(_drive_response, response, spec_version, "content")

    assert len(closed_on) == 1
    assert closed_on[0] != get_ident()
    assert not consumed_after_disconnect


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_disconnect_on_done_closes_without_consuming_past_terminal_event(spec_version: str) -> None:
    state: list[str] = []

    def content() -> Iterator[str]:
        try:
            yield "event: content\ndata: complete\n\n"
            state.append("saved")
            yield "event: done\ndata: saved\n\n"
            state.append("advanced past done")
        finally:
            state.append("closed")

    original = content()
    response = ClosingStreamingResponse(original, media_type="text/event-stream")
    anyio.run(_drive_response, response, spec_version, "done")

    assert state == ["saved", "closed"]


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_header_disconnect_cleans_reservation_when_content_was_never_started(spec_version: str) -> None:
    state: list[str] = ["reserved"]
    cleanup_threads: list[int] = []

    def content() -> Iterator[str]:
        state.append("started")
        try:
            yield "event: content\ndata: unused\n\n"
        finally:
            state.append("generator closed")

    def release_unstarted() -> None:
        cleanup_threads.append(get_ident())
        state.append("reservation released")

    original = content()
    response = ClosingStreamingResponse(original, on_unstarted_close=release_unstarted)
    anyio.run(_drive_response, response, spec_version, "headers")

    assert state == ["reserved", "reservation released"]
    assert len(cleanup_threads) == 1
    assert cleanup_threads[0] != get_ident()


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_natural_completion_does_not_run_unstarted_cleanup(spec_version: str) -> None:
    state: list[str] = []

    def content() -> Iterator[str]:
        try:
            yield "event: done\ndata: saved\n\n"
        finally:
            state.append("closed")

    response = ClosingStreamingResponse(content(), on_unstarted_close=lambda: state.append("unstarted cleanup"))
    messages = anyio.run(_drive_response, response, spec_version, None)

    assert state == ["closed"]
    assert messages[-1] == {"type": "http.response.body", "body": b"", "more_body": False}


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_cancellation_cannot_skip_content_cleanup(spec_version: str) -> None:
    state: list[str] = []

    def content() -> Iterator[str]:
        try:
            yield "event: content\ndata: partial\n\n"
            state.append("unexpected continuation")
        finally:
            state.append("closed")

    original = content()
    response = ClosingStreamingResponse(original, on_unstarted_close=lambda: state.append("unstarted cleanup"))

    async def run() -> None:
        async def receive() -> Message:
            await anyio.sleep_forever()
            raise AssertionError("receive should be cancelled")

        with anyio.fail_after(2):
            with anyio.CancelScope() as cancellation:

                async def send(message: Message) -> None:
                    if message["type"] == "http.response.body":
                        cancellation.cancel()
                        await anyio.sleep(0)

                await response(_scope(spec_version), receive, send)

    anyio.run(run)

    assert state == ["closed"]

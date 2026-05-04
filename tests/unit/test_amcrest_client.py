"""Tests for cat_watcher.amcrest_client."""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003  # runtime import: respx.mock calls inspect.getfullargspec, which forces annotation evaluation
from typing import TYPE_CHECKING, override
from zoneinfo import ZoneInfo

import httpx
import pytest
import respx
from pydantic import SecretStr

if TYPE_CHECKING:
    from collections.abc import Iterator

from cat_watcher.amcrest_client import (
    AmcrestClient,
    CameraAPIError,
    CameraAuthError,
    CameraUnreachableError,
    Recording,
)
from cat_watcher.config import CameraConfig, CameraSecrets

_BASE_URL = "http://cam.example.com:80"
_FIND_URL = f"{_BASE_URL}/cgi-bin/mediaFileFind.cgi"
_FIND_HANDLE = "99"
_FIND_HANDLE_INT = 99
_UTC_ZONE = ZoneInfo("UTC")


def _camera(*, host: str = "cam.example.com", port: int = 80) -> CameraConfig:
    return CameraConfig(name="pantry", display_name="Pantry", host=host, port=port, timezone=None)


def _secrets() -> CameraSecrets:
    return CameraSecrets(username="u", password=SecretStr("p"))


def _make_client(
    *,
    camera_tz: ZoneInfo = _UTC_ZONE,
    retry_attempts: int = 2,
    retry_delay_seconds: float = 0.0,
) -> AmcrestClient:
    """Test-tuned client: short retries, no inter-attempt sleep."""
    return AmcrestClient(
        _camera(),
        _secrets(),
        camera_tz=camera_tz,
        retry_attempts=retry_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )


def _factory_create(result: int = _FIND_HANDLE_INT) -> respx.Route:
    return respx.get(_FIND_URL, params={"action": "factory.create"}).mock(
        return_value=httpx.Response(200, text=f"result={result}\r\n"),
    )


def _find_file_route() -> respx.Route:
    return respx.get(_FIND_URL, params__contains={"action": "findFile", "object": _FIND_HANDLE}).mock(
        return_value=httpx.Response(200, text="OK\r\n"),
    )


def _find_next_file_route() -> respx.Route:
    return respx.get(_FIND_URL, params={"action": "findNextFile", "object": _FIND_HANDLE, "count": "100"})


def _close_route() -> respx.Route:
    return respx.get(_FIND_URL, params={"action": "close", "object": _FIND_HANDLE}).mock(
        return_value=httpx.Response(200, text="OK\r\n"),
    )


def _destroy_route() -> respx.Route:
    return respx.get(_FIND_URL, params={"action": "destroy", "object": _FIND_HANDLE}).mock(
        return_value=httpx.Response(200, text="OK\r\n"),
    )


_PAGE_TWO_ITEMS = (
    "found=2\r\n"
    "items[0].FilePath=/mnt/sd/2026-05-01/001/dav/06/06.47.04-06.48.58[M][0@0][0].mp4\r\n"
    "items[0].StartTime=2026-05-01 06:47:04\r\n"
    "items[0].EndTime=2026-05-01 06:48:58\r\n"
    "items[0].Length=1234567\r\n"
    "items[1].FilePath=/mnt/sd/2026-05-01/001/dav/07/07.10.00-07.11.00[M][0@0][0].mp4\r\n"
    "items[1].StartTime=2026-05-01 07:10:00\r\n"
    "items[1].EndTime=2026-05-01 07:11:00\r\n"
    "items[1].Length=2345\r\n"
)
_PAGE_EMPTY = "found=0\r\n"


@respx.mock
def test_iter_recordings_short_page_ends_pagination() -> None:
    """Two records (< page size) on page 1: loop exits after one fetch, handle destroyed."""
    factory = _factory_create()
    find_file = _find_file_route()
    page = _find_next_file_route().mock(return_value=httpx.Response(200, text=_PAGE_TWO_ITEMS))
    close = _close_route()
    destroy = _destroy_route()

    client = _make_client(camera_tz=ZoneInfo("America/New_York"))
    items = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC), until=datetime(2026, 5, 2, tzinfo=UTC)))

    assert factory.called
    assert find_file.called
    assert page.call_count == 1
    assert close.called
    assert destroy.called
    assert len(items) == 2
    assert isinstance(items[0], Recording)
    assert items[0].source_filename == "06.47.04-06.48.58[M][0@0][0].mp4"
    assert items[0].camera_path == "/mnt/sd/2026-05-01/001/dav/06/06.47.04-06.48.58[M][0@0][0].mp4"
    assert items[0].file_size_bytes == 1234567
    # America/New_York is UTC-04:00 in May (EDT), so 06:47:04 local -> 10:47:04 UTC.
    assert items[0].start_ts == datetime(2026, 5, 1, 10, 47, 4, tzinfo=UTC)
    assert items[0].end_ts == datetime(2026, 5, 1, 10, 48, 58, tzinfo=UTC)


@respx.mock
def test_iter_recordings_full_page_triggers_next_fetch() -> None:
    """A page with exactly _FIND_PAGE_SIZE rows triggers a follow-up call; ``found=0`` ends it."""
    full_page_lines = ["found=100\r\n"]
    for i in range(100):
        full_page_lines.append(f"items[{i}].FilePath=/c/{i:03d}.mp4\r\n")
        full_page_lines.append(f"items[{i}].StartTime=2026-05-01 06:00:{i % 60:02d}\r\n")
        full_page_lines.append(f"items[{i}].EndTime=2026-05-01 06:01:{i % 60:02d}\r\n")
        full_page_lines.append(f"items[{i}].Length={i}\r\n")
    full_page = "".join(full_page_lines)

    _ = _factory_create()
    _ = _find_file_route()
    page = _find_next_file_route().mock(
        side_effect=[
            httpx.Response(200, text=full_page),
            httpx.Response(200, text=_PAGE_EMPTY),
        ],
    )
    _ = _close_route()
    _ = _destroy_route()

    client = _make_client()
    items = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))

    assert page.call_count == 2
    assert len(items) == 100


@respx.mock
def test_iter_recordings_sends_channel_one_in_findfile_request() -> None:
    """``condition.Channel=1`` must appear in the findFile request (PDF says channels are 1-indexed).

    Guards against regressing the channel value to ``0`` (the original implementation choice that
    was caught by reading the Amcrest spec).
    """
    _ = _factory_create()
    find_file = respx.get(
        _FIND_URL,
        params={"action": "findFile", "object": _FIND_HANDLE, "condition.Channel": "1"},
    ).mock(return_value=httpx.Response(200, text="OK\r\n"))
    _ = _find_next_file_route().mock(return_value=httpx.Response(200, text=_PAGE_EMPTY))
    _ = _close_route()
    _ = _destroy_route()

    client = _make_client()
    _ = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))

    assert find_file.called, "findFile must be called with condition.Channel=1"


def test_iter_recordings_naive_since_raises_value_error() -> None:
    """``since`` must be tz-aware. A naive datetime is rejected up-front (no HTTP traffic).

    Without this guard, ``datetime.astimezone`` silently treats naive values as system-local time
    and produces wrong window bounds on any non-UTC host.
    """
    client = _make_client()
    naive = datetime(2026, 4, 30, 12, 0, 0)  # noqa: DTZ001  # intentionally tz-naive for the precondition
    with pytest.raises(ValueError, match="naive datetimes rejected"):
        _ = list(client.iter_recordings(since=naive))


@respx.mock
def test_iter_recordings_handles_missing_length_field() -> None:
    """A row with no ``Length`` field parses cleanly to ``file_size_bytes=0`` (defensive default)."""
    page_no_length = (
        "found=1\r\n"  # dprint-ignore
        "items[0].FilePath=/clip.mp4\r\n"
        "items[0].StartTime=2026-05-01 06:00:00\r\n"
        "items[0].EndTime=2026-05-01 06:01:00\r\n"
        # No Length line.
    )
    _ = _factory_create()
    _ = _find_file_route()
    _ = _find_next_file_route().mock(return_value=httpx.Response(200, text=page_no_length))
    _ = _close_route()
    _ = _destroy_route()

    client = _make_client()
    items = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))

    assert len(items) == 1
    assert items[0].file_size_bytes == 0


@respx.mock
def test_iter_recordings_destroys_handle_on_exception() -> None:
    """If a page fetch raises mid-iteration, the search handle is still destroyed (try/finally)."""
    _ = _factory_create()
    _ = _find_file_route()
    _ = _find_next_file_route().mock(side_effect=httpx.ReadTimeout("page hung"))
    _ = _close_route()
    destroy = _destroy_route()

    client = _make_client(retry_attempts=1)
    with pytest.raises(CameraUnreachableError):
        _ = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))

    assert destroy.called


@respx.mock
def test_iter_recordings_retries_then_succeeds_on_connect_error() -> None:
    """Two ConnectErrors on the first transient call followed by a clean response succeeds."""
    factory = respx.get(_FIND_URL, params={"action": "factory.create"}).mock(
        side_effect=[
            httpx.ConnectError("net fail"),
            httpx.ConnectError("net fail again"),
            httpx.Response(200, text=f"result={_FIND_HANDLE}\r\n"),
        ],
    )
    _ = _find_file_route()
    _ = _find_next_file_route().mock(return_value=httpx.Response(200, text=_PAGE_EMPTY))
    _ = _close_route()
    _ = _destroy_route()

    client = _make_client(retry_attempts=3)
    items = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))

    assert factory.call_count == 3
    assert not items


@respx.mock
def test_iter_recordings_exhausted_raises_unreachable() -> None:
    """All retry attempts raise transient errors -> CameraUnreachableError."""
    route = respx.get(_FIND_URL, params={"action": "factory.create"}).mock(side_effect=httpx.ConnectError("dead"))

    client = _make_client(retry_attempts=3)
    with pytest.raises(CameraUnreachableError):
        _ = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))

    assert route.call_count == 3


@pytest.mark.parametrize("status", [401, 403])
@respx.mock
def test_iter_recordings_auth_status_raises_auth_error_without_retry(status: int) -> None:
    """Both 401 and 403 surface as CameraAuthError after a single attempt (4xx is not retryable)."""
    route = respx.get(_FIND_URL, params={"action": "factory.create"}).mock(return_value=httpx.Response(status))

    client = _make_client(retry_attempts=3)
    with pytest.raises(CameraAuthError):
        _ = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))

    assert route.call_count == 1


@respx.mock
def test_iter_recordings_other_4xx_raises_api_error_without_retry() -> None:
    """A 404 raises CameraAPIError, not CameraAuthError, and is not retried."""
    route = respx.get(_FIND_URL, params={"action": "factory.create"}).mock(return_value=httpx.Response(404))

    client = _make_client(retry_attempts=3)
    with pytest.raises(CameraAPIError):
        _ = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))

    assert route.call_count == 1


@respx.mock
def test_iter_recordings_5xx_is_retried() -> None:
    """A 5xx response is retried (treated like a transient network failure)."""
    factory = respx.get(_FIND_URL, params={"action": "factory.create"}).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, text=f"result={_FIND_HANDLE}\r\n"),
        ],
    )
    _ = _find_file_route()
    _ = _find_next_file_route().mock(return_value=httpx.Response(200, text=_PAGE_EMPTY))
    _ = _close_route()
    _ = _destroy_route()

    client = _make_client(retry_attempts=3)
    _ = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))

    assert factory.call_count == 2


# --- download_recording ----------------------------------------------------

_DOWNLOAD_URL = f"{_BASE_URL}/cgi-bin/RPC_Loadfile/mnt/sd/2026-05-01/001/dav/06/clip.mp4"
_CAMERA_PATH = "/mnt/sd/2026-05-01/001/dav/06/clip.mp4"


@respx.mock
def test_download_recording_writes_atomic_with_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful download: bytes land under final name; .part is gone; fsync precedes the rename."""
    payload = b"\x00\x01\x02" * 4096
    _ = respx.get(_DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=payload))

    real_fsync = os.fsync
    real_replace = os.replace
    order: list[str] = []

    def recording_fsync(fd: int) -> None:
        order.append("fsync")
        real_fsync(fd)

    def recording_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        order.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr("cat_watcher.amcrest_client.os.fsync", recording_fsync)
    monkeypatch.setattr("cat_watcher.amcrest_client.os.replace", recording_replace)

    dest = tmp_path / "clip.mp4"
    client = _make_client()
    client.download_recording(_CAMERA_PATH, dest=dest)

    assert dest.read_bytes() == payload
    assert not dest.with_suffix(".mp4.part").exists()
    assert order == ["fsync", "replace"], f"fsync must precede rename, got {order}"


_PARTIAL_CHUNK = b"x" * (128 * 1024)  # > 64 KiB so httpx.iter_bytes flushes at least one chunk to disk


class _FlakyStream(httpx.SyncByteStream):
    """Yields one chunk worth of bytes, then raises ``ReadTimeout`` mid-iteration."""

    @override
    def __iter__(self) -> Iterator[bytes]:
        yield _PARTIAL_CHUNK
        msg = "hung mid-stream"
        raise httpx.ReadTimeout(msg)

    @override
    def close(self) -> None:
        return None


@respx.mock
def test_download_recording_leaves_part_file_on_mid_stream_failure(tmp_path: Path) -> None:
    """A mid-stream ReadTimeout leaves the .part file (with whatever made it to disk).

    The final filename is not created. A sweep / retention pass detects the partial via the
    ``.part`` suffix.
    """
    _ = respx.get(_DOWNLOAD_URL).mock(return_value=httpx.Response(200, stream=_FlakyStream()))

    dest = tmp_path / "clip.mp4"
    part = dest.with_name("clip.mp4.part")
    # retry_attempts=1: a second attempt would truncate (O_TRUNC) the partial bytes from the first
    # before failing the same way (respx returns the same Response, whose stream is exhausted).
    client = _make_client(retry_attempts=1)

    with pytest.raises(CameraUnreachableError):
        client.download_recording(_CAMERA_PATH, dest=dest)

    assert not dest.exists()
    assert part.exists(), "partial download must remain so a sweep can detect it"
    assert part.stat().st_size > 0, "expected at least one chunk flushed before the timeout"


@respx.mock
def test_download_recording_401_raises_auth_error_without_retry(tmp_path: Path) -> None:
    """401 during download raises CameraAuthError immediately; no .part file remains."""
    route = respx.get(_DOWNLOAD_URL).mock(return_value=httpx.Response(401))

    dest = tmp_path / "clip.mp4"
    client = _make_client(retry_attempts=3)
    with pytest.raises(CameraAuthError):
        client.download_recording(_CAMERA_PATH, dest=dest)

    assert route.call_count == 1
    assert not dest.exists()


@respx.mock
def test_download_recording_other_4xx_raises_api_error_without_retry(tmp_path: Path) -> None:
    """404 during download raises CameraAPIError immediately; no .part file remains."""
    route = respx.get(_DOWNLOAD_URL).mock(return_value=httpx.Response(404))

    dest = tmp_path / "clip.mp4"
    client = _make_client(retry_attempts=3)
    with pytest.raises(CameraAPIError):
        client.download_recording(_CAMERA_PATH, dest=dest)

    assert route.call_count == 1
    assert not dest.exists()


@respx.mock
def test_download_recording_retries_then_succeeds(tmp_path: Path) -> None:
    """A transient ConnectError followed by a 200 succeeds and produces the final file."""
    payload = b"hello"
    _ = respx.get(_DOWNLOAD_URL).mock(
        side_effect=[
            httpx.ConnectError("dead"),
            httpx.Response(200, content=payload),
        ],
    )

    dest = tmp_path / "clip.mp4"
    client = _make_client(retry_attempts=3)
    client.download_recording(_CAMERA_PATH, dest=dest)

    assert dest.read_bytes() == payload


@respx.mock
def test_download_recording_5xx_is_retried(tmp_path: Path) -> None:
    """A 5xx response during download is retried (matches iter_recordings semantics)."""
    payload = b"hello"
    route = respx.get(_DOWNLOAD_URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, content=payload),
        ],
    )

    dest = tmp_path / "clip.mp4"
    client = _make_client(retry_attempts=3)
    client.download_recording(_CAMERA_PATH, dest=dest)

    assert route.call_count == 2
    assert dest.read_bytes() == payload


@respx.mock
def test_iter_recordings_malformed_factory_create_raises_api_error() -> None:
    """A 200 response from factory.create with no ``result=`` line raises CameraAPIError."""
    _ = respx.get(_FIND_URL, params={"action": "factory.create"}).mock(
        return_value=httpx.Response(200, text="garbage with no result line\r\n"),
    )

    client = _make_client()
    with pytest.raises(CameraAPIError, match=r"factory\.create returned no result handle"):
        _ = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))


@respx.mock
def test_iter_recordings_destroy_failure_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    """If close/destroy itself fails, the cleanup logs a warning and the original flow completes."""
    _ = _factory_create()
    _ = _find_file_route()
    _ = _find_next_file_route().mock(return_value=httpx.Response(200, text=_PAGE_EMPTY))
    # close + destroy both raise — cleanup must swallow the error so the iter_recordings caller
    # still gets the empty result list rather than seeing a finally-block exception.
    _ = respx.get(_FIND_URL, params={"action": "close", "object": _FIND_HANDLE}).mock(side_effect=httpx.ConnectError("close failed"))
    _ = respx.get(_FIND_URL, params={"action": "destroy", "object": _FIND_HANDLE}).mock(side_effect=httpx.ConnectError("destroy failed"))

    # alembic's env.py invokes ``logging.config.fileConfig`` during the integration test suite,
    # which by default disables every existing logger including ours. Re-enable so caplog sees
    # records emitted from this module under any test ordering.
    logging.getLogger("cat_watcher.amcrest_client").disabled = False

    client = _make_client(retry_attempts=1)
    with caplog.at_level("WARNING", logger="cat_watcher.amcrest_client"):
        items = list(client.iter_recordings(since=datetime(2026, 4, 30, tzinfo=UTC)))

    assert not items
    cleanup_warnings = [r for r in caplog.records if "close failed" in r.message or "destroy failed" in r.message]
    assert cleanup_warnings, "expected a WARNING log for the close/destroy failure"


def test_client_context_manager_closes_underlying_httpx() -> None:
    """``with AmcrestClient(...) as c:`` exits cleanly and the httpx client is closed."""
    with _make_client() as client:
        assert isinstance(client._client, httpx.Client)
    assert client._client.is_closed

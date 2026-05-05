"""Wrapper over the Amcrest IP-camera HTTP API.

Implementation choice — ``httpx.Client`` with ``httpx.DigestAuth`` against the camera's
``mediaFileFind.cgi`` (factory.create / findFile / findNextFile / close / destroy) and
``RPC_Loadfile`` endpoints. See ``docs/resources/Amcrest-HTTP_API_V3.26.pdf`` for the wire format.

Reasons not to depend on ``python-amcrest``: GPL-2.0-only license (incompatible with this project's
AGPL-3.0-or-later), heavy transitive deps (requests + urllib3 + argcomplete alongside httpx), and
no declared Python 3.14 support. The API surface we need is small enough to implement directly.

The module surfaces a typed exception per failure mode (network / auth / other API problem) so
callers can react without inspecting status codes themselves. Retries are bounded and only fire for
transient errors (network exceptions + 5xx); 4xx is fail-fast because it indicates a config problem
the next attempt won't solve.

Datetime contract: every :class:`Recording` field is tz-aware UTC. The Amcrest API documents
``StartTime``/``EndTime`` as bare ``"Y-M-D H-m-S"`` strings with no explicit timezone metadata; the
camera has its own NTP/locale clock config that callers can read via
``configManager.cgi?action=getConfig&name=NTP``. This module treats the strings as wall-clock times
in whatever zone the caller declares via ``camera_tz`` and converts to UTC before returning.
Resolving the right ``ZoneInfo`` (per-camera config vs. a display-timezone fallback) is the
caller's job — the client itself knows nothing about project-wide config.
"""

import logging
import os
import re
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Self
from urllib.parse import quote, urlencode

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from zoneinfo import ZoneInfo

    from cat_watcher.config import CameraConfig, CameraSecrets


logger = logging.getLogger(__name__)

_FIND_PAGE_SIZE = 100
_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_RETRY_ATTEMPTS = 3
_DEFAULT_RETRY_DELAY_SECONDS = 10.0
_DOWNLOAD_CHUNK_SIZE = 64 * 1024
_PART_SUFFIX = ".part"
_FIND_ENDPOINT = "/cgi-bin/mediaFileFind.cgi"
_DOWNLOAD_ENDPOINT_PREFIX = "/cgi-bin/RPC_Loadfile"
_AMCREST_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Transient HTTP failures that warrant retry. ``httpx.RemoteProtocolError`` covers the
# camera-disconnected-mid-response case the spec calls out alongside connect/read timeouts.
_RETRYABLE_HTTPX_ERRORS = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)
_HTTP_BAD_REQUEST = 400
_HTTP_INTERNAL_ERROR = 500
_AUTH_STATUSES = frozenset({401, 403})


class CameraError(RuntimeError):
    """Base for all camera HTTP failures."""


class CameraUnreachableError(CameraError):
    """Network-layer failure or 5xx response after exhausting retries."""


class CameraAuthError(CameraError):
    """Camera returned 401 or 403; credentials or ACLs are wrong."""


class CameraAPIError(CameraError):
    """Camera returned another 4xx, or its response was malformed.

    ``status`` carries the HTTP status code when the error originated from a non-2xx response; it is
    ``None`` for errors raised from response-body parsing (e.g. ``factory.create`` returning a body
    without a ``result=`` line). Callers that need to distinguish specific status codes — most
    notably ``findFile``'s overloaded use of HTTP 400 to mean "no recordings in window" — read this
    attribute instead of parsing the message text.
    """

    status: int | None

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Recording:
    """One row from the camera's recording index. All datetimes are tz-aware UTC."""

    source_filename: str
    camera_path: str
    start_ts: datetime
    end_ts: datetime
    file_size_bytes: int


_ITEM_PATTERN = re.compile(r"^items\[(\d+)\]\.([A-Za-z]+)=(.+?)\s*$")
_RESULT_PATTERN = re.compile(r"^result=(\d+)\s*$", re.MULTILINE)


def _amcrest_query(params: dict[str, str]) -> str:
    """Build a query string Amcrest's CGI parser will accept.

    The parser has two non-standard requirements that ``httpx``'s default ``params=`` dict
    violates:

    1. ``[`` / ``]`` must be **literal**, not percent-encoded (``%5B`` / ``%5D`` → 400).
    2. Spaces must be encoded as ``%20``, not ``+`` (the form-encoded ``+`` → 400).

    Every example in ``docs/resources/Amcrest-HTTP_API_V3.26.pdf`` follows both rules
    (e.g. ``condition.Types[0]=dav&condition.StartTime=2014-1-1%2012:00:00``). ``urllib``'s default
    ``quote_via=quote_plus`` produces ``+`` for spaces, so we force ``quote_via=quote``. Call sites
    with bracket-bearing parameters or whitespace-bearing values must build their query string
    through this helper and append it to the path so ``httpx`` forwards the URL unchanged. Full
    diagnosis in ``docs/resources/amcrest-bracket-quirk.md``.
    """
    return urlencode(params, safe="[]", quote_via=quote)


def _parse_find_page(body: str) -> list[dict[str, str]]:
    """Parse a ``findNextFile`` page body into a list of ``items[n]`` dicts in index order."""
    items: dict[int, dict[str, str]] = {}
    for line in body.splitlines():
        match = _ITEM_PATTERN.match(line)
        if match is None:
            continue
        idx = int(match.group(1))
        items.setdefault(idx, {})[match.group(2)] = match.group(3)
    return [items[k] for k in sorted(items)]


class AmcrestClient:
    """One client per camera. Holds a long-lived ``httpx.Client`` for connection reuse."""

    _camera: CameraConfig
    _tz: ZoneInfo
    _retry_attempts: int
    _retry_delay_seconds: float
    _client: httpx.Client

    def __init__(
        self,
        camera: CameraConfig,
        secrets: CameraSecrets,
        *,
        camera_tz: ZoneInfo,
        retry_attempts: int = _DEFAULT_RETRY_ATTEMPTS,
        retry_delay_seconds: float = _DEFAULT_RETRY_DELAY_SECONDS,
    ) -> None:
        self._camera = camera
        self._tz = camera_tz
        self._retry_attempts = retry_attempts
        self._retry_delay_seconds = retry_delay_seconds
        self._client = httpx.Client(
            base_url=f"http://{camera.host}:{camera.port}",
            auth=httpx.DigestAuth(secrets.username, secrets.password.get_secret_value()),
            timeout=_DEFAULT_TIMEOUT_SECONDS,
        )

    def __enter__(self) -> Self:
        """Context-manager entry."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Context-manager exit. Closes the underlying httpx client."""
        self.close()

    def close(self) -> None:
        """Close the underlying httpx client; safe to call multiple times."""
        self._client.close()

    def _classify_status(self, status_code: int) -> None:
        """Raise the right typed exception for a non-success status; return cleanly for 2xx/3xx."""
        if status_code in _AUTH_STATUSES:
            msg = f"camera {self._camera.name!r} auth failed (HTTP {status_code})"
            raise CameraAuthError(msg)
        if _HTTP_BAD_REQUEST <= status_code < _HTTP_INTERNAL_ERROR:
            msg = f"camera {self._camera.name!r} client error (HTTP {status_code})"
            raise CameraAPIError(msg, status=status_code)

    def _request_with_retries(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        """HTTP call with bounded retries on transient failures. 4xx fails fast."""
        last_exc: BaseException | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                response = self._client.request(method, path, params=params)
            except _RETRYABLE_HTTPX_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    "camera %s %s %s attempt %d/%d failed: %s",
                    self._camera.name,
                    method,
                    path,
                    attempt,
                    self._retry_attempts,
                    exc,
                )
                if attempt < self._retry_attempts:
                    time.sleep(self._retry_delay_seconds)
                continue

            self._classify_status(response.status_code)
            if response.status_code >= _HTTP_INTERNAL_ERROR:
                last_exc = httpx.HTTPStatusError(
                    f"server error {response.status_code}",
                    request=response.request,
                    response=response,
                )
                if attempt < self._retry_attempts:
                    time.sleep(self._retry_delay_seconds)
                continue
            return response

        msg = f"camera {self._camera.name!r} unreachable after {self._retry_attempts} attempts"
        raise CameraUnreachableError(msg) from last_exc

    def iter_recordings(
        self,
        *,
        since: datetime,
        until: datetime | None = None,
    ) -> Iterator[Recording]:
        """Yield :class:`Recording`s newer than ``since`` (and older than ``until``).

        ``since`` and ``until`` are tz-aware UTC; the camera receives the equivalent local-time
        bounds. Pagination is lazy — the search handle is closed/destroyed even if the consumer
        bails partway through.
        """
        until = until or datetime.now(UTC)
        # Reject naive datetimes loudly. ``datetime.astimezone`` silently treats them as
        # system-local time, which on a non-UTC host produces window bounds that are hours off
        # — and the camera happily accepts them, returning wrong recordings. Fail-fast at the
        # boundary instead.
        if since.tzinfo is None or until.tzinfo is None:
            msg = "iter_recordings: since/until must be tz-aware (naive datetimes rejected)"
            raise ValueError(msg)
        local_since = since.astimezone(self._tz).strftime(_AMCREST_TIME_FORMAT)
        local_until = until.astimezone(self._tz).strftime(_AMCREST_TIME_FORMAT)

        # Per Amcrest-HTTP_API_V3.26.pdf §"Create a media files finder", factory.create is GET.
        create_response = self._request_with_retries("GET", _FIND_ENDPOINT, params={"action": "factory.create"})
        result_match = _RESULT_PATTERN.search(create_response.text)
        if result_match is None:
            msg = f"camera {self._camera.name!r} factory.create returned no result handle"
            raise CameraAPIError(msg)
        find_handle = result_match.group(1)

        try:
            yield from self._iter_pages(find_handle, local_since=local_since, local_until=local_until)
        finally:
            self._destroy_find_handle(find_handle)

    def _iter_pages(self, find_handle: str, *, local_since: str, local_until: str) -> Iterator[Recording]:
        """Inner loop: drain the camera's pagination once the search handle is open."""
        # ``condition.Types[0]`` carries literal brackets the Amcrest CGI parser refuses to decode
        # from ``%5B`` / ``%5D``; build the query string via ``_amcrest_query`` and pass it as part
        # of the path so ``httpx`` forwards it unchanged. ``condition.Channel`` is 1-indexed per
        # the API spec ("starting from 1") — our single-channel litter-box cameras are always 1.
        find_query = _amcrest_query(
            {
                "action": "findFile",
                "object": find_handle,
                "condition.Channel": "1",
                "condition.StartTime": local_since,
                "condition.EndTime": local_until,
                "condition.Types[0]": "dav",
            },
        )
        # Amcrest firmware overloads HTTP 400 on findFile to mean "no recordings in window" (the API
        # spec buckets it under "request inherently impossible to be satisfied"). Empty windows
        # happen on every quiet poll tick, so swallow that specific status and yield zero results;
        # other 4xx classes (auth, malformed query, endpoint typos) keep failing loudly. The
        # URL-encoding regression test ensures a real syntax break can't sneak through this branch.
        # See ``docs/resources/amcrest-bracket-quirk.md`` for the firmware behavior.
        try:
            _ = self._request_with_retries("GET", f"{_FIND_ENDPOINT}?{find_query}")
        except CameraAPIError as exc:
            if exc.status != _HTTP_BAD_REQUEST:
                raise
            logger.info(
                "camera %s: findFile returned 400 — treating as empty window (%s..%s)",
                self._camera.name,
                local_since,
                local_until,
            )
            return
        while True:
            page = self._request_with_retries(
                "GET",
                _FIND_ENDPOINT,
                params={"action": "findNextFile", "object": find_handle, "count": str(_FIND_PAGE_SIZE)},
            )
            rows = _parse_find_page(page.text)
            if not rows:
                return
            for row in rows:
                yield self._row_to_recording(row)
            if len(rows) < _FIND_PAGE_SIZE:
                return

    def _destroy_find_handle(self, find_handle: str) -> None:
        """Best-effort cleanup of the camera-side search handle. Failures are logged, not raised."""
        for action in ("close", "destroy"):
            try:
                _ = self._request_with_retries(
                    "GET",
                    _FIND_ENDPOINT,
                    params={"action": action, "object": find_handle},
                )
            except CameraError as exc:
                logger.warning(
                    "camera %s %s for object=%s failed: %s",
                    self._camera.name,
                    action,
                    find_handle,
                    exc,
                )

    def _row_to_recording(self, row: dict[str, str]) -> Recording:
        path = row["FilePath"]
        local_start = datetime.strptime(row["StartTime"], _AMCREST_TIME_FORMAT).replace(tzinfo=self._tz)
        local_end = datetime.strptime(row["EndTime"], _AMCREST_TIME_FORMAT).replace(tzinfo=self._tz)
        return Recording(
            source_filename=path.rsplit("/", 1)[-1],
            camera_path=path,
            start_ts=local_start.astimezone(UTC),
            end_ts=local_end.astimezone(UTC),
            file_size_bytes=int(row.get("Length", "0") or "0"),
        )

    def download_recording(self, camera_path: str, *, dest: Path) -> None:
        """Stream a recording to ``dest`` atomically.

        Writes to ``<dest>.part`` first, fsyncs, then ``os.replace``s onto the final name. On
        retryable failure the ``.part`` file is left in place so a sweep can detect the partial.
        """
        url = f"{_DOWNLOAD_ENDPOINT_PREFIX}{camera_path}"
        part_path = dest.with_name(dest.name + _PART_SUFFIX)
        last_exc: BaseException | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                self._stream_to_part(url, part_path)
            except _RETRYABLE_HTTPX_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    "camera %s download attempt %d/%d for %s failed: %s",
                    self._camera.name,
                    attempt,
                    self._retry_attempts,
                    camera_path,
                    exc,
                )
                if attempt < self._retry_attempts:
                    time.sleep(self._retry_delay_seconds)
                continue
            _ = part_path.replace(dest)
            return

        msg = f"camera {self._camera.name!r} download failed after {self._retry_attempts} attempts"
        raise CameraUnreachableError(msg) from last_exc

    def _stream_to_part(self, url: str, part_path: Path) -> None:
        """Open the camera URL and write bytes into ``part_path`` with fsync before close."""
        with self._client.stream("GET", url) as response:
            self._classify_status(response.status_code)
            if response.status_code >= _HTTP_INTERNAL_ERROR:
                # Translate 5xx into a retryable transport error so the outer download loop's
                # ``except _RETRYABLE_HTTPX_ERRORS`` reacts the same way it does for connect /
                # read-timeout failures, mirroring ``_request_with_retries``'s 5xx handling.
                msg = f"camera {self._camera.name!r} returned HTTP {response.status_code} during download"
                raise httpx.RemoteProtocolError(msg, request=response.request)
            fd = os.open(part_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with closing(os.fdopen(fd, "wb")) as fp:
                for chunk in response.iter_bytes(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    _ = fp.write(chunk)
                fp.flush()
                os.fsync(fp.fileno())

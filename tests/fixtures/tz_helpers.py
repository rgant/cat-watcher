"""Shared OS-local timezone manipulation helpers for tests.

Some code under test (the poller's ``_parse_iso_datetime``, the logs viewer's ``parse_since``)
interprets naive datetimes as system-local. Tests that exercise that path need to pin the system
timezone so the round-trip is deterministic regardless of where the test runs.
"""

import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def pinned_tz(tz: str) -> Generator[None]:
    """Set ``$TZ`` to ``tz`` for the duration of the block; restore on exit.

    Calls ``time.tzset()`` after both the set and the restore so libc picks up the new value
    immediately (Python's :mod:`datetime` honors libc's tz tables for ``astimezone``).
    """
    saved = os.environ.get("TZ")
    os.environ["TZ"] = tz
    time.tzset()
    try:
        yield
    finally:
        if saved is None:
            _ = os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved
        time.tzset()

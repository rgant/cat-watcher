"""One rendering rule for every user-visible timestamp.

Lives at package root rather than under ``web/`` because the CLI and the log viewer format the same
values the web UI does, and the point of the module is that all three agree character-for-character.

Wire and storage formats are deliberately *not* routed through here: ``<time datetime="…">``
attributes, ``/health``, the JSONL log ``ts``, on-disk clip paths, and the Amcrest ``findFile``
format are machine-readable and stay ISO 8601 UTC.
"""

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime, tzinfo

    from jinja2 import Environment

LOCAL_STAMP_FORMAT = "%Y-%m-%d %H:%M:%S %Z"
LOCAL_DATE_FORMAT = "%Y-%m-%d"
_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60


def local_stamp(value: datetime, *, tz: tzinfo) -> str:
    """Render ``value`` in ``tz`` as ``2026-07-02 08:19:05 EDT``."""
    return _in_zone(value, tz).strftime(LOCAL_STAMP_FORMAT)


def local_date(value: datetime, *, tz: tzinfo) -> str:
    """Render ``value``'s calendar day in ``tz`` as ``2026-07-02``."""
    return _in_zone(value, tz).strftime(LOCAL_DATE_FORMAT)


def ago(dt: datetime, now: datetime) -> str:
    """Render ``now - dt`` as ``45s ago`` / ``12m ago`` / ``14h 18m ago`` per spec §4.14 examples."""
    delta = now - dt
    seconds = max(0, int(delta.total_seconds()))
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds}s ago"
    minutes_total = seconds // _SECONDS_PER_MINUTE
    if minutes_total < _MINUTES_PER_HOUR:
        return f"{minutes_total}m ago"
    hours, minutes = divmod(minutes_total, _MINUTES_PER_HOUR)
    return f"{hours}h {minutes}m ago"


def _in_zone(value: datetime, tz: tzinfo) -> datetime:
    """Convert ``value`` to ``tz``, rejecting naive input.

    ``astimezone`` on a naive datetime silently assumes system local time, which yields a wrong
    answer that looks right. Same principle as :meth:`cat_watcher.db.UtcDateTime.process_bind_param`.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        msg = "naive datetime rejected; cat-watcher requires tz-aware datetimes"
        raise ValueError(msg)
    return value.astimezone(tz)


def register_datetime_filters(env: Environment, *, tz: tzinfo) -> None:
    """Bind ``tz`` into the ``localstamp`` / ``localdate`` Jinja filters on ``env``.

    The filter names live here beside the functions so ``build_app`` carries no knowledge of what
    templates call them.
    """
    env.filters["localstamp"] = partial(local_stamp, tz=tz)
    env.filters["localdate"] = partial(local_date, tz=tz)

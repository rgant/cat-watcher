"""Unit tests for :mod:`cat_watcher.timefmt`.

Assertions are on exact full output strings: the point of the module is that every surface renders
one identical string, so a substring match would not prove it.
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from jinja2 import Environment

from cat_watcher.timefmt import local_date, local_stamp, register_datetime_filters

_EASTERN = ZoneInfo("America/New_York")


def test_local_stamp_renders_daylight_time_with_its_marker() -> None:
    """A summer instant renders the Eastern wall clock with the ``EDT`` marker."""
    assert local_stamp(datetime(2026, 7, 2, 12, 19, 5, tzinfo=UTC), tz=_EASTERN) == "2026-07-02 08:19:05 EDT"


def test_local_stamp_renders_standard_time_with_its_marker() -> None:
    """A winter instant renders ``EST`` — the marker is not hardcoded to one side of DST."""
    assert local_stamp(datetime(2026, 1, 15, 12, 19, 5, tzinfo=UTC), tz=_EASTERN) == "2026-01-15 07:19:05 EST"


def test_local_stamp_crosses_midnight_into_the_previous_local_day() -> None:
    """An instant just after UTC midnight belongs to the previous calendar day locally."""
    assert local_stamp(datetime(2026, 7, 2, 2, 0, tzinfo=UTC), tz=_EASTERN) == "2026-07-01 22:00:00 EDT"


def test_local_stamp_converts_an_already_local_input() -> None:
    """An input already in another zone is converted, not reinterpreted at the same wall clock."""
    pacific = datetime(2026, 7, 1, 19, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
    assert local_stamp(pacific, tz=_EASTERN) == "2026-07-01 22:00:00 EDT"


def test_local_stamp_honors_a_utc_target_zone() -> None:
    """Rendering into UTC yields a ``UTC`` marker, confirming the format follows the target zone."""
    assert local_stamp(datetime(2026, 7, 2, 2, 0, tzinfo=UTC), tz=UTC) == "2026-07-02 02:00:00 UTC"


def test_local_date_uses_the_local_calendar_day() -> None:
    """The date-only form crosses midnight with the stamp, not with UTC."""
    assert local_date(datetime(2026, 7, 2, 2, 0, tzinfo=UTC), tz=_EASTERN) == "2026-07-01"


def test_local_stamp_rejects_a_naive_datetime() -> None:
    """A naive input raises rather than being silently read as system-local."""
    with pytest.raises(ValueError, match="naive datetime rejected"):
        _ = local_stamp(datetime(2026, 7, 2, 2, 0), tz=_EASTERN)  # noqa: DTZ001  # the naive value under test


def test_local_date_rejects_a_naive_datetime() -> None:
    """Both formatters reject naive input; neither is a silent fallback for the other."""
    with pytest.raises(ValueError, match="naive datetime rejected"):
        _ = local_date(datetime(2026, 7, 2, 2, 0), tz=_EASTERN)  # noqa: DTZ001  # the naive value under test


def test_register_datetime_filters_binds_both_names_to_the_given_zone() -> None:
    """A template rendered through the registered filters produces the bound zone's strings."""
    env = Environment(autoescape=True)  # autoescape is on; the rule cannot see the kwarg
    register_datetime_filters(env, tz=_EASTERN)
    rendered = env.from_string("{{ d | localstamp }}|{{ d | localdate }}").render(d=datetime(2026, 7, 2, 2, 0, tzinfo=UTC))
    assert rendered == "2026-07-01 22:00:00 EDT|2026-07-01"

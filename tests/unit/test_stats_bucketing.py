"""Unit tests for :func:`cat_watcher.web.routes.bucket_clips_by_local_day`.

The DST cases live here rather than in the route tests because through HTTP they are reachable only
during a 30-day window in spring or autumn.
"""

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from cat_watcher.web.routes import bucket_clips_by_local_day

_EASTERN = ZoneInfo("America/New_York")


def test_empty_input_yields_an_empty_mapping() -> None:
    """No clips in the window renders the empty state, not a zero row."""
    assert not bucket_clips_by_local_day([], tz=_EASTERN)


def test_clip_after_utc_midnight_buckets_to_the_previous_local_day() -> None:
    """02:00 UTC is 22:00 the previous day in Eastern — the case UTC day buckets got wrong."""
    rows = [(1, datetime(2026, 7, 2, 2, 0, tzinfo=UTC), False)]
    assert bucket_clips_by_local_day(rows, tz=_EASTERN) == {(1, date(2026, 7, 1)): (1, 0)}


def test_clips_on_different_utc_days_merge_into_one_local_bucket() -> None:
    """Two clips either side of UTC midnight but the same local day are one row, not two."""
    rows = [
        (1, datetime(2026, 7, 2, 2, 0, tzinfo=UTC), False),
        (1, datetime(2026, 7, 1, 20, 0, tzinfo=UTC), False),
    ]
    assert bucket_clips_by_local_day(rows, tz=_EASTERN) == {(1, date(2026, 7, 1)): (2, 0)}


def test_cameras_bucket_independently() -> None:
    """Two cameras on the same local day stay separate rows."""
    rows = [
        (1, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), False),
        (2, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), False),
    ]
    buckets = bucket_clips_by_local_day(rows, tz=_EASTERN)
    assert buckets == {(1, date(2026, 7, 2)): (1, 0), (2, date(2026, 7, 2)): (1, 0)}


def test_cat_total_counts_the_effective_value_passed_in() -> None:
    """The cat column reads ``effective_has_cat``, so an operator override moves the count."""
    rows = [
        (1, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), True),
        (1, datetime(2026, 7, 2, 17, 0, tzinfo=UTC), False),
        (1, datetime(2026, 7, 2, 18, 0, tzinfo=UTC), True),
    ]
    assert bucket_clips_by_local_day(rows, tz=_EASTERN) == {(1, date(2026, 7, 2)): (3, 2)}


def test_fall_back_transition_splits_on_the_local_midnight() -> None:
    """Both instants are past the 2026-11-01 transition, so a fixed -04:00 offset misplaces the first.

    04:30Z is 00:30 EST on Nov 2 under a wrong -04:00 reading but 23:30 EST on Nov 1 in truth.
    """
    rows = [
        (1, datetime(2026, 11, 2, 4, 30, tzinfo=UTC), False),
        (1, datetime(2026, 11, 2, 5, 0, tzinfo=UTC), False),
    ]
    assert bucket_clips_by_local_day(rows, tz=_EASTERN) == {
        (1, date(2026, 11, 1)): (1, 0),
        (1, date(2026, 11, 2)): (1, 0),
    }


def test_spring_forward_transition_splits_on_the_local_midnight() -> None:
    """Both instants are past the 2026-03-08 transition; a fixed -05:00 reading misplaces the second."""
    rows = [
        (1, datetime(2026, 3, 9, 3, 30, tzinfo=UTC), False),
        (1, datetime(2026, 3, 9, 4, 30, tzinfo=UTC), False),
    ]
    assert bucket_clips_by_local_day(rows, tz=_EASTERN) == {
        (1, date(2026, 3, 8)): (1, 0),
        (1, date(2026, 3, 9)): (1, 0),
    }

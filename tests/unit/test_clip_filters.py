"""Unit tests for :mod:`cat_watcher.web.clip_filters`.

The SQL cases run against ``alembic_engine`` rather than ``db_engine``: ``effective_has_cat`` lives
in the ``clip_label_summary`` view, and ``create_all`` emits no view DDL.
"""

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest
from db_helpers import build_test_clip, seed_cat_subject, stamp_reviewed_at, tag_clip_frame
from sqlalchemy import func, select
from starlette.datastructures import QueryParams

from cat_watcher.db import Clip, get_session
from cat_watcher.web.clip_filters import (
    RECOGNIZED_KEYS,
    ClipsFilter,
    IgnoredFilter,
    ParsedClipsFilter,
    apply_clip_filters,
    build_filter_qs,
    build_ignored_notice,
    parse_clips_filter,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.engine import Engine

_EASTERN = ZoneInfo("America/New_York")
_CAMERAS = ("pantry", "garage")


def _parse(**params: str) -> ParsedClipsFilter:
    """Parse ``params`` against the standard two-camera vocabulary."""
    return parse_clips_filter(params, camera_names=_CAMERAS)


# --- parsing -------------------------------------------------------------------------------------


def test_empty_mapping_yields_defaults_and_no_keys_present() -> None:
    """No querystring at all means the default review queue and legacy detail navigation."""
    parsed = parse_clips_filter({}, camera_names=_CAMERAS)
    assert parsed.clips_filter == ClipsFilter(reviewed="no", camera=None, has_cat=None, day=None)
    assert not parsed.ignored
    assert parsed.any_key_present is False


def test_only_unrecognized_keys_are_invisible_to_the_parser() -> None:
    """A tracking parameter must not flip the detail page out of legacy navigation."""
    parsed = parse_clips_filter({"utm_source": "x"}, camera_names=_CAMERAS)
    assert parsed.any_key_present is False
    assert not parsed.ignored


@pytest.mark.parametrize("key", RECOGNIZED_KEYS)
def test_empty_value_selects_the_default_without_being_reported(key: str) -> None:
    """The filter form submits ``key=`` for "unset"; that is a choice, not a bad value."""
    parsed = parse_clips_filter({key: ""}, camera_names=_CAMERAS)
    assert parsed.any_key_present is True
    assert not parsed.ignored
    assert parsed.clips_filter == ClipsFilter()


@pytest.mark.parametrize("value", ["any", "no", "yes"])
def test_reviewed_round_trips_each_accepted_value(value: str) -> None:
    """Every option the Reviewed select offers survives parsing."""
    assert _parse(reviewed=value).clips_filter.reviewed == value


@pytest.mark.parametrize(("raw", "expected"), [("true", True), ("false", False)])
def test_has_cat_accepts_the_two_rendered_options(raw: str, *, expected: bool) -> None:
    """``has_cat`` is tri-state; only the two explicit words select a value."""
    assert _parse(has_cat=raw).clips_filter.has_cat is expected


def test_date_str_parses_to_a_date() -> None:
    """The filter carries a parsed ``date``, so nothing downstream re-parses the string."""
    assert _parse(date_str="2026-07-02").clips_filter.day == date(2026, 7, 2)


def test_date_str_accepts_the_compact_iso_form() -> None:
    """``date.fromisoformat`` accepts more than ``YYYY-MM-DD``; those are valid days, not bad input."""
    parsed = _parse(date_str="20260702")
    assert parsed.clips_filter.day == date(2026, 7, 2)
    assert not parsed.ignored


def test_camera_matching_the_vocabulary_is_kept() -> None:
    """A camera the page offers filters normally."""
    assert _parse(camera="garage").clips_filter.camera == "garage"


def test_unknown_camera_snaps_to_unfiltered_and_is_reported() -> None:
    """An unknown camera behaves like every other control: default plus a notice entry."""
    parsed = _parse(camera="nope")
    assert parsed.clips_filter.camera is None
    assert parsed.ignored == (IgnoredFilter(param="camera", value="nope"),)


def test_unrecognized_has_cat_snaps_to_unfiltered_and_is_reported() -> None:
    """``has_cat=maybe`` is the shape that used to 422 as a bool route parameter."""
    parsed = _parse(has_cat="maybe")
    assert parsed.clips_filter.has_cat is None
    assert parsed.ignored == (IgnoredFilter(param="has_cat", value="maybe"),)


@pytest.mark.parametrize("raw", ["abc", "2026-5-2", "2026-02-30"])
def test_unparseable_date_str_snaps_to_unfiltered_and_is_reported(raw: str) -> None:
    """These three are the values that raised ``ValueError`` straight out of the route."""
    parsed = _parse(date_str=raw)
    assert parsed.clips_filter.day is None
    assert parsed.ignored == (IgnoredFilter(param="date_str", value=raw),)


def test_unrecognized_reviewed_snaps_to_the_default_queue_and_is_reported() -> None:
    """``reviewed=bogus`` is the shape that used to 422 as a ``Literal`` route parameter."""
    parsed = _parse(reviewed="bogus")
    assert parsed.clips_filter.reviewed == "no"
    assert parsed.ignored == (IgnoredFilter(param="reviewed", value="bogus"),)


def test_ignored_entries_follow_recognized_keys_order_not_insertion_order() -> None:
    """Keys are inserted has_cat-then-camera; the notice must still read camera first."""
    parsed = parse_clips_filter({"has_cat": "maybe", "camera": "nope"}, camera_names=_CAMERAS)
    assert parsed.ignored == (
        IgnoredFilter(param="camera", value="nope"),
        IgnoredFilter(param="has_cat", value="maybe"),
    )


def test_every_key_invalid_at_once_reports_all_four_in_order() -> None:
    """The reported production URL carried several keys; one-at-a-time coverage would miss this."""
    parsed = parse_clips_filter(
        {"camera": "nope", "has_cat": "maybe", "date_str": "abc", "reviewed": "bogus"},
        camera_names=_CAMERAS,
    )
    assert [item.param for item in parsed.ignored] == list(RECOGNIZED_KEYS)
    assert parsed.clips_filter == ClipsFilter()


def test_repeated_key_takes_the_last_occurrence() -> None:
    """``QueryParams.get`` returns the last value; a plain dict cannot express this case at all."""
    parsed = parse_clips_filter(QueryParams("has_cat=true&has_cat=false"), camera_names=_CAMERAS)
    assert parsed.clips_filter.has_cat is False


# --- notice text ---------------------------------------------------------------------------------


def test_notice_is_empty_when_nothing_was_rejected() -> None:
    """An empty sequence renders no banner at all, not an empty-looking one."""
    assert build_ignored_notice(()) == ""


def test_notice_renders_one_entry_exactly() -> None:
    """The unescaped form; autoescaping rewrites the quotes when this reaches a page."""
    notice = build_ignored_notice((IgnoredFilter(param="date_str", value="abc"),))
    assert notice == 'Ignored invalid filter values: date_str="abc".'


def test_notice_joins_entries_with_a_comma() -> None:
    """Multiple entries share one banner rather than stacking."""
    notice = build_ignored_notice(
        (IgnoredFilter(param="reviewed", value="bogus"), IgnoredFilter(param="date_str", value="abc")),
    )
    assert notice == 'Ignored invalid filter values: reviewed="bogus", date_str="abc".'


# --- querystring ---------------------------------------------------------------------------------


def test_default_filter_serializes_only_reviewed() -> None:
    """``reviewed`` is always emitted so the detail page can rebuild a complete back-link."""
    assert build_filter_qs(ClipsFilter()) == "reviewed=no"


def test_populated_filter_serializes_every_key() -> None:
    """``has_cat`` serializes as a lowercase word and the day under the ``date_str`` key."""
    f = ClipsFilter(reviewed="any", camera="pantry", has_cat=True, day=date(2026, 7, 2))
    assert build_filter_qs(f) == "reviewed=any&camera=pantry&has_cat=true&date_str=2026-07-02"


@pytest.mark.parametrize(
    "f",
    [
        ClipsFilter(),
        ClipsFilter(reviewed="yes"),
        ClipsFilter(reviewed="any", camera="pantry", has_cat=False, day=date(2026, 7, 2)),
        ClipsFilter(reviewed="no", has_cat=True),
    ],
)
def test_querystring_round_trips_through_the_parser(f: ClipsFilter) -> None:
    """Serializing then re-parsing yields an equal filter, so a row click preserves the queue."""
    parsed = parse_clips_filter(QueryParams(build_filter_qs(f)), camera_names=_CAMERAS)
    assert parsed.clips_filter == f
    assert not parsed.ignored


# --- SQL -----------------------------------------------------------------------------------------


def _add(engine: Engine, cam_id: int, start_ts: datetime, *, name: str, has_cat: bool = False) -> int:
    clip = build_test_clip(cam_id, start_ts=start_ts, source_filename=name, has_cat=has_cat)
    with get_session(engine) as session:
        session.add(clip)
    return clip.id


def _ids(engine: Engine, f: ClipsFilter) -> set[int]:
    stmt = apply_clip_filters(select(Clip.id), f, display_tz=_EASTERN)
    with get_session(engine) as session:
        return set(session.scalars(stmt))


def test_camera_filter_returns_only_that_cameras_clips(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A camera filter joins ``Camera`` and matches on name."""
    pantry = seed_camera(alembic_engine, name="pantry")
    garage = seed_camera(alembic_engine, name="garage", display_name="Garage")
    mine = _add(alembic_engine, pantry, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4")
    theirs = _add(alembic_engine, garage, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="b.mp4")

    found = _ids(alembic_engine, ClipsFilter(reviewed="any", camera="pantry"))
    assert found == {mine}
    assert theirs not in found


def test_has_cat_filters_effective_not_raw(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """The control must agree with the Cat? badge, which renders ``effective_has_cat``.

    Both clips are reviewed on purpose: the view sets ``effective_has_cat = has_cat`` while
    ``reviewed_at IS NULL``, so on the default queue the two columns can never disagree and this
    test would pass against the old, wrong column.
    """
    cam_id = seed_camera(alembic_engine)
    reviewed_at = datetime(2026, 7, 2, 18, 0, tzinfo=UTC)
    subject_id = seed_cat_subject(alembic_engine)
    # Detector said no cat, operator tagged one → effective TRUE.
    tagged = _add(alembic_engine, cam_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="tagged.mp4", has_cat=False)
    tag_clip_frame(alembic_engine, clip_id=tagged, subject_id=subject_id, reviewed_at=reviewed_at)
    # Detector said cat, operator tagged nothing → effective FALSE.
    untagged = _add(alembic_engine, cam_id, datetime(2026, 7, 2, 17, 0, tzinfo=UTC), name="untagged.mp4", has_cat=True)
    stamp_reviewed_at(alembic_engine, untagged, reviewed_at)

    assert _ids(alembic_engine, ClipsFilter(reviewed="any", has_cat=True)) == {tagged}
    assert _ids(alembic_engine, ClipsFilter(reviewed="any", has_cat=False)) == {untagged}


def test_day_filter_spans_the_whole_local_day(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Midnight and the last second of the local day are both inside the window."""
    cam_id = seed_camera(alembic_engine)
    # 2026-07-02 local is UTC-04:00, so local midnight is 04:00Z and 23:59:59 local is 03:59:59Z next day.
    first = _add(alembic_engine, cam_id, datetime(2026, 7, 2, 4, 0, 0, tzinfo=UTC), name="first.mp4")
    last = _add(alembic_engine, cam_id, datetime(2026, 7, 3, 3, 59, 59, tzinfo=UTC), name="last.mp4")

    assert _ids(alembic_engine, ClipsFilter(reviewed="any", day=date(2026, 7, 2))) == {first, last}


def test_day_filter_excludes_the_neighboring_local_days(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A 02:00 UTC clip belongs to the previous local day — the case the UTC window got wrong."""
    cam_id = seed_camera(alembic_engine)
    # 02:00 UTC on 2026-07-02 is 22:00 EDT on 2026-07-01.
    previous_day = _add(alembic_engine, cam_id, datetime(2026, 7, 2, 2, 0, tzinfo=UTC), name="prev.mp4")
    inside = _add(alembic_engine, cam_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="inside.mp4")

    found = _ids(alembic_engine, ClipsFilter(reviewed="any", day=date(2026, 7, 2)))
    assert found == {inside}
    assert previous_day not in found
    assert _ids(alembic_engine, ClipsFilter(reviewed="any", day=date(2026, 7, 1))) == {previous_day}


def test_day_filter_covers_the_twenty_five_hour_fall_back_day(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """2026-11-01 local spans 25 hours; a ``+24h`` end bound would drop its last hour.

    A clip at 01:30 local that day proves nothing — the repeated hour sits inside a 24-hour window
    too. These two instants straddle the local midnight that a fixed-offset bound gets wrong.
    """
    cam_id = seed_camera(alembic_engine)
    last_hour = _add(alembic_engine, cam_id, datetime(2026, 11, 2, 4, 30, tzinfo=UTC), name="late.mp4")
    next_day = _add(alembic_engine, cam_id, datetime(2026, 11, 2, 5, 0, tzinfo=UTC), name="next.mp4")

    found = _ids(alembic_engine, ClipsFilter(reviewed="any", day=date(2026, 11, 1)))
    assert found == {last_hour}
    assert next_day not in found


def test_day_filter_covers_the_twenty_three_hour_spring_forward_day(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """2026-03-08 local spans 23 hours; a ``+24h`` end bound would admit an hour of the next day."""
    cam_id = seed_camera(alembic_engine)
    last_hour = _add(alembic_engine, cam_id, datetime(2026, 3, 9, 3, 30, tzinfo=UTC), name="late.mp4")
    next_day = _add(alembic_engine, cam_id, datetime(2026, 3, 9, 4, 30, tzinfo=UTC), name="next.mp4")

    found = _ids(alembic_engine, ClipsFilter(reviewed="any", day=date(2026, 3, 8)))
    assert found == {last_hour}
    assert next_day not in found


def test_reviewed_modes_select_their_own_sets(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """The three Reviewed options partition the clips, with ``any`` adding no clause."""
    cam_id = seed_camera(alembic_engine)
    unreviewed = _add(alembic_engine, cam_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="open.mp4")
    reviewed = _add(alembic_engine, cam_id, datetime(2026, 7, 2, 17, 0, tzinfo=UTC), name="done.mp4")
    stamp_reviewed_at(alembic_engine, reviewed, datetime(2026, 7, 2, 18, 0, tzinfo=UTC))

    assert _ids(alembic_engine, ClipsFilter(reviewed="no")) == {unreviewed}
    assert _ids(alembic_engine, ClipsFilter(reviewed="yes")) == {reviewed}
    assert _ids(alembic_engine, ClipsFilter(reviewed="any")) == {unreviewed, reviewed}


def test_multi_field_filter_composes_with_and_semantics(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Every active filter narrows; none replaces another."""
    pantry = seed_camera(alembic_engine, name="pantry")
    garage = seed_camera(alembic_engine, name="garage", display_name="Garage")
    wanted = _add(alembic_engine, pantry, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="wanted.mp4")
    _ = _add(alembic_engine, garage, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="wrong-cam.mp4")
    _ = _add(alembic_engine, pantry, datetime(2026, 7, 3, 16, 0, tzinfo=UTC), name="wrong-day.mp4")

    f = ClipsFilter(reviewed="any", camera="pantry", day=date(2026, 7, 2))
    assert _ids(alembic_engine, f) == {wanted}


def test_apply_clip_filters_composes_onto_a_count_statement(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """The progress indicator's shape: ``select_from`` plus a join must still compile and count."""
    cam_id = seed_camera(alembic_engine)
    _ = _add(alembic_engine, cam_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4", has_cat=True)
    _ = _add(alembic_engine, cam_id, datetime(2026, 7, 3, 16, 0, tzinfo=UTC), name="b.mp4", has_cat=True)

    stmt = apply_clip_filters(
        select(func.count()).select_from(Clip),  # pylint: disable=not-callable  # sqlalchemy func.count() is a generative construct, not the builtin; pylint false positive
        ClipsFilter(reviewed="any", camera="pantry", has_cat=True, day=date(2026, 7, 2)),
        display_tz=_EASTERN,
    )
    with get_session(alembic_engine) as session:
        assert session.scalar(stmt) == 1


def test_apply_clip_filters_composes_onto_an_entity_select(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """``select(Clip)`` is the list page's shape; the joins must not duplicate or drop rows."""
    cam_id = seed_camera(alembic_engine)
    _ = _add(alembic_engine, cam_id, datetime(2026, 7, 2, 16, 0, tzinfo=UTC), name="a.mp4", has_cat=True)

    stmt = apply_clip_filters(select(Clip), ClipsFilter(reviewed="any", camera="pantry", has_cat=True), display_tz=_EASTERN)
    with get_session(alembic_engine) as session:
        assert len(list(session.scalars(stmt))) == 1

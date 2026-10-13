"""Tests for the visit rule in ``cat_watcher.visits``."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from db_helpers import add_clip, add_clip_frame, apply_clip_label_summary_view, recorded_statements, seed_cat_subject, tag_frame

from cat_watcher.db import Clip, get_session
from cat_watcher.identity import ClipIdentity, NamedCat, query_cat_subjects
from cat_watcher.visits import (
    VISIT_GAP,
    CatClip,
    Visit,
    group_visits,
    last_visits_for,
    query_cat_clips,
    query_last_visits,
    query_visits,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from sqlalchemy.engine import Engine

    from cat_watcher.config import Config

THRESHOLD = 0.70
T0 = datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC)
CLIP_LENGTH = timedelta(seconds=2)  # The length that ``add_clip`` gives every clip.


def _cat_clip(clip_id: int, start: datetime, *, camera_id: int = 1, slugs: tuple[str, ...] = ("marcel",)) -> CatClip:
    identity = ClipIdentity(cats=tuple(NamedCat(slug, "operator") for slug in slugs), fallback=None)
    return CatClip(clip_id=clip_id, camera_id=camera_id, start_ts=start, end_ts=start + CLIP_LENGTH, identity=identity)


@pytest.fixture(autouse=True)
def _label_summary_view(db_engine: Engine) -> None:
    """Create the ``clip_label_summary`` view, which ``create_all`` does not emit."""
    apply_clip_label_summary_view(db_engine)


def _seed_tagged_clip(engine: Engine, cam: int, subject_ids: list[int], start: datetime, name: str) -> int:
    clip_id = add_clip(engine, cam, start_ts=start, name=name)
    frame_id = add_clip_frame(engine, clip_id, 0)
    for subject_id in subject_ids:
        tag_frame(engine, frame_id, subject_id)
    return clip_id


def test_gap_under_two_minutes_joins_one_visit() -> None:
    """Two clips of one cat on one camera less than 2 minutes apart make one visit."""
    second_start = T0 + CLIP_LENGTH + VISIT_GAP - timedelta(seconds=1)
    visits = group_visits([_cat_clip(1, T0), _cat_clip(2, second_start)])

    assert visits == {
        "marcel": [Visit(slug="marcel", camera_id=1, start_ts=T0, end_ts=second_start + CLIP_LENGTH, clip_ids=(1, 2))],
    }


def test_gap_of_exactly_two_minutes_starts_a_new_visit() -> None:
    """A gap of exactly 2 minutes starts a second visit."""
    second_start = T0 + CLIP_LENGTH + VISIT_GAP
    visits = group_visits([_cat_clip(1, T0), _cat_clip(2, second_start)])

    assert [visit.clip_ids for visit in visits["marcel"]] == [(1,), (2,)]


def test_two_cameras_make_two_visits() -> None:
    """Clips of one cat on two cameras never share a visit."""
    visits = group_visits([_cat_clip(1, T0, camera_id=1), _cat_clip(2, T0 + timedelta(seconds=10), camera_id=2)])

    assert [(visit.camera_id, visit.clip_ids) for visit in visits["marcel"]] == [(1, (1,)), (2, (2,))]


def test_visits_of_one_cat_are_sorted_by_start_across_cameras() -> None:
    """The visits of one cat are oldest first, whatever the camera."""
    visits = group_visits([_cat_clip(1, T0 + timedelta(minutes=10), camera_id=1), _cat_clip(2, T0, camera_id=2)])

    assert [visit.clip_ids for visit in visits["marcel"]] == [(2,), (1,)]


def test_clip_naming_two_cats_joins_a_visit_of_each() -> None:
    """A clip that names two cats counts in the visits of both."""
    visits = group_visits([_cat_clip(1, T0, slugs=("marcel", "rufus"))])

    assert set(visits) == {"marcel", "rufus"}
    assert visits["rufus"][0].clip_ids == (1,)


def test_unknown_cat_clip_makes_no_visit() -> None:
    """A cat clip that names no cat is in no visit."""
    unknown = CatClip(clip_id=1, camera_id=1, start_ts=T0, end_ts=T0 + CLIP_LENGTH, identity=ClipIdentity(cats=(), fallback="unsure"))

    assert not group_visits([unknown])


def test_query_cat_clips_skips_clips_that_are_not_cat_clips(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A clip without a cat is not in the result. The window includes ``since`` and excludes ``until``."""
    cam = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine, slug="marcel", display_name="Marcel", display_order=1)
    cat = _seed_tagged_clip(db_engine, cam, [marcel], T0, "cat.mp4")
    _ = add_clip(db_engine, cam, start_ts=T0 + timedelta(minutes=5), name="empty.mp4", has_cat=False)
    _ = _seed_tagged_clip(db_engine, cam, [marcel], T0 + timedelta(hours=1), "late.mp4")

    with get_session(db_engine) as session:
        clips = query_cat_clips(
            session,
            since=T0,
            until=T0 + timedelta(hours=1),
            unsure_threshold=THRESHOLD,
            cats=query_cat_subjects(session),
        )

    assert [clip.clip_id for clip in clips] == [cat]
    assert clips[0].identity.cats == (NamedCat("marcel", "operator"),)
    assert clips[0].end_ts == T0 + CLIP_LENGTH


def test_query_last_visits_ignores_visits_outside_retention(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A visit older than ``retention_days`` is not a last visit."""
    cam = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine, slug="marcel", display_name="Marcel", display_order=1)
    rufus = seed_cat_subject(db_engine, slug="rufus", display_name="Rufus", display_order=2)
    now = T0 + timedelta(days=40)
    _ = _seed_tagged_clip(db_engine, cam, [marcel], now - timedelta(days=31), "old.mp4")
    _ = _seed_tagged_clip(db_engine, cam, [rufus], now - timedelta(days=2), "recent.mp4")
    newest = _seed_tagged_clip(db_engine, cam, [rufus], now - timedelta(hours=1), "newest.mp4")

    with get_session(db_engine) as session:
        last = query_last_visits(session, now=now, retention_days=30, unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))

    assert set(last) == {"rufus"}
    assert last["rufus"].clip_ids == (newest,)


def test_last_visits_for_reads_retention_from_config(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    make_config: Callable[..., Config],
    tmp_path: Path,
) -> None:
    """The config sets the retention window. Without a ``[classifier]`` table, operator tags still name the cats."""
    cam = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine, slug="marcel", display_name="Marcel", display_order=1)
    rufus = seed_cat_subject(db_engine, slug="rufus", display_name="Rufus", display_order=2)
    now = T0 + timedelta(days=10)
    _ = _seed_tagged_clip(db_engine, cam, [marcel], now - timedelta(days=4), "old.mp4")
    recent = _seed_tagged_clip(db_engine, cam, [rufus], now - timedelta(days=2), "recent.mp4")
    base = make_config(tmp_path, tmp_path)
    config = base.model_copy(update={"retention": base.retention.model_copy(update={"clip_days": 3})})
    assert config.classifier is None

    with get_session(db_engine) as session:
        last = last_visits_for(session, config, now=now, cats=query_cat_subjects(session))

    assert {slug: visit.clip_ids for slug, visit in last.items()} == {"rufus": (recent,)}


def _count_statements(engine: Engine, now: datetime) -> int:
    with recorded_statements(engine) as statements, get_session(engine) as session:
        _ = query_last_visits(session, now=now, retention_days=30, unsure_threshold=THRESHOLD, cats=query_cat_subjects(session))
    return len(statements)


def test_query_last_visits_statement_count_does_not_grow_with_clips(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """The statement count is the same for 1 clip and for 20 clips."""
    cam = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine, slug="marcel", display_name="Marcel", display_order=1)
    now = T0 + timedelta(days=1)
    _ = _seed_tagged_clip(db_engine, cam, [marcel], T0, "c0.mp4")
    one_clip = _count_statements(db_engine, now)
    for i in range(1, 20):
        _ = _seed_tagged_clip(db_engine, cam, [marcel], T0 + timedelta(minutes=10 * i), f"c{i}.mp4")

    assert _count_statements(db_engine, now) == one_clip


def _visits_since(engine: Engine, since: datetime) -> dict[str, list[Visit]]:
    with get_session(engine) as session:
        return query_visits(
            session,
            since=since,
            until=since + timedelta(hours=1),
            unsure_threshold=THRESHOLD,
            cats=query_cat_subjects(session),
        )


def test_query_visits_drops_a_visit_that_starts_before_the_window(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A clip in the window that continues a visit from before the window gives no visit."""
    cam = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine, slug="marcel", display_name="Marcel", display_order=1)
    early = add_clip(db_engine, cam, start_ts=T0 - timedelta(seconds=30), name="early.mp4")
    with get_session(db_engine) as session:
        clip = session.get(Clip, early)
        assert clip is not None
        clip.end_ts = T0 + timedelta(seconds=10)
    tag_frame(db_engine, add_clip_frame(db_engine, early, 0), marcel)
    _ = _seed_tagged_clip(db_engine, cam, [marcel], T0 + timedelta(seconds=60), "chained.mp4")

    assert not _visits_since(db_engine, T0)


def test_query_visits_keeps_a_lone_clip_that_starts_in_the_window(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A clip more than ``VISIT_GAP`` after the last clip before the window starts a visit in the window."""
    cam = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine, slug="marcel", display_name="Marcel", display_order=1)
    _ = _seed_tagged_clip(db_engine, cam, [marcel], T0 - timedelta(minutes=10), "before.mp4")
    lone = _seed_tagged_clip(db_engine, cam, [marcel], T0 + timedelta(seconds=60), "lone.mp4")

    visits = _visits_since(db_engine, T0)

    assert [visit.clip_ids for visit in visits["marcel"]] == [(lone,)]


def _count_query_visits_statements(engine: Engine) -> int:
    with recorded_statements(engine) as statements:
        _ = _visits_since(engine, T0)
    return len(statements)


def test_query_visits_statement_count_does_not_grow_with_clips(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """The statement count is the same for 1 clip and for 20 clips, in and before the window."""
    cam = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine, slug="marcel", display_name="Marcel", display_order=1)
    _ = _seed_tagged_clip(db_engine, cam, [marcel], T0, "c0.mp4")
    one_clip = _count_query_visits_statements(db_engine)
    for i in range(1, 20):
        _ = _seed_tagged_clip(db_engine, cam, [marcel], T0 + timedelta(minutes=5 * i - 50), f"c{i}.mp4")

    assert _count_query_visits_statements(db_engine) == one_clip

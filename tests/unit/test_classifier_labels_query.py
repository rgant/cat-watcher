"""Unit tests for :mod:`cat_watcher.classifier.labels_query`.

Runs against ``alembic_engine`` because the module reads through the full migrated schema.
``db_helpers`` covers a single tag on a single frame. Several of these cases need several tags on
one frame, or several frames on one clip, so those rows go in directly through ``get_session``.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from db_helpers import add_clip, add_clip_frame, add_event_subject, seed_cat_subject, tag_frame

from cat_watcher.classifier.labels_query import CatFrameRow, query_single_cat_frames, resolve_cat_classes
from cat_watcher.db import Clip, Subject, get_session

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.engine import Engine

_START = datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)


def _add_clip(engine: Engine, cam_id: int, *, name: str) -> int:
    """Insert a ``Clip`` row at ``_START`` and return its id."""
    return add_clip(engine, cam_id, start_ts=_START, name=name)


def _add_frame(engine: Engine, clip_id: int, ordinal: int, *, thumb_path: str | None = None) -> int:
    """Insert a ``ClipFrame`` row at ``ordinal`` and return its id."""
    return add_clip_frame(engine, clip_id, ordinal, thumb_path=thumb_path)


def _tag(engine: Engine, frame_id: int, subject_id: int) -> None:
    """Link a ``ClipFrame`` to a ``Subject`` through ``ClipFrameSubject``."""
    tag_frame(engine, frame_id, subject_id)


def _seed_event_subject(engine: Engine, *, slug: str, display_order: int) -> int:
    """Insert a ``kind='event'`` ``Subject`` row and return its id."""
    return add_event_subject(engine, slug=slug, display_order=display_order)


def _archive(engine: Engine, subject_id: int) -> None:
    """Stamp ``archived_at`` on a ``Subject`` row."""
    with get_session(engine) as session:
        subj = session.get(Subject, subject_id)
        assert subj is not None
        subj.archived_at = _START


# --- resolve_cat_classes ---------------------------------------------------------------------------


def test_resolve_cat_classes_returns_active_cats_in_display_order(alembic_engine: Engine) -> None:
    """The class order follows ``display_order``, not insertion order or slug alphabetically."""
    _ = seed_cat_subject(alembic_engine, slug="rufus", display_name="Rufus", display_order=2)
    _ = seed_cat_subject(alembic_engine, slug="marcel", display_name="Marcel", display_order=1)
    _ = _seed_event_subject(alembic_engine, slug="cleaning", display_order=1)
    archived_id = seed_cat_subject(alembic_engine, slug="old_cat", display_name="Old Cat", display_order=3)
    _archive(alembic_engine, archived_id)

    assert resolve_cat_classes(alembic_engine) == ("marcel", "rufus")


def test_resolve_cat_classes_on_an_empty_database_returns_an_empty_tuple(alembic_engine: Engine) -> None:
    """No subject rows at all still returns a valid, empty result."""
    assert not resolve_cat_classes(alembic_engine)


# --- query_single_cat_frames -------------------------------------------------------------------------


def test_query_single_cat_frames_returns_a_row_with_the_correct_fields(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A frame tagged with one cat returns the cat slug plus the clip and frame paths."""
    cam_id = seed_camera(alembic_engine)
    marcel_id = seed_cat_subject(alembic_engine, slug="marcel", display_name="Marcel", display_order=1)
    clip_id = _add_clip(alembic_engine, cam_id, name="a.mp4")
    frame_id = _add_frame(alembic_engine, clip_id, 2, thumb_path="thumbs/a_2.jpg")
    _tag(alembic_engine, frame_id, marcel_id)

    with get_session(alembic_engine) as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        expected_file_path = clip.file_path

    assert query_single_cat_frames(alembic_engine) == [
        CatFrameRow(
            clip_id=clip_id,
            frame_id=frame_id,
            ordinal=2,
            t_offset_seconds=2.0,
            cat_slug="marcel",
            clip_file_path=expected_file_path,
            frame_thumb_path="thumbs/a_2.jpg",
        ),
    ]


def test_query_single_cat_frames_excludes_a_frame_tagged_with_two_cats(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Two cat tags on one frame fail the 'exactly one' rule and drop the frame from the result."""
    cam_id = seed_camera(alembic_engine)
    marcel_id = seed_cat_subject(alembic_engine, slug="marcel", display_name="Marcel", display_order=1)
    rufus_id = seed_cat_subject(alembic_engine, slug="rufus", display_name="Rufus", display_order=2)
    clip_id = _add_clip(alembic_engine, cam_id, name="a.mp4")
    frame_id = _add_frame(alembic_engine, clip_id, 0)
    _tag(alembic_engine, frame_id, marcel_id)
    _tag(alembic_engine, frame_id, rufus_id)

    assert query_single_cat_frames(alembic_engine) == []


def test_query_single_cat_frames_includes_a_frame_tagged_with_one_cat_and_one_event(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """An event tag does not count toward the cat-tag total, so the frame stays included."""
    cam_id = seed_camera(alembic_engine)
    marcel_id = seed_cat_subject(alembic_engine, slug="marcel", display_name="Marcel", display_order=1)
    cleaning_id = _seed_event_subject(alembic_engine, slug="cleaning", display_order=1)
    clip_id = _add_clip(alembic_engine, cam_id, name="a.mp4")
    frame_id = _add_frame(alembic_engine, clip_id, 0)
    _tag(alembic_engine, frame_id, marcel_id)
    _tag(alembic_engine, frame_id, cleaning_id)

    rows = query_single_cat_frames(alembic_engine)

    assert len(rows) == 1
    assert rows[0].frame_id == frame_id
    assert rows[0].cat_slug == "marcel"


def test_query_single_cat_frames_excludes_untagged_and_event_only_frames(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A frame with zero cat tags is excluded, whether it carries an event tag or no tag at all."""
    cam_id = seed_camera(alembic_engine)
    cleaning_id = _seed_event_subject(alembic_engine, slug="cleaning", display_order=1)
    clip_id = _add_clip(alembic_engine, cam_id, name="a.mp4")
    _ = _add_frame(alembic_engine, clip_id, 0)
    event_only_id = _add_frame(alembic_engine, clip_id, 1)
    _tag(alembic_engine, event_only_id, cleaning_id)

    assert not query_single_cat_frames(alembic_engine)


def test_query_single_cat_frames_orders_by_clip_id_then_ordinal(alembic_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Rows sort by ``clip_id`` then ``ordinal``, regardless of insertion order."""
    cam_id = seed_camera(alembic_engine)
    marcel_id = seed_cat_subject(alembic_engine, slug="marcel", display_name="Marcel", display_order=1)
    clip_a = _add_clip(alembic_engine, cam_id, name="a.mp4")
    clip_b = _add_clip(alembic_engine, cam_id, name="b.mp4")
    frame_b0 = _add_frame(alembic_engine, clip_b, 0)
    frame_a1 = _add_frame(alembic_engine, clip_a, 1)
    frame_a0 = _add_frame(alembic_engine, clip_a, 0)
    for frame_id in (frame_b0, frame_a1, frame_a0):
        _tag(alembic_engine, frame_id, marcel_id)

    rows = query_single_cat_frames(alembic_engine)

    assert [(r.clip_id, r.ordinal) for r in rows] == [(clip_a, 0), (clip_a, 1), (clip_b, 0)]


def test_query_single_cat_frames_returns_both_cats_from_one_clip_on_different_frames(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """A clip whose frames name different cats returns both, each with its own cat slug."""
    cam_id = seed_camera(alembic_engine)
    marcel_id = seed_cat_subject(alembic_engine, slug="marcel", display_name="Marcel", display_order=1)
    rufus_id = seed_cat_subject(alembic_engine, slug="rufus", display_name="Rufus", display_order=2)
    clip_id = _add_clip(alembic_engine, cam_id, name="a.mp4")
    marcel_frame = _add_frame(alembic_engine, clip_id, 0)
    rufus_frame = _add_frame(alembic_engine, clip_id, 1)
    _tag(alembic_engine, marcel_frame, marcel_id)
    _tag(alembic_engine, rufus_frame, rufus_id)

    rows = query_single_cat_frames(alembic_engine)

    assert [(r.ordinal, r.cat_slug) for r in rows] == [(0, "marcel"), (1, "rufus")]


def test_query_single_cat_frames_excludes_a_frame_tagged_with_an_archived_cat(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """An archived cat drops out of ``resolve_cat_classes``, so its tagged frames must not export."""
    cam_id = seed_camera(alembic_engine)
    old_cat_id = seed_cat_subject(alembic_engine, slug="old_cat", display_name="Old Cat", display_order=1)
    _archive(alembic_engine, old_cat_id)
    clip_id = _add_clip(alembic_engine, cam_id, name="a.mp4")
    frame_id = _add_frame(alembic_engine, clip_id, 0)
    _tag(alembic_engine, frame_id, old_cat_id)

    assert not query_single_cat_frames(alembic_engine)


def test_query_single_cat_frames_excludes_a_frame_tagged_with_an_archived_cat_and_an_active_cat(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """Two cat tags already fail the 'exactly one' rule. The archived tag is not why this frame drops."""
    cam_id = seed_camera(alembic_engine)
    old_cat_id = seed_cat_subject(alembic_engine, slug="old_cat", display_name="Old Cat", display_order=1)
    _archive(alembic_engine, old_cat_id)
    marcel_id = seed_cat_subject(alembic_engine, slug="marcel", display_name="Marcel", display_order=2)
    clip_id = _add_clip(alembic_engine, cam_id, name="a.mp4")
    frame_id = _add_frame(alembic_engine, clip_id, 0)
    _tag(alembic_engine, frame_id, old_cat_id)
    _tag(alembic_engine, frame_id, marcel_id)

    assert not query_single_cat_frames(alembic_engine)


def test_query_single_cat_frames_excludes_a_frame_tagged_with_an_archived_cat_and_an_event(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """An event tag does not rescue an archived cat's frame. The archived filter still excludes it."""
    cam_id = seed_camera(alembic_engine)
    old_cat_id = seed_cat_subject(alembic_engine, slug="old_cat", display_name="Old Cat", display_order=1)
    _archive(alembic_engine, old_cat_id)
    cleaning_id = _seed_event_subject(alembic_engine, slug="cleaning", display_order=1)
    clip_id = _add_clip(alembic_engine, cam_id, name="a.mp4")
    frame_id = _add_frame(alembic_engine, clip_id, 0)
    _tag(alembic_engine, frame_id, old_cat_id)
    _tag(alembic_engine, frame_id, cleaning_id)

    assert not query_single_cat_frames(alembic_engine)


def test_query_single_cat_frames_returns_only_the_active_cats_frames_when_one_cat_is_archived(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """After someone archives a cat, only the active cat's frames come back. This is the real-DB shape."""
    cam_id = seed_camera(alembic_engine)
    old_cat_id = seed_cat_subject(alembic_engine, slug="old_cat", display_name="Old Cat", display_order=1)
    _archive(alembic_engine, old_cat_id)
    marcel_id = seed_cat_subject(alembic_engine, slug="marcel", display_name="Marcel", display_order=2)
    clip_id = _add_clip(alembic_engine, cam_id, name="a.mp4")
    old_cat_frame = _add_frame(alembic_engine, clip_id, 0)
    marcel_frame = _add_frame(alembic_engine, clip_id, 1)
    _tag(alembic_engine, old_cat_frame, old_cat_id)
    _tag(alembic_engine, marcel_frame, marcel_id)

    rows = query_single_cat_frames(alembic_engine)

    assert [(r.ordinal, r.cat_slug) for r in rows] == [(1, "marcel")]


def test_query_single_cat_frames_on_an_empty_database_returns_an_empty_list(alembic_engine: Engine) -> None:
    """No clip or frame rows at all still returns a valid, empty result."""
    assert query_single_cat_frames(alembic_engine) == []

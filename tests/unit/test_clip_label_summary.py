"""Unit tests for the ``clip_label_summary`` view contract.

The view exposes three derived columns per clip:

* ``has_manual_cat`` — TRUE iff at least one frame in the clip has a subject of kind ``'cat'``
  tagged (archived subjects still count).
* ``effective_has_cat`` — when the clip has been reviewed (``reviewed_at IS NOT NULL``) this
  reflects ``has_manual_cat``; otherwise it reflects the detector verdict (``clips.has_cat``).
* ``tagged_subject_slugs`` — comma-separated, subjects ordered by ``kind`` then ``display_order``;
  empty string when no subjects are tagged.
"""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

import pytest
from alembic import command
from alembic.config import Config
from db_helpers import make_clip_frame
from sqlalchemy import text

from cat_watcher.db import Camera, Clip, ClipFrame, ClipFrameSubject, PollStatus, Subject, create_engine, get_session

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    """File-backed SQLite engine with full schema applied via Alembic (includes the view)."""
    db_path = tmp_path / "test_view.sqlite"
    monkeypatch.setenv("CAT_WATCHER_DB_URL", f"sqlite:///{db_path}")
    cfg = Config("alembic.ini")
    cfg.set_main_option("script_location", "migrations")
    command.upgrade(cfg, "head")
    eng = create_engine(f"sqlite:///{db_path}")
    try:
        yield eng
    finally:
        eng.dispose()


def _seed_camera(session: Session) -> int:
    cam = Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)
    session.add(cam)
    session.flush()
    return cam.id


def _make_clip(camera_id: int, *, has_cat: bool, reviewed_at: datetime | None = None, idx: int = 0) -> Clip:
    ts = _T0 + timedelta(hours=idx)
    return Clip(
        camera_id=camera_id,
        source_filename=f"clip_{idx:03d}.mp4",
        start_ts=ts,
        end_ts=ts + timedelta(seconds=30),
        duration_seconds=30.0,
        file_path=f"clips/{idx:03d}.mp4",
        thumb_path=f"thumbs/{idx:03d}.jpg",
        file_size_bytes=1024,
        has_cat=has_cat,
        max_score=0.9 if has_cat else 0.0,
        frames_sampled=1,
        frames_with_cat=1 if has_cat else 0,
        detector_version="test@0.0.1",
        ingested_at=ts,
        reviewed_at=reviewed_at,
    )


def _make_frame(clip_id: int, ordinal: int = 0) -> ClipFrame:
    return make_clip_frame(clip_id, ordinal)


def _make_subject(slug: str, kind: str, display_order: int) -> Subject:
    return Subject(
        slug=slug,
        display_name=slug.capitalize(),
        kind=kind,
        display_order=display_order,
    )


def _query_view(eng: Engine, clip_id: int) -> tuple[bool, bool, str]:
    """Return ``(has_manual_cat, effective_has_cat, tagged_subject_slugs)`` from the view."""
    with eng.connect() as conn:
        row = conn.execute(
            text("SELECT has_manual_cat, effective_has_cat, tagged_subject_slugs FROM clip_label_summary WHERE clip_id = :id"),
            {"id": clip_id},
        ).one()
    # SQLAlchemy Row.__getitem__ returns Any; cast narrows to concrete types for the callers.
    has_manual_cat = cast("int", row[0])
    effective_has_cat = cast("int", row[1])
    tagged_subject_slugs = cast("str", row[2])
    return bool(has_manual_cat), bool(effective_has_cat), tagged_subject_slugs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_clip_unreviewed(engine: Engine) -> None:
    """Clip with no frame subjects and ``reviewed_at IS NULL`` → all false, empty slugs."""
    with get_session(engine) as session:
        cam_id = _seed_camera(session)
        clip = _make_clip(cam_id, has_cat=False)
        session.add(clip)
        session.flush()
        clip_id = clip.id

    has_manual_cat, effective_has_cat, tagged_slugs = _query_view(engine, clip_id)

    assert has_manual_cat is False
    assert effective_has_cat is False
    assert tagged_slugs == ""


def test_empty_clip_unreviewed_detector_cat(engine: Engine) -> None:
    """Unreviewed clip where detector says cat → ``effective_has_cat`` reflects detector."""
    with get_session(engine) as session:
        cam_id = _seed_camera(session)
        clip = _make_clip(cam_id, has_cat=True)
        session.add(clip)
        session.flush()
        clip_id = clip.id

    has_manual_cat, effective_has_cat, tagged_slugs = _query_view(engine, clip_id)

    assert has_manual_cat is False
    assert effective_has_cat is True  # detector verdict
    assert tagged_slugs == ""


def test_cat_membership_unreviewed(engine: Engine) -> None:
    """One cat membership, unreviewed → ``has_manual_cat=TRUE``, ``effective_has_cat`` = detector."""
    with get_session(engine) as session:
        cam_id = _seed_camera(session)
        subj = _make_subject("marcel", "cat", 1)
        session.add(subj)
        clip = _make_clip(cam_id, has_cat=False)  # detector says no-cat
        session.add(clip)
        session.flush()
        frame = _make_frame(clip.id)
        session.add(frame)
        session.flush()
        session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=subj.id))
        clip_id = clip.id

    has_manual_cat, effective_has_cat, tagged_slugs = _query_view(engine, clip_id)

    assert has_manual_cat is True
    assert effective_has_cat is False  # unreviewed → still detector (False)
    assert tagged_slugs == "marcel"


def test_cat_membership_reviewed(engine: Engine) -> None:
    """One cat membership, reviewed → ``effective_has_cat`` uses ``has_manual_cat``."""
    with get_session(engine) as session:
        cam_id = _seed_camera(session)
        subj = _make_subject("marcel", "cat", 1)
        session.add(subj)
        clip = _make_clip(cam_id, has_cat=False, reviewed_at=_T0)
        session.add(clip)
        session.flush()
        frame = _make_frame(clip.id)
        session.add(frame)
        session.flush()
        session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=subj.id))
        clip_id = clip.id

    has_manual_cat, effective_has_cat, tagged_slugs = _query_view(engine, clip_id)

    assert has_manual_cat is True
    assert effective_has_cat is True
    assert tagged_slugs == "marcel"


def test_false_positive_correction(engine: Engine) -> None:
    """Detector says cat, reviewer tags nothing, reviewed → ``effective_has_cat=FALSE``."""
    with get_session(engine) as session:
        cam_id = _seed_camera(session)
        clip = _make_clip(cam_id, has_cat=True, reviewed_at=_T0)
        session.add(clip)
        session.flush()
        clip_id = clip.id

    has_manual_cat, effective_has_cat, tagged_slugs = _query_view(engine, clip_id)

    assert has_manual_cat is False
    assert effective_has_cat is False  # FP correction
    assert tagged_slugs == ""


def test_event_only_memberships(engine: Engine) -> None:
    """Event subjects → ``has_manual_cat=FALSE``, slug still included in ``tagged_subject_slugs``."""
    with get_session(engine) as session:
        cam_id = _seed_camera(session)
        subj = _make_subject("litter-change", "event", 1)
        session.add(subj)
        clip = _make_clip(cam_id, has_cat=False)
        session.add(clip)
        session.flush()
        frame = _make_frame(clip.id)
        session.add(frame)
        session.flush()
        session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=subj.id))
        clip_id = clip.id

    has_manual_cat, _effective, tagged_slugs = _query_view(engine, clip_id)

    assert has_manual_cat is False
    assert tagged_slugs == "litter-change"


def test_archived_cat_counts_toward_has_manual_cat(engine: Engine) -> None:
    """Archived cat subject still counts toward ``has_manual_cat`` and appears in slugs with no marker."""
    with get_session(engine) as session:
        cam_id = _seed_camera(session)
        subj = _make_subject("old-cat", "cat", 1)
        subj.archived_at = _T0  # archive the subject
        session.add(subj)
        clip = _make_clip(cam_id, has_cat=False, reviewed_at=_T0)
        session.add(clip)
        session.flush()
        frame = _make_frame(clip.id)
        session.add(frame)
        session.flush()
        session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=subj.id))
        clip_id = clip.id

    has_manual_cat, effective_has_cat, tagged_slugs = _query_view(engine, clip_id)

    assert has_manual_cat is True
    assert effective_has_cat is True
    # The view emits the bare slug; archived-state annotation is a presentation concern for callers.
    assert tagged_slugs == "old-cat"


def test_same_subject_on_multiple_frames_dedupes_slug(engine: Engine) -> None:
    """A subject tagged on several frames appears once in ``tagged_subject_slugs`` (view uses DISTINCT)."""
    with get_session(engine) as session:
        cam_id = _seed_camera(session)
        subj = _make_subject("marcel", "cat", 1)
        session.add(subj)
        clip = _make_clip(cam_id, has_cat=False)
        session.add(clip)
        session.flush()
        for ordinal in range(3):
            frame = _make_frame(clip.id, ordinal)
            session.add(frame)
            session.flush()
            session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=subj.id))
        clip_id = clip.id

    has_manual_cat, _effective, tagged_slugs = _query_view(engine, clip_id)

    assert has_manual_cat is True
    assert tagged_slugs == "marcel"


def test_has_manual_cat_is_never_null_for_empty_clip(engine: Engine) -> None:
    """``has_manual_cat`` is a concrete 0 (never SQL NULL) when a clip has no memberships.

    Queries the raw column rather than the ``bool()``-coercing ``_query_view`` helper — a regressed
    view returning NULL would silently read as ``False`` through that helper and the
    ``effective_has_cat`` COALESCE-collapse the spec warns about would go undetected.
    """
    with get_session(engine) as session:
        cam_id = _seed_camera(session)
        clip = _make_clip(cam_id, has_cat=False)
        session.add(clip)
        session.flush()
        clip_id = clip.id

    with engine.connect() as conn:
        # scalar_one() is typed Any; cast narrows it so the not-None / value checks are type-checked.
        raw = cast(
            "int | None",
            conn.execute(
                text("SELECT has_manual_cat FROM clip_label_summary WHERE clip_id = :id"),
                {"id": clip_id},
            ).scalar_one(),
        )

    assert raw is not None
    assert raw == 0


def test_multi_kind_ordering(engine: Engine) -> None:
    """Cats appear before events; within kind ordered by ``display_order``."""
    with get_session(engine) as session:
        cam_id = _seed_camera(session)
        # Cats — intentionally added in reverse display_order to verify sort
        cat_b = _make_subject("zsolt", "cat", 2)
        cat_a = _make_subject("asha", "cat", 1)
        # Events
        evt = _make_subject("litter-change", "event", 1)
        session.add_all([cat_b, cat_a, evt])

        clip = _make_clip(cam_id, has_cat=False)
        session.add(clip)
        session.flush()

        frame = _make_frame(clip.id)
        session.add(frame)
        session.flush()

        # Tag all three subjects on the same frame
        session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=cat_b.id))
        session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=cat_a.id))
        session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=evt.id))
        clip_id = clip.id

    _has_manual, _effective, tagged_slugs = _query_view(engine, clip_id)

    assert tagged_slugs == "asha,zsolt,litter-change"

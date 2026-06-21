"""Unit tests for cat_watcher.subjects_sync."""

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from cat_watcher.db import Base, Camera, Clip, ClipFrame, ClipFrameSubject, PollStatus, Subject, create_engine, get_session
from cat_watcher.subjects_sync import ConfiguredSubject, SyncError, sync_subjects, sync_subjects_at_startup

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """File-backed SQLite engine with the full schema applied; disposed in teardown."""
    db_path = tmp_path / "test.sqlite"
    eng = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


def _subject_rows(session: Session) -> list[Subject]:
    return list(session.query(Subject).order_by(Subject.slug).all())


def _six_subjects() -> list[ConfiguredSubject]:
    return [
        ConfiguredSubject(slug="marcel", display_name="Marcel", kind="cat", display_order=1),
        ConfiguredSubject(slug="callie", display_name="Callie", kind="cat", display_order=2),
        ConfiguredSubject(slug="shadow", display_name="Shadow", kind="cat", display_order=3),
        ConfiguredSubject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1),
        ConfiguredSubject(slug="feeding", display_name="Feeding", kind="event", display_order=2),
        ConfiguredSubject(slug="playing", display_name="Playing", kind="event", display_order=3),
    ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_empty_db(engine: Engine) -> None:
    """Empty DB + 6-subject config → 6 added, 0 reactivated, 0 archived; rows present."""
    subjects = _six_subjects()
    with get_session(engine) as session:
        report = sync_subjects(session, subjects)

    assert report.added == 6
    assert report.reactivated == 0
    assert report.archived == 0

    with get_session(engine) as session:
        rows = _subject_rows(session)
    assert len(rows) == 6
    slugs = {r.slug for r in rows}
    assert slugs == {"marcel", "callie", "shadow", "cleaning", "feeding", "playing"}
    # All should be active (no archived_at)
    for row in rows:
        assert row.archived_at is None


def test_happy_path_fields_round_trip(engine: Engine) -> None:
    """Fields set in config land correctly on the DB row."""
    subjects = [
        ConfiguredSubject(
            slug="marcel",
            display_name="Marcel the Cat",
            kind="cat",
            display_order=1,
            description="Our oldest cat",
            color="#ff9900",
        ),
    ]
    with get_session(engine) as session:
        _ = sync_subjects(session, subjects)

    with get_session(engine) as session:
        row = session.query(Subject).filter_by(slug="marcel").one()
    assert row.display_name == "Marcel the Cat"
    assert row.kind == "cat"
    assert row.display_order == 1
    assert row.description == "Our oldest cat"
    assert row.color == "#ff9900"
    assert row.archived_at is None


# ---------------------------------------------------------------------------
# Idempotent
# ---------------------------------------------------------------------------


def test_idempotent_second_run_reports_zero(engine: Engine) -> None:
    """Running sync twice with the same config → second run reports 0/0/0."""
    subjects = _six_subjects()
    with get_session(engine) as session:
        _ = sync_subjects(session, subjects)

    with get_session(engine) as session:
        report = sync_subjects(session, subjects)

    assert report.added == 0
    assert report.reactivated == 0
    assert report.archived == 0


# ---------------------------------------------------------------------------
# Reactivation
# ---------------------------------------------------------------------------


def test_reactivation(engine: Engine) -> None:
    """Archive a row manually, re-add it to config → 0 added, 1 reactivated, 0 archived."""
    subjects = _six_subjects()
    with get_session(engine) as session:
        _ = sync_subjects(session, subjects)

    # Manually archive one row
    with get_session(engine) as session:
        row = session.query(Subject).filter_by(slug="callie").one()
        row.archived_at = datetime.now(UTC)

    # Run sync with "callie" still in config
    with get_session(engine) as session:
        report = sync_subjects(session, subjects)

    assert report.added == 0
    assert report.reactivated == 1
    assert report.archived == 0

    with get_session(engine) as session:
        row = session.query(Subject).filter_by(slug="callie").one()
    assert row.archived_at is None
    # display_order should be refreshed to config value
    assert row.display_order == 2


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


def test_removal(engine: Engine) -> None:
    """Drop a slug from config → 0 added, 0 reactivated, 1 archived; archived_at set."""
    subjects = _six_subjects()
    with get_session(engine) as session:
        _ = sync_subjects(session, subjects)

    # Remove "shadow" from the configured list
    reduced = [s for s in subjects if s.slug != "shadow"]
    with get_session(engine) as session:
        report = sync_subjects(session, reduced)

    assert report.added == 0
    assert report.reactivated == 0
    assert report.archived == 1

    with get_session(engine) as session:
        row = session.query(Subject).filter_by(slug="shadow").one()
    assert row.archived_at is not None
    # Other fields untouched
    assert row.display_name == "Shadow"
    assert row.kind == "cat"


# ---------------------------------------------------------------------------
# Kind change propagates
# ---------------------------------------------------------------------------


def test_kind_change_propagates(engine: Engine) -> None:
    """Change a slug's kind in config → row's kind updated; existing memberships survive.

    A ``ClipFrameSubject`` row linked to the subject must not be deleted by the kind change, proving
    that ``ON DELETE RESTRICT`` on the subject FK is not violated and the kind column is updated
    in-place.
    """
    subjects = [
        ConfiguredSubject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1),
        ConfiguredSubject(slug="special", display_name="Special", kind="cat", display_order=1),
    ]
    with get_session(engine) as session:
        _ = sync_subjects(session, subjects)

    # Seed a Camera → Clip → ClipFrame → ClipFrameSubject chain against "special".
    with get_session(engine) as session:
        cam = Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)
        session.add(cam)
        session.flush()

        now = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        clip = Clip(
            camera_id=cam.id,
            source_filename="120000.mp4",
            start_ts=now,
            end_ts=now,
            duration_seconds=2.0,
            file_path="clips/pantry/2025-01-01/120000.mp4",
            thumb_path="thumbs/pantry/2025-01-01/120000.mp4.jpg",
            file_size_bytes=1024,
            has_cat=False,
            ingested_at=now,
            detector_version="test@deadbeef",
        )
        session.add(clip)
        session.flush()

        frame = ClipFrame(clip_id=clip.id, ordinal=0, t_offset_seconds=0.0, score=0.0, thumb_path="t.jpg")
        session.add(frame)
        session.flush()

        subject_id = session.query(Subject).filter_by(slug="special").one().id
        membership = ClipFrameSubject(clip_frame_id=frame.id, subject_id=subject_id)
        session.add(membership)

    # Change "special" from cat → event; adjust order to avoid collision with "cleaning".
    changed = [
        ConfiguredSubject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1),
        ConfiguredSubject(slug="special", display_name="Special", kind="event", display_order=2),
    ]
    with get_session(engine) as session:
        report = sync_subjects(session, changed)

    assert report.added == 0

    with get_session(engine) as session:
        row = session.query(Subject).filter_by(slug="special").one()
    assert row.kind == "event"
    assert row.display_order == 2

    # Membership must survive — kind change must not cascade-delete or violate RESTRICT.
    with get_session(engine) as session:
        count = session.query(ClipFrameSubject).filter_by(subject_id=row.id).count()
    assert count == 1


# ---------------------------------------------------------------------------
# Abort: duplicate slug
# ---------------------------------------------------------------------------


def test_abort_duplicate_slug(engine: Engine) -> None:
    """Duplicate slug in config → SyncError(reason='duplicate_slug'); DB unchanged."""
    subjects = [
        ConfiguredSubject(slug="marcel", display_name="Marcel", kind="cat", display_order=1),
        ConfiguredSubject(slug="marcel", display_name="Marcel Dup", kind="cat", display_order=2),
    ]
    with get_session(engine) as session, pytest.raises(SyncError) as exc_info:
        _ = sync_subjects(session, subjects)

    assert exc_info.value.reason == "duplicate_slug"
    assert exc_info.value.offending_slug == "marcel"

    # DB must be unchanged
    with get_session(engine) as session:
        rows = _subject_rows(session)
    assert not rows


# ---------------------------------------------------------------------------
# Abort: invalid kind
# ---------------------------------------------------------------------------


def test_abort_invalid_kind(engine: Engine) -> None:
    """Invalid kind value → SyncError(reason='invalid_kind'); DB unchanged."""
    subjects = [
        ConfiguredSubject(slug="oops", display_name="Oops", kind="dog", display_order=1),
    ]
    with get_session(engine) as session, pytest.raises(SyncError) as exc_info:
        _ = sync_subjects(session, subjects)

    assert exc_info.value.reason == "invalid_kind"
    assert exc_info.value.offending_slug == "oops"
    # The human-readable message names the offending kind so the operator can find the bad entry.
    assert "dog" in exc_info.value.message

    with get_session(engine) as session:
        rows = _subject_rows(session)
    assert not rows


# ---------------------------------------------------------------------------
# Abort: display_order collision
# ---------------------------------------------------------------------------


def test_abort_display_order_collision(engine: Engine) -> None:
    """Two entries of same kind + same display_order → SyncError(reason='display_order_collision')."""
    subjects = [
        ConfiguredSubject(slug="marcel", display_name="Marcel", kind="cat", display_order=1),
        ConfiguredSubject(slug="callie", display_name="Callie", kind="cat", display_order=1),
    ]
    with get_session(engine) as session, pytest.raises(SyncError) as exc_info:
        _ = sync_subjects(session, subjects)

    assert exc_info.value.reason == "display_order_collision"

    with get_session(engine) as session:
        rows = _subject_rows(session)
    assert not rows


# ---------------------------------------------------------------------------
# Abort: first-letter collision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name_a", "name_b"),
    [
        ("Cleaning", "Crafting"),  # same-case first letter
        ("cleaning", "Crafting"),  # case-insensitive: lowercase 'c' collides with 'C'
    ],
)
def test_abort_first_letter_collision(engine: Engine, name_a: str, name_b: str) -> None:
    """Two active entries of same kind sharing a first letter → SyncError, case-insensitively; DB unchanged."""
    subjects = [
        ConfiguredSubject(slug="cleaning", display_name=name_a, kind="event", display_order=1),
        ConfiguredSubject(slug="crafting", display_name=name_b, kind="event", display_order=2),
    ]
    with get_session(engine) as session, pytest.raises(SyncError) as exc_info:
        _ = sync_subjects(session, subjects)

    assert exc_info.value.reason == "first_letter_collision"

    with get_session(engine) as session:
        rows = _subject_rows(session)
    assert not rows


# ---------------------------------------------------------------------------
# Abort: (kind, display_order) collision introduced by a kind change
# ---------------------------------------------------------------------------


def test_abort_display_order_collision_from_kind_change(engine: Engine) -> None:
    """Kind-change that creates a (kind, display_order) collision → SyncError(reason='display_order_collision').

    Setup: "cleaning" is event order=1; "marcel" is cat order=1 (different kinds → no collision).
    Action: change "marcel" to kind=event, order=1 → both are now (event, 1) → pre-sync validation
    catches the collision and raises SyncError before any DB writes.
    """
    initial = [
        ConfiguredSubject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1),
        ConfiguredSubject(slug="marcel", display_name="Marcel", kind="cat", display_order=1),
    ]
    with get_session(engine) as session:
        _ = sync_subjects(session, initial)

    # Change marcel's kind to event, same display_order → (event, 1) collision with "cleaning"
    changed = [
        ConfiguredSubject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1),
        ConfiguredSubject(slug="marcel", display_name="Marcel", kind="event", display_order=1),
    ]
    with get_session(engine) as session, pytest.raises(SyncError) as exc_info:
        _ = sync_subjects(session, changed)

    assert exc_info.value.reason == "display_order_collision"
    assert exc_info.value.offending_slug == "marcel"

    # DB unchanged — both subjects still in their original state
    with get_session(engine) as session:
        cleaning = session.query(Subject).filter_by(slug="cleaning").one()
        marcel = session.query(Subject).filter_by(slug="marcel").one()
    assert cleaning.kind == "event"
    assert cleaning.display_order == 1
    assert marcel.kind == "cat"
    assert marcel.display_order == 1


# ---------------------------------------------------------------------------
# Archived rows exempt from uniqueness checks
# ---------------------------------------------------------------------------


def test_archived_rows_exempt_from_uniqueness_checks(engine: Engine) -> None:
    """Archived subjects don't count toward (kind, display_order) or first-letter uniqueness.

    Two previously-active subjects both named "Cleaning / C..." can exist archived, and the active
    set can reuse their positions.
    """
    # Seed two archived rows that would collide if they were active
    with get_session(engine) as session:
        session.add(
            Subject(
                slug="cleaning-old",
                display_name="Cleaning Old",
                kind="event",
                display_order=1,
                archived_at=datetime.now(UTC),
            ),
        )
        session.add(
            Subject(
                slug="crafting-old",
                display_name="Crafting Old",
                kind="event",
                display_order=1,
                archived_at=datetime.now(UTC),
            ),
        )

    # Now sync with an active subject that reuses order=1 for event and starts with 'C'
    subjects = [
        ConfiguredSubject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1),
    ]
    with get_session(engine) as session:
        report = sync_subjects(session, subjects)

    assert report.added == 1
    assert report.archived == 0


# ---------------------------------------------------------------------------
# Abort: DB unchanged after failure (transaction rollback)
# ---------------------------------------------------------------------------


def test_db_unchanged_after_abort_display_order_collision(engine: Engine) -> None:
    """Failed sync leaves the DB in its pre-call state (no partial writes)."""
    # Seed a valid row first
    with get_session(engine) as session:
        session.add(
            Subject(
                slug="feeding",
                display_name="Feeding",
                kind="event",
                display_order=2,
            ),
        )

    bad_subjects = [
        ConfiguredSubject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1),
        ConfiguredSubject(slug="crafting", display_name="Crafting", kind="event", display_order=1),
    ]
    with get_session(engine) as session, pytest.raises(SyncError):
        _ = sync_subjects(session, bad_subjects)

    # Only the seeded row remains; the bad sync added nothing
    with get_session(engine) as session:
        rows = _subject_rows(session)
    assert len(rows) == 1
    assert rows[0].slug == "feeding"


# ---------------------------------------------------------------------------
# Post-sync first-letter collision from reactivation
# ---------------------------------------------------------------------------


def test_abort_post_sync_first_letter_collision_from_reactivation(engine: Engine) -> None:
    """Reactivating an archived slug that causes a first-letter collision is rejected.

    Setup: "cleaning" active (event, order=1). "crafting" archived (event, order=2).
    Action: include "crafting" in config → reactivation causes first-letter collision.
    """
    # Seed: cleaning active, crafting archived
    with get_session(engine) as session:
        session.add(
            Subject(
                slug="cleaning",
                display_name="Cleaning",
                kind="event",
                display_order=1,
            ),
        )
        session.add(
            Subject(
                slug="crafting",
                display_name="Crafting",
                kind="event",
                display_order=2,
                archived_at=datetime.now(UTC),
            ),
        )

    # Try to reactivate crafting by including it in config
    subjects = [
        ConfiguredSubject(slug="cleaning", display_name="Cleaning", kind="event", display_order=1),
        ConfiguredSubject(slug="crafting", display_name="Crafting", kind="event", display_order=2),
    ]
    with get_session(engine) as session, pytest.raises(SyncError) as exc_info:
        _ = sync_subjects(session, subjects)

    assert exc_info.value.reason == "first_letter_collision"

    # DB unchanged: cleaning still active, crafting still archived
    with get_session(engine) as session:
        cleaning = session.query(Subject).filter_by(slug="cleaning").one()
        crafting = session.query(Subject).filter_by(slug="crafting").one()
    assert cleaning.archived_at is None
    assert crafting.archived_at is not None


# ---------------------------------------------------------------------------
# Startup wrapper: sync_subjects_at_startup
# ---------------------------------------------------------------------------


def test_sync_at_startup_returns_report_and_applies_on_success(engine: Engine) -> None:
    """The startup wrapper opens its own session, applies the sync, and returns the SyncReport."""
    logger = logging.getLogger("cat_watcher.test.subjects_sync")

    report = sync_subjects_at_startup(engine, _six_subjects(), logger)

    assert report is not None
    assert report.added == 6
    with get_session(engine) as session:
        assert len(_subject_rows(session)) == 6


def test_sync_at_startup_returns_none_and_rolls_back_on_error(engine: Engine) -> None:
    """On a SyncError the wrapper returns None and leaves the subjects table unchanged (rolled back).

    A pre-existing row must survive, proving the failed sync's transaction did not partially apply.
    """
    with get_session(engine) as session:
        session.add(Subject(slug="keep", display_name="Keep", kind="cat", display_order=9))
    logger = logging.getLogger("cat_watcher.test.subjects_sync")
    bad = [ConfiguredSubject(slug="oops", display_name="Oops", kind="dog", display_order=1)]

    result = sync_subjects_at_startup(engine, bad, logger)

    assert result is None
    with get_session(engine) as session:
        rows = _subject_rows(session)
    assert [r.slug for r in rows] == ["keep"]

"""Tests for cat_watcher.db: engine factory, ORM models, session lifecycle."""

from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import Dialect, select
from sqlalchemy.exc import IntegrityError, StatementError

from cat_watcher.db import (
    AlertSent,
    AlertType,
    Base,
    Camera,
    Clip,
    Heartbeat,
    PollStatus,
    UtcDateTime,
    create_engine,
    get_session,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sqlalchemy.engine import Engine


def _file_engine(tmp_path: Path) -> Engine:
    """Build a file-based SQLite engine; in-memory SQLite cannot enable WAL mode."""
    db_path = tmp_path / "test.sqlite"
    return create_engine(f"sqlite:///{db_path}")


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Per-test file-backed SQLite engine with the schema materialized.

    Disposes the engine in teardown so SQLAlchemy's connection pool releases all sqlite3 handles
    before pytest's ``filterwarnings = error`` escalates a ``ResourceWarning`` from a GC-finalized
    connection.
    """
    eng = _file_engine(tmp_path)
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


def _make_camera() -> Camera:
    """Build a minimally-valid Camera row for tests."""
    return Camera(
        name="pantry",
        display_name="Pantry Litter Box Camera",
        host="10.0.0.50",
        poll_status=PollStatus.OK,
    )


def _make_clip(camera: Camera, *, source_filename: str = "2026/05/03/12.00.00-12.00.30[M][0@0][0].mp4") -> Clip:
    """Build a minimally-valid Clip row attached to ``camera``."""
    start = datetime(2026, 5, 3, 12, 0, 0, tzinfo=UTC)
    end = datetime(2026, 5, 3, 12, 0, 30, tzinfo=UTC)
    return Clip(
        camera=camera,
        source_filename=source_filename,
        start_ts=start,
        end_ts=end,
        duration_seconds=30.0,
        file_path="clips/pantry/2026/05/03/12.00.00.mp4",
        thumb_path="thumbs/pantry/2026/05/03/12.00.00.jpg",
        file_size_bytes=1_234_567,
        has_cat=True,
        max_score=0.92,
        frames_sampled=5,
        frames_with_cat=3,
        detector_version="yolov11n.pt@deadbeef",
        ingested_at=datetime(2026, 5, 3, 12, 1, 0, tzinfo=UTC),
    )


def test_create_engine_enables_wal_and_foreign_keys(tmp_path: Path) -> None:
    """A fresh connection must report ``journal_mode=wal`` and ``foreign_keys=1``.

    SQLite's WAL mode is persistent on the database file, so we verify it is set after the
    connect-time PRAGMA fires. ``foreign_keys`` is per-connection, so confirming it here proves the
    connect listener actually ran for this new connection.
    """
    eng = _file_engine(tmp_path)
    try:
        with eng.connect() as conn:
            journal_mode = cast("str", conn.exec_driver_sql("PRAGMA journal_mode").scalar_one())
            foreign_keys = cast("int", conn.exec_driver_sql("PRAGMA foreign_keys").scalar_one())
            synchronous = cast("int", conn.exec_driver_sql("PRAGMA synchronous").scalar_one())
    finally:
        eng.dispose()

    assert journal_mode == "wal"
    assert foreign_keys == 1
    # ``synchronous=NORMAL`` is the documented WAL companion (level 1).
    assert synchronous == 1


def test_round_trip_camera_clip_alert(engine: Engine) -> None:
    """Insert Camera + Clip + AlertSent, then reload from a fresh session and verify values."""
    camera = _make_camera()
    clip = _make_clip(camera)
    alert = AlertSent(
        alert_type=AlertType.INACTIVITY,
        camera=camera,
        sent_at=datetime(2026, 5, 3, 13, 0, 0, tzinfo=UTC),
        subject="Pantry: no cat seen for 12h",
        body="Inactivity threshold exceeded.",
        email_ok=True,
        macos_ok=False,
    )

    with get_session(engine) as session:
        session.add_all([camera, clip, alert])

    # Fresh session: reload from disk to prove the commit persisted the rows
    # and the FK relationships round-trip correctly.
    with get_session(engine) as session:
        loaded_camera = session.scalars(select(Camera).where(Camera.name == "pantry")).one()
        assert loaded_camera.display_name == "Pantry Litter Box Camera"
        assert loaded_camera.poll_status is PollStatus.OK

        assert len(loaded_camera.clips) == 1
        loaded_clip = loaded_camera.clips[0]
        assert loaded_clip.has_cat is True
        # ``pytest.approx`` is partially-typed in the upstream stubs; the comparison is
        # well-defined at runtime and the suppression keeps basedpyright clean.
        assert loaded_clip.max_score == pytest.approx(0.92)  # pyright: ignore[reportUnknownMemberType]
        assert loaded_clip.start_ts == datetime(2026, 5, 3, 12, 0, 0, tzinfo=UTC)
        # tz-aware DateTime columns must hand back tz-aware datetimes.
        assert loaded_clip.start_ts.tzinfo is not None
        assert loaded_clip.manual_has_cat is None

        assert len(loaded_camera.alerts) == 1
        loaded_alert = loaded_camera.alerts[0]
        assert loaded_alert.alert_type is AlertType.INACTIVITY
        assert loaded_alert.email_ok is True
        assert loaded_alert.macos_ok is False


def test_clip_unique_camera_source_filename(engine: Engine) -> None:
    """Inserting two clips with the same (camera_id, source_filename) raises IntegrityError."""
    camera = _make_camera()
    clip = _make_clip(camera, source_filename="dup.mp4")
    with get_session(engine) as session:
        session.add_all([camera, clip])

    duplicate = _make_clip(camera, source_filename="dup.mp4")
    with pytest.raises(IntegrityError), get_session(engine) as session:
        session.add(duplicate)


def test_get_session_rolls_back_on_exception(engine: Engine) -> None:
    """An exception inside the ``with`` block triggers rollback; no rows persist."""
    msg = "boom"

    def _add_then_raise() -> None:
        with get_session(engine) as session:
            session.add(_make_camera())
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match=msg):
        _add_then_raise()

    with get_session(engine) as session:
        assert session.scalars(select(Camera)).all() == []


def test_camera_cascade_deletes_clips(engine: Engine) -> None:
    """Deleting a Camera removes its dependent Clip rows (FK ON DELETE CASCADE)."""
    camera = _make_camera()
    clip = _make_clip(camera)
    with get_session(engine) as session:
        session.add_all([camera, clip])

    with get_session(engine) as session:
        loaded = session.scalars(select(Camera).where(Camera.name == "pantry")).one()
        session.delete(loaded)

    with get_session(engine) as session:
        assert session.scalars(select(Camera)).all() == []
        assert session.scalars(select(Clip)).all() == []


def test_alert_sent_survives_camera_deletion(engine: Engine) -> None:
    """AlertSent rows persist after their Camera is deleted (no cascade — keep history).

    Uses a nullable ``camera_id`` FK with no ``ondelete`` action: SQLite blocks the delete
    (RESTRICT) unless the application clears the column first. The test clears it explicitly, then
    verifies the alert row survives.
    """
    camera = _make_camera()
    alert = AlertSent(
        alert_type=AlertType.INACTIVITY,
        camera=camera,
        sent_at=datetime(2026, 5, 3, 13, 0, 0, tzinfo=UTC),
        subject="subj",
        body="body",
        email_ok=True,
        macos_ok=True,
    )
    with get_session(engine) as session:
        session.add_all([camera, alert])

    # Detach the alert from the camera before delete; otherwise SQLite blocks the delete on the FK
    # (no ON DELETE action to clear the column for us).
    with get_session(engine) as session:
        loaded_alert = session.scalars(select(AlertSent)).one()
        loaded_alert.camera = None
        loaded_camera = session.scalars(select(Camera)).one()
        session.delete(loaded_camera)

    with get_session(engine) as session:
        surviving = session.scalars(select(AlertSent)).all()
        assert len(surviving) == 1
        assert surviving[0].camera_id is None


def test_heartbeat_round_trip(engine: Engine) -> None:
    """Heartbeat (PK on agent_name) inserts and reads back correctly.

    The PK-on-string pattern is non-default for SQLAlchemy ORM models, so we verify the round-trip
    explicitly rather than relying on the broader Camera/Clip/Alert round-trip.
    """
    now = datetime(2026, 5, 3, 14, 0, 0, tzinfo=UTC)
    with get_session(engine) as session:
        session.add(Heartbeat(agent_name="poller", last_seen_at=now))

    with get_session(engine) as session:
        hb = session.scalars(select(Heartbeat).where(Heartbeat.agent_name == "poller")).one()
        assert hb.last_seen_at == now


def test_naive_datetime_is_rejected_on_insert(engine: Engine) -> None:
    """Binding a naive datetime to a UTC column raises ``ValueError`` (no silent drift).

    The ``UtcDateTime`` decorator rejects naive datetimes at bind time so a stray ``datetime.now()``
    (without ``tz=UTC``) cannot quietly land in the DB labeled as UTC.
    """
    naive = datetime(2026, 5, 3, 12, 0, 0, tzinfo=None)  # noqa: DTZ001  # intentional: this is the failure path under test
    camera = Camera(
        name="naive",
        display_name="naive",
        host="x",
        poll_status=PollStatus.OK,
        last_polled_at=naive,
    )

    def _flush() -> None:
        with get_session(engine) as session:
            session.add(camera)
            session.flush()

    # SQLAlchemy wraps the bind-time ValueError in StatementError with the original ValueError
    # chained as ``orig``; assert on both layers.
    with pytest.raises(StatementError) as exc_info:
        _flush()
    assert isinstance(exc_info.value.orig, ValueError)
    assert "naive datetime rejected" in str(exc_info.value.orig)


def test_utc_datetime_result_value_normalizes_aware_input_to_utc() -> None:
    """``process_result_value`` converts a tz-aware non-UTC datetime to UTC.

    SQLite always hands back naive datetimes, so the "input already has tzinfo" branch is only
    reachable on dialects that round-trip tz-aware values (Postgres, MySQL with proper config).
    Test the decorator directly to cover the branch.
    """
    decorator = UtcDateTime()
    # ``dialect`` is ignored by both methods at runtime; cast a sentinel object so type-checkers
    # accept the call without us having to construct a real dialect.
    fake_dialect = cast("Dialect", object())

    naive_input = datetime(2026, 5, 3, 12, 0, 0, tzinfo=None)  # noqa: DTZ001  # exercising the SQLite-style naive return path
    aware_naive = decorator.process_result_value(naive_input, dialect=fake_dialect)
    assert aware_naive == datetime(2026, 5, 3, 12, 0, 0, tzinfo=UTC)

    # Branch under test: input already has a non-UTC tzinfo.
    eastern = timezone(timedelta(hours=-5))
    converted = decorator.process_result_value(datetime(2026, 5, 3, 7, 0, 0, tzinfo=eastern), dialect=fake_dialect)
    assert converted == datetime(2026, 5, 3, 12, 0, 0, tzinfo=UTC)
    assert converted is not None
    assert converted.tzinfo is UTC

    # Coverage of the ``value is None`` early-return.
    assert decorator.process_result_value(None, dialect=fake_dialect) is None
    assert decorator.process_bind_param(None, dialect=fake_dialect) is None

    # Cover ``process_literal_param`` (round-trips through bind-param normalization).
    literal = decorator.process_literal_param(datetime(2026, 5, 3, 12, 0, 0, tzinfo=eastern), dialect=fake_dialect)
    # Eastern (-5h) 12:00 should normalize to UTC 17:00 in the rendered literal.
    assert "17, 0" in literal
    assert "2026, 5, 3" in literal

    # Cover ``python_type`` accessor.
    assert decorator.python_type is datetime


def test_utc_datetime_normalizes_aware_non_utc_input_on_bind(engine: Engine) -> None:
    """An aware, non-UTC datetime is converted to UTC before persistence (no wall-clock drift).

    Guards against a future "simplification" that drops ``value.astimezone(UTC)`` from
    ``process_bind_param``: without the conversion, an aware non-UTC datetime would store with the
    wrong wall-clock value (the original local time labeled UTC) — a silent timezone bug.
    """
    eastern = ZoneInfo("America/New_York")
    # 2026-05-03 08:00:00-04:00 (EDT) is the same instant as 2026-05-03 12:00:00 UTC.
    eastern_input = datetime(2026, 5, 3, 8, 0, 0, tzinfo=eastern)
    expected_utc = datetime(2026, 5, 3, 12, 0, 0, tzinfo=UTC)

    camera = _make_camera()
    camera.last_polled_at = eastern_input
    with get_session(engine) as session:
        session.add(camera)

    with get_session(engine) as session:
        loaded = session.scalars(select(Camera).where(Camera.name == "pantry")).one()
        assert loaded.last_polled_at == expected_utc
        # Critically: the wall-clock fields match UTC (12:00), not the original Eastern (08:00).
        assert loaded.last_polled_at is not None
        assert loaded.last_polled_at.hour == 12
        assert loaded.last_polled_at.tzinfo is not None
        assert loaded.last_polled_at.utcoffset() == timedelta(0)


def test_get_session_commits_on_clean_exit(engine: Engine) -> None:
    """A clean ``with`` block commits the changes; a fresh session sees them.

    The rollback-on-exception path is covered separately; this is the symmetric happy-path commit.
    Guards against a future refactor that accidentally drops ``session.commit()`` from the context
    manager.
    """
    with get_session(engine) as session:
        session.add(_make_camera())
        # No explicit commit; the context manager handles it on clean exit.

    with get_session(engine) as session:
        loaded = session.scalars(select(Camera).where(Camera.name == "pantry")).all()
        assert len(loaded) == 1
        assert loaded[0].name == "pantry"

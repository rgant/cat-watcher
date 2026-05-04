"""Tests for cat_watcher.retention.

The sweep operates against a real file-backed SQLite (``db_engine`` fixture) and a real ``tmp_path``
filesystem so we exercise the actual SQLAlchemy queries and ``os.unlink`` / ``os.rmdir`` calls.
Only ``time`` is mocked (via the explicit ``now`` argument) for determinism.
"""

import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path  # runtime: ``monkeypatch.setattr(Path, ...)`` patches the class itself
from typing import TYPE_CHECKING

from cat_watcher.config import RetentionConfig
from cat_watcher.db import AgentStart, AlertSent, AlertType, Camera, Clip, get_session
from cat_watcher.retention import RetentionReport, sweep

if TYPE_CHECKING:
    import pytest
    from sqlalchemy.engine import Engine


_NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


def _retention(*, clip_days: int = 30, agent_starts_days: int = 30, alerts_sent_days: int = 30) -> RetentionConfig:
    return RetentionConfig(clip_days=clip_days, agent_starts_days=agent_starts_days, alerts_sent_days=alerts_sent_days)


def _seed_camera(engine: Engine, *, name: str = "pantry") -> int:
    """Insert a Camera row and return its id."""
    cam = Camera(name=name, display_name=name.title(), host=f"{name}.local")
    with get_session(engine) as session:
        session.add(cam)
        session.flush()
        return cam.id


def _seed_clip(  # noqa: PLR0913  # test-fixture builder; inlining the args at every callsite is noisier
    engine: Engine,
    *,
    camera_id: int,
    storage_root: Path,
    rel_clip: str,
    rel_thumb: str,
    start_ts: datetime,
    file_mtime: datetime | None = None,
) -> int:
    """Insert a Clip row and create the corresponding files on disk; return clip id."""
    clip_path = storage_root / rel_clip
    thumb_path = storage_root / rel_thumb
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    _ = clip_path.write_bytes(b"clip-bytes")
    _ = thumb_path.write_bytes(b"thumb-bytes")
    if file_mtime is not None:
        ts = file_mtime.timestamp()
        os.utime(clip_path, (ts, ts))
        os.utime(thumb_path, (ts, ts))

    clip = Clip(
        camera_id=camera_id,
        source_filename=rel_clip.rsplit("/", 1)[-1],
        start_ts=start_ts,
        end_ts=start_ts + timedelta(seconds=2),
        duration_seconds=2.0,
        file_path=rel_clip,
        thumb_path=rel_thumb,
        file_size_bytes=10,
        detector_version="test@deadbeef",
        ingested_at=start_ts,
    )
    with get_session(engine) as session:
        session.add(clip)
        session.flush()
        return clip.id


def _make_orphan(storage_root: Path, rel_path: str, *, mtime: datetime) -> Path:
    """Create a file with no matching DB row; set its mtime explicitly."""
    full = storage_root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    _ = full.write_bytes(b"orphan-bytes")
    ts = mtime.timestamp()
    os.utime(full, (ts, ts))
    return full


# --- pass 1: DB-driven clip sweep -----------------------------------------------------------------


def test_pass1_removes_old_clips_and_files(db_engine: Engine, tmp_path: Path) -> None:
    """Clips with start_ts older than the cutoff are deleted from the DB and disk."""
    cam_id = _seed_camera(db_engine)
    old_ts = _NOW - timedelta(days=31)
    rel_clip = "clips/pantry/2026-03-31/060000.mp4"
    rel_thumb = "thumbs/pantry/2026-03-31/060000.jpg"
    clip_id = _seed_clip(db_engine, camera_id=cam_id, storage_root=tmp_path, rel_clip=rel_clip, rel_thumb=rel_thumb, start_ts=old_ts)

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.clips_removed_pass1 == 1
    assert not (tmp_path / rel_clip).exists()
    assert not (tmp_path / rel_thumb).exists()
    with get_session(db_engine) as session:
        assert session.get(Clip, clip_id) is None


def test_pass1_preserves_recent_clips(db_engine: Engine, tmp_path: Path) -> None:
    """Clips inside the retention window are left alone."""
    cam_id = _seed_camera(db_engine)
    young_ts = _NOW - timedelta(days=10)
    rel_clip = "clips/pantry/2026-04-21/060000.mp4"
    rel_thumb = "thumbs/pantry/2026-04-21/060000.jpg"
    clip_id = _seed_clip(db_engine, camera_id=cam_id, storage_root=tmp_path, rel_clip=rel_clip, rel_thumb=rel_thumb, start_ts=young_ts)

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.clips_removed_pass1 == 0
    assert (tmp_path / rel_clip).exists()
    assert (tmp_path / rel_thumb).exists()
    with get_session(db_engine) as session:
        assert session.get(Clip, clip_id) is not None


def test_pass1_oserror_during_unlink_does_not_raise(
    db_engine: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If a file unlink fails (race / already-gone), the row is still committed-removed and a WARNING is logged."""
    cam_id = _seed_camera(db_engine)
    old_ts = _NOW - timedelta(days=31)
    rel_clip = "clips/pantry/2026-03-31/060000.mp4"
    rel_thumb = "thumbs/pantry/2026-03-31/060000.jpg"
    _ = _seed_clip(db_engine, camera_id=cam_id, storage_root=tmp_path, rel_clip=rel_clip, rel_thumb=rel_thumb, start_ts=old_ts)

    def boom_unlink(_self: Path, *, missing_ok: bool = False) -> None:  # noqa: ARG001  # mimics Path.unlink signature  # pyright: ignore[reportUnusedParameter]
        msg = "simulated unlink failure"
        raise OSError(msg)

    monkeypatch.setattr(Path, "unlink", boom_unlink)
    logging.getLogger("cat_watcher.retention").disabled = False

    with caplog.at_level("WARNING", logger="cat_watcher.retention"):
        report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.clips_removed_pass1 == 1  # the row was deleted; only the unlink failed
    assert any("simulated unlink failure" in r.message for r in caplog.records)


# --- pass 2: orphan filesystem sweep --------------------------------------------------------------


def test_pass2_removes_old_orphans(db_engine: Engine, tmp_path: Path) -> None:
    """Files with no matching Clip row and old mtime are unlinked."""
    _ = _seed_camera(db_engine)
    old_mtime = _NOW - timedelta(days=45)
    orphan = _make_orphan(tmp_path, "clips/pantry/2026-03-17/000000.mp4", mtime=old_mtime)

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.orphans_removed_pass2 == 1
    assert not orphan.exists()


def test_pass2_preserves_recent_orphans(db_engine: Engine, tmp_path: Path) -> None:
    """A young orphan (mtime inside retention window) is preserved — it might be a partial download in progress."""
    _ = _seed_camera(db_engine)
    young_mtime = _NOW - timedelta(hours=2)
    orphan = _make_orphan(tmp_path, "clips/pantry/2026-04-30/100000.mp4", mtime=young_mtime)

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.orphans_removed_pass2 == 0
    assert orphan.exists()


def test_pass2_preserves_files_with_matching_clip_row(db_engine: Engine, tmp_path: Path) -> None:
    """A file whose path matches a surviving Clip.file_path / thumb_path is NOT treated as an orphan."""
    cam_id = _seed_camera(db_engine)
    young_ts = _NOW - timedelta(days=5)
    rel_clip = "clips/pantry/2026-04-26/120000.mp4"
    rel_thumb = "thumbs/pantry/2026-04-26/120000.jpg"
    _ = _seed_clip(
        db_engine,
        camera_id=cam_id,
        storage_root=tmp_path,
        rel_clip=rel_clip,
        rel_thumb=rel_thumb,
        start_ts=young_ts,
        # Backdate the mtime so it WOULD be orphan-swept if not for the matching DB row.
        file_mtime=_NOW - timedelta(days=60),
    )

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.orphans_removed_pass2 == 0
    assert (tmp_path / rel_clip).exists()
    assert (tmp_path / rel_thumb).exists()


def test_crash_tolerance_pass2_cleans_up_pass1_orphan(db_engine: Engine, tmp_path: Path) -> None:
    """A file whose Clip row was deleted (simulating a pass1 crash mid-sequence) gets swept by pass2."""
    cam_id = _seed_camera(db_engine)
    old_ts = _NOW - timedelta(days=60)
    rel_clip = "clips/pantry/2026-03-02/060000.mp4"
    rel_thumb = "thumbs/pantry/2026-03-02/060000.jpg"
    _ = _seed_clip(db_engine, camera_id=cam_id, storage_root=tmp_path, rel_clip=rel_clip, rel_thumb=rel_thumb, start_ts=old_ts)
    # Simulate the crash: row deleted directly, files left behind with old mtime.
    with get_session(db_engine) as session:
        for clip in session.query(Clip).all():
            session.delete(clip)
    ts = (_NOW - timedelta(days=60)).timestamp()
    os.utime(tmp_path / rel_clip, (ts, ts))
    os.utime(tmp_path / rel_thumb, (ts, ts))

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.orphans_removed_pass2 == 2
    assert not (tmp_path / rel_clip).exists()
    assert not (tmp_path / rel_thumb).exists()


# --- empty date directory cleanup -----------------------------------------------------------------


def test_empty_date_dir_removed_after_pass1(db_engine: Engine, tmp_path: Path) -> None:
    """A clips/<slug>/<date>/ directory that becomes empty during the sweep is rmdir'd."""
    cam_id = _seed_camera(db_engine)
    old_ts = _NOW - timedelta(days=45)
    rel_clip = "clips/pantry/2026-03-17/060000.mp4"
    rel_thumb = "thumbs/pantry/2026-03-17/060000.jpg"
    _ = _seed_clip(db_engine, camera_id=cam_id, storage_root=tmp_path, rel_clip=rel_clip, rel_thumb=rel_thumb, start_ts=old_ts)

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    # The clips/pantry/2026-03-17 and thumbs/pantry/2026-03-17 dirs both held one file -> both empty after pass1 -> both rmdir'd.
    assert report.dirs_removed == 2
    assert not (tmp_path / "clips/pantry/2026-03-17").exists()
    assert not (tmp_path / "thumbs/pantry/2026-03-17").exists()


def test_non_empty_date_dir_preserved(db_engine: Engine, tmp_path: Path) -> None:
    """A date directory containing a surviving (young) file is kept even if a sibling was swept."""
    cam_id = _seed_camera(db_engine)
    old_ts = _NOW - timedelta(days=45)
    young_ts = _NOW - timedelta(days=2)
    # Both clips share the same date directory; one is old, one is young.
    _ = _seed_clip(
        db_engine,
        camera_id=cam_id,
        storage_root=tmp_path,
        rel_clip="clips/pantry/2026-04-29/050000.mp4",
        rel_thumb="thumbs/pantry/2026-04-29/050000.jpg",
        start_ts=old_ts,
    )
    _ = _seed_clip(
        db_engine,
        camera_id=cam_id,
        storage_root=tmp_path,
        rel_clip="clips/pantry/2026-04-29/060000.mp4",
        rel_thumb="thumbs/pantry/2026-04-29/060000.jpg",
        start_ts=young_ts,
    )

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.clips_removed_pass1 == 1
    assert report.dirs_removed == 0
    assert (tmp_path / "clips/pantry/2026-04-29").is_dir()
    assert (tmp_path / "clips/pantry/2026-04-29/060000.mp4").exists()


# --- configurability + idempotency ----------------------------------------------------------------


def test_clip_days_configurable(db_engine: Engine, tmp_path: Path) -> None:
    """``retention.clip_days = 7`` removes clips older than 7 days; the default 30 would keep them."""
    cam_id = _seed_camera(db_engine)
    ten_day_ts = _NOW - timedelta(days=10)
    rel_clip = "clips/pantry/2026-04-21/060000.mp4"
    rel_thumb = "thumbs/pantry/2026-04-21/060000.jpg"
    _ = _seed_clip(db_engine, camera_id=cam_id, storage_root=tmp_path, rel_clip=rel_clip, rel_thumb=rel_thumb, start_ts=ten_day_ts)

    # With clip_days=30 the clip survives.
    report_default = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(clip_days=30), now=_NOW)
    assert report_default.clips_removed_pass1 == 0
    assert (tmp_path / rel_clip).exists()

    # With clip_days=7 the same clip is now past the cutoff.
    report_short = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(clip_days=7), now=_NOW)
    assert report_short.clips_removed_pass1 == 1
    assert not (tmp_path / rel_clip).exists()


def test_sweep_is_idempotent(db_engine: Engine, tmp_path: Path) -> None:
    """A second sweep with identical inputs deletes nothing additional."""
    cam_id = _seed_camera(db_engine)
    old_ts = _NOW - timedelta(days=45)
    _ = _seed_clip(
        db_engine,
        camera_id=cam_id,
        storage_root=tmp_path,
        rel_clip="clips/pantry/2026-03-17/060000.mp4",
        rel_thumb="thumbs/pantry/2026-03-17/060000.jpg",
        start_ts=old_ts,
    )

    first = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)
    second = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert first.clips_removed_pass1 == 1
    assert second == RetentionReport(
        clips_removed_pass1=0,
        orphans_removed_pass2=0,
        dirs_removed=0,
        agent_starts_pruned=0,
        alerts_sent_pruned=0,
    )


# --- agent_starts + alerts_sent pruning ------------------------------------------------------------


def test_agent_starts_pruned_by_age(db_engine: Engine, tmp_path: Path) -> None:
    """``agent_starts`` rows older than agent_starts_days are removed; recent ones survive."""
    old_start = AgentStart(agent_name="poller", started_at=_NOW - timedelta(days=45))
    young_start = AgentStart(agent_name="poller", started_at=_NOW - timedelta(days=5))
    with get_session(db_engine) as session:
        session.add_all([old_start, young_start])

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.agent_starts_pruned == 1
    with get_session(db_engine) as session:
        remaining = session.query(AgentStart).all()
        assert len(remaining) == 1
        assert remaining[0].started_at == young_start.started_at


def test_alerts_sent_pruned_by_age(db_engine: Engine, tmp_path: Path) -> None:
    """``alerts_sent`` rows older than alerts_sent_days are removed; recent ones survive."""
    cam_id = _seed_camera(db_engine)
    old_alert = AlertSent(
        alert_type=AlertType.INACTIVITY,
        camera_id=cam_id,
        sent_at=_NOW - timedelta(days=45),
        subject="old",
        body="old body",
    )
    young_alert = AlertSent(
        alert_type=AlertType.INACTIVITY,
        camera_id=cam_id,
        sent_at=_NOW - timedelta(days=5),
        subject="young",
        body="young body",
    )
    with get_session(db_engine) as session:
        session.add_all([old_alert, young_alert])

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.alerts_sent_pruned == 1
    with get_session(db_engine) as session:
        remaining = session.query(AlertSent).all()
        assert len(remaining) == 1
        assert remaining[0].subject == "young"


def test_independent_retention_windows(db_engine: Engine, tmp_path: Path) -> None:
    """Each table's prune cutoff is independent: agent_starts_days=7, alerts_sent_days=30, clip_days=30."""
    cam_id = _seed_camera(db_engine)
    # 10-day-old agent_start: pruned under agent_starts_days=7 but kept under default 30.
    with get_session(db_engine) as session:
        session.add(AgentStart(agent_name="poller", started_at=_NOW - timedelta(days=10)))
        # 10-day-old alert: kept under alerts_sent_days=30.
        session.add(
            AlertSent(
                alert_type=AlertType.INACTIVITY,
                camera_id=cam_id,
                sent_at=_NOW - timedelta(days=10),
                subject="s",
                body="b",
            ),
        )

    report = sweep(
        engine=db_engine,
        storage_root=tmp_path,
        retention=_retention(clip_days=30, agent_starts_days=7, alerts_sent_days=30),
        now=_NOW,
    )

    assert report.agent_starts_pruned == 1
    assert report.alerts_sent_pruned == 0


# --- integration / day-zero ----------------------------------------------------------------------


def test_sweep_on_fresh_storage_root_does_not_crash(db_engine: Engine, tmp_path: Path) -> None:
    """Day-zero deploy: ``storage_root`` exists but has no ``clips/`` or ``thumbs/`` subdirs yet."""
    # tmp_path is empty; no clips/, no thumbs/. The sweep must short-circuit gracefully.
    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report == RetentionReport(
        clips_removed_pass1=0,
        orphans_removed_pass2=0,
        dirs_removed=0,
        agent_starts_pruned=0,
        alerts_sent_pruned=0,
    )


def test_sweep_full_pipeline_against_representative_state(db_engine: Engine, tmp_path: Path) -> None:
    """Single sweep over a mix of all data classes; verifies the passes compose without interference.

    Seeds:
      - aged Clip with files (pass 1 should remove)
      - young Clip with files (everything should preserve)
      - aged orphan file (pass 2 should remove)
      - young orphan file (pass 2 should preserve)
      - aged AgentStart row (prune should remove)
      - young AgentStart row (preserve)
      - aged AlertSent row (prune should remove)
      - young AlertSent row (preserve)
      - a date directory whose only file is the aged Clip (becomes empty after pass 1 -> rmdir)
    """
    cam_id = _seed_camera(db_engine)
    aged = _NOW - timedelta(days=45)
    young = _NOW - timedelta(days=2)

    aged_clip_id = _seed_clip(
        db_engine,
        camera_id=cam_id,
        storage_root=tmp_path,
        rel_clip="clips/pantry/2026-03-17/050000.mp4",
        rel_thumb="thumbs/pantry/2026-03-17/050000.jpg",
        start_ts=aged,
    )
    young_clip_id = _seed_clip(
        db_engine,
        camera_id=cam_id,
        storage_root=tmp_path,
        rel_clip="clips/pantry/2026-04-29/060000.mp4",
        rel_thumb="thumbs/pantry/2026-04-29/060000.jpg",
        start_ts=young,
    )
    aged_orphan = _make_orphan(tmp_path, "clips/pantry/2026-03-01/000000.mp4", mtime=aged)
    young_orphan = _make_orphan(tmp_path, "clips/pantry/2026-04-30/100000.mp4", mtime=young)
    with get_session(db_engine) as session:
        session.add_all(
            [
                AgentStart(agent_name="poller", started_at=aged),
                AgentStart(agent_name="poller", started_at=young),
                AlertSent(alert_type=AlertType.INACTIVITY, camera_id=cam_id, sent_at=aged, subject="aged", body="b"),
                AlertSent(alert_type=AlertType.INACTIVITY, camera_id=cam_id, sent_at=young, subject="young", body="b"),
            ],
        )

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report == RetentionReport(
        clips_removed_pass1=1,  # the aged Clip
        orphans_removed_pass2=1,  # the aged orphan (the aged-Clip files were already cleaned by pass 1)
        # The aged Clip's date dir (2026-03-17) under clips/ AND thumbs/ + the aged orphan's
        # 2026-03-01 dir under clips/. The young clip's date dir survives (still has files).
        dirs_removed=3,
        agent_starts_pruned=1,
        alerts_sent_pruned=1,
    )
    # Surviving artifacts:
    with get_session(db_engine) as session:
        assert session.get(Clip, aged_clip_id) is None
        assert session.get(Clip, young_clip_id) is not None
    assert young_orphan.exists()
    assert (tmp_path / "clips/pantry/2026-04-29/060000.mp4").exists()
    assert not aged_orphan.exists()
    assert not (tmp_path / "clips/pantry/2026-03-17").exists()
    assert not (tmp_path / "thumbs/pantry/2026-03-17").exists()

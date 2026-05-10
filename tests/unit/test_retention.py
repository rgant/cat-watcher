"""Tests for cat_watcher.retention.

Sweeps run against a real file-backed SQLite (``db_engine``) and real ``tmp_path`` filesystem so the
actual SQLAlchemy queries and ``os.unlink`` / ``os.rmdir`` calls are exercised; only ``time`` is
mocked (via the explicit ``now`` argument) for determinism.
"""

import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path  # runtime: ``monkeypatch.setattr(Path, ...)`` patches the class itself
from typing import TYPE_CHECKING

from cat_watcher.config import RetentionConfig
from cat_watcher.db import AgentStart, AlertSent, AlertType, Camera, Clip, ClipFrame, get_session
from cat_watcher.retention import RetentionReport, sweep

if TYPE_CHECKING:
    import pytest
    from sqlalchemy.engine import Engine


_NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


def _retention(*, clip_days: int = 30, agent_starts_days: int = 30, alerts_sent_days: int = 30) -> RetentionConfig:
    return RetentionConfig(clip_days=clip_days, agent_starts_days=agent_starts_days, alerts_sent_days=alerts_sent_days)


def _seed_camera(engine: Engine, *, name: str = "pantry") -> int:
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
    """Insert a Clip row, create the on-disk files, and return the clip id."""
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
    full = storage_root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    _ = full.write_bytes(b"orphan-bytes")
    ts = mtime.timestamp()
    os.utime(full, (ts, ts))
    return full


# --- pass 1: DB-driven clip sweep -----------------------------------------------------------------


def test_pass1_removes_old_clips_and_files(db_engine: Engine, tmp_path: Path) -> None:
    """A clip past ``clip_days`` is deleted from the DB and both files (mp4 + thumb) unlinked."""
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
    """Clips inside the retention window survive — sweep is age-bounded, not aggressive."""
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
    """If unlink fails (race / already-gone), the row is still committed-removed and a WARNING is logged."""
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
    """Old files in the storage tree without a matching DB row are unlinked in pass 2."""
    _ = _seed_camera(db_engine)
    old_mtime = _NOW - timedelta(days=45)
    orphan = _make_orphan(tmp_path, "clips/pantry/2026-03-17/000000.mp4", mtime=old_mtime)

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.orphans_removed_pass2 == 1
    assert not orphan.exists()


def test_pass2_preserves_recent_orphans(db_engine: Engine, tmp_path: Path) -> None:
    """Young orphans are preserved — likely a partial download still in progress."""
    _ = _seed_camera(db_engine)
    young_mtime = _NOW - timedelta(hours=2)
    orphan = _make_orphan(tmp_path, "clips/pantry/2026-04-30/100000.mp4", mtime=young_mtime)

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.orphans_removed_pass2 == 0
    assert orphan.exists()


def test_pass2_preserves_files_with_matching_clip_row(db_engine: Engine, tmp_path: Path) -> None:
    """Files matching a surviving Clip.file_path / thumb_path are kept even with old mtimes."""
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
    """Files left behind by a pass1 crash (row deleted, file alive) get swept by pass2."""
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
    """A ``clips/<slug>/<date>/`` dir that becomes empty during the sweep gets rmdir'd."""
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
    """A date dir with at least one surviving file is kept even when a sibling is swept."""
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
    """A clip_days override changes the cutoff: same clip survives at 30, gets swept at 7."""
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
    """A second sweep over the same state reports zeros — no double-deletion or counter inflation."""
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
    """``agent_starts`` rows past ``agent_starts_days`` are pruned; rows inside the window survive."""
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
    """``alerts_sent`` rows past ``alerts_sent_days`` are pruned; rows inside the window survive."""
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
    """Each table's prune cutoff is independent — same 10-day-old row is pruned or kept by table-specific config."""
    cam_id = _seed_camera(db_engine)
    # 10-day-old agent_start: pruned at agent_starts_days=7 but kept at default 30.
    with get_session(db_engine) as session:
        session.add(AgentStart(agent_name="poller", started_at=_NOW - timedelta(days=10)))
        # 10-day-old alert: kept at alerts_sent_days=30.
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
    """Day-zero deploy: ``storage_root`` exists but has no ``clips/`` or ``thumbs/`` yet."""
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


# --- per-frame thumb retention (Task 9) ----------------------------------------------------------


def _write_per_frame_thumbs(storage_root: Path, per_clip_dir: str, frame_count: int) -> list[str]:
    relpaths: list[str] = []
    for ordinal in range(frame_count):
        rel = f"{per_clip_dir}/{ordinal:02d}.jpg"
        full = storage_root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        _ = full.write_bytes(b"frame-bytes")
        relpaths.append(rel)
    return relpaths


def _seed_per_frame_clip(  # noqa: PLR0913  # test-fixture builder; inlining the args at every callsite is noisier
    engine: Engine,
    *,
    camera_id: int,
    storage_root: Path,
    rel_clip: str,
    per_clip_dir: str,
    frame_count: int,
    best_ordinal: int,
    start_ts: datetime,
) -> tuple[int, list[str]]:
    """Insert a Clip with ``frame_count`` ClipFrame rows + thumbs at ``per_clip_dir/<NN>.jpg``.

    Returns ``(clip_id, frame_relpaths)``. ``Clip.thumb_path`` is set to the ordinal indicated by
    ``best_ordinal``.
    """
    clip_path = storage_root / rel_clip
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    _ = clip_path.write_bytes(b"clip-bytes")

    frame_relpaths = _write_per_frame_thumbs(storage_root, per_clip_dir, frame_count)
    clip = Clip(
        camera_id=camera_id,
        source_filename=rel_clip.rsplit("/", 1)[-1],
        start_ts=start_ts,
        end_ts=start_ts + timedelta(seconds=2),
        duration_seconds=2.0,
        file_path=rel_clip,
        thumb_path=frame_relpaths[best_ordinal],
        file_size_bytes=10,
        detector_version="test@deadbeef",
        ingested_at=start_ts,
    )
    with get_session(engine) as session:
        session.add(clip)
        session.flush()
        clip_id = clip.id
        for ordinal, rel in enumerate(frame_relpaths):
            session.add(
                ClipFrame(
                    clip_id=clip_id,
                    ordinal=ordinal,
                    t_offset_seconds=float(ordinal),
                    score=0.5,
                    thumb_path=rel,
                ),
            )
    return clip_id, frame_relpaths


def test_retention_pass1_unlinks_per_frame_thumbs(db_engine: Engine, tmp_path: Path) -> None:
    """Pass 1 unlinks every per-frame thumb and rmdirs the per-clip ``<HHMMSS>/`` subdir."""
    cam_id = _seed_camera(db_engine)
    old_ts = _NOW - timedelta(days=45)
    rel_clip = "clips/pantry/2026-03-17/103045.mp4"
    per_clip_dir = "thumbs/pantry/2026-03-17/103045"
    clip_id, frame_relpaths = _seed_per_frame_clip(
        db_engine,
        camera_id=cam_id,
        storage_root=tmp_path,
        rel_clip=rel_clip,
        per_clip_dir=per_clip_dir,
        frame_count=4,
        best_ordinal=1,
        start_ts=old_ts,
    )

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    assert report.clips_removed_pass1 == 1
    with get_session(db_engine) as session:
        assert session.get(Clip, clip_id) is None
    for rel in frame_relpaths:
        assert not (tmp_path / rel).exists()
    assert not (tmp_path / per_clip_dir).exists()
    # No flat-file leftover at <HHMMSS>.jpg either.
    assert not (tmp_path / "thumbs/pantry/2026-03-17/103045.jpg").exists()


def test_retention_pass2_treats_clip_frames_as_survivors(db_engine: Engine, tmp_path: Path) -> None:
    """Pass 2 keeps every ClipFrame.thumb_path as a survivor and only sweeps the unrelated orphan."""
    cam_id = _seed_camera(db_engine)
    young_ts = _NOW - timedelta(days=2)
    rel_clip = "clips/pantry/2026-04-29/120000.mp4"
    per_clip_dir = "thumbs/pantry/2026-04-29/120000"
    _, frame_relpaths = _seed_per_frame_clip(
        db_engine,
        camera_id=cam_id,
        storage_root=tmp_path,
        rel_clip=rel_clip,
        per_clip_dir=per_clip_dir,
        frame_count=4,
        best_ordinal=2,
        start_ts=young_ts,
    )
    # Backdate every per-frame thumb so it WOULD be orphan-swept if not protected by ClipFrame.
    old_ts = (_NOW - timedelta(days=60)).timestamp()
    for rel in frame_relpaths:
        os.utime(tmp_path / rel, (old_ts, old_ts))
    orphan = _make_orphan(
        tmp_path,
        "thumbs/pantry/2026-04-29/_orphan.jpg",
        mtime=_NOW - timedelta(days=60),
    )

    report = sweep(engine=db_engine, storage_root=tmp_path, retention=_retention(), now=_NOW)

    for rel in frame_relpaths:
        assert (tmp_path / rel).is_file()
    assert not orphan.exists()
    assert report.orphans_removed_pass2 == 1

"""Tests for cat_watcher.alerts.

Coverage roadmap (per Task 18 plan):

* Per-rule branch verification (every branch of INACTIVITY, FREQUENCY, the watchdogs, the storage
  rules) — each evaluator's trigger and non-trigger paths.
* ``poll_status != ok`` immediate-fire (no time threshold).
* Configurable thresholds (``poller_stuck_minutes`` override path).
* Cool-down: sent path writes exactly one row; suppressed path logs INFO + writes nothing; hard-
  coded overrides honored; non-camera cool-down key.
* ``BACKUP_STALE`` suppression during ``STORAGE_UNAVAILABLE`` cool-down; ``DISK_LOW`` skipped when
  storage offline.
* Manual-label override on ``FREQUENCY``.
* End-to-end: alerts agent runs even when ``storage_root`` is unmounted.

The tests disable both notifier channels via ``EmailRulesConfig(enabled=False)`` /
``MacOsRulesConfig(enabled=False)`` so dispatch_alert never reaches real SMTP / osascript. The
disabled path returns ``ok=True`` per :mod:`cat_watcher.notifier`'s contract; that's plenty to
exercise the cool-down + alerts_sent bookkeeping this module owns.
"""

import logging
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from cat_watcher.alert_templates import AlertContent
from cat_watcher.alerts import (
    DispatchEnv,
    cooldown_for,
    dispatch_alert,
    evaluate_backup_stale,
    evaluate_frequency,
    evaluate_heartbeat_watchdog,
    evaluate_inactivity,
    evaluate_inactivity_no_cats_global,
    evaluate_storage,
    evaluate_web_flapping,
    run_alerts_tick,
)
from cat_watcher.config import EmailRulesConfig, MacOsRulesConfig
from cat_watcher.db import (
    AgentStart,
    AlertSent,
    AlertType,
    Camera,
    Heartbeat,
    PollStatus,
    get_session,
)
from cat_watcher.notifier import EmailResult, NotifResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


_NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
_TZ = "UTC"
_URL = "http://localhost:8000"
_TRIVIAL_CONTENT = AlertContent(subject="subj", body="body", macos_summary="summary")


def _channels_disabled(base_config: Config) -> Config:
    """Return a copy of ``base_config`` with email + macos channels disabled (no real I/O)."""
    return base_config.model_copy(
        update={
            "alerts": base_config.alerts.model_copy(
                update={
                    "email": EmailRulesConfig(enabled=False),
                    "macos": MacOsRulesConfig(enabled=False),
                },
            ),
        },
    )


@pytest.fixture
def cfg(make_config: Callable[..., Config], tmp_path: Path) -> Config:
    """Disabled-channels config for tests that exercise dispatch_alert / run_alerts_tick."""
    return _channels_disabled(make_config(tmp_path, tmp_path))


def _dispatch_env(cfg: Config, session: Session, *, now: datetime) -> DispatchEnv:  # pylint: disable=redefined-outer-name
    """Build a ``DispatchEnv`` for tests that hand the dispatcher a live session."""
    return DispatchEnv(secrets=cfg.email, rules=cfg.alerts, session=session, now=now)


# --- INACTIVITY ----------------------------------------------------------------------------------


def test_inactivity_fires_immediately_when_poll_status_unreachable(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """``poll_status != ok`` fires on the same tick the camera goes unreachable, no time threshold."""
    cam_id = seed_camera(
        db_engine,
        last_clip_at=_NOW - timedelta(minutes=2),
        last_cat_seen_at=_NOW - timedelta(minutes=2),
        poll_status=PollStatus.UNREACHABLE,
        poll_status_since=_NOW - timedelta(minutes=2),
    )

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        cand = evaluate_inactivity(cam, inactivity_hours=12, public_url=_URL, tz_name=_TZ, now=_NOW)

    assert cand is not None
    assert cand.alert_type == AlertType.INACTIVITY
    assert cand.camera_id == cam_id
    assert "unreachable" in cand.content.body.lower()


def test_inactivity_fires_when_no_clips_for_inactivity_window(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """``last_clip_at`` older than ``inactivity_hours`` fires the ``no clips received`` branch."""
    cam_id = seed_camera(
        db_engine,
        last_polled_at=_NOW - timedelta(minutes=2),
        last_clip_at=_NOW - timedelta(hours=14),
        last_cat_seen_at=_NOW - timedelta(hours=14),
    )

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        cand = evaluate_inactivity(cam, inactivity_hours=12, public_url=_URL, tz_name=_TZ, now=_NOW)

    assert cand is not None
    assert "no clips received" in cand.content.body


def test_inactivity_per_camera_ignores_stale_last_cat_seen_at(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A stale ``last_cat_seen_at`` alone does NOT fire per-camera INACTIVITY.

    The fleet-wide no-cats branch lives in :func:`evaluate_inactivity_no_cats_global`. The per-
    camera evaluator only handles unreachable + no-clips; a camera with recent clips but stale
    cat sightings must not fire its own per-camera INACTIVITY (otherwise a cat using only one box
    today would alert on the other box).
    """
    cam_id = seed_camera(
        db_engine,
        last_polled_at=_NOW - timedelta(minutes=2),
        last_clip_at=_NOW - timedelta(hours=2),
        last_cat_seen_at=_NOW - timedelta(hours=13),
    )

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        cand = evaluate_inactivity(cam, inactivity_hours=12, public_url=_URL, tz_name=_TZ, now=_NOW)

    assert cand is None


def test_inactivity_does_not_fire_without_baseline(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Both ``last_cat_seen_at`` and ``last_clip_at`` NULL + ``poll_status=ok`` → no fire (no baseline)."""
    cam_id = seed_camera(db_engine)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        cand = evaluate_inactivity(cam, inactivity_hours=12, public_url=_URL, tz_name=_TZ, now=_NOW)
    assert cand is None


def test_inactivity_does_not_fire_within_threshold(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Recent ``last_clip_at`` and ``last_cat_seen_at`` + ok status → no fire."""
    cam_id = seed_camera(
        db_engine,
        last_polled_at=_NOW - timedelta(minutes=2),
        last_clip_at=_NOW - timedelta(hours=1),
        last_cat_seen_at=_NOW - timedelta(hours=1),
    )
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        cand = evaluate_inactivity(cam, inactivity_hours=12, public_url=_URL, tz_name=_TZ, now=_NOW)
    assert cand is None


def test_inactivity_global_no_cats_fires_when_max_last_cat_seen_at_stale(
    db_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """Two cameras with both ``last_cat_seen_at`` older than the threshold → global candidate fires."""
    _ = seed_camera(db_engine, name="pantry", display_name="Pantry", last_cat_seen_at=_NOW - timedelta(hours=14))
    _ = seed_camera(
        db_engine,
        name="bathroom",
        display_name="Bathroom",
        host="cam2.example.com",
        last_cat_seen_at=_NOW - timedelta(hours=13),
    )

    with get_session(db_engine) as session:
        cand = evaluate_inactivity_no_cats_global(session, inactivity_hours=12, public_url=_URL, tz_name=_TZ, now=_NOW)

    assert cand is not None
    assert cand.alert_type == AlertType.INACTIVITY
    assert cand.camera_id is None
    assert "no cats seen on any camera" in cand.content.body
    # The renderer identifies the most-recent camera (Bathroom at -13h, more recent than Pantry at -14h).
    assert "(Bathroom)" in cand.content.body


def test_inactivity_global_no_cats_does_not_fire_when_one_camera_recent(
    db_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """Any camera with a recent ``last_cat_seen_at`` keeps the fleet-wide branch silent."""
    _ = seed_camera(db_engine, name="pantry", display_name="Pantry", last_cat_seen_at=_NOW - timedelta(hours=1))
    _ = seed_camera(
        db_engine,
        name="bathroom",
        display_name="Bathroom",
        host="cam2.example.com",
        last_cat_seen_at=_NOW - timedelta(hours=13),
    )

    with get_session(db_engine) as session:
        cand = evaluate_inactivity_no_cats_global(session, inactivity_hours=12, public_url=_URL, tz_name=_TZ, now=_NOW)

    assert cand is None


def test_inactivity_global_no_cats_does_not_fire_without_baseline(
    db_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """No camera has ever seen a cat (fresh install before backfill) → no fire."""
    _ = seed_camera(db_engine, name="pantry", display_name="Pantry")
    _ = seed_camera(db_engine, name="bathroom", display_name="Bathroom", host="cam2.example.com")

    with get_session(db_engine) as session:
        cand = evaluate_inactivity_no_cats_global(session, inactivity_hours=12, public_url=_URL, tz_name=_TZ, now=_NOW)

    assert cand is None


def test_inactivity_global_no_cats_single_camera_still_fires(
    db_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """Single-camera deployments keep the same semantics: stale cat sightings fire the global branch."""
    _ = seed_camera(db_engine, last_cat_seen_at=_NOW - timedelta(hours=13))

    with get_session(db_engine) as session:
        cand = evaluate_inactivity_no_cats_global(session, inactivity_hours=12, public_url=_URL, tz_name=_TZ, now=_NOW)

    assert cand is not None
    assert cand.camera_id is None


def test_inactivity_no_cats_does_not_fire_for_silent_camera_when_other_camera_active(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
) -> None:
    """Two cameras: cat used box A in the last hour but skipped box B today.

    Cats don't use both litter boxes every day, so a stale ``last_cat_seen_at`` on the silent camera
    must not fire INACTIVITY when the other camera saw a cat within the threshold. The
    ``no cats seen`` check is fleet-wide, not per-camera.
    """
    active_cam = seed_camera(
        db_engine,
        name="pantry",
        display_name="Pantry",
        last_polled_at=_NOW - timedelta(minutes=2),
        last_clip_at=_NOW - timedelta(hours=1),
        last_cat_seen_at=_NOW - timedelta(hours=1),
    )
    silent_cam = seed_camera(
        db_engine,
        name="bathroom",
        display_name="Bathroom",
        host="cam2.example.com",
        last_polled_at=_NOW - timedelta(minutes=2),
        last_clip_at=_NOW - timedelta(hours=1),
        last_cat_seen_at=_NOW - timedelta(hours=13),
    )

    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).filter(AlertSent.alert_type == AlertType.INACTIVITY).all()
    assert rows == [], (
        f"INACTIVITY must not fire when any camera saw a cat within the threshold; "
        f"got {[(r.camera_id, r.subject) for r in rows]} (active_cam={active_cam}, silent_cam={silent_cam})"
    )


# --- FREQUENCY -----------------------------------------------------------------------------------


def test_frequency_fires_when_threshold_reached(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """``count(cat_positive) >= threshold`` in the window → ``FREQUENCY`` candidate with subject + body."""
    cam_id = seed_camera(db_engine)
    for i in range(10):
        seed_clip(db_engine, camera_id=cam_id, start_ts=_NOW - timedelta(hours=1, minutes=i * 5), has_cat=True)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        cand = evaluate_frequency(session, cam, window_hours=6, threshold=8, public_url=_URL, tz_name=_TZ, now=_NOW)

    assert cand is not None
    assert cand.alert_type == AlertType.FREQUENCY
    assert "(10 in 6h)" in cand.content.subject


def test_frequency_does_not_fire_below_threshold(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """``count < threshold`` → no fire."""
    cam_id = seed_camera(db_engine)
    for i in range(7):
        seed_clip(db_engine, camera_id=cam_id, start_ts=_NOW - timedelta(hours=1, minutes=i * 5), has_cat=True)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        cand = evaluate_frequency(session, cam, window_hours=6, threshold=8, public_url=_URL, tz_name=_TZ, now=_NOW)
    assert cand is None


def test_frequency_excludes_clips_with_manual_has_cat_false(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """Clips marked ``manual_has_cat=False`` (operator override) do NOT count even if ``has_cat=True``."""
    cam_id = seed_camera(db_engine)
    for i in range(8):
        seed_clip(
            db_engine,
            camera_id=cam_id,
            start_ts=_NOW - timedelta(hours=1, minutes=i * 5),
            has_cat=True,
            manual_has_cat=False if i < 5 else None,
        )
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        cand = evaluate_frequency(session, cam, window_hours=6, threshold=8, public_url=_URL, tz_name=_TZ, now=_NOW)
    assert cand is None


def test_frequency_includes_model_false_clips_marked_manual_true(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """``has_cat=False`` + ``manual_has_cat=True`` (operator promote) DOES count toward the threshold."""
    cam_id = seed_camera(db_engine)
    for i in range(8):
        seed_clip(
            db_engine,
            camera_id=cam_id,
            start_ts=_NOW - timedelta(hours=1, minutes=i * 5),
            has_cat=False,
            manual_has_cat=True,
        )
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        cand = evaluate_frequency(session, cam, window_hours=6, threshold=8, public_url=_URL, tz_name=_TZ, now=_NOW)
    assert cand is not None
    assert "(8 in 6h)" in cand.content.subject


# --- watchdog rules ------------------------------------------------------------------------------


def test_poller_stuck_fires_when_heartbeat_older_than_threshold(db_engine: Engine) -> None:
    """``POLLER_STUCK`` fires when the poller heartbeat is older than ``poller_stuck_minutes``."""
    with get_session(db_engine) as session:
        session.add(Heartbeat(agent_name="poller", last_seen_at=_NOW - timedelta(minutes=20)))

    with get_session(db_engine) as session:
        cand = evaluate_heartbeat_watchdog(
            session,
            alert_type=AlertType.POLLER_STUCK,
            agent_name="poller",
            stale_minutes=15,
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand is not None
    assert cand.alert_type == AlertType.POLLER_STUCK
    assert cand.camera_id is None


def test_poller_stuck_does_not_fire_within_threshold(db_engine: Engine) -> None:
    """Recent heartbeat → no fire."""
    with get_session(db_engine) as session:
        session.add(Heartbeat(agent_name="poller", last_seen_at=_NOW - timedelta(minutes=5)))
    with get_session(db_engine) as session:
        cand = evaluate_heartbeat_watchdog(
            session,
            alert_type=AlertType.POLLER_STUCK,
            agent_name="poller",
            stale_minutes=15,
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand is None


def test_watchdog_does_not_fire_with_missing_heartbeat(db_engine: Engine) -> None:
    """No heartbeat row at all → no fire (covers fresh-install case before agents have run once)."""
    with get_session(db_engine) as session:
        cand = evaluate_heartbeat_watchdog(
            session,
            alert_type=AlertType.WEB_DOWN,
            agent_name="web",
            stale_minutes=5,
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand is None


def test_poller_stuck_respects_configurable_threshold(db_engine: Engine) -> None:
    """Override ``poller_stuck_minutes=1``: 90s-stale heartbeat fires; 30s-stale does not."""
    stale_minutes = 1
    with get_session(db_engine) as session:
        session.add(Heartbeat(agent_name="poller", last_seen_at=_NOW - timedelta(seconds=90)))

    with get_session(db_engine) as session:
        cand_fires = evaluate_heartbeat_watchdog(
            session,
            alert_type=AlertType.POLLER_STUCK,
            agent_name="poller",
            stale_minutes=stale_minutes,
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand_fires is not None

    # Now refresh to 30s-stale; should NOT fire.
    with get_session(db_engine) as session:
        hb = session.get(Heartbeat, "poller")
        assert hb is not None
        hb.last_seen_at = _NOW - timedelta(seconds=30)

    with get_session(db_engine) as session:
        cand_silent = evaluate_heartbeat_watchdog(
            session,
            alert_type=AlertType.POLLER_STUCK,
            agent_name="poller",
            stale_minutes=stale_minutes,
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand_silent is None


def test_web_flapping_fires_when_threshold_restarts_in_window(db_engine: Engine) -> None:
    """``WEB_FLAPPING`` fires when ``≥threshold`` web restarts land in the window."""
    with get_session(db_engine) as session:
        for i in range(5):
            session.add(AgentStart(agent_name="web", started_at=_NOW - timedelta(minutes=i * 5)))

    with get_session(db_engine) as session:
        cand = evaluate_web_flapping(
            session,
            window_minutes=30,
            threshold=5,
            log_path=Path("/nonexistent.log"),
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand is not None
    assert cand.alert_type == AlertType.WEB_FLAPPING
    assert "(5 restarts in last 30m)" in cand.content.subject


def test_web_flapping_does_not_fire_below_threshold(db_engine: Engine) -> None:
    """4 restarts under a threshold of 5 → no fire."""
    with get_session(db_engine) as session:
        for i in range(4):
            session.add(AgentStart(agent_name="web", started_at=_NOW - timedelta(minutes=i * 5)))
    with get_session(db_engine) as session:
        cand = evaluate_web_flapping(
            session,
            window_minutes=30,
            threshold=5,
            log_path=Path("/nonexistent.log"),
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand is None


# --- storage rules -------------------------------------------------------------------------------


def test_storage_unavailable_fires_when_storage_root_missing(tmp_path: Path) -> None:
    """``storage_root`` not a directory → ``STORAGE_UNAVAILABLE`` candidate (and ``DISK_LOW`` skipped)."""
    missing = tmp_path / "doesnt-exist"
    candidates = evaluate_storage(
        storage_root=missing,
        threshold_fraction=0.10,
        public_url=_URL,
        tz_name=_TZ,
        now=_NOW,
    )
    assert len(candidates) == 1
    assert candidates[0].alert_type == AlertType.STORAGE_UNAVAILABLE


def test_disk_low_fires_when_fraction_below_threshold(tmp_path: Path) -> None:
    """``disk_usage.free / total < threshold_fraction`` → ``DISK_LOW`` candidate."""
    fake_usage = shutil._ntuple_diskusage(total=1_000_000_000_000, used=950_000_000_000, free=50_000_000_000)
    with patch("cat_watcher.alerts.shutil.disk_usage", return_value=fake_usage):
        candidates = evaluate_storage(
            storage_root=tmp_path,
            threshold_fraction=0.10,
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
    assert len(candidates) == 1
    assert candidates[0].alert_type == AlertType.DISK_LOW


def test_disk_low_does_not_fire_above_threshold(tmp_path: Path) -> None:
    """Plenty of free space → no fire."""
    fake_usage = shutil._ntuple_diskusage(total=1_000_000_000_000, used=100_000_000_000, free=900_000_000_000)
    with patch("cat_watcher.alerts.shutil.disk_usage", return_value=fake_usage):
        candidates = evaluate_storage(
            storage_root=tmp_path,
            threshold_fraction=0.10,
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
    assert not candidates


def test_disk_low_skipped_when_storage_offline(tmp_path: Path) -> None:
    """``storage_root`` unmounted → only ``STORAGE_UNAVAILABLE`` fires; ``DISK_LOW`` is suppressed.

    The disk_usage call is poisoned to assert_never_called: the rule must not even probe disk usage
    when the drive is offline.
    """
    missing = tmp_path / "doesnt-exist"
    with patch(
        "cat_watcher.alerts.shutil.disk_usage",
        side_effect=AssertionError("disk_usage must not be called when storage offline"),
    ):
        candidates = evaluate_storage(
            storage_root=missing,
            threshold_fraction=0.10,
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
    assert [c.alert_type for c in candidates] == [AlertType.STORAGE_UNAVAILABLE]


def _seed_backup_file(backups_dir: Path, name: str, mtime: datetime) -> Path:
    """Create ``backups_dir/name`` with mtime set to ``mtime``."""
    backups_dir.mkdir(parents=True, exist_ok=True)
    f = backups_dir / name
    _ = f.write_bytes(b"x")
    ts = mtime.timestamp()
    os.utime(f, (ts, ts))
    return f


def test_backup_stale_fires_when_newest_mtime_old(db_engine: Engine, tmp_path: Path) -> None:
    """Newest backup older than ``threshold_hours`` → ``BACKUP_STALE`` candidate."""
    storage_root = tmp_path
    _ = _seed_backup_file(
        storage_root / "backups",
        "cat_watcher-2026-04-29.sqlite",
        _NOW - timedelta(hours=40),
    )
    with get_session(db_engine) as session:
        cand = evaluate_backup_stale(
            session,
            storage_root=storage_root,
            threshold_hours=36,
            storage_unavailable_cooldown_hours=6,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand is not None
    assert cand.alert_type == AlertType.BACKUP_STALE
    assert "no backup in 40h" in cand.content.subject


def test_backup_stale_does_not_fire_when_recent(db_engine: Engine, tmp_path: Path) -> None:
    """Newest backup within threshold → no fire."""
    storage_root = tmp_path
    _ = _seed_backup_file(
        storage_root / "backups",
        "cat_watcher-2026-04-30.sqlite",
        _NOW - timedelta(hours=2),
    )
    with get_session(db_engine) as session:
        cand = evaluate_backup_stale(
            session,
            storage_root=storage_root,
            threshold_hours=36,
            storage_unavailable_cooldown_hours=6,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand is None


def test_backup_stale_no_fire_when_backups_dir_absent(db_engine: Engine, tmp_path: Path) -> None:
    """Missing ``backups/`` dir (storage unmounted, fresh install) → no fire."""
    with get_session(db_engine) as session:
        cand = evaluate_backup_stale(
            session,
            storage_root=tmp_path / "no-storage",
            threshold_hours=36,
            storage_unavailable_cooldown_hours=6,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand is None


def test_backup_stale_suppressed_during_storage_unavailable_cooldown(db_engine: Engine, tmp_path: Path) -> None:
    """A recent ``STORAGE_UNAVAILABLE`` row inside its cool-down window suppresses ``BACKUP_STALE``.

    Spec §4.5: same root cause; both alerts firing would be noise.
    """
    storage_root = tmp_path
    _ = _seed_backup_file(
        storage_root / "backups",
        "cat_watcher-2026-04-29.sqlite",
        _NOW - timedelta(hours=40),
    )
    with get_session(db_engine) as session:
        session.add(
            AlertSent(
                alert_type=AlertType.STORAGE_UNAVAILABLE,
                camera_id=None,
                sent_at=_NOW - timedelta(hours=1),
                subject="s",
                body="b",
            ),
        )

    with get_session(db_engine) as session:
        cand = evaluate_backup_stale(
            session,
            storage_root=storage_root,
            threshold_hours=36,
            storage_unavailable_cooldown_hours=6,
            tz_name=_TZ,
            now=_NOW,
        )
    assert cand is None


# --- cooldown_for + dispatch_alert ---------------------------------------------------------------


def test_cooldown_for_returns_hardcoded_overrides(cfg: Config) -> None:
    """``WEB_FLAPPING=1h``, ``DISK_LOW=24h``, ``BACKUP_STALE=24h`` regardless of ``cooldown_hours``."""
    rules = cfg.alerts.model_copy(update={"cooldown_hours": 6})
    assert cooldown_for(AlertType.WEB_FLAPPING, rules) == 1
    assert cooldown_for(AlertType.DISK_LOW, rules) == 24
    assert cooldown_for(AlertType.BACKUP_STALE, rules) == 24


def test_cooldown_for_returns_default_for_other_types(cfg: Config) -> None:
    """Types not in the override dict fall back to ``rules.cooldown_hours``.

    One camera-bound (``INACTIVITY``) and one non-camera (``POLLER_STUCK``) check covers the
    fallback path; the helper itself is :meth:`dict.get`, not a per-type lookup, so additional
    enum-value assertions don't add coverage.
    """
    rules = cfg.alerts.model_copy(update={"cooldown_hours": 6})
    assert cooldown_for(AlertType.INACTIVITY, rules) == 6
    assert cooldown_for(AlertType.POLLER_STUCK, rules) == 6


def test_dispatch_alert_writes_one_row_when_cooldown_clear(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
) -> None:
    """Sent path writes exactly one ``alerts_sent`` row with ok flags set per the disabled channels."""
    cam_id = seed_camera(db_engine)
    with get_session(db_engine) as session:
        dispatch_alert(
            AlertType.INACTIVITY,
            camera_id=cam_id,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW),
        )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.alert_type == AlertType.INACTIVITY
    assert row.camera_id == cam_id
    # Disabled channels return ok=True; delivery_error stays None.
    assert row.email_ok is True
    assert row.macos_ok is True
    assert row.delivery_error is None


def test_dispatch_alert_suppressed_logs_info_and_writes_no_row(
    db_engine: Engine,
    cfg: Config,
    caplog: pytest.LogCaptureFixture,
    seed_camera: Callable[..., int],
) -> None:
    """Suppressed path logs INFO with alert type + remaining cool-down and does NOT write a second row.

    Spec §6.2 has no ``suppressed`` column by design — an active 24h cool-down would otherwise
    accumulate ~96 rows/day per type.
    """
    # alembic's env.py invokes ``logging.config.fileConfig`` during the integration test suite,
    # which by default disables every existing logger including ours. Re-enable so caplog sees
    # records emitted from this module under any test ordering.
    logging.getLogger("cat_watcher.alerts").disabled = False

    cam_id = seed_camera(db_engine)
    # First dispatch at t=0 — writes a row.
    with get_session(db_engine) as session:
        dispatch_alert(
            AlertType.INACTIVITY,
            camera_id=cam_id,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW),
        )

    # Second dispatch at t=10min, cool-down still open — should suppress.
    with caplog.at_level(logging.INFO, logger="cat_watcher.alerts"), get_session(db_engine) as session:
        dispatch_alert(
            AlertType.INACTIVITY,
            camera_id=cam_id,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW + timedelta(minutes=10)),
        )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).all()
    assert len(rows) == 1, f"suppressed path must NOT write a row; got {len(rows)}"

    suppression_logs = [r for r in caplog.records if "alert suppressed" in r.message and r.levelno == logging.INFO]
    assert len(suppression_logs) == 1, f"expected 1 suppression INFO log; got {len(suppression_logs)}: {caplog.text!r}"
    formatted = suppression_logs[0].getMessage()
    assert "INACTIVITY" in formatted
    assert "remaining_cooldown_seconds=" in formatted


def test_dispatch_alert_uses_global_cooldown_for_non_camera_alerts(db_engine: Engine, cfg: Config) -> None:
    """``camera_id=None`` cool-down key is global per alert type — both fires share the same window.

    Per plan §13 resolution 5: one cool-down per non-camera alert type, regardless of which camera
    triggered it. This test seeds a recent ``POLLER_STUCK`` row with ``camera_id=None`` and verifies
    a second dispatch with ``camera_id=None`` is suppressed (stays at one row).
    """
    with get_session(db_engine) as session:
        dispatch_alert(
            AlertType.POLLER_STUCK,
            camera_id=None,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW),
        )

    with get_session(db_engine) as session:
        dispatch_alert(
            AlertType.POLLER_STUCK,
            camera_id=None,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW + timedelta(minutes=30)),
        )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).filter(AlertSent.alert_type == AlertType.POLLER_STUCK).all()
    assert len(rows) == 1


def test_dispatch_alert_honors_hardcoded_cooldown_overrides(db_engine: Engine, cfg: Config) -> None:
    """``WEB_FLAPPING`` uses 1h cool-down; second dispatch 30min later is suppressed even with cooldown_hours=6."""
    with get_session(db_engine) as session:
        dispatch_alert(
            AlertType.WEB_FLAPPING,
            camera_id=None,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW),
        )
    # Within 30min still inside the WEB_FLAPPING 1h override → suppressed.
    with get_session(db_engine) as session:
        dispatch_alert(
            AlertType.WEB_FLAPPING,
            camera_id=None,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW + timedelta(minutes=30)),
        )
    # Past 1h, the override window has elapsed → second dispatch lands.
    with get_session(db_engine) as session:
        dispatch_alert(
            AlertType.WEB_FLAPPING,
            camera_id=None,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW + timedelta(hours=1, minutes=5)),
        )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).filter(AlertSent.alert_type == AlertType.WEB_FLAPPING).all()
    assert len(rows) == 2


def test_dispatch_alert_camera_keyed_cooldown_does_not_cross_cameras(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
) -> None:
    """Two cameras both fire ``INACTIVITY`` independently — different cool-down keys."""
    cam1 = seed_camera(db_engine, name="pantry", display_name="Pantry")
    cam2 = seed_camera(db_engine, name="bathroom", display_name="Bathroom")
    for cam_id in (cam1, cam2):
        with get_session(db_engine) as session:
            dispatch_alert(
                AlertType.INACTIVITY,
                camera_id=cam_id,
                content=_TRIVIAL_CONTENT,
                env=_dispatch_env(cfg, session, now=_NOW),
            )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).filter(AlertSent.alert_type == AlertType.INACTIVITY).all()
    assert len(rows) == 2
    assert {r.camera_id for r in rows} == {cam1, cam2}


# --- run_alerts_tick end-to-end ------------------------------------------------------------------


def test_run_alerts_tick_inserts_agent_starts_and_heartbeat(db_engine: Engine, cfg: Config) -> None:
    """A clean tick (no firing rules) still records ``agent_starts`` + heartbeat."""
    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        starts = session.query(AgentStart).filter(AgentStart.agent_name == "alerts").all()
        hb = session.get(Heartbeat, "alerts")
    assert len(starts) == 1
    assert hb is not None
    assert hb.last_seen_at == _NOW


def test_run_alerts_tick_runs_without_external_drive(
    make_config: Callable[..., Config],
    tmp_path: Path,
    db_engine: Engine,
) -> None:
    """Alerts agent does NOT exit when ``storage_root`` is unmounted — fires ``STORAGE_UNAVAILABLE``.

    Per spec §4.5: the alerts agent's state lives on internal storage, so it skips the §4.13
    storage wait and runs even when the drive is offline. This is what lets STORAGE_UNAVAILABLE
    fire as a normal DB-recorded alert (no marker-file hack).
    """
    storage_root = tmp_path / "external-drive-not-mounted"
    base = make_config(tmp_path, storage_root)
    cfg_disabled = _channels_disabled(base)

    run_alerts_tick(config=cfg_disabled, engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).filter(AlertSent.alert_type == AlertType.STORAGE_UNAVAILABLE).all()
        starts = session.query(AgentStart).filter(AgentStart.agent_name == "alerts").all()
        hb = session.get(Heartbeat, "alerts")
    assert len(rows) == 1
    assert rows[0].camera_id is None
    assert len(starts) == 1
    assert hb is not None


def test_run_alerts_tick_dispatches_inactivity_for_unreachable_camera(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
) -> None:
    """End-to-end: orchestrator wires ``evaluate_inactivity`` → ``dispatch_alert`` → ``alerts_sent``.

    Rule-level + dispatch-level tests exist separately; this one pins the wiring inside
    :func:`run_alerts_tick` (``_camera_candidates`` → ``_dispatch_each``). A regression that broke
    that orchestration would silently make the agent stop emitting per-camera alerts.
    """
    cam_id = seed_camera(
        db_engine,
        poll_status=PollStatus.UNREACHABLE,
        poll_status_since=_NOW - timedelta(minutes=5),
    )

    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).filter(AlertSent.alert_type == AlertType.INACTIVITY).all()
    assert len(rows) == 1
    assert rows[0].camera_id == cam_id


# --- dispatch_alert: failure-path bookkeeping ----------------------------------------------------


def test_dispatch_alert_records_delivery_error_when_email_fails(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
) -> None:
    """A failing ``send_email`` populates ``email_ok=False`` + ``delivery_error`` on the row.

    Without this test, a regression that swallowed the email error (e.g., always set ``email_ok``
    to True) would silently break the operator-facing audit trail in ``alerts_sent``.
    """
    cam_id = seed_camera(db_engine)
    with (
        patch(
            "cat_watcher.alerts.send_email",
            return_value=EmailResult(ok=False, error="connection refused"),
        ),
        get_session(db_engine) as session,
    ):
        dispatch_alert(
            AlertType.INACTIVITY,
            camera_id=cam_id,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW),
        )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.email_ok is False
    # macOS channel still uses the disabled path (ok=True, error="disabled"); only email failed.
    assert row.macos_ok is True
    assert row.delivery_error is not None
    assert "email:" in row.delivery_error
    assert "connection refused" in row.delivery_error


def test_dispatch_alert_logs_critical_when_both_channels_fail(
    db_engine: Engine,
    cfg: Config,
    caplog: pytest.LogCaptureFixture,
    seed_camera: Callable[..., int],
) -> None:
    """Both senders failing → CRITICAL log + ``alerts_sent`` row with both ok-flags False.

    The most operationally severe failure mode (no alert reached the operator). The row is still
    written so the failure shows up in the alerts history; the CRITICAL log surfaces it in real
    time.
    """
    # See the cool-down suppression test for why this re-enable is needed under random ordering.
    logging.getLogger("cat_watcher.alerts").disabled = False

    cam_id = seed_camera(db_engine)
    with (
        patch(
            "cat_watcher.alerts.send_email",
            return_value=EmailResult(ok=False, error="smtp timeout"),
        ),
        patch(
            "cat_watcher.alerts.send_macos_notification",
            return_value=NotifResult(ok=False, error="osascript exit 1"),
        ),
        caplog.at_level(logging.CRITICAL, logger="cat_watcher.alerts"),
        get_session(db_engine) as session,
    ):
        dispatch_alert(
            AlertType.INACTIVITY,
            camera_id=cam_id,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW),
        )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.email_ok is False
    assert row.macos_ok is False
    assert row.delivery_error is not None
    assert "smtp timeout" in row.delivery_error
    assert "osascript exit 1" in row.delivery_error

    critical_logs = [r for r in caplog.records if r.levelno == logging.CRITICAL and "dispatch failed on both channels" in r.message]
    assert len(critical_logs) == 1

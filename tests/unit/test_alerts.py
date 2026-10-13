"""Tests for cat_watcher.alerts.

Tests disable both notifier channels via ``EmailRulesConfig(enabled=False)`` /
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
from unittest.mock import MagicMock, patch

import pytest
from db_helpers import add_clip, recorded_statements, seed_cat_subject, tag_clip_frame
from sqlalchemy import select
from sqlalchemy.engine import Engine

from cat_watcher.alert_templates import AlertContent
from cat_watcher.alerts import (
    DispatchEnv,
    cooldown_for,
    dispatch_alert,
    evaluate_backup_stale,
    evaluate_camera_clock,
    evaluate_cat_frequency,
    evaluate_cat_inactivity,
    evaluate_heartbeat_watchdog,
    evaluate_inactivity,
    evaluate_storage,
    evaluate_web_flapping,
    run_alerts_tick,
)
from cat_watcher.alerts import main as alerts_main
from cat_watcher.db import (
    AgentStart,
    AlertSent,
    AlertType,
    Camera,
    Heartbeat,
    PollStatus,
    Subject,
    get_session,
)
from cat_watcher.notifier import EmailResult, NotifResult
from cat_watcher.visits import Visit

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


_NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
_TZ = "UTC"
_URL = "http://localhost:8000"
_TRIVIAL_CONTENT = AlertContent(subject="subj", body="body", macos_summary="summary")


@pytest.fixture
def db_engine(alembic_engine: Engine) -> Engine:
    """Alias for ``alembic_engine`` — all alerts tests need the view DDL that Alembic applies."""
    return alembic_engine


@pytest.fixture
def cfg(
    make_config: Callable[..., Config],
    disable_alert_channels: Callable[[Config], Config],
    tmp_path: Path,
) -> Config:
    """Disabled-channels config for tests that exercise dispatch_alert / run_alerts_tick."""
    return disable_alert_channels(make_config(tmp_path, tmp_path))


def _dispatch_env(cfg: Config, session: Session, *, now: datetime) -> DispatchEnv:  # pylint: disable=redefined-outer-name  # params intentionally share the cfg/session fixture names; this is a helper, not a fixture
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

    The per-camera evaluator only handles unreachable + no-clips. A cat that uses only one box today
    must not alert on the other box. Cat absence is a per-cat rule.
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


def test_tick_with_no_cat_visits_gives_no_fleet_wide_inactivity(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
) -> None:
    """Cameras that saw no cat for longer than ``inactivity_hours`` fire no INACTIVITY without a named cat."""
    for name, host in (("pantry", "cam.example.com"), ("bathroom", "cam2.example.com")):
        _ = seed_camera(
            db_engine,
            name=name,
            display_name=name.title(),
            host=host,
            last_polled_at=_NOW - timedelta(minutes=2),
            last_clip_at=_NOW - timedelta(hours=1),
            last_cat_seen_at=_NOW - timedelta(hours=20),
        )

    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).filter(AlertSent.alert_type == AlertType.INACTIVITY).all()
    assert rows == []


# --- per-cat INACTIVITY + FREQUENCY ----------------------------------------------------------------


_MARCEL = Subject(id=7, slug="marcel", display_name="Marcel", kind="cat", display_order=1)
_CAMERA_NAMES = {1: "Pantry", 2: "Bathroom"}


def _visit(start: datetime, *, camera_id: int = 1, clip_ids: tuple[int, ...] = (1,)) -> Visit:
    return Visit(slug="marcel", camera_id=camera_id, start_ts=start, end_ts=start + timedelta(seconds=2), clip_ids=clip_ids)


def _seed_cat_clip(engine: Engine, cam_id: int, start: datetime, *, cat_id: int | None) -> int:
    """Seed a detector cat clip. With ``cat_id``, an operator tag names that cat. Without it, the clip is an unknown-cat clip."""
    clip_id = add_clip(engine, cam_id, start_ts=start, name=f"{cam_id}-{start.isoformat()}.mp4")
    if cat_id is not None:
        tag_clip_frame(engine, clip_id=clip_id, subject_id=cat_id)
    return clip_id


def _cat_alerts(engine: Engine, alert_type: AlertType) -> list[AlertSent]:
    with get_session(engine) as session:
        return session.query(AlertSent).filter(AlertSent.alert_type == alert_type, AlertSent.subject_id.is_not(None)).all()


def _with_frequency(base: Config, *, threshold: int) -> Config:
    return base.model_copy(update={"alerts": base.alerts.model_copy(update={"frequency_threshold_count": threshold})})


def test_cat_inactivity_fires_for_last_visit_older_than_threshold() -> None:
    """A last visit that starts more than ``inactivity_hours`` ago fires a per-cat candidate."""
    cand = evaluate_cat_inactivity(
        cat=_MARCEL,
        last_visit=_visit(_NOW - timedelta(hours=14), camera_id=2),
        inactivity_hours=12,
        camera_name_by_id=_CAMERA_NAMES,
        public_url=_URL,
        tz_name=_TZ,
        now=_NOW,
    )

    assert cand is not None
    assert cand.alert_type == AlertType.INACTIVITY
    assert cand.camera_id is None
    assert cand.subject_id == _MARCEL.id
    assert "Marcel" in cand.content.subject
    assert "(Bathroom)" in cand.content.body


def test_cat_inactivity_does_not_fire_for_visit_inside_threshold() -> None:
    """A last visit inside ``inactivity_hours`` fires nothing."""
    cand = evaluate_cat_inactivity(
        cat=_MARCEL,
        last_visit=_visit(_NOW - timedelta(hours=11)),
        inactivity_hours=12,
        camera_name_by_id=_CAMERA_NAMES,
        public_url=_URL,
        tz_name=_TZ,
        now=_NOW,
    )

    assert cand is None


def test_cat_inactivity_does_not_fire_without_last_visit() -> None:
    """A cat with no visit in the retention window fires nothing."""
    cand = evaluate_cat_inactivity(
        cat=_MARCEL,
        last_visit=None,
        inactivity_hours=12,
        camera_name_by_id=_CAMERA_NAMES,
        public_url=_URL,
        tz_name=_TZ,
        now=_NOW,
    )

    assert cand is None


def test_cat_frequency_counts_visits_over_all_cameras() -> None:
    """Visits on two cameras add up. The candidate names Pantry and Bathroom."""
    visits = [_visit(_NOW - timedelta(hours=3), camera_id=1), _visit(_NOW - timedelta(hours=2), camera_id=2, clip_ids=(2,))]

    cand = evaluate_cat_frequency(
        cat=_MARCEL,
        visits=visits,
        window_hours=6,
        threshold=2,
        camera_name_by_id=_CAMERA_NAMES,
        public_url=_URL,
        tz_name=_TZ,
    )

    assert cand is not None
    assert cand.alert_type == AlertType.FREQUENCY
    assert cand.camera_id is None
    assert cand.subject_id == _MARCEL.id
    assert "Pantry, Bathroom" in cand.content.body


def test_cat_frequency_counts_every_visit_it_gets() -> None:
    """The caller owns the window. A visit that starts before ``window_hours`` still counts."""
    visits = [_visit(_NOW - timedelta(hours=7)), _visit(_NOW - timedelta(hours=2), clip_ids=(2,))]

    cand = evaluate_cat_frequency(
        cat=_MARCEL,
        visits=visits,
        window_hours=6,
        threshold=2,
        camera_name_by_id=_CAMERA_NAMES,
        public_url=_URL,
        tz_name=_TZ,
    )

    assert cand is not None


def test_tick_fires_cat_inactivity_for_stale_cat_only(db_engine: Engine, cfg: Config, seed_camera: Callable[..., int]) -> None:
    """Only the cat whose last visit is older than ``inactivity_hours`` gets an alert.

    The config has no ``[classifier]`` table. The operator tags still name the cats.
    """
    assert cfg.classifier is None
    cam_id = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine)
    rufus = seed_cat_subject(db_engine, slug="rufus", display_name="Rufus", display_order=2)
    _ = _seed_cat_clip(db_engine, cam_id, _NOW - timedelta(hours=14), cat_id=marcel)
    _ = _seed_cat_clip(db_engine, cam_id, _NOW - timedelta(hours=1), cat_id=rufus)

    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW)

    rows = _cat_alerts(db_engine, AlertType.INACTIVITY)
    assert [(r.subject_id, r.camera_id) for r in rows] == [(marcel, None)]


def test_tick_reads_the_visit_clips_once(db_engine: Engine, cfg: Config, seed_camera: Callable[..., int]) -> None:
    """One read of the cat clips serves the INACTIVITY rule and the FREQUENCY rule of every cat."""
    cam_id = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine)
    _ = _seed_cat_clip(db_engine, cam_id, _NOW - timedelta(hours=14), cat_id=marcel)
    _ = _seed_cat_clip(db_engine, cam_id, _NOW - timedelta(hours=1), cat_id=marcel)

    with recorded_statements(db_engine) as statements:
        run_alerts_tick(config=_with_frequency(cfg, threshold=1), engine=db_engine, now=_NOW)

    assert len([statement for statement in statements if "clips.end_ts >=" in statement]) == 1
    assert [r.subject_id for r in _cat_alerts(db_engine, AlertType.FREQUENCY)] == [marcel]
    assert not _cat_alerts(db_engine, AlertType.INACTIVITY)


def test_tick_unknown_cat_clip_does_not_reset_cat_inactivity(db_engine: Engine, cfg: Config, seed_camera: Callable[..., int]) -> None:
    """A cat clip that names no cat, after the last visit of a cat, does not stop its alert."""
    cam_id = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine)
    _ = _seed_cat_clip(db_engine, cam_id, _NOW - timedelta(hours=14), cat_id=marcel)
    _ = _seed_cat_clip(db_engine, cam_id, _NOW - timedelta(hours=1), cat_id=None)

    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW)

    assert [r.subject_id for r in _cat_alerts(db_engine, AlertType.INACTIVITY)] == [marcel]


def test_tick_reviewed_tag_on_missed_clip_resets_cat_inactivity(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., int],
) -> None:
    """A reviewed cat tag on a clip that the detector missed is a visit, so the cat fires no INACTIVITY."""
    cam_id = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine)
    _ = _seed_cat_clip(db_engine, cam_id, _NOW - timedelta(hours=14), cat_id=marcel)
    missed = seed_clip(db_engine, camera_id=cam_id, start_ts=_NOW - timedelta(hours=1), has_cat=False)
    tag_clip_frame(db_engine, clip_id=missed, subject_id=marcel, reviewed_at=_NOW - timedelta(minutes=30))

    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW)

    assert _cat_alerts(db_engine, AlertType.INACTIVITY) == []


def test_tick_unreviewed_tag_on_missed_clip_keeps_cat_inactivity(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., int],
) -> None:
    """Without the review mark, a cat tag on a missed clip is not a visit, so the cat still fires INACTIVITY."""
    cam_id = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine)
    _ = _seed_cat_clip(db_engine, cam_id, _NOW - timedelta(hours=14), cat_id=marcel)
    missed = seed_clip(db_engine, camera_id=cam_id, start_ts=_NOW - timedelta(hours=1), has_cat=False)
    tag_clip_frame(db_engine, clip_id=missed, subject_id=marcel)

    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW)

    assert [r.subject_id for r in _cat_alerts(db_engine, AlertType.INACTIVITY)] == [marcel]


@pytest.mark.parametrize(("visit_count", "fires"), [(3, True), (2, False)])
def test_tick_cat_frequency_fires_at_threshold_over_two_cameras(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
    visit_count: int,
    *,
    fires: bool,
) -> None:
    """FREQUENCY fires at ``frequency_threshold_count`` visits over two cameras, and not at one fewer."""
    cams = [seed_camera(db_engine), seed_camera(db_engine, name="bathroom", display_name="Bathroom", host="cam2.example.com")]
    marcel = seed_cat_subject(db_engine)
    for i in range(visit_count):
        _ = _seed_cat_clip(db_engine, cams[i % 2], _NOW - timedelta(hours=1, minutes=10 * i), cat_id=marcel)

    run_alerts_tick(config=_with_frequency(cfg, threshold=3), engine=db_engine, now=_NOW)

    rows = _cat_alerts(db_engine, AlertType.FREQUENCY)
    assert [(r.subject_id, r.camera_id) for r in rows] == ([(marcel, None)] if fires else [])


def test_tick_cat_frequency_counts_a_visit_of_two_clips_once(db_engine: Engine, cfg: Config, seed_camera: Callable[..., int]) -> None:
    """Two clips less than 2 minutes apart are one visit, so two such pairs miss a threshold of 3."""
    cam_id = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine)
    for start in (_NOW - timedelta(hours=2), _NOW - timedelta(hours=1)):
        _ = _seed_cat_clip(db_engine, cam_id, start, cat_id=marcel)
        _ = _seed_cat_clip(db_engine, cam_id, start + timedelta(seconds=30), cat_id=marcel)

    run_alerts_tick(config=_with_frequency(cfg, threshold=3), engine=db_engine, now=_NOW)

    assert _cat_alerts(db_engine, AlertType.FREQUENCY) == []


def test_tick_archived_cat_gets_no_alert(db_engine: Engine, cfg: Config, seed_camera: Callable[..., int]) -> None:
    """Archived cats with tagged clips get no alert. If the cats are active, Marcel fires FREQUENCY and Rufus fires INACTIVITY."""
    cam_id = seed_camera(db_engine)
    marcel = seed_cat_subject(db_engine)
    rufus = seed_cat_subject(db_engine, slug="rufus", display_name="Rufus", display_order=2)
    _ = _seed_cat_clip(db_engine, cam_id, _NOW - timedelta(hours=1), cat_id=marcel)
    _ = _seed_cat_clip(db_engine, cam_id, _NOW - timedelta(hours=14), cat_id=rufus)
    with get_session(db_engine) as session:
        for subject in session.scalars(select(Subject)):
            subject.archived_at = _NOW - timedelta(days=1)

    run_alerts_tick(config=_with_frequency(cfg, threshold=1), engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        assert session.query(AlertSent).filter(AlertSent.subject_id.is_not(None)).all() == []


def test_tick_cat_alert_keeps_cool_down_when_cat_changes_cameras(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
) -> None:
    """A per-cat INACTIVITY alert stays in cool-down after the last visit moves to a second camera."""
    pantry = seed_camera(db_engine)
    bathroom = seed_camera(db_engine, name="bathroom", display_name="Bathroom", host="cam2.example.com")
    marcel = seed_cat_subject(db_engine)
    _ = _seed_cat_clip(db_engine, pantry, _NOW - timedelta(hours=14), cat_id=marcel)
    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW)

    _ = _seed_cat_clip(db_engine, bathroom, _NOW - timedelta(hours=13), cat_id=marcel)
    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW + timedelta(hours=1))

    assert [r.sent_at for r in _cat_alerts(db_engine, AlertType.INACTIVITY)] == [_NOW]


def test_tick_many_cat_clips_on_one_camera_give_no_camera_frequency(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., int],
) -> None:
    """Many cat clips that name no cat fire no FREQUENCY alert of any kind."""
    cam_id = seed_camera(db_engine, last_polled_at=_NOW - timedelta(minutes=2), last_clip_at=_NOW - timedelta(minutes=5))
    for i in range(20):
        _ = seed_clip(db_engine, camera_id=cam_id, start_ts=_NOW - timedelta(hours=1, minutes=i * 3), has_cat=True)

    run_alerts_tick(config=cfg, engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        assert session.query(AlertSent).filter(AlertSent.alert_type == AlertType.FREQUENCY).all() == []


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


def _clock_camera(
    *,
    streak: int = 0,
    ntp_enabled: bool | None = None,
    drift_seconds: float | None = -7198.3,
) -> Camera:
    """Build a camera row carrying only the clock state the evaluator reads."""
    return Camera(
        id=1,
        name="office",
        display_name="Office Litter Box Camera",
        host="cam.example.com",
        poll_status=PollStatus.OK,
        clock_correction_streak=streak,
        clock_ntp_enabled=ntp_enabled,
        clock_drift_seconds=drift_seconds,
    )


def test_camera_clock_fires_at_streak_threshold() -> None:
    """Reaching the threshold means the clock is not holding between ticks."""
    candidate = evaluate_camera_clock(
        _clock_camera(streak=3),
        streak_threshold=3,
        public_url=_URL,
        tz_name=_TZ,
        now=_NOW,
    )

    assert candidate is not None
    assert candidate.alert_type is AlertType.CAMERA_CLOCK
    assert candidate.camera_id == 1


def test_camera_clock_silent_below_streak_threshold() -> None:
    """One correction is the expected result of an unplug, so it must stay quiet."""
    assert (
        evaluate_camera_clock(
            _clock_camera(streak=2),
            streak_threshold=3,
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
        is None
    )


def test_camera_clock_fires_on_ntp_enabled_regardless_of_streak() -> None:
    """NTP back on overwrites the clock every sync, so it warrants an alert on its own."""
    candidate = evaluate_camera_clock(
        _clock_camera(streak=0, ntp_enabled=True),
        streak_threshold=3,
        public_url=_URL,
        tz_name=_TZ,
        now=_NOW,
    )

    assert candidate is not None
    assert "NTP.Enable" in candidate.content.body


def test_camera_clock_silent_when_ntp_state_never_read() -> None:
    """``None`` means the poller has not had cause to read NTP, which is not a fault."""
    assert (
        evaluate_camera_clock(
            _clock_camera(streak=0, ntp_enabled=None),
            streak_threshold=3,
            public_url=_URL,
            tz_name=_TZ,
            now=_NOW,
        )
        is None
    )


def test_camera_clock_cooldown_suppresses_repeat(
    db_engine: Engine,
    cfg: Config,
    seed_camera: Callable[..., int],
) -> None:
    """A second dispatch inside the cool-down writes no new row."""
    cam_id = seed_camera(db_engine)
    for _ in range(2):
        with get_session(db_engine) as session:
            dispatch_alert(
                AlertType.CAMERA_CLOCK,
                camera_id=cam_id,
                content=_TRIVIAL_CONTENT,
                env=_dispatch_env(cfg, session, now=_NOW),
            )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).where(AlertSent.alert_type == AlertType.CAMERA_CLOCK).all()
    assert len(rows) == 1


def test_cooldown_for_returns_hardcoded_overrides(cfg: Config) -> None:
    """``WEB_FLAPPING=1h``, ``DISK_LOW=24h``, ``BACKUP_STALE=24h`` regardless of ``cooldown_hours``."""
    rules = cfg.alerts.model_copy(update={"cooldown_hours": 6})
    assert cooldown_for(AlertType.WEB_FLAPPING, rules) == 1
    assert cooldown_for(AlertType.DISK_LOW, rules) == 24
    assert cooldown_for(AlertType.BACKUP_STALE, rules) == 24


def test_cooldown_for_returns_default_for_other_types(cfg: Config) -> None:
    """Types not in the override dict fall back to ``rules.cooldown_hours``."""
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
    with get_session(db_engine) as session:
        dispatch_alert(
            AlertType.INACTIVITY,
            camera_id=cam_id,
            content=_TRIVIAL_CONTENT,
            env=_dispatch_env(cfg, session, now=_NOW),
        )

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


def test_dispatch_alert_per_cat_cooldown_blocks_same_cat_only(db_engine: Engine, cfg: Config) -> None:
    """A per-cat alert blocks a repeat for that cat and lets another cat's alert through."""
    marcel = seed_cat_subject(db_engine, slug="marcel", display_name="Marcel", display_order=1)
    rufus = seed_cat_subject(db_engine, slug="rufus", display_name="Rufus", display_order=2)
    schedule = [(marcel, 0), (marcel, 10), (rufus, 20)]
    for subject_id, minutes in schedule:
        with get_session(db_engine) as session:
            dispatch_alert(
                AlertType.INACTIVITY,
                camera_id=None,
                subject_id=subject_id,
                content=_TRIVIAL_CONTENT,
                env=_dispatch_env(cfg, session, now=_NOW + timedelta(minutes=minutes)),
            )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).order_by(AlertSent.id).all()
        assert [r.subject_id for r in rows] == [marcel, rufus]
        assert rows[0].cat is not None
        assert rows[0].cat.slug == "marcel"


def test_dispatch_alert_cat_cooldown_is_separate_from_camera_cooldown(db_engine: Engine, cfg: Config) -> None:
    """A per-cat row does not block the NULL-subject alert of the same type, and the reverse."""
    marcel = seed_cat_subject(db_engine)
    for subject_id, minutes in [(marcel, 0), (None, 5), (None, 10)]:
        with get_session(db_engine) as session:
            dispatch_alert(
                AlertType.INACTIVITY,
                camera_id=None,
                subject_id=subject_id,
                content=_TRIVIAL_CONTENT,
                env=_dispatch_env(cfg, session, now=_NOW + timedelta(minutes=minutes)),
            )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).order_by(AlertSent.id).all()
    assert [r.subject_id for r in rows] == [marcel, None]


def test_dispatch_alert_null_subject_keeps_cooldown(db_engine: Engine, cfg: Config, seed_camera: Callable[..., int]) -> None:
    """A camera alert without a subject still suppresses its repeat (NULL matches NULL)."""
    cam_id = seed_camera(db_engine)
    for minutes in (0, 10):
        with get_session(db_engine) as session:
            dispatch_alert(
                AlertType.INACTIVITY,
                camera_id=cam_id,
                content=_TRIVIAL_CONTENT,
                env=_dispatch_env(cfg, session, now=_NOW + timedelta(minutes=minutes)),
            )

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).all()
    assert len(rows) == 1
    assert rows[0].subject_id is None


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
    disable_alert_channels: Callable[[Config], Config],
    tmp_path: Path,
    db_engine: Engine,
) -> None:
    """Alerts agent does NOT exit when ``storage_root`` is unmounted — fires ``STORAGE_UNAVAILABLE``.

    Per spec §4.5: the alerts agent's state lives on internal storage, so it skips the §4.13 storage
    wait and runs even when the drive is offline. This is what lets STORAGE_UNAVAILABLE fire as a
    normal DB-recorded alert (no marker-file hack).
    """
    storage_root = tmp_path / "external-drive-not-mounted"
    base = make_config(tmp_path, storage_root)
    cfg_disabled = disable_alert_channels(base)

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
    """End-to-end: orchestrator wires ``evaluate_inactivity`` → ``dispatch_alert`` → ``alerts_sent``."""
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
    """A failing ``send_email`` populates ``email_ok=False`` + ``delivery_error`` on the row."""
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

    The row is still written so the failure shows up in the alerts history; the CRITICAL log
    surfaces it in real time.
    """
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


# --- main() smoke test ---------------------------------------------------------------------------


def test_main_wires_log_level_from_config(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    restore_root_logger: logging.Logger,
) -> None:
    """``main()`` reads ``config.log_level`` and passes it through to the root logger.

    Tripwire for the logging-wiring contract: if ``setup_agent_logging`` ever ignores
    ``config.log_level`` (e.g. hardcodes WARNING), ``root.level != logging.INFO`` will catch it.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with (
        patch("cat_watcher.alerts.load_config", return_value=config),
        patch("cat_watcher.alerts.storage_available", return_value=True),
        patch("cat_watcher.alerts.ensure_storage_layout"),
        patch("cat_watcher.alerts.engine_for", return_value=MagicMock(spec=Engine)),
        patch("cat_watcher.alerts.run_alerts_tick"),
    ):
        rc = alerts_main([])

    assert rc == 0
    assert restore_root_logger.level == logging.INFO


def test_main_skips_storage_layout_when_storage_offline(
    cfg: Config,
    restore_root_logger: logging.Logger,
) -> None:
    """``main()`` skips ``ensure_storage_layout`` when the drive is offline but still runs ``run_alerts_tick``.

    The alerts agent must keep running with the external drive offline so it can fire
    ``STORAGE_UNAVAILABLE``. ``ensure_storage_layout`` would fail with a missing mount, so the guard
    exits that branch early; the tick itself must still execute.
    """
    _ = restore_root_logger
    config = cfg

    mock_run_alerts_tick = MagicMock()
    mock_ensure_storage_layout = MagicMock()

    with (
        patch("cat_watcher.alerts.load_config", return_value=config),
        patch("cat_watcher.alerts.storage_available", return_value=False),
        patch("cat_watcher.alerts.ensure_storage_layout", mock_ensure_storage_layout),
        patch("cat_watcher.alerts.engine_for", return_value=MagicMock(spec=Engine)),
        patch("cat_watcher.alerts.run_alerts_tick", mock_run_alerts_tick),
    ):
        rc = alerts_main([])

    assert rc == 0
    assert mock_run_alerts_tick.called
    mock_ensure_storage_layout.assert_not_called()

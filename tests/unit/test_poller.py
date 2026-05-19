"""Tests for cat_watcher.poller.

The poller composes existing modules (amcrest_client, detector, retention, storage, db). Most unit
tests target small helper functions in isolation against the ``db_engine`` fixture; the end-to-end
poll-tick exercise lives in tests/integration/test_poller_end_to_end.py.
"""

import fcntl
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # runtime: pytest fixture annotations are evaluated by collectors
from typing import TYPE_CHECKING, Self, final
from unittest.mock import MagicMock

import pytest
from tz_helpers import pinned_tz

from cat_watcher.amcrest_client import CameraUnreachableError, Recording
from cat_watcher.config import EmailRulesConfig, MacOsRulesConfig
from cat_watcher.db import AlertSent, AlertType, Camera, Clip, Heartbeat, PollStatus, get_session
from cat_watcher.detector import DetectionResult, Detector, DetectorError
from cat_watcher.poller import (
    PollerArgs,
    PollerError,
    PollerLockedError,
    _check_alerts_stuck,
    _limited,
    _parse_args,
    _parse_iso_datetime,
    _poll_camera,
    _resolve_window,
    detection_fields_for,
    extract_thumbnail,
    pid_lock,
    relative_paths_for,
    run_tick,
    update_camera_state_failure,
    update_camera_state_success,
    upsert_heartbeat,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.engine import Engine

    from cat_watcher.config import Config


_NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


# --- update_camera_state_success ------------------------------------------------------------------


def test_update_state_success_advances_last_polled_when_no_clips(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Empty ingest still bumps ``last_polled_at`` (minus the overlap) so the heartbeat advances even on quiet polls."""
    cam_id = seed_camera(db_engine, last_clip_at=_NOW - timedelta(days=2), last_cat_seen_at=_NOW - timedelta(days=3))
    previous_clip_at = _NOW - timedelta(days=2)
    previous_cat_at = _NOW - timedelta(days=3)

    with get_session(db_engine) as session:
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=[], now=_NOW, overlap_minutes=0)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_polled_at == _NOW
        # Preservation: not overwritten when no new clips were ingested.
        assert cam.last_clip_at == previous_clip_at
        assert cam.last_cat_seen_at == previous_cat_at
        assert cam.poll_status == PollStatus.OK
        assert cam.poll_status_since is None
        assert cam.poll_error is None


def test_update_state_success_advances_last_clip_at_when_clips_ingested(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """``last_clip_at`` becomes ``max(start_ts)`` of new clips when any are ingested."""
    cam_id = seed_camera(db_engine)
    clip1_ts = _NOW - timedelta(minutes=30)
    clip2_ts = _NOW - timedelta(minutes=10)
    seed_clip(db_engine, camera_id=cam_id, start_ts=clip1_ts, has_cat=False)
    seed_clip(db_engine, camera_id=cam_id, start_ts=clip2_ts, has_cat=False)

    with get_session(db_engine) as session:
        clips = session.query(Clip).order_by(Clip.start_ts).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW, overlap_minutes=0)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_clip_at == clip2_ts  # the later one wins


def test_update_state_success_advances_last_cat_seen_when_cat_positive_clip(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """``last_cat_seen_at`` advances to the latest cat-positive clip; later non-cat clips don't move it back."""
    cam_id = seed_camera(db_engine)
    no_cat_ts = _NOW - timedelta(minutes=30)
    cat_ts = _NOW - timedelta(minutes=20)
    later_no_cat_ts = _NOW - timedelta(minutes=10)
    seed_clip(db_engine, camera_id=cam_id, start_ts=no_cat_ts, has_cat=False)
    seed_clip(db_engine, camera_id=cam_id, start_ts=cat_ts, has_cat=True)
    seed_clip(db_engine, camera_id=cam_id, start_ts=later_no_cat_ts, has_cat=False)

    with get_session(db_engine) as session:
        clips = session.query(Clip).order_by(Clip.start_ts).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW, overlap_minutes=0)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_cat_seen_at == cat_ts  # latest TRUE wins; later no-cat doesn't move it back
        assert cam.last_clip_at == later_no_cat_ts  # last_clip_at sees all clips


def test_update_state_success_preserves_last_cat_seen_when_no_cat_positive(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """A polling round with only no-cat clips leaves ``last_cat_seen_at`` untouched (no false reset)."""
    previous_cat_at = _NOW - timedelta(days=3)
    cam_id = seed_camera(db_engine, last_cat_seen_at=previous_cat_at)
    new_ts = _NOW - timedelta(minutes=10)
    seed_clip(db_engine, camera_id=cam_id, start_ts=new_ts, has_cat=False)

    with get_session(db_engine) as session:
        clips = session.query(Clip).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW, overlap_minutes=0)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_cat_seen_at == previous_cat_at  # NOT moved to ``new_ts``
        assert cam.last_clip_at == new_ts


def test_update_state_success_respects_manual_has_cat_override(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """``COALESCE(manual_has_cat, has_cat)``: a model-false clip with ``manual_has_cat=True`` is cat-positive."""
    cam_id = seed_camera(db_engine)
    ts = _NOW - timedelta(minutes=5)
    seed_clip(db_engine, camera_id=cam_id, start_ts=ts, has_cat=False, manual_has_cat=True)

    with get_session(db_engine) as session:
        clips = session.query(Clip).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW, overlap_minutes=0)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_cat_seen_at == ts


def test_update_state_success_clears_poll_status_since_on_recovery(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Successful poll after a failure clears ``poll_status_since`` and ``poll_error`` — recovery is observable."""
    cam_id = seed_camera(
        db_engine,
        poll_status=PollStatus.UNREACHABLE,
        poll_status_since=_NOW - timedelta(hours=1),
        poll_error="prior failure",
    )

    with get_session(db_engine) as session:
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=[], now=_NOW, overlap_minutes=0)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.poll_status == PollStatus.OK
        assert cam.poll_status_since is None
        assert cam.poll_error is None


# --- update_camera_state_success: overlap_minutes cursor semantics --------------------------------


def test_update_state_success_advances_cursor_by_cadence_minus_overlap(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Steady-state: with cadence 5min and overlap 15min, the cursor net-advances 5min per 20min wall-clock.

    Two consecutive ticks at ``T`` and ``T + 5min`` with ``overlap_minutes = 15`` move the cursor
    from ``T - 15min`` to ``T + 5min - 15min``: a 5-minute net advance per tick, which is exactly
    ``cadence - overlap = 20 - 15``. The window's left edge keeps reaching back into already-covered
    ground so a delayed ``findFile`` index does not silently drop the clip.
    """
    cam_id = seed_camera(db_engine, last_polled_at=None)
    overlap_minutes = 15

    tick1 = _NOW
    with get_session(db_engine) as session:
        update_camera_state_success(
            session,
            camera_id=cam_id,
            ingested_clips=[],
            now=tick1,
            overlap_minutes=overlap_minutes,
        )
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_polled_at == tick1 - timedelta(minutes=overlap_minutes)

    tick2 = _NOW + timedelta(minutes=5)
    with get_session(db_engine) as session:
        update_camera_state_success(
            session,
            camera_id=cam_id,
            ingested_clips=[],
            now=tick2,
            overlap_minutes=overlap_minutes,
        )
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        cursor_after_tick2 = cam.last_polled_at
        assert cursor_after_tick2 is not None
        assert cursor_after_tick2 == tick2 - timedelta(minutes=overlap_minutes)
        # Net advance from tick1's cursor to tick2's cursor equals cadence (5min), not cadence + overlap.
        assert cursor_after_tick2 - (tick1 - timedelta(minutes=overlap_minutes)) == timedelta(minutes=5)


def test_update_state_success_never_rewinds_cursor(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """If the prior cursor is already further forward than ``now - overlap``, the cursor stays put.

    Defensive against clock skew, manual DB edits, or a prior tick that landed cursor unusually
    forward. The cursor only ever moves forward in time on success.
    """
    overlap_minutes = 15
    already_forward = _NOW - timedelta(minutes=5)  # newer than _NOW - 15min
    cam_id = seed_camera(db_engine, last_polled_at=already_forward)

    with get_session(db_engine) as session:
        update_camera_state_success(
            session,
            camera_id=cam_id,
            ingested_clips=[],
            now=_NOW,
            overlap_minutes=overlap_minutes,
        )

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_polled_at == already_forward  # NOT rewound to _NOW - 15min


# --- update_camera_state_failure ------------------------------------------------------------------


def test_update_state_failure_sets_poll_status_since_on_first_failure(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """OK -> non-OK transition pins ``poll_status_since`` to ``now`` and records the error message."""
    cam_id = seed_camera(db_engine)  # starts OK

    with get_session(db_engine) as session:
        update_camera_state_failure(
            session,
            camera_id=cam_id,
            status=PollStatus.UNREACHABLE,
            error="connect refused",
            now=_NOW,
        )

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.poll_status == PollStatus.UNREACHABLE
        assert cam.poll_status_since == _NOW
        assert cam.poll_error == "connect refused"
        assert cam.last_polled_at == _NOW  # still advanced


def test_update_state_failure_preserves_poll_status_since_on_repeat(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A repeat non-OK tick leaves ``poll_status_since`` pointing at the original transition."""
    earlier = _NOW - timedelta(hours=1)
    cam_id = seed_camera(
        db_engine,
        poll_status=PollStatus.UNREACHABLE,
        poll_status_since=earlier,
        poll_error="initial error",
    )

    with get_session(db_engine) as session:
        update_camera_state_failure(
            session,
            camera_id=cam_id,
            status=PollStatus.UNREACHABLE,
            error="still unreachable",
            now=_NOW,
        )

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.poll_status == PollStatus.UNREACHABLE
        assert cam.poll_status_since == earlier  # NOT advanced; still pointing at the original transition
        assert cam.poll_error == "still unreachable"  # error message updates each tick


def test_update_state_success_preserves_last_polled_at_when_cursor_locked(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """``advance_cursor=False`` preserves ``last_polled_at`` while still updating observation fields.

    A scoped tick (``--since`` / ``--until`` / ``--limit``) cannot prove it covered the full
    ``[last_polled_at, now]`` window, so the resume cursor must stay where it is. The other fields
    (``last_clip_at``, ``last_cat_seen_at``, ``poll_status``) are real observations and advance
    normally.
    """
    earlier = _NOW - timedelta(hours=1)
    cam_id = seed_camera(
        db_engine,
        last_polled_at=earlier,
        poll_status=PollStatus.ERROR,
        poll_status_since=earlier,
        poll_error="prior failure",
    )
    clip_ts = _NOW - timedelta(minutes=30)
    seed_clip(db_engine, camera_id=cam_id, start_ts=clip_ts, has_cat=True)

    with get_session(db_engine) as session:
        clips = session.query(Clip).all()
        update_camera_state_success(
            session,
            camera_id=cam_id,
            ingested_clips=clips,
            now=_NOW,
            overlap_minutes=15,
            advance_cursor=False,
        )

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_polled_at == earlier  # NOT advanced — scoped query
        assert cam.last_clip_at == clip_ts  # observation: ingested clip
        assert cam.last_cat_seen_at == clip_ts  # observation: cat-positive clip
        assert cam.poll_status == PollStatus.OK  # observation: camera reachable
        assert cam.poll_status_since is None
        assert cam.poll_error is None


def test_update_state_failure_preserves_last_polled_at_when_cursor_locked(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Failure path with ``advance_cursor=False``: cursor stays put while status fields update."""
    earlier = _NOW - timedelta(hours=1)
    cam_id = seed_camera(db_engine, last_polled_at=earlier)

    with get_session(db_engine) as session:
        update_camera_state_failure(
            session,
            camera_id=cam_id,
            status=PollStatus.UNREACHABLE,
            error="connection refused",
            now=_NOW,
            advance_cursor=False,
        )

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_polled_at == earlier  # NOT advanced
        assert cam.poll_status == PollStatus.UNREACHABLE
        assert cam.poll_status_since == _NOW  # observation: first transition into non-OK
        assert cam.poll_error == "connection refused"


# --- PollerArgs.truncates_default_window ----------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        (PollerArgs(), False),
        (PollerArgs(since=_NOW), True),
        (PollerArgs(until=_NOW), True),
        (PollerArgs(limit=50), True),
        (PollerArgs(since=_NOW, until=_NOW, limit=50), True),
        # list_only handled separately at the run_tick level, not via this property
        (PollerArgs(list_only=True), False),
        # camera / no_detect / verbose do not truncate the search window
        (PollerArgs(camera="pantry"), False),
        (PollerArgs(no_detect=True), False),
        (PollerArgs(verbose=True), False),
    ],
    ids=[
        "default",
        "since_only",
        "until_only",
        "limit_only",
        "all_three",
        "list_only_excluded",
        "camera_excluded",
        "no_detect_excluded",
        "verbose_excluded",
    ],
)
def test_poller_args_truncates_default_window(case: tuple[PollerArgs, bool]) -> None:
    """Only ``--since`` / ``--until`` / ``--limit`` mark the run as scoped — gates cursor advancement."""
    args, expected = case
    assert args.truncates_default_window is expected


# --- upsert_heartbeat -----------------------------------------------------------------------------


def test_upsert_heartbeat_inserts_when_absent(db_engine: Engine) -> None:
    """First call for an unseen ``agent_name`` inserts a fresh row at ``now``."""
    with get_session(db_engine) as session:
        upsert_heartbeat(session, agent_name="poller", now=_NOW)

    with get_session(db_engine) as session:
        hb = session.get(Heartbeat, "poller")
        assert hb is not None
        assert hb.last_seen_at == _NOW


def test_upsert_heartbeat_updates_when_present(db_engine: Engine) -> None:
    """Subsequent calls advance ``last_seen_at`` without creating a duplicate row."""
    with get_session(db_engine) as session:
        upsert_heartbeat(session, agent_name="poller", now=_NOW - timedelta(minutes=10))
    with get_session(db_engine) as session:
        upsert_heartbeat(session, agent_name="poller", now=_NOW)

    with get_session(db_engine) as session:
        rows = session.query(Heartbeat).all()
        assert len(rows) == 1
        assert rows[0].last_seen_at == _NOW


# --- pid_lock -------------------------------------------------------------------------------------


def test_pid_lock_writes_pid_file_and_releases_on_exit(tmp_path: Path) -> None:
    """Inside the context the ``.poller.pid`` file holds the current PID; after exit the lock can be re-acquired."""
    with pid_lock(tmp_path):
        pid_file = tmp_path / ".poller.pid"
        assert pid_file.is_file()
        assert pid_file.read_text().strip() == str(os.getpid())
    # After exit the file may still exist (we only release the flock); a re-acquire works.
    with pid_lock(tmp_path):
        pass


def test_pid_lock_raises_when_already_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second concurrent acquisition raises ``PollerLockedError``.

    Same-process flock is reentrant on the same FD, so simulate cross-process contention by mocking
    ``fcntl.flock`` to raise ``BlockingIOError`` on the second call (what ``LOCK_NB`` does when the
    lock is held).
    """
    real_flock = fcntl.flock
    calls = {"count": 0}

    def flaky_flock(fd: int, op: int) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            real_flock(fd, op)
            return
        if op & fcntl.LOCK_EX:
            msg = "simulated lock contention"
            raise BlockingIOError(msg)
        real_flock(fd, op)

    monkeypatch.setattr(fcntl, "flock", flaky_flock)

    with pid_lock(tmp_path), pytest.raises(PollerLockedError, match="poller PID lock held"), pid_lock(tmp_path):
        pass


# --- relative_paths_for --------------------------------------------------------------------------


def test_relative_paths_for_uses_camera_local_date_and_time() -> None:
    """On-disk layout uses the camera-local clock so date dirs line up with operator expectations."""
    local_dt = datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)
    rel_clip, rel_thumb = relative_paths_for("pantry", local_dt)
    assert rel_clip == "clips/pantry/2026-05-01/064704.mp4"
    assert rel_thumb == "thumbs/pantry/2026-05-01/064704.jpg"


# --- _resolve_window ------------------------------------------------------------------------------


def test_resolve_window_uses_args_since_when_provided(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """``--since`` overrides every other source (cursor, retention backfill)."""
    cam_id = seed_camera(db_engine, last_polled_at=_NOW - timedelta(hours=1))
    args = PollerArgs(since=_NOW - timedelta(days=2))
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        since, until = _resolve_window(db_camera=cam, args=args, retention_days=30, now=_NOW)
    assert since == _NOW - timedelta(days=2)
    assert until == _NOW


def test_resolve_window_uses_last_polled_at_when_no_since(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Steady state: the resume cursor is ``camera.last_polled_at``."""
    last_poll = _NOW - timedelta(minutes=10)
    cam_id = seed_camera(db_engine, last_polled_at=last_poll)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        since, _ = _resolve_window(db_camera=cam, args=PollerArgs(), retention_days=30, now=_NOW)
    assert since == last_poll


def test_resolve_window_first_run_uses_retention_days_backfill(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Fresh install with no cursor and no ``--since`` falls back to ``now - retention``."""
    cam_id = seed_camera(db_engine, last_polled_at=None)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        since, _ = _resolve_window(db_camera=cam, args=PollerArgs(), retention_days=30, now=_NOW)
    assert since == _NOW - timedelta(days=30)


def test_resolve_window_args_until_overrides_now(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """``--until`` caps the upper bound (default is ``now``)."""
    cam_id = seed_camera(db_engine)
    explicit_until = _NOW - timedelta(hours=1)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        _, until = _resolve_window(db_camera=cam, args=PollerArgs(until=explicit_until), retention_days=30, now=_NOW)
    assert until == explicit_until


# --- extract_thumbnail ---------------------------------------------------------------------------


def test_extract_thumbnail_produces_valid_jpeg(synthetic_clip_path: Path, tmp_path: Path) -> None:
    """Real ffmpeg against the synthetic clip yields a non-empty JPEG (no mocks)."""
    thumb = tmp_path / "thumb.jpg"
    extract_thumbnail(synthetic_clip_path, thumb)
    assert thumb.is_file()
    head = thumb.read_bytes()[:3]
    assert head == b"\xff\xd8\xff", f"expected JPEG magic bytes, got {head!r}"


# --- detection_fields_for ------------------------------------------------------------------------


def _make_mock_detector(*, has_cat: bool, version: str = "test@deadbeef", side_effect: BaseException | None = None) -> MagicMock:
    mock_detector = MagicMock(spec=Detector)
    mock_detector.version = version
    if side_effect is not None:
        mock_detector.detect.side_effect = side_effect
    else:
        mock_detector.detect.return_value = DetectionResult(
            has_cat=has_cat,
            max_score=0.9 if has_cat else 0.0,
            frames_sampled=5,
            frames_with_cat=5 if has_cat else 0,
            best_box_xyxy=(1.0, 2.0, 3.0, 4.0) if has_cat else None,
            detector_version=version,
        )
    return mock_detector


def test_detection_fields_for_returns_no_detect_markers_when_detector_is_none(tmp_path: Path) -> None:
    """``detector=None`` (the ``--no-detect`` path) yields the standard skip markers on the Clip row."""
    fields = detection_fields_for(None, tmp_path / "anyfile.mp4")

    assert fields["analysis_error"] == "skipped: --no-detect"
    assert fields["has_cat"] is False
    assert fields["detector_version"] == "skipped"


def test_detection_fields_for_records_detector_error(tmp_path: Path) -> None:
    """A DetectorError gets captured into ``analysis_error`` so the clip row still inserts."""
    detector = _make_mock_detector(has_cat=False, side_effect=DetectorError("ffprobe died"))
    fields = detection_fields_for(detector, tmp_path / "anyfile.mp4")

    assert fields["has_cat"] is False
    assert fields["analysis_error"] is not None
    assert "ffprobe died" in str(fields["analysis_error"])


# --- update_camera_state_* missing-camera guards --------------------------------------------------


def test_update_state_success_raises_when_camera_missing(db_engine: Engine) -> None:
    """Missing ``camera_id`` is a programming error, not a soft skip."""
    with pytest.raises(ValueError, match="not found"), get_session(db_engine) as session:
        update_camera_state_success(session, camera_id=99999, ingested_clips=[], now=_NOW, overlap_minutes=0)


def test_update_state_failure_raises_when_camera_missing(db_engine: Engine) -> None:
    """Missing ``camera_id`` is a programming error, not a soft skip."""
    with pytest.raises(ValueError, match="not found"), get_session(db_engine) as session:
        update_camera_state_failure(session, camera_id=99999, status=PollStatus.ERROR, error="x", now=_NOW)


# --- extract_thumbnail error paths --------------------------------------------------------------


def test_extract_thumbnail_raises_when_ffmpeg_missing(synthetic_clip_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ``ffmpeg`` on PATH surfaces as ``PollerError`` rather than a raw ``FileNotFoundError``."""

    def missing_which(_name: str) -> str | None:
        return None

    monkeypatch.setattr("cat_watcher.poller.shutil.which", missing_which)

    with pytest.raises(PollerError, match="ffmpeg not on PATH"):
        extract_thumbnail(synthetic_clip_path, tmp_path / "thumb.jpg")


def test_extract_thumbnail_raises_on_ffmpeg_failure(tmp_path: Path) -> None:
    """Ffmpeg's non-zero exit (bogus input) surfaces as ``PollerError`` rather than a raw stderr."""
    bogus = tmp_path / "not-a-video.txt"
    _ = bogus.write_text("definitely not an mp4")
    with pytest.raises(PollerError, match="thumbnail failed"):
        extract_thumbnail(bogus, tmp_path / "thumb.jpg")


# --- CLI parsing ---------------------------------------------------------------------------------


def test_parse_args_supports_full_flag_surface(tmp_path: Path) -> None:
    """All CLI flags round-trip into a populated PollerArgs.

    ``--since`` / ``--until`` use explicit ``+00:00`` offsets so the assertion is independent of
    the test runner's OS timezone (naive values are interpreted as OS-local by ``_parse_iso_datetime``).
    """
    cfg_path = tmp_path / "cfg.toml"
    args = _parse_args(
        [
            "--config",
            str(cfg_path),
            "--camera",
            "pantry",
            "--since",
            "2026-04-30T00:00:00+00:00",
            "--until",
            "2026-05-01T00:00:00+00:00",
            "--limit",
            "3",
            "--no-detect",
            "--list-only",
            "--verbose",
        ],
    )

    assert args.config_path == cfg_path
    assert args.camera == "pantry"
    assert args.since == datetime(2026, 4, 30, tzinfo=UTC)
    assert args.until == datetime(2026, 5, 1, tzinfo=UTC)
    assert args.limit == 3
    assert args.no_detect is True
    assert args.list_only is True
    assert args.verbose is True


def test_parse_args_defaults_are_empty() -> None:
    """No CLI flags means PollerArgs() defaults; ``run_tick`` resolves them at the boundary."""
    args = _parse_args([])

    assert args.config_path is None
    assert args.camera is None
    assert args.since is None
    assert args.no_detect is False
    assert args.verbose is False


def test_parse_iso_datetime_naive_treated_as_os_local() -> None:
    """Naive ISO 8601 input is interpreted as OS-local time and converted to UTC.

    Pinning the system tz makes the round-trip deterministic regardless of the host's actual zone.
    """
    with pinned_tz("America/New_York"):  # EDT in May = UTC-04:00
        # Naive: "May 4 midnight in New York" -> "May 4 04:00 UTC".
        naive_result = _parse_iso_datetime("2026-05-04T00:00:00")
        assert naive_result == datetime(2026, 5, 4, 4, 0, 0, tzinfo=UTC)
        explicit_result = _parse_iso_datetime("2026-05-04T00:00:00+00:00")
        assert explicit_result == datetime(2026, 5, 4, 0, 0, 0, tzinfo=UTC)
        offset_result = _parse_iso_datetime("2026-05-04T00:00:00-04:00")
        assert offset_result == datetime(2026, 5, 4, 4, 0, 0, tzinfo=UTC)


# --- _limited ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("items", "limit", "expected"),
    [
        ([1, 2, 3, 4, 5], 2, [1, 2]),
        ([1, 2, 3], None, [1, 2, 3]),
        ([], 5, []),
    ],
    ids=["caps_to_limit", "none_means_all", "empty_input"],
)
def test_limited_caps_iterable(items: list[int], limit: int | None, expected: list[int]) -> None:
    """``_limited`` caps to ``limit``, returns all when None, and yields nothing for an empty input."""
    assert list(_limited(iter(items), limit)) == expected


# --- run_tick: --camera filter -------------------------------------------------------------------


def test_run_tick_raises_when_camera_filter_does_not_match_config(
    db_engine: Engine,
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """``--camera unknown`` is fail-fast, not a silent no-op."""
    config = make_config(tmp_path, tmp_path)
    args = PollerArgs(camera="not-a-real-camera")

    with pytest.raises(PollerError, match="not-a-real-camera"):
        run_tick(config=config, args=args, engine=db_engine, detector=None, now=_NOW)


# --- _check_alerts_stuck ------------------------------------------------------------------------


def _disabled_alerts_config(make_config: Callable[..., Config], internal_root: Path, storage_root: Path) -> Config:
    """Build a Config with both alert channels disabled — wiring tests don't need real I/O."""
    base = make_config(internal_root, storage_root)
    return base.model_copy(
        update={
            "alerts": base.alerts.model_copy(
                update={
                    "email": EmailRulesConfig(enabled=False),
                    "macos": MacOsRulesConfig(enabled=False),
                },
            ),
        },
    )


def test_check_alerts_stuck_dispatches_when_alerts_heartbeat_stale(
    db_engine: Engine,
    tmp_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """A stale ``alerts`` heartbeat fires ``ALERTS_STUCK`` and writes exactly one ``alerts_sent`` row."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    with get_session(db_engine) as session:
        session.add(Heartbeat(agent_name="alerts", last_seen_at=_NOW - timedelta(minutes=45)))

    _check_alerts_stuck(config=config, engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).filter(AlertSent.alert_type == AlertType.ALERTS_STUCK).all()
    assert len(rows) == 1
    assert rows[0].camera_id is None


def test_check_alerts_stuck_silent_when_heartbeat_recent(
    db_engine: Engine,
    tmp_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """A recent ``alerts`` heartbeat suppresses the stuck-alert dispatch — no row written."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    with get_session(db_engine) as session:
        session.add(Heartbeat(agent_name="alerts", last_seen_at=_NOW - timedelta(minutes=5)))

    _check_alerts_stuck(config=config, engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).all()
    assert rows == []


def test_check_alerts_stuck_silent_when_heartbeat_missing(
    db_engine: Engine,
    tmp_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """A missing heartbeat row is treated as silence, matching the watchdog evaluator's contract."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)

    _check_alerts_stuck(config=config, engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).all()
    assert rows == []


# --- safety-net override: _poll_camera + run_tick -------------------------------------------------


@final
class _StubAmcrestClient:
    """Drop-in stub for :class:`cat_watcher.amcrest_client.AmcrestClient` in poller-tick tests.

    Records the ``(since, until)`` each call resolved to so the test can assert on the window after
    the safety-net override fires. ``recordings`` is the canned ``findFile`` result the iterator
    yields. ``raises`` is a queue of exceptions consumed in order — the Nth call raises the Nth
    queued exception (instead of yielding ``recordings``); when the queue empties, subsequent calls
    fall back to yielding ``recordings``. The queue model lets one stub drive a multi-tick recovery
    test (fail once, succeed after) without re-monkeypatching.
    """

    recordings: tuple[Recording, ...]
    calls: list[tuple[datetime, datetime]]
    pending_raises: list[Exception]

    def __init__(self, recordings: tuple[Recording, ...] = (), *, raises: tuple[Exception, ...] = ()) -> None:
        self.recordings = recordings
        self.calls = []
        self.pending_raises = list(raises)

    def __call__(self, *_args: object, **_kwargs: object) -> Self:
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def iter_recordings(self, *, since: datetime, until: datetime) -> list[Recording]:
        """Record the window; raise the next queued exception if any, else yield ``recordings``."""
        self.calls.append((since, until))
        if self.pending_raises:
            raise self.pending_raises.pop(0)
        return list(self.recordings)


def test_safety_net_not_triggered_at_exact_threshold(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``last_clip_at`` exactly ``safety_net_hours`` ago: strict ``>`` inequality means no trigger."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    safety_hours = config.poller.safety_net_hours
    cam_id = seed_camera(
        db_engine,
        last_clip_at=_NOW - timedelta(hours=safety_hours),
        last_polled_at=_NOW - timedelta(minutes=5),
    )
    stub = _StubAmcrestClient()
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        outcome = _poll_camera(
            config=config,
            db_camera=cam,
            cam_cfg=config.cameras[0],
            engine=db_engine,
            args=PollerArgs(),
            detector=None,
            now=_NOW,
        )
    assert outcome.query.safety_net_triggered is False
    # Default window: since = cam.last_polled_at, until = now.
    assert stub.calls == [(_NOW - timedelta(minutes=5), _NOW)]


def test_safety_net_triggers_just_past_threshold_and_overrides_window(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``last_clip_at`` ``safety_net_hours + 1min`` ago: trigger fires, window anchors on last clip."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    safety_hours = config.poller.safety_net_hours
    overlap_minutes = config.poller.overlap_minutes
    last_clip_at = _NOW - timedelta(hours=safety_hours, minutes=1)
    cam_id = seed_camera(db_engine, last_clip_at=last_clip_at, last_polled_at=_NOW - timedelta(minutes=5))
    stub = _StubAmcrestClient()
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        outcome = _poll_camera(
            config=config,
            db_camera=cam,
            cam_cfg=config.cameras[0],
            engine=db_engine,
            args=PollerArgs(),
            detector=None,
            now=_NOW,
        )
    assert outcome.query.safety_net_triggered is True
    expected_since = last_clip_at - timedelta(minutes=overlap_minutes)
    assert stub.calls == [(expected_since, _NOW)]
    assert outcome.query.queried_since == expected_since
    assert outcome.query.queried_until == _NOW


def test_safety_net_does_not_apply_when_last_clip_at_is_none(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A never-seen-a-clip camera has no clip-anchor; safety net does not apply, default window used."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    cam_id = seed_camera(db_engine, last_clip_at=None, last_polled_at=None)
    stub = _StubAmcrestClient()
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        outcome = _poll_camera(
            config=config,
            db_camera=cam,
            cam_cfg=config.cameras[0],
            engine=db_engine,
            args=PollerArgs(),
            detector=None,
            now=_NOW,
        )
    assert outcome.query.safety_net_triggered is False
    # Default first-run window: since = now - retention.clip_days.
    expected_since = _NOW - timedelta(days=config.retention.clip_days)
    assert stub.calls == [(expected_since, _NOW)]


def test_safety_net_does_not_override_scoped_since_query(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--since`` (and any scoped query) wins over the safety-net override."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    safety_hours = config.poller.safety_net_hours
    last_clip_at = _NOW - timedelta(hours=safety_hours + 2)
    cam_id = seed_camera(db_engine, last_clip_at=last_clip_at)
    stub = _StubAmcrestClient()
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)
    scoped_since = _NOW - timedelta(hours=1)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        outcome = _poll_camera(
            config=config,
            db_camera=cam,
            cam_cfg=config.cameras[0],
            engine=db_engine,
            args=PollerArgs(since=scoped_since),
            detector=None,
            now=_NOW,
        )
    assert outcome.query.safety_net_triggered is False
    assert stub.calls == [(scoped_since, _NOW)]


def test_safety_net_no_alert_when_find_file_returns_rows(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safety net triggered + ``findFile`` returned ≥ 1 row → no ``POLLER_EMPTY_AFTER_QUIET`` alert.

    Duplicate-row paths (all rows already ingested) take this branch too; the contract checks rows
    returned by ``findFile``, not new clips ingested.
    """
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    safety_hours = config.poller.safety_net_hours
    last_clip_at = _NOW - timedelta(hours=safety_hours + 2)
    cam_id = seed_camera(db_engine, last_clip_at=last_clip_at)
    # Pre-seed the existing clip as a duplicate so _ingest_recording skips it (`find_file_row_count`
    # increments but `len(ingested) == 0`); the safety-net alert must still NOT fire.
    existing = Recording(
        source_filename="duplicate.mp4",
        camera_path="/path/duplicate.mp4",
        start_ts=last_clip_at,
        end_ts=last_clip_at + timedelta(seconds=10),
        file_size_bytes=10,
    )
    with get_session(db_engine) as session:
        session.add(
            Clip(
                camera_id=cam_id,
                source_filename="duplicate.mp4",
                start_ts=last_clip_at,
                end_ts=last_clip_at + timedelta(seconds=10),
                duration_seconds=10.0,
                file_path="clips/pantry/x.mp4",
                thumb_path="thumbs/pantry/x.jpg",
                file_size_bytes=10,
                detector_version="test",
                ingested_at=last_clip_at,
            ),
        )
    stub = _StubAmcrestClient(recordings=(existing,))
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)

    run_tick(config=config, args=PollerArgs(), engine=db_engine, detector=None, now=_NOW)

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).filter(AlertSent.alert_type == AlertType.POLLER_EMPTY_AFTER_QUIET).all()
    assert rows == []


def test_safety_net_fires_alert_when_find_file_returns_zero_rows_and_honors_cooldown(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safety net triggered + zero rows: alert fires once; a second tick inside the cool-down does not re-fire."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    safety_hours = config.poller.safety_net_hours
    last_clip_at = _NOW - timedelta(hours=safety_hours + 2)
    cam_id = seed_camera(db_engine, last_clip_at=last_clip_at)
    stub = _StubAmcrestClient()
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)

    run_tick(config=config, args=PollerArgs(), engine=db_engine, detector=None, now=_NOW)
    run_tick(config=config, args=PollerArgs(), engine=db_engine, detector=None, now=_NOW + timedelta(minutes=5))

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).filter(AlertSent.alert_type == AlertType.POLLER_EMPTY_AFTER_QUIET).all()
    assert len(rows) == 1
    assert rows[0].camera_id == cam_id
    assert rows[0].subject.startswith("[cat-watcher] POLLER_EMPTY_AFTER_QUIET")


# --- run_tick: cursor preservation on failure -----------------------------------------------------


def test_run_tick_preserves_cursor_on_camera_unreachable(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CameraUnreachableError`` keeps ``last_polled_at`` at its prior value; status -> UNREACHABLE."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    prior = _NOW - timedelta(hours=1)
    cam_id = seed_camera(db_engine, last_polled_at=prior)
    stub = _StubAmcrestClient(raises=(CameraUnreachableError("connect refused"),))
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)

    run_tick(config=config, args=PollerArgs(), engine=db_engine, detector=None, now=_NOW)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_polled_at == prior  # cursor preserved — failed tick proves no window coverage
        assert cam.poll_status == PollStatus.UNREACHABLE
        assert cam.poll_status_since == _NOW
        assert cam.poll_error == "connect refused"


def test_run_tick_preserves_cursor_on_unexpected_exception(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert an uncaught exception in ``_poll_camera`` still preserves the cursor.

    The bare ``except Exception`` handler in ``run_tick`` catches it; the error message is the
    generic ``"unexpected exception"`` sentinel so logs (not the DB row) carry the traceback
    detail.
    """
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    prior = _NOW - timedelta(hours=1)
    cam_id = seed_camera(db_engine, last_polled_at=prior)
    stub = _StubAmcrestClient(raises=(RuntimeError("boom"),))
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)

    run_tick(config=config, args=PollerArgs(), engine=db_engine, detector=None, now=_NOW)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_polled_at == prior
        assert cam.poll_status == PollStatus.ERROR
        assert cam.poll_status_since == _NOW
        assert cam.poll_error == "unexpected exception"


def test_run_tick_advances_cursor_on_successful_recovery_after_failure(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed tick at T1 keeps cursor at T0; successful tick at T2 jumps cursor to ``T2 - overlap``.

    Single-tick cursor jump from ``T0`` to ``T2 - overlap`` is intentional: the recovery tick's
    ``findFile`` query covers ``[T0, T2]`` (since the cursor was never advanced), so its successful
    completion proves coverage of the whole stretch — the cursor catches up in one move.
    """
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    t0 = _NOW - timedelta(hours=2)
    t1 = _NOW - timedelta(hours=1)
    t2 = _NOW
    cam_id = seed_camera(db_engine, last_polled_at=t0)
    stub = _StubAmcrestClient(raises=(CameraUnreachableError("transient"),))
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)

    run_tick(config=config, args=PollerArgs(), engine=db_engine, detector=None, now=t1)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_polled_at == t0  # unchanged after failed tick

    run_tick(config=config, args=PollerArgs(), engine=db_engine, detector=None, now=t2)

    expected_cursor = t2 - timedelta(minutes=config.poller.overlap_minutes)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_polled_at == expected_cursor
        assert cam.poll_status == PollStatus.OK
        assert cam.poll_status_since is None
        assert cam.poll_error is None
    # Recovery tick's query covered the full lagged window [t0, t2].
    assert stub.calls[-1] == (t0, t2)


# --- run_tick: structured per-tick INFO logging ---------------------------------------------------


def _seed_pantry_with_cursor(engine: Engine, last_polled_at: datetime) -> None:
    """Insert a ``pantry`` row with the given cursor.

    Used by the structured-logging tests to set up the prior ``last_polled_at`` without paying the
    6-fixture cost of ``seed_camera``.
    """
    with get_session(engine) as session:
        session.add(
            Camera(
                name="pantry",
                display_name="Pantry",
                host="cam.example.com",
                poll_status=PollStatus.OK,
                last_polled_at=last_polled_at,
            ),
        )


def test_run_tick_emits_poll_tick_info_on_success(
    db_engine: Engine,
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One INFO ``poll_tick`` record per successful camera tick, with the full ``extras`` schema."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    # Prior cursor lags far enough back that the new cursor will be ``now - overlap_minutes``
    # rather than clamped to ``prior`` by the no-rewind ``max(target, prior)`` rule.
    prior = _NOW - timedelta(hours=1)
    _seed_pantry_with_cursor(db_engine, last_polled_at=prior)
    stub = _StubAmcrestClient()
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)

    logging.getLogger("cat_watcher.poller").disabled = False
    with caplog.at_level(logging.INFO, logger="cat_watcher.poller"):
        run_tick(config=config, args=PollerArgs(), engine=db_engine, detector=None, now=_NOW)

    records = [r for r in caplog.records if r.message == "poll_tick"]
    assert len(records) == 1
    rec = records[0]
    assert rec.levelno == logging.INFO
    assert vars(rec)["camera_name"] == "pantry"
    assert vars(rec)["window_since"] == prior.isoformat()
    assert vars(rec)["window_until"] == _NOW.isoformat()
    assert vars(rec)["findFile_rows"] == 0
    assert vars(rec)["ingested_clips"] == 0
    assert vars(rec)["skipped_duplicates"] == 0
    assert vars(rec)["safety_net_triggered"] is False
    assert vars(rec)["cursor_before"] == prior.isoformat()
    assert vars(rec)["cursor_after"] == (_NOW - timedelta(minutes=config.poller.overlap_minutes)).isoformat()


def test_run_tick_emits_poll_tick_failed_warning_on_typed_amcrest_failure(
    db_engine: Engine,
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One WARNING ``poll_tick_failed`` record per failed tick, with ``error_type`` and ``error_msg``."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    prior = _NOW - timedelta(hours=1)
    _seed_pantry_with_cursor(db_engine, last_polled_at=prior)
    stub = _StubAmcrestClient(raises=(CameraUnreachableError("connect refused"),))
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)

    logging.getLogger("cat_watcher.poller").disabled = False
    with caplog.at_level(logging.WARNING, logger="cat_watcher.poller"):
        run_tick(config=config, args=PollerArgs(), engine=db_engine, detector=None, now=_NOW)

    records = [r for r in caplog.records if r.message == "poll_tick_failed"]
    assert len(records) == 1
    rec = records[0]
    assert rec.levelno == logging.WARNING
    assert vars(rec)["camera_name"] == "pantry"
    assert vars(rec)["error_type"] == "CameraUnreachableError"
    assert vars(rec)["error_msg"] == "connect refused"
    # Cursor preserved per Task 5.
    assert vars(rec)["cursor_before"] == prior.isoformat()
    assert vars(rec)["cursor_after"] == prior.isoformat()


def test_run_tick_emits_poll_tick_failed_warning_on_unexpected_exception(
    db_engine: Engine,
    tmp_path: Path,
    make_config: Callable[..., Config],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Assert the bare-``except`` path emits ``poll_tick_failed`` with the exception class + message.

    ``window_since`` / ``window_until`` are ``None`` because ``_poll_camera`` raised before the
    window resolved.
    """
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)
    prior = _NOW - timedelta(hours=1)
    _seed_pantry_with_cursor(db_engine, last_polled_at=prior)
    stub = _StubAmcrestClient(raises=(RuntimeError("boom"),))
    monkeypatch.setattr("cat_watcher.poller.AmcrestClient", stub)

    logging.getLogger("cat_watcher.poller").disabled = False
    with caplog.at_level(logging.WARNING, logger="cat_watcher.poller"):
        run_tick(config=config, args=PollerArgs(), engine=db_engine, detector=None, now=_NOW)

    records = [r for r in caplog.records if r.message == "poll_tick_failed"]
    assert len(records) == 1
    rec = records[0]
    assert rec.levelno == logging.WARNING
    assert vars(rec)["camera_name"] == "pantry"
    assert vars(rec)["window_since"] is None
    assert vars(rec)["window_until"] is None
    assert vars(rec)["error_type"] == "RuntimeError"
    assert vars(rec)["error_msg"] == "boom"
    assert vars(rec)["cursor_before"] == prior.isoformat()
    assert vars(rec)["cursor_after"] == prior.isoformat()

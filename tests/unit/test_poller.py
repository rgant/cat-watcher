"""Tests for cat_watcher.poller.

The poller composes existing modules (amcrest_client, detector, retention, storage, db). Most unit
tests target small helper functions in isolation against the ``db_engine`` fixture; the end-to-end
poll-tick exercise lives in tests/integration/test_poller_end_to_end.py.
"""

import fcntl
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # runtime: pytest fixture annotations are evaluated by collectors
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from cat_watcher.db import Camera, Clip, Heartbeat, PollStatus, get_session
from cat_watcher.detector import DetectionResult, Detector, DetectorError
from cat_watcher.poller import (
    PollerArgs,
    PollerError,
    PollerLockedError,
    _detection_fields_for,
    _extract_thumbnail,
    _limited,
    _parse_args,
    _relative_paths_for,
    _resolve_window,
    pid_lock,
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


def _seed_camera(engine: Engine, **overrides: object) -> int:
    """Insert a Camera row with sensible defaults; overrides override fields."""
    defaults: dict[str, object] = {
        "name": "pantry",
        "display_name": "Pantry",
        "host": "pantry.local",
        "poll_status": PollStatus.OK,
    }
    defaults.update(overrides)
    cam = Camera(**defaults)
    with get_session(engine) as session:
        session.add(cam)
        session.flush()
        return cam.id


def _seed_clip(engine: Engine, *, camera_id: int, start_ts: datetime, has_cat: bool, manual_has_cat: bool | None = None) -> None:
    clip = Clip(
        camera_id=camera_id,
        source_filename=f"{start_ts.strftime('%H%M%S')}.mp4",
        start_ts=start_ts,
        end_ts=start_ts + timedelta(seconds=2),
        duration_seconds=2.0,
        file_path=f"clips/pantry/{start_ts.strftime('%Y-%m-%d')}/{start_ts.strftime('%H%M%S')}.mp4",
        thumb_path=f"thumbs/pantry/{start_ts.strftime('%Y-%m-%d')}/{start_ts.strftime('%H%M%S')}.jpg",
        file_size_bytes=10,
        has_cat=has_cat,
        manual_has_cat=manual_has_cat,
        detector_version="test@deadbeef",
        ingested_at=start_ts,
    )
    with get_session(engine) as session:
        session.add(clip)


# --- update_camera_state_success ------------------------------------------------------------------


def test_update_state_success_advances_last_polled_when_no_clips(db_engine: Engine) -> None:
    """A successful tick with no new clips advances ``last_polled_at`` only."""
    cam_id = _seed_camera(db_engine, last_clip_at=_NOW - timedelta(days=2), last_cat_seen_at=_NOW - timedelta(days=3))
    previous_clip_at = _NOW - timedelta(days=2)
    previous_cat_at = _NOW - timedelta(days=3)

    with get_session(db_engine) as session:
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=[], now=_NOW)

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


def test_update_state_success_advances_last_clip_at_when_clips_ingested(db_engine: Engine) -> None:
    """``last_clip_at`` becomes ``max(start_ts of new clips)`` when clips were ingested."""
    cam_id = _seed_camera(db_engine)
    clip1_ts = _NOW - timedelta(minutes=30)
    clip2_ts = _NOW - timedelta(minutes=10)
    _seed_clip(db_engine, camera_id=cam_id, start_ts=clip1_ts, has_cat=False)
    _seed_clip(db_engine, camera_id=cam_id, start_ts=clip2_ts, has_cat=False)

    with get_session(db_engine) as session:
        clips = session.query(Clip).order_by(Clip.start_ts).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_clip_at == clip2_ts  # the later one wins


def test_update_state_success_advances_last_cat_seen_when_cat_positive_clip(db_engine: Engine) -> None:
    """``last_cat_seen_at`` advances to the latest cat-positive clip's start_ts."""
    cam_id = _seed_camera(db_engine)
    no_cat_ts = _NOW - timedelta(minutes=30)
    cat_ts = _NOW - timedelta(minutes=20)
    later_no_cat_ts = _NOW - timedelta(minutes=10)
    _seed_clip(db_engine, camera_id=cam_id, start_ts=no_cat_ts, has_cat=False)
    _seed_clip(db_engine, camera_id=cam_id, start_ts=cat_ts, has_cat=True)
    _seed_clip(db_engine, camera_id=cam_id, start_ts=later_no_cat_ts, has_cat=False)

    with get_session(db_engine) as session:
        clips = session.query(Clip).order_by(Clip.start_ts).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_cat_seen_at == cat_ts  # latest TRUE wins; later no-cat doesn't move it back
        assert cam.last_clip_at == later_no_cat_ts  # last_clip_at sees all clips


def test_update_state_success_preserves_last_cat_seen_when_no_cat_positive(db_engine: Engine) -> None:
    """If new clips are ingested but none are cat-positive, ``last_cat_seen_at`` is preserved."""
    previous_cat_at = _NOW - timedelta(days=3)
    cam_id = _seed_camera(db_engine, last_cat_seen_at=previous_cat_at)
    new_ts = _NOW - timedelta(minutes=10)
    _seed_clip(db_engine, camera_id=cam_id, start_ts=new_ts, has_cat=False)

    with get_session(db_engine) as session:
        clips = session.query(Clip).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_cat_seen_at == previous_cat_at  # NOT moved to ``new_ts``
        assert cam.last_clip_at == new_ts


def test_update_state_success_respects_manual_has_cat_override(db_engine: Engine) -> None:
    """``COALESCE(manual_has_cat, has_cat)`` semantics: a model-false clip with ``manual_has_cat=True`` counts as cat-positive."""
    cam_id = _seed_camera(db_engine)
    ts = _NOW - timedelta(minutes=5)
    _seed_clip(db_engine, camera_id=cam_id, start_ts=ts, has_cat=False, manual_has_cat=True)

    with get_session(db_engine) as session:
        clips = session.query(Clip).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_cat_seen_at == ts


def test_update_state_success_clears_poll_status_since_on_recovery(db_engine: Engine) -> None:
    """A successful tick after a non-OK status clears ``poll_status_since`` (transition back to OK)."""
    cam_id = _seed_camera(
        db_engine,
        poll_status=PollStatus.UNREACHABLE,
        poll_status_since=_NOW - timedelta(hours=1),
        poll_error="prior failure",
    )

    with get_session(db_engine) as session:
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=[], now=_NOW)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.poll_status == PollStatus.OK
        assert cam.poll_status_since is None
        assert cam.poll_error is None


# --- update_camera_state_failure ------------------------------------------------------------------


def test_update_state_failure_sets_poll_status_since_on_first_failure(db_engine: Engine) -> None:
    """OK -> non-OK transition sets ``poll_status_since`` to ``now`` and records the error."""
    cam_id = _seed_camera(db_engine)  # starts OK

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


def test_update_state_failure_preserves_poll_status_since_on_repeat(db_engine: Engine) -> None:
    """A second consecutive non-OK tick leaves ``poll_status_since`` unchanged (still failing)."""
    earlier = _NOW - timedelta(hours=1)
    cam_id = _seed_camera(
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


# --- upsert_heartbeat -----------------------------------------------------------------------------


def test_upsert_heartbeat_inserts_when_absent(db_engine: Engine) -> None:
    """First call for an agent name inserts the row."""
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
    """The lock context manager writes the current PID and removes the lock on clean exit."""
    with pid_lock(tmp_path):
        pid_file = tmp_path / ".poller.pid"
        assert pid_file.is_file()
        assert pid_file.read_text().strip() == str(os.getpid())
    # After exit the file may still exist (we only release the flock); a re-acquire works.
    with pid_lock(tmp_path):
        pass


def test_pid_lock_raises_when_already_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second concurrent acquisition raises ``PollerLockedError``.

    Mocks ``fcntl.flock`` to raise ``BlockingIOError`` on the second call (mirroring what
    ``LOCK_NB`` does cross-process when the lock is held). Same-process flock is reentrant on the
    same FD so we can't exercise this with a literal nested ``with``.
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


# --- _relative_paths_for --------------------------------------------------------------------------


def test_relative_paths_for_uses_camera_local_date_and_time() -> None:
    """The on-disk layout matches the camera-local clock (operator-friendly date dirs)."""
    local_dt = datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)
    rel_clip, rel_thumb = _relative_paths_for("pantry", local_dt)
    assert rel_clip == "clips/pantry/2026-05-01/064704.mp4"
    assert rel_thumb == "thumbs/pantry/2026-05-01/064704.jpg"


# --- _resolve_window ------------------------------------------------------------------------------


def test_resolve_window_uses_args_since_when_provided(db_engine: Engine) -> None:
    """``--since`` overrides every other source."""
    cam_id = _seed_camera(db_engine, last_polled_at=_NOW - timedelta(hours=1))
    args = PollerArgs(since=_NOW - timedelta(days=2))
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        since, until = _resolve_window(db_camera=cam, args=args, retention_days=30, now=_NOW)
    assert since == _NOW - timedelta(days=2)
    assert until == _NOW


def test_resolve_window_uses_last_polled_at_when_no_since(db_engine: Engine) -> None:
    """Steady-state: ``camera.last_polled_at`` is the start of the window."""
    last_poll = _NOW - timedelta(minutes=10)
    cam_id = _seed_camera(db_engine, last_polled_at=last_poll)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        since, _ = _resolve_window(db_camera=cam, args=PollerArgs(), retention_days=30, now=_NOW)
    assert since == last_poll


def test_resolve_window_first_run_uses_retention_days_backfill(db_engine: Engine) -> None:
    """Fresh-install: no ``last_polled_at`` and no ``--since`` -> fall back to ``now - retention``."""
    cam_id = _seed_camera(db_engine, last_polled_at=None)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        since, _ = _resolve_window(db_camera=cam, args=PollerArgs(), retention_days=30, now=_NOW)
    assert since == _NOW - timedelta(days=30)


def test_resolve_window_args_until_overrides_now(db_engine: Engine) -> None:
    """``--until`` caps the upper bound; default is ``now``."""
    cam_id = _seed_camera(db_engine)
    explicit_until = _NOW - timedelta(hours=1)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        _, until = _resolve_window(db_camera=cam, args=PollerArgs(until=explicit_until), retention_days=30, now=_NOW)
    assert until == explicit_until


# --- _extract_thumbnail ---------------------------------------------------------------------------


def test_extract_thumbnail_produces_valid_jpeg(synthetic_clip_path: Path, tmp_path: Path) -> None:
    """Real ffmpeg invocation against the synthetic clip yields a non-empty JPEG."""
    thumb = tmp_path / "thumb.jpg"
    _extract_thumbnail(synthetic_clip_path, thumb)
    assert thumb.is_file()
    head = thumb.read_bytes()[:3]
    assert head == b"\xff\xd8\xff", f"expected JPEG magic bytes, got {head!r}"


# --- _detection_fields_for ------------------------------------------------------------------------


def _make_mock_detector(*, has_cat: bool, version: str = "test@deadbeef", side_effect: BaseException | None = None) -> MagicMock:
    """Build a Detector mock with a canned ``detect()`` return or side effect."""
    mock_detector: MagicMock = MagicMock(spec=Detector)
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
    """``--no-detect`` (detector=None) yields the standard skip markers stored in the Clip row."""
    from cat_watcher.poller import _detection_fields_for

    fields = _detection_fields_for(None, tmp_path / "anyfile.mp4")

    assert fields["analysis_error"] == "skipped: --no-detect"
    assert fields["has_cat"] is False
    assert fields["detector_version"] == "skipped"


def test_detection_fields_for_records_detector_error(tmp_path: Path) -> None:
    """A DetectorError during detect() is captured into ``analysis_error`` (clip still inserts)."""
    detector = _make_mock_detector(has_cat=False, side_effect=DetectorError("ffprobe died"))
    fields = _detection_fields_for(detector, tmp_path / "anyfile.mp4")

    assert fields["has_cat"] is False
    assert fields["analysis_error"] is not None
    assert "ffprobe died" in str(fields["analysis_error"])


# --- update_camera_state_* missing-camera guards --------------------------------------------------


def test_update_state_success_raises_when_camera_missing(db_engine: Engine) -> None:
    """Missing ``camera_id`` is a programming error, not a soft skip."""
    with pytest.raises(ValueError, match="not found"), get_session(db_engine) as session:
        update_camera_state_success(session, camera_id=99999, ingested_clips=[], now=_NOW)


def test_update_state_failure_raises_when_camera_missing(db_engine: Engine) -> None:
    """Missing ``camera_id`` is a programming error, not a soft skip."""
    with pytest.raises(ValueError, match="not found"), get_session(db_engine) as session:
        update_camera_state_failure(session, camera_id=99999, status=PollStatus.ERROR, error="x", now=_NOW)


# --- _extract_thumbnail error paths --------------------------------------------------------------


def test_extract_thumbnail_raises_when_ffmpeg_missing(synthetic_clip_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``ffmpeg`` missing from PATH surfaces as ``PollerError``."""

    def missing_which(_name: str) -> str | None:
        return None

    monkeypatch.setattr("cat_watcher.poller.shutil.which", missing_which)

    with pytest.raises(PollerError, match="ffmpeg not on PATH"):
        _extract_thumbnail(synthetic_clip_path, tmp_path / "thumb.jpg")


def test_extract_thumbnail_raises_on_ffmpeg_failure(tmp_path: Path) -> None:
    """A bogus input file causes ffmpeg to exit non-zero, surfaced as ``PollerError``."""
    bogus = tmp_path / "not-a-video.txt"
    _ = bogus.write_text("definitely not an mp4")
    with pytest.raises(PollerError, match="thumbnail failed"):
        _extract_thumbnail(bogus, tmp_path / "thumb.jpg")


# --- CLI parsing ---------------------------------------------------------------------------------


def test_parse_args_supports_full_flag_surface(tmp_path: Path) -> None:
    """All seven CLI flags from spec §4.10 round-trip into a populated PollerArgs."""
    cfg_path = tmp_path / "cfg.toml"
    args = _parse_args(
        [
            "--config",
            str(cfg_path),
            "--camera",
            "pantry",
            "--since",
            "2026-04-30T00:00:00",
            "--until",
            "2026-05-01T00:00:00",
            "--limit",
            "3",
            "--no-detect",
            "--list-only",
        ],
    )

    assert args.config_path == cfg_path
    assert args.camera == "pantry"
    assert args.since == datetime(2026, 4, 30, tzinfo=UTC)
    assert args.until == datetime(2026, 5, 1, tzinfo=UTC)
    assert args.limit == 3
    assert args.no_detect is True
    assert args.list_only is True


def test_parse_args_defaults_are_empty() -> None:
    """No CLI flags means PollerArgs() defaults — ``run_tick`` falls back at the boundary."""
    args = _parse_args([])

    assert args.config_path is None
    assert args.camera is None
    assert args.since is None
    assert args.no_detect is False


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
    """``_limited`` yields at most ``limit`` items; ``None`` yields everything."""
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

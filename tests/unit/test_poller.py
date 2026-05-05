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
    """A successful tick with no new clips advances ``last_polled_at`` only."""
    cam_id = seed_camera(db_engine, last_clip_at=_NOW - timedelta(days=2), last_cat_seen_at=_NOW - timedelta(days=3))
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


def test_update_state_success_advances_last_clip_at_when_clips_ingested(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """``last_clip_at`` becomes ``max(start_ts of new clips)`` when clips were ingested."""
    cam_id = seed_camera(db_engine)
    clip1_ts = _NOW - timedelta(minutes=30)
    clip2_ts = _NOW - timedelta(minutes=10)
    seed_clip(db_engine, camera_id=cam_id, start_ts=clip1_ts, has_cat=False)
    seed_clip(db_engine, camera_id=cam_id, start_ts=clip2_ts, has_cat=False)

    with get_session(db_engine) as session:
        clips = session.query(Clip).order_by(Clip.start_ts).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_clip_at == clip2_ts  # the later one wins


def test_update_state_success_advances_last_cat_seen_when_cat_positive_clip(
    db_engine: Engine,
    seed_camera: Callable[..., int],
    seed_clip: Callable[..., None],
) -> None:
    """``last_cat_seen_at`` advances to the latest cat-positive clip's start_ts."""
    cam_id = seed_camera(db_engine)
    no_cat_ts = _NOW - timedelta(minutes=30)
    cat_ts = _NOW - timedelta(minutes=20)
    later_no_cat_ts = _NOW - timedelta(minutes=10)
    seed_clip(db_engine, camera_id=cam_id, start_ts=no_cat_ts, has_cat=False)
    seed_clip(db_engine, camera_id=cam_id, start_ts=cat_ts, has_cat=True)
    seed_clip(db_engine, camera_id=cam_id, start_ts=later_no_cat_ts, has_cat=False)

    with get_session(db_engine) as session:
        clips = session.query(Clip).order_by(Clip.start_ts).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW)

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
    """If new clips are ingested but none are cat-positive, ``last_cat_seen_at`` is preserved."""
    previous_cat_at = _NOW - timedelta(days=3)
    cam_id = seed_camera(db_engine, last_cat_seen_at=previous_cat_at)
    new_ts = _NOW - timedelta(minutes=10)
    seed_clip(db_engine, camera_id=cam_id, start_ts=new_ts, has_cat=False)

    with get_session(db_engine) as session:
        clips = session.query(Clip).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW)

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
    """``COALESCE(manual_has_cat, has_cat)`` semantics: a model-false clip with ``manual_has_cat=True`` counts as cat-positive."""
    cam_id = seed_camera(db_engine)
    ts = _NOW - timedelta(minutes=5)
    seed_clip(db_engine, camera_id=cam_id, start_ts=ts, has_cat=False, manual_has_cat=True)

    with get_session(db_engine) as session:
        clips = session.query(Clip).all()
        update_camera_state_success(session, camera_id=cam_id, ingested_clips=clips, now=_NOW)

    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        assert cam.last_cat_seen_at == ts


def test_update_state_success_clears_poll_status_since_on_recovery(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """A successful tick after a non-OK status clears ``poll_status_since`` (transition back to OK)."""
    cam_id = seed_camera(
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


def test_update_state_failure_sets_poll_status_since_on_first_failure(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """OK -> non-OK transition sets ``poll_status_since`` to ``now`` and records the error."""
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
    """A second consecutive non-OK tick leaves ``poll_status_since`` unchanged (still failing)."""
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
    """``advance_cursor=False`` on the failure path: cursor stays put, status fields update."""
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
    """``--since`` / ``--until`` / ``--limit`` mark the run as scoped; other flags do not."""
    args, expected = case
    assert args.truncates_default_window is expected


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


# --- relative_paths_for --------------------------------------------------------------------------


def test_relative_paths_for_uses_camera_local_date_and_time() -> None:
    """The on-disk layout matches the camera-local clock (operator-friendly date dirs)."""
    local_dt = datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)
    rel_clip, rel_thumb = relative_paths_for("pantry", local_dt)
    assert rel_clip == "clips/pantry/2026-05-01/064704.mp4"
    assert rel_thumb == "thumbs/pantry/2026-05-01/064704.jpg"


# --- _resolve_window ------------------------------------------------------------------------------


def test_resolve_window_uses_args_since_when_provided(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """``--since`` overrides every other source."""
    cam_id = seed_camera(db_engine, last_polled_at=_NOW - timedelta(hours=1))
    args = PollerArgs(since=_NOW - timedelta(days=2))
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        since, until = _resolve_window(db_camera=cam, args=args, retention_days=30, now=_NOW)
    assert since == _NOW - timedelta(days=2)
    assert until == _NOW


def test_resolve_window_uses_last_polled_at_when_no_since(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Steady-state: ``camera.last_polled_at`` is the start of the window."""
    last_poll = _NOW - timedelta(minutes=10)
    cam_id = seed_camera(db_engine, last_polled_at=last_poll)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        since, _ = _resolve_window(db_camera=cam, args=PollerArgs(), retention_days=30, now=_NOW)
    assert since == last_poll


def test_resolve_window_first_run_uses_retention_days_backfill(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """Fresh-install: no ``last_polled_at`` and no ``--since`` -> fall back to ``now - retention``."""
    cam_id = seed_camera(db_engine, last_polled_at=None)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        since, _ = _resolve_window(db_camera=cam, args=PollerArgs(), retention_days=30, now=_NOW)
    assert since == _NOW - timedelta(days=30)


def test_resolve_window_args_until_overrides_now(db_engine: Engine, seed_camera: Callable[..., int]) -> None:
    """``--until`` caps the upper bound; default is ``now``."""
    cam_id = seed_camera(db_engine)
    explicit_until = _NOW - timedelta(hours=1)
    with get_session(db_engine) as session:
        cam = session.get(Camera, cam_id)
        assert cam is not None
        _, until = _resolve_window(db_camera=cam, args=PollerArgs(until=explicit_until), retention_days=30, now=_NOW)
    assert until == explicit_until


# --- extract_thumbnail ---------------------------------------------------------------------------


def test_extract_thumbnail_produces_valid_jpeg(synthetic_clip_path: Path, tmp_path: Path) -> None:
    """Real ffmpeg invocation against the synthetic clip yields a non-empty JPEG."""
    thumb = tmp_path / "thumb.jpg"
    extract_thumbnail(synthetic_clip_path, thumb)
    assert thumb.is_file()
    head = thumb.read_bytes()[:3]
    assert head == b"\xff\xd8\xff", f"expected JPEG magic bytes, got {head!r}"


# --- detection_fields_for ------------------------------------------------------------------------


def _make_mock_detector(*, has_cat: bool, version: str = "test@deadbeef", side_effect: BaseException | None = None) -> MagicMock:
    """Build a Detector mock with a canned ``detect()`` return or side effect."""
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
    """``--no-detect`` (detector=None) yields the standard skip markers stored in the Clip row."""
    fields = detection_fields_for(None, tmp_path / "anyfile.mp4")

    assert fields["analysis_error"] == "skipped: --no-detect"
    assert fields["has_cat"] is False
    assert fields["detector_version"] == "skipped"


def test_detection_fields_for_records_detector_error(tmp_path: Path) -> None:
    """A DetectorError during detect() is captured into ``analysis_error`` (clip still inserts)."""
    detector = _make_mock_detector(has_cat=False, side_effect=DetectorError("ffprobe died"))
    fields = detection_fields_for(detector, tmp_path / "anyfile.mp4")

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


# --- extract_thumbnail error paths --------------------------------------------------------------


def test_extract_thumbnail_raises_when_ffmpeg_missing(synthetic_clip_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``ffmpeg`` missing from PATH surfaces as ``PollerError``."""

    def missing_which(_name: str) -> str | None:
        return None

    monkeypatch.setattr("cat_watcher.poller.shutil.which", missing_which)

    with pytest.raises(PollerError, match="ffmpeg not on PATH"):
        extract_thumbnail(synthetic_clip_path, tmp_path / "thumb.jpg")


def test_extract_thumbnail_raises_on_ffmpeg_failure(tmp_path: Path) -> None:
    """A bogus input file causes ffmpeg to exit non-zero, surfaced as ``PollerError``."""
    bogus = tmp_path / "not-a-video.txt"
    _ = bogus.write_text("definitely not an mp4")
    with pytest.raises(PollerError, match="thumbnail failed"):
        extract_thumbnail(bogus, tmp_path / "thumb.jpg")


# --- CLI parsing ---------------------------------------------------------------------------------


def test_parse_args_supports_full_flag_surface(tmp_path: Path) -> None:
    """All CLI flags round-trip into a populated PollerArgs.

    ``--since`` / ``--until`` use explicit ``+00:00`` offsets so the test is independent of the OS
    timezone the test runner happens to be in (naive values are now interpreted as OS-local by
    ``_parse_iso_datetime``).
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
    """No CLI flags means PollerArgs() defaults — ``run_tick`` falls back at the boundary."""
    args = _parse_args([])

    assert args.config_path is None
    assert args.camera is None
    assert args.since is None
    assert args.no_detect is False
    assert args.verbose is False


def test_parse_iso_datetime_naive_treated_as_os_local() -> None:
    """Naive ISO 8601 input is interpreted as OS-local time and converted to UTC.

    This is what ``_parse_iso_datetime`` documents — it relies on ``datetime.astimezone()`` treating
    naive values as system-local. The test pins the system timezone via the ``TZ`` env var +
    ``time.tzset()`` so the round-trip is deterministic regardless of the host's actual zone (the
    alternative — OS-dependent expected values — would be flaky).
    """
    import os
    import time

    original_tz = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"  # EDT in May = UTC-04:00
    time.tzset()
    try:
        # Naive: "May 4 midnight in New York" → "May 4 04:00 UTC"
        naive_result = _parse_iso_datetime("2026-05-04T00:00:00")
        assert naive_result == datetime(2026, 5, 4, 4, 0, 0, tzinfo=UTC)
        # Explicit UTC offset: honored as-is.
        explicit_result = _parse_iso_datetime("2026-05-04T00:00:00+00:00")
        assert explicit_result == datetime(2026, 5, 4, 0, 0, 0, tzinfo=UTC)
        # Explicit non-UTC offset: converted.
        offset_result = _parse_iso_datetime("2026-05-04T00:00:00-04:00")
        assert offset_result == datetime(2026, 5, 4, 4, 0, 0, tzinfo=UTC)
    finally:
        if original_tz is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = original_tz
        time.tzset()


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


# --- _check_alerts_stuck ------------------------------------------------------------------------


def _disabled_alerts_config(make_config: Callable[..., Config], internal_root: Path, storage_root: Path) -> Config:
    """Build a Config with both alert channels disabled — Task 18 wiring tests don't need real I/O."""
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
    """A stale ``alerts`` heartbeat fires ``ALERTS_STUCK`` via :func:`dispatch_alert` (one row written)."""
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
    """Recent ``alerts`` heartbeat → no fire (no ``alerts_sent`` row written)."""
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
    """No ``alerts`` heartbeat row at all → no fire (matches the watchdog evaluator's contract)."""
    config = _disabled_alerts_config(make_config, tmp_path, tmp_path)

    _check_alerts_stuck(config=config, engine=db_engine, now=_NOW)

    with get_session(db_engine) as session:
        rows = session.query(AlertSent).all()
    assert rows == []

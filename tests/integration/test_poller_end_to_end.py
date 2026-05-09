"""End-to-end poller tick.

Wires the real ``run_tick`` against a real DB (file-backed SQLite) + real ``storage_root``
(``tmp_path``) + real ffmpeg (for thumbnails) + a mocked Amcrest API (respx) + a mocked Detector.
Verifies the file-before-row ordering invariant: the ``clips`` row only commits after the .mp4
and .jpg files exist on disk.
"""

from collections.abc import Callable  # noqa: TC003  # runtime: respx.mock evaluates fixture annotations at decoration time
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # runtime: respx.mock evaluates fixture annotations at decoration time
from unittest.mock import MagicMock

import httpx  # respx returns httpx types; httpxyz aliases httpx to httpxyz at runtime.
import numpy as np
import pytest  # noqa: TC002  # runtime: respx.mock evaluates pytest.MonkeyPatch annotations at decoration time
import respx
from sqlalchemy.engine import Engine  # noqa: TC002  # runtime: Engine-annotated helper called at module level via fixtures
from sqlalchemy.orm import Session

from cat_watcher.config import CameraConfig, Config
from cat_watcher.db import AgentStart, Base, Camera, Clip, Heartbeat, PollStatus, create_engine, get_session
from cat_watcher.detector import DetectionResult, Detector, DetectorError, ScoredFrame
from cat_watcher.poller import PollerArgs, run_tick

_BASE_URL = "http://cam.example.com:80"
_FIND_URL = f"{_BASE_URL}/cgi-bin/mediaFileFind.cgi"
_FIND_HANDLE = "99"
_DOWNLOAD_PATH = "/mnt/sd/2026-05-01/001/dav/06/06.47.04-06.48.58[M][0@0][0].mp4"
_DOWNLOAD_URL = f"{_BASE_URL}/cgi-bin/RPC_Loadfile{_DOWNLOAD_PATH}"
_NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


def _seed_amcrest_mocks(payload: bytes) -> None:
    """Stand up the respx routes the AmcrestClient walks for one recording."""
    _ = respx.get(_FIND_URL, params={"action": "factory.create"}).mock(
        return_value=httpx.Response(200, text=f"result={_FIND_HANDLE}\r\n"),
    )
    _ = respx.get(_FIND_URL, params__contains={"action": "findFile", "object": _FIND_HANDLE}).mock(
        return_value=httpx.Response(200, text="OK\r\n"),
    )
    page = (
        "found=1\r\n"
        f"items[0].FilePath={_DOWNLOAD_PATH}\r\n"
        "items[0].StartTime=2026-05-01 06:47:04\r\n"
        "items[0].EndTime=2026-05-01 06:48:58\r\n"
        f"items[0].Length={len(payload)}\r\n"
    )
    _ = respx.get(_FIND_URL, params={"action": "findNextFile", "object": _FIND_HANDLE, "count": "100"}).mock(
        return_value=httpx.Response(200, text=page),
    )
    _ = respx.get(_FIND_URL, params={"action": "close", "object": _FIND_HANDLE}).mock(return_value=httpx.Response(200, text="OK\r\n"))
    _ = respx.get(_FIND_URL, params={"action": "destroy", "object": _FIND_HANDLE}).mock(return_value=httpx.Response(200, text="OK\r\n"))
    _ = respx.get(_DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=payload))


def _make_detector(
    *,
    has_cat: bool,
    scored_frames: tuple[ScoredFrame, ...] = (),
) -> MagicMock:
    """A Detector mock that returns a canned DetectionResult.

    Pass ``scored_frames`` to populate ``DetectionResult.scored_frames`` so the success-path
    per-frame thumbnail pipeline has frames to encode.
    """
    mock_detector = MagicMock(spec=Detector)
    mock_detector.version = "yolo11n.pt@deadbeef"
    mock_detector.detect.return_value = DetectionResult(
        has_cat=has_cat,
        max_score=0.92 if has_cat else 0.0,
        frames_sampled=5,
        frames_with_cat=5 if has_cat else 0,
        best_box_xyxy=(10.0, 20.0, 30.0, 40.0) if has_cat else None,
        detector_version="yolo11n.pt@deadbeef",
        scored_frames=scored_frames,
    )
    return mock_detector


def _stub_scored_frames(scores: tuple[float, ...]) -> tuple[ScoredFrame, ...]:
    """Build ``ScoredFrame``s with stub ndarrays for tests that exercise the per-frame thumb path."""
    stub_frame = np.zeros((180, 320, 3), dtype=np.uint8)
    return tuple(ScoredFrame(ordinal=i, t_offset_seconds=float(i + 1), score=score, frame=stub_frame) for i, score in enumerate(scores))


@respx.mock
def test_full_tick_ingests_clip_to_canonical_layout(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """One end-to-end tick: discover one Recording, download, thumbnail, detect, insert.

    Verifies file-before-row ordering: both files exist on disk and the Clip row points at them with
    matching paths. The downloaded "video bytes" are the synthetic test clip, so ffmpeg's thumbnail
    step succeeds against real H.264.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    payload = synthetic_clip_path.read_bytes()
    _seed_amcrest_mocks(payload)
    detector = _make_detector(has_cat=True)

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC))
    try:
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
    finally:
        engine.dispose()

    expected_clip = storage_root / "clips/pantry/2026-05-01/064704.mp4"
    expected_thumb = storage_root / "thumbs/pantry/2026-05-01/064704.jpg"
    assert expected_clip.is_file()
    assert expected_thumb.is_file()
    assert expected_clip.read_bytes() == payload

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            clip = session.query(Clip).one()
            assert clip.source_filename == "06.47.04-06.48.58[M][0@0][0].mp4"
            assert clip.file_path == "clips/pantry/2026-05-01/064704.mp4"
            assert clip.thumb_path == "thumbs/pantry/2026-05-01/064704.jpg"
            assert clip.has_cat is True
            assert clip.max_score == 0.92
            assert clip.detector_version == "yolo11n.pt@deadbeef"
            assert clip.analysis_error is None

            cam = session.query(Camera).filter_by(name="pantry").one()
            assert cam.poll_status == PollStatus.OK
            # ``--since`` makes this a scoped query that cannot prove it covered the full default
            # window; ``last_polled_at`` therefore stays unset (the resume cursor only advances on
            # a default-window tick). Observation fields below still update.
            assert cam.last_polled_at is None
            assert cam.last_clip_at == datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)
            assert cam.last_cat_seen_at == datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_with_no_detect_skips_inference_and_marks_clip(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """``--no-detect`` (passed as ``detector=None``): clip is ingested with the skip marker."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC), no_detect=True)
    try:
        run_tick(config=config, args=args, engine=engine, detector=None, now=_NOW)
    finally:
        engine.dispose()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            clip = session.query(Clip).one()
            assert clip.analysis_error == "skipped: --no-detect"
            assert clip.has_cat is False
            assert clip.detector_version == "skipped"
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_list_only_is_strict_dry_run(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """``--list-only`` writes nothing persisted.

    Contract under test:

    - No ``Clip`` rows or clip / thumb files on disk.
    - ``cameras.last_polled_at`` / ``last_clip_at`` / ``poll_status`` are not mutated.
    - No ``agent_starts`` row inserted.
    - No ``heartbeats`` row written, no retention sweep run.

    ``ensure_db_camera`` is the one allowed write: creating the row on first run is benign init that
    ``_resolve_window`` needs to read ``last_polled_at`` from. The flag's reason for existing is
    repeatable testing — any state mutation would invalidate the next run's ``since`` window.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC), list_only=True)
    try:
        run_tick(config=config, args=args, engine=engine, detector=None, now=_NOW)
    finally:
        engine.dispose()

    assert not any((storage_root / "clips").rglob("*.mp4")) if (storage_root / "clips").exists() else True
    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            assert session.query(Clip).count() == 0
            # Camera row created by ensure_db_camera, but its state is untouched.
            cameras = session.query(Camera).all()
            assert len(cameras) == 1
            assert cameras[0].last_polled_at is None, "list-only must not advance last_polled_at"
            assert cameras[0].last_clip_at is None
            assert cameras[0].poll_status == PollStatus.OK
            # No agent_starts row: list-only is not "the poller ran".
            assert session.query(AgentStart).count() == 0
            # No heartbeat: list-only doesn't liveness-signal.
            assert session.query(Heartbeat).count() == 0
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_default_window_advances_last_polled_at(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """A default-window tick (no ``--since`` / ``--until`` / ``--limit``) advances the cursor.

    Complement to ``test_full_tick_ingests_clip_to_canonical_layout`` which uses ``--since``.
    Together they pin the contract: ``last_polled_at`` advances only when the run can prove it
    covered the full ``[last_polled_at, now]`` window.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())
    detector = _make_detector(has_cat=False)

    # PollerArgs() with no since/until/limit — runs against the default retention-derived window.
    args = PollerArgs()
    try:
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
    finally:
        engine.dispose()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            cam = session.query(Camera).filter_by(name="pantry").one()
            assert cam.last_polled_at == _NOW  # cursor advanced — default-window tick
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_with_limit_does_not_advance_cursor(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """``--limit`` truncates the result set, so the cursor must stay where it was.

    Without this guard, a manual ``--limit 1`` run that ingests the first clip and stops would
    advance ``last_polled_at`` past the unprocessed clips, silently dropping them on the next
    default tick. Observation fields (``last_clip_at``, ``poll_status``) still update.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())

    args = PollerArgs(limit=1, no_detect=True)
    try:
        run_tick(config=config, args=args, engine=engine, detector=None, now=_NOW)
    finally:
        engine.dispose()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            cam = session.query(Camera).filter_by(name="pantry").one()
            assert cam.last_polled_at is None  # NOT advanced — scoped (--limit) tick
            assert cam.last_clip_at == datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)
            assert cam.poll_status == PollStatus.OK  # observation: camera reachable
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_idempotent_on_second_invocation(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Running the same tick twice ingests the recording exactly once (UNIQUE constraint guard)."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())
    detector = _make_detector(has_cat=False)

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC))
    try:
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
    finally:
        engine.dispose()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            assert session.query(Clip).count() == 1
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_camera_unreachable_records_status(
    storage_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """A connection failure surfaces as ``PollStatus.UNREACHABLE`` on the Camera row."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    # Skip the 10-second inter-attempt sleep so the test runs in milliseconds rather than 30s.
    def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("cat_watcher.amcrest_client.time.sleep", no_sleep)
    # Every factory.create attempt raises ConnectError; AmcrestClient retries 3x then surfaces
    # CameraUnreachableError.
    _ = respx.get(_FIND_URL, params={"action": "factory.create"}).mock(side_effect=httpx.ConnectError("dead"))

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC))
    detector = _make_detector(has_cat=False)
    try:
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
    finally:
        engine.dispose()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            cam = session.query(Camera).filter_by(name="pantry").one()
            assert cam.poll_status == PollStatus.UNREACHABLE
            assert cam.poll_status_since == _NOW
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_isolates_per_camera_failures(  # pylint: disable=too-many-locals  # multi-camera setup needs them
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., Config],
) -> None:
    """Two cameras: A is unreachable, B succeeds. As state -> UNREACHABLE, Bs clip lands."""
    internal_root, storage_root = storage_dirs
    cameras = [
        CameraConfig(name="cam_a", display_name="A", host="cam-a.example.com", port=80, timezone="UTC"),
        CameraConfig(name="cam_b", display_name="B", host="cam.example.com", port=80, timezone="UTC"),
    ]
    config = make_config(internal_root, storage_root, cameras=cameras)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("cat_watcher.amcrest_client.time.sleep", no_sleep)
    # Camera A: every factory.create raises ConnectError -> CameraUnreachableError after retries.
    _ = respx.post("http://cam-a.example.com:80/cgi-bin/mediaFileFind.cgi", params={"action": "factory.create"}).mock(
        side_effect=httpx.ConnectError("dead"),
    )
    _ = respx.get("http://cam-a.example.com:80/cgi-bin/mediaFileFind.cgi", params={"action": "factory.create"}).mock(
        side_effect=httpx.ConnectError("dead"),
    )
    # Camera B: normal happy path against the standard cam.example.com mocks.
    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())
    detector = _make_detector(has_cat=True)

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC))
    try:
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
    finally:
        engine.dispose()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            cam_a = session.query(Camera).filter_by(name="cam_a").one()
            cam_b = session.query(Camera).filter_by(name="cam_b").one()
            assert cam_a.poll_status == PollStatus.UNREACHABLE
            assert cam_b.poll_status == PollStatus.OK
            # Cam B`s clip is the only one ingested.
            clips = session.query(Clip).all()
            assert len(clips) == 1
            assert clips[0].camera_id == cam_b.id
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_records_analysis_error_when_detector_fails(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """A DetectorError during detect() is captured into clips.analysis_error; the clip still inserts."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())
    detector = MagicMock(spec=Detector)
    detector.version = "yolo11n.pt@deadbeef"
    detector.detect.side_effect = DetectorError("ffprobe died")

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC))
    try:
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
    finally:
        engine.dispose()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            clip = session.query(Clip).one()
            assert clip.has_cat is False
            assert clip.analysis_error is not None
            assert "ffprobe died" in clip.analysis_error
            assert clip.detector_version == "yolo11n.pt@deadbeef"
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_respects_limit_cap(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """``--limit 1`` ingests at most one recording even when the camera reports several."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    payload = synthetic_clip_path.read_bytes()
    second_path = "/mnt/sd/2026-05-01/001/dav/07/07.10.00-07.11.00[M][0@0][0].mp4"
    page = (
        "found=2\r\n"
        f"items[0].FilePath={_DOWNLOAD_PATH}\r\n"
        "items[0].StartTime=2026-05-01 06:47:04\r\n"
        "items[0].EndTime=2026-05-01 06:48:58\r\n"
        f"items[0].Length={len(payload)}\r\n"
        f"items[1].FilePath={second_path}\r\n"
        "items[1].StartTime=2026-05-01 07:10:00\r\n"
        "items[1].EndTime=2026-05-01 07:11:00\r\n"
        f"items[1].Length={len(payload)}\r\n"
    )
    _ = respx.get(_FIND_URL, params={"action": "factory.create"}).mock(return_value=httpx.Response(200, text=f"result={_FIND_HANDLE}\r\n"))
    _ = respx.get(_FIND_URL, params__contains={"action": "findFile", "object": _FIND_HANDLE}).mock(
        return_value=httpx.Response(200, text="OK\r\n"),
    )
    _ = respx.get(_FIND_URL, params={"action": "findNextFile", "object": _FIND_HANDLE, "count": "100"}).mock(
        return_value=httpx.Response(200, text=page),
    )
    _ = respx.get(_FIND_URL, params={"action": "close", "object": _FIND_HANDLE}).mock(return_value=httpx.Response(200, text="OK\r\n"))
    _ = respx.get(_FIND_URL, params={"action": "destroy", "object": _FIND_HANDLE}).mock(return_value=httpx.Response(200, text="OK\r\n"))
    # Only the first recording's URL is mocked because --limit=1 stops after the first download.
    _ = respx.get(_DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=payload))

    detector = _make_detector(has_cat=False)
    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC), limit=1)
    try:
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
    finally:
        engine.dispose()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            clips = session.query(Clip).all()
            assert len(clips) == 1
            assert clips[0].source_filename == "06.47.04-06.48.58[M][0@0][0].mp4"
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_file_before_row_ordering_leaves_only_orphan_file_on_db_failure(  # pylint: disable=too-many-locals  # crash-simulation test needs the setup
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Callable[..., Config],
) -> None:
    """Simulate a crash AFTER files land but BEFORE the Clip row commits.

    Per spec §4.4 step 4 (strict file-before-row ordering): a DB failure during the per-clip insert
    must leave the .mp4 + .jpg on disk (so retention pass 2 can sweep them later) and NOT leave a
    Clip row pointing at a partial file.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())
    detector = _make_detector(has_cat=False)

    real_add = Session.add

    def failing_add(self: Session, instance: object, _warn: bool = True) -> None:  # noqa: FBT001, FBT002
        if isinstance(instance, Clip):
            msg = "simulated DB failure during clip insert"
            raise RuntimeError(msg)  # noqa: TRY004  # not a type check; targeted simulated crash to exercise file-before-row ordering
        real_add(self, instance, _warn=_warn)

    monkeypatch.setattr(Session, "add", failing_add)

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC))
    try:
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
    finally:
        engine.dispose()

    # The download + thumbnail landed on disk before the (failing) DB insert.
    expected_clip = storage_root / "clips/pantry/2026-05-01/064704.mp4"
    expected_thumb = storage_root / "thumbs/pantry/2026-05-01/064704.jpg"
    assert expected_clip.is_file(), "the clip file should have been written before the DB insert"
    assert expected_thumb.is_file(), "the thumbnail should have been written before the DB insert"

    # No Clip row — the orphan files will be cleaned up by retention pass 2.
    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            assert session.query(Clip).count() == 0
            cam = session.query(Camera).filter_by(name="pantry").one()
            # The unexpected exception flips the camera into ERROR per ``run_tick``'s catch-all.
            assert cam.poll_status == PollStatus.ERROR
    finally:
        engine.dispose()


@respx.mock
def test_full_tick_writes_heartbeat_and_agent_starts_row(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """A successful tick records its start (``agent_starts``) and end (``heartbeats``) liveness rows.

    These rows feed the alerts agent's ``POLLER_STUCK`` watchdog (heartbeat staleness) and the
    ``WEB_FLAPPING``-style retention/flap detection on ``agent_starts``. A regression that dropped
    either insertion would silently break those checks; ``test_full_tick_list_only_is_strict_dry_run``
    pins the absent-on-list-only contract — this is the present-on-real-tick complement.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC), no_detect=True)
    try:
        run_tick(config=config, args=args, engine=engine, detector=None, now=_NOW)
    finally:
        engine.dispose()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            starts = session.query(AgentStart).filter(AgentStart.agent_name == "poller").all()
            heartbeat = session.get(Heartbeat, "poller")
        assert len(starts) == 1
        assert starts[0].started_at == _NOW
        assert heartbeat is not None
        assert heartbeat.last_seen_at == _NOW
    finally:
        engine.dispose()


def _seed_aged_camera_and_clip(engine: Engine, storage_root: Path) -> tuple[Path, Path]:
    """Seed a Camera + a Clip + matching files dated 60 days ago (outside the default 30-day window).

    Returns ``(aged_clip_path, aged_thumb_path)`` for post-sweep existence assertions.
    """
    aged_ts = _NOW - timedelta(days=60)
    clip_rel = "clips/pantry/2026-03-02/100000.mp4"
    thumb_rel = "thumbs/pantry/2026-03-02/100000.jpg"
    aged_clip = storage_root / clip_rel
    aged_thumb = storage_root / thumb_rel
    aged_clip.parent.mkdir(parents=True, exist_ok=True)
    aged_thumb.parent.mkdir(parents=True, exist_ok=True)
    _ = aged_clip.write_bytes(b"old clip bytes")
    _ = aged_thumb.write_bytes(b"old thumb bytes")
    with get_session(engine) as session:
        cam = Camera(name="pantry", display_name="Pantry", host="cam.example.com")
        session.add(cam)
        session.flush()
        session.add(
            Clip(
                camera_id=cam.id,
                source_filename="aged.mp4",
                start_ts=aged_ts,
                end_ts=aged_ts + timedelta(seconds=10),
                duration_seconds=10.0,
                file_path=clip_rel,
                thumb_path=thumb_rel,
                file_size_bytes=14,
                ingested_at=aged_ts,
                detector_version="legacy",
            ),
        )
    return aged_clip, aged_thumb


@respx.mock
def test_full_tick_invokes_retention_sweep_to_prune_aged_clips(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Spec §4.4 step 7: a normal tick fires :func:`cat_watcher.retention.sweep` at end of tick.

    Seeds a Clip row + matching files dated outside ``[retention].clip_days`` (default 30); after
    the tick, the row is gone and the files unlinked. Pins the ``run_tick`` → ``retention.sweep``
    call site at the integration level — the unit ``test_retention.py`` proves the sweep itself
    works, but only this test catches a regression that drops the call entirely.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)
    aged_clip_path, aged_thumb_path = _seed_aged_camera_and_clip(engine, storage_root)
    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())

    args = PollerArgs(no_detect=True)  # default-window tick
    try:
        run_tick(config=config, args=args, engine=engine, detector=None, now=_NOW)
    finally:
        engine.dispose()

    assert not aged_clip_path.exists(), "retention sweep should have unlinked the aged clip file"
    assert not aged_thumb_path.exists(), "retention sweep should have unlinked the aged thumbnail"
    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            aged_rows = session.query(Clip).filter(Clip.source_filename == "aged.mp4").all()
        assert aged_rows == []
    finally:
        engine.dispose()


@respx.mock
def test_poller_writes_per_frame_thumbs_and_clip_frames(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Success path: one JPEG per scored frame, one ``ClipFrame`` row per scored frame.

    ``Clip.thumb_path`` points at the highest-scoring frame's relpath (ordinal 1, score 0.85).
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())
    scored = _stub_scored_frames((0.1, 0.85, 0.3, 0.6))
    detector = _make_detector(has_cat=True, scored_frames=scored)

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC))
    try:
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
    finally:
        engine.dispose()

    per_clip_dir = storage_root / "thumbs/pantry/2026-05-01/064704"
    for ordinal in range(4):
        assert (per_clip_dir / f"{ordinal:02d}.jpg").is_file()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            clip = session.query(Clip).one()
            assert clip.thumb_path.endswith("/01.jpg")
            assert clip.thumb_path == "thumbs/pantry/2026-05-01/064704/01.jpg"
            assert len(clip.frames) == 4
            assert clip.frames[0].score == 0.1
            assert clip.frames[1].score == 0.85
            assert clip.frames[2].score == 0.3
            assert clip.frames[3].score == 0.6
            assert clip.frames[1].thumb_path == clip.thumb_path
            assert clip.frames[0].ordinal == 0
            assert clip.frames[0].t_offset_seconds == 1.0
    finally:
        engine.dispose()


@respx.mock
def test_poller_no_detect_falls_back_to_single_thumb(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """``--no-detect`` path: legacy single-frame thumb at ``<HHMMSS>.jpg`` and no ``ClipFrame`` rows."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC), no_detect=True)
    try:
        run_tick(config=config, args=args, engine=engine, detector=None, now=_NOW)
    finally:
        engine.dispose()

    legacy_thumb = storage_root / "thumbs/pantry/2026-05-01/064704.jpg"
    assert legacy_thumb.is_file()
    assert not (storage_root / "thumbs/pantry/2026-05-01/064704").is_dir()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            clip = session.query(Clip).one()
            assert clip.thumb_path == "thumbs/pantry/2026-05-01/064704.jpg"
            assert len(clip.frames) == 0
    finally:
        engine.dispose()


@respx.mock
def test_poller_detection_error_falls_back_to_single_thumb(
    storage_dirs: tuple[Path, Path],
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """A ``DetectorError`` during detection routes the clip through the legacy fallback thumb path."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    Base.metadata.create_all(engine)

    _seed_amcrest_mocks(synthetic_clip_path.read_bytes())
    detector = MagicMock(spec=Detector)
    detector.version = "yolo11n.pt@deadbeef"
    detector.detect.side_effect = DetectorError("test failure")

    args = PollerArgs(since=datetime(2026, 4, 30, tzinfo=UTC))
    try:
        run_tick(config=config, args=args, engine=engine, detector=detector, now=_NOW)
    finally:
        engine.dispose()

    legacy_thumb = storage_root / "thumbs/pantry/2026-05-01/064704.jpg"
    assert legacy_thumb.is_file()
    assert not (storage_root / "thumbs/pantry/2026-05-01/064704").is_dir()

    engine = create_engine(f"sqlite:///{internal_root / 'test.sqlite'}")
    try:
        with get_session(engine) as session:
            clip = session.query(Clip).one()
            assert clip.thumb_path == "thumbs/pantry/2026-05-01/064704.jpg"
            assert len(clip.frames) == 0
            assert clip.analysis_error is not None
            assert clip.analysis_error.startswith("detect failed")
    finally:
        engine.dispose()

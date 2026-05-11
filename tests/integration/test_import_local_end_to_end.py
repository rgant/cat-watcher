"""End-to-end tests for cat_watcher.import_local.

Wires the real ``import_local`` against a real DB (file-backed SQLite) + a synthetic SD-card tree
under ``tmp_path`` + a mocked detector. Verifies the file-before-row invariant: the ``clips`` row
only commits after the .mp4 and .jpg files exist on disk.
"""

# pylint: disable=duplicate-code  # boilerplate engine-open / session blocks repeat across tests
from collections.abc import Callable  # noqa: TC003  # runtime: pytest evaluates fixture annotations during collection
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003  # runtime: pytest evaluates fixture annotations during collection
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy.orm import Session

from cat_watcher.config import Config  # noqa: TC001  # runtime: make_config callable annotation
from cat_watcher.db import Base, Clip, create_engine, get_session
from cat_watcher.detector import DetectionResult, Detector, ScoredFrame
from cat_watcher.import_local import ImportReport, import_local
from cat_watcher.poller import PollerLockedError, pid_lock

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


_NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)


def _setup_dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    source_root = tmp_path / "sd"
    internal_root.mkdir()
    storage_root.mkdir()
    source_root.mkdir()
    return internal_root, storage_root, source_root


def _make_detector(*, scored_frames: tuple[ScoredFrame, ...] = ()) -> MagicMock:
    """Build a Detector mock; pass ``scored_frames`` to populate ``DetectionResult.scored_frames``.

    Tests that exercise the success-path per-frame thumbnail pipeline need frames to encode.
    """
    detector = MagicMock(spec=Detector)
    detector.version = "yolo11n.pt@deadbeef"
    detector.detect.return_value = DetectionResult(
        has_cat=True,
        max_score=0.92,
        frames_sampled=5,
        frames_with_cat=5,
        best_box_xyxy=(10.0, 20.0, 30.0, 40.0),
        detector_version="yolo11n.pt@deadbeef",
        scored_frames=scored_frames,
    )
    return detector


def _stub_scored_frames(scores: tuple[float, ...]) -> tuple[ScoredFrame, ...]:
    stub_frame = np.zeros((180, 320, 3), dtype=np.uint8)
    return tuple(ScoredFrame(ordinal=i, t_offset_seconds=float(i + 1), score=score, frame=stub_frame) for i, score in enumerate(scores))


def _build_sd_tree(  # noqa: PLR0913  # pylint: disable=too-many-locals  # synthesizes one Amcrest SD path; each axis is a real test variation
    source_root: Path,
    *,
    clip_payload: bytes,
    date_str: str = "2026-04-27",
    hour: int = 5,
    minute: int = 50,
    start_sec: int = 17,
    end_sec: int = 20,
    end_minute: int | None = None,
    thumb_seconds: tuple[int, ...] | None = (24, 25, 26),
) -> tuple[Path, Path | None]:
    """Build a synthetic Amcrest SD-card tree under ``source_root`` for one clip.

    ``primary_thumb_path`` is ``None`` when ``thumb_seconds`` is ``None`` or empty (used to test
    the ffmpeg fallback).
    """
    end_min = end_minute if end_minute is not None else minute + 1
    fname = f"{hour:02d}.{minute:02d}.{start_sec:02d}-{hour:02d}.{end_min:02d}.{end_sec:02d}[M][0@0][0].mp4"
    base = source_root / date_str / "001"
    clip_path = base / "dav" / f"{hour:02d}" / fname
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    _ = clip_path.write_bytes(clip_payload)

    primary_thumb: Path | None = None
    if thumb_seconds:
        thumb_dir = base / "jpg" / f"{hour:02d}" / f"{minute:02d}"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        for sec in thumb_seconds:
            tpath = thumb_dir / f"{sec:02d}[M][0@0][0].jpg"
            _ = tpath.write_bytes(b"\xff\xd8\xff\xe0sd-thumb-payload")
            if primary_thumb is None:
                primary_thumb = tpath
    return clip_path, primary_thumb


def _materialize_engine(internal_root: Path) -> None:
    db_path = internal_root / "cat_watcher.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    engine.dispose()


def _open_engine(internal_root: Path) -> Engine:
    return create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")


def test_import_local_ingests_clip_to_canonical_layout(  # pylint: disable=too-many-locals  # integration test asserts on many fields of the resulting Clip row
    tmp_path: Path,
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """One end-to-end run: synthesize SD tree, import, verify canonical files + Clip row."""
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    config = make_config(internal_root, storage_root)
    _materialize_engine(internal_root)

    payload = synthetic_clip_path.read_bytes()
    _, sd_thumb = _build_sd_tree(source_root, clip_payload=payload)
    assert sd_thumb is not None
    sd_thumb_payload = sd_thumb.read_bytes()

    engine = _open_engine(internal_root)
    try:
        report = import_local(
            engine=engine,
            config=config,
            camera_name="pantry",
            source_dir=source_root,
            detector=_make_detector(),
            limit=None,
            now=_NOW,
        )
    finally:
        engine.dispose()

    assert report == ImportReport(inspected=1, ingested=1, duplicates=0, skipped=0, errors=0)

    expected_clip = storage_root / "clips/pantry/2026-04-27/055017.mp4"
    expected_thumb = storage_root / "thumbs/pantry/2026-04-27/055017.jpg"
    assert expected_clip.is_file()
    assert expected_thumb.is_file()
    assert expected_clip.read_bytes() == payload
    # SD-card thumb was preferred over ffmpeg extraction.
    assert expected_thumb.read_bytes() == sd_thumb_payload

    engine = _open_engine(internal_root)
    try:
        with get_session(engine) as session:
            clip = session.query(Clip).one()
            assert clip.source_filename == "05.50.17-05.51.20[M][0@0][0].mp4"
            assert clip.file_path == "clips/pantry/2026-04-27/055017.mp4"
            assert clip.thumb_path == "thumbs/pantry/2026-04-27/055017.jpg"
            assert clip.has_cat is True
            assert clip.max_score == 0.92
            assert clip.detector_version == "yolo11n.pt@deadbeef"
            assert clip.analysis_error is None
            assert clip.start_ts == datetime(2026, 4, 27, 5, 50, 17, tzinfo=UTC)
            assert clip.end_ts == datetime(2026, 4, 27, 5, 51, 20, tzinfo=UTC)
    finally:
        engine.dispose()


def test_import_local_writes_per_frame_thumbs(
    tmp_path: Path,
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Success path: one JPEG per scored frame, one ``ClipFrame`` row per scored frame.

    ``Clip.thumb_path`` points at the highest-scoring frame's relpath (ordinal 1, score 0.85). The
    SD-card jpg sibling is NOT consulted on the success path — confirming the shared per-frame
    pipeline takes precedence over import_local's SD-card-thumb shortcut.
    """
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    _materialize_engine(internal_root)

    _, sd_thumb = _build_sd_tree(source_root, clip_payload=synthetic_clip_path.read_bytes())
    assert sd_thumb is not None
    sd_thumb_payload = sd_thumb.read_bytes()

    detector = _make_detector(scored_frames=_stub_scored_frames((0.1, 0.85, 0.3, 0.6)))

    engine = _open_engine(internal_root)
    try:
        report = import_local(
            engine=engine,
            config=make_config(internal_root, storage_root),
            camera_name="pantry",
            source_dir=source_root,
            detector=detector,
            limit=None,
            now=_NOW,
        )
    finally:
        engine.dispose()

    assert report == ImportReport(inspected=1, ingested=1, duplicates=0, skipped=0, errors=0)

    per_clip_dir = storage_root / "thumbs/pantry/2026-04-27/055017"
    for ordinal in range(4):
        assert (per_clip_dir / f"{ordinal:02d}.jpg").is_file()

    engine = _open_engine(internal_root)
    try:
        with get_session(engine) as session:
            clip = session.query(Clip).one()
            assert clip.thumb_path == "thumbs/pantry/2026-04-27/055017/01.jpg"
            assert clip.thumb_path.endswith("/01.jpg")
            assert len(clip.frames) == 4
            assert clip.frames[0].ordinal == 0
            assert clip.frames[0].score == 0.1
            assert clip.frames[1].score == 0.85
            assert clip.frames[2].score == 0.3
            assert clip.frames[3].score == 0.6
            assert clip.frames[0].t_offset_seconds == 1.0
            assert clip.frames[1].thumb_path == clip.thumb_path
            # The SD-card jpg shortcut is fallback-only; on the success path the per-frame writer
            # encodes from in-memory ndarrays, so the primary thumb's bytes must NOT match the
            # SD-card jpg's bytes.
            assert (storage_root / clip.thumb_path).read_bytes() != sd_thumb_payload
    finally:
        engine.dispose()


def test_import_local_no_detect_skips_inference_and_marks_clip(
    tmp_path: Path,
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """``--no-detect`` (passed as ``detector=None``): clip is ingested with the skip marker."""
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    config = make_config(internal_root, storage_root)
    _materialize_engine(internal_root)
    _ = _build_sd_tree(source_root, clip_payload=synthetic_clip_path.read_bytes())

    engine = _open_engine(internal_root)
    try:
        report = import_local(
            engine=engine,
            config=config,
            camera_name="pantry",
            source_dir=source_root,
            detector=None,
            limit=None,
            now=_NOW,
        )
    finally:
        engine.dispose()

    assert report.ingested == 1

    engine = _open_engine(internal_root)
    try:
        with get_session(engine) as session:
            clip = session.query(Clip).one()
            assert clip.analysis_error == "skipped: --no-detect"
            assert clip.has_cat is False
            assert clip.detector_version == "skipped"
    finally:
        engine.dispose()


def test_import_local_is_idempotent_via_unique_constraint(
    tmp_path: Path,
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Re-running on the same source: every clip reports as duplicate; no new rows or files."""
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    config = make_config(internal_root, storage_root)
    _materialize_engine(internal_root)
    _ = _build_sd_tree(source_root, clip_payload=synthetic_clip_path.read_bytes())

    for run_index in range(2):
        engine = _open_engine(internal_root)
        try:
            report = import_local(
                engine=engine,
                config=config,
                camera_name="pantry",
                source_dir=source_root,
                detector=None,
                limit=None,
                now=_NOW,
            )
        finally:
            engine.dispose()
        if run_index == 0:
            assert report.ingested == 1
            assert report.duplicates == 0
        else:
            assert report.ingested == 0
            assert report.duplicates == 1

    engine = _open_engine(internal_root)
    try:
        with get_session(engine) as session:
            assert session.query(Clip).count() == 1
    finally:
        engine.dispose()


def test_import_local_falls_back_to_ffmpeg_when_no_sd_thumb(
    tmp_path: Path,
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Missing parallel jpgs are not an error; the helper extracts a thumbnail via ffmpeg."""
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    config = make_config(internal_root, storage_root)
    _materialize_engine(internal_root)
    _, sd_thumb = _build_sd_tree(
        source_root,
        clip_payload=synthetic_clip_path.read_bytes(),
        thumb_seconds=None,  # no SD-card thumbnails
    )
    assert sd_thumb is None

    engine = _open_engine(internal_root)
    try:
        report = import_local(
            engine=engine,
            config=config,
            camera_name="pantry",
            source_dir=source_root,
            detector=None,
            limit=None,
            now=_NOW,
        )
    finally:
        engine.dispose()

    assert report.ingested == 1
    extracted = storage_root / "thumbs/pantry/2026-04-27/055017.jpg"
    assert extracted.is_file()
    # ffmpeg-produced JPEG starts with the JFIF magic bytes.
    assert extracted.read_bytes()[:3] == b"\xff\xd8\xff"


def test_import_local_skips_orphans_and_non_amcrest_files(
    tmp_path: Path,
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Files at the source root or with non-Amcrest names are counted as skipped, not errored."""
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    config = make_config(internal_root, storage_root)
    _materialize_engine(internal_root)

    # one valid clip:
    _ = _build_sd_tree(source_root, clip_payload=synthetic_clip_path.read_bytes())
    # an orphan at source root (no date-dir ancestor):
    _ = (source_root / "06.47.04-06.48.58[M][0@0][0].mp4").write_bytes(b"")
    # a non-matching .mp4 inside the date tree:
    _ = (source_root / "2026-04-27" / "001" / "dav" / "05" / "stray.mp4").write_text("nope")

    engine = _open_engine(internal_root)
    try:
        report = import_local(
            engine=engine,
            config=config,
            camera_name="pantry",
            source_dir=source_root,
            detector=None,
            limit=None,
            now=_NOW,
        )
    finally:
        engine.dispose()

    assert report.ingested == 1
    assert report.skipped == 2  # orphan + stray.mp4


def test_import_local_limit_caps_processed_clips(
    tmp_path: Path,
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """``--limit N`` halts the loop after N matched clips have been considered."""
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    config = make_config(internal_root, storage_root)
    _materialize_engine(internal_root)

    payload = synthetic_clip_path.read_bytes()
    # three clips, all on the same day:
    _ = _build_sd_tree(source_root, clip_payload=payload, hour=5, minute=50, start_sec=17, end_sec=20)
    _ = _build_sd_tree(source_root, clip_payload=payload, hour=6, minute=10, start_sec=0, end_sec=15)
    _ = _build_sd_tree(source_root, clip_payload=payload, hour=7, minute=20, start_sec=30, end_sec=45)

    engine = _open_engine(internal_root)
    try:
        report = import_local(
            engine=engine,
            config=config,
            camera_name="pantry",
            source_dir=source_root,
            detector=None,
            limit=2,
            now=_NOW,
        )
    finally:
        engine.dispose()

    assert report.inspected == 2
    assert report.ingested == 2

    engine = _open_engine(internal_root)
    try:
        with get_session(engine) as session:
            assert session.query(Clip).count() == 2
    finally:
        engine.dispose()


def test_import_local_refuses_when_poller_lock_is_held(
    tmp_path: Path,
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Concurrent poller tick raises PollerLockedError; CLI translates to non-zero exit."""
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    config = make_config(internal_root, storage_root)
    _materialize_engine(internal_root)
    _ = _build_sd_tree(source_root, clip_payload=synthetic_clip_path.read_bytes())

    engine = _open_engine(internal_root)
    try:
        with pid_lock(internal_root), pytest.raises(PollerLockedError):  # holds the PID file open
            _ = import_local(
                engine=engine,
                config=config,
                camera_name="pantry",
                source_dir=source_root,
                detector=None,
                limit=None,
                now=_NOW,
            )
    finally:
        engine.dispose()


def test_import_local_unknown_camera_raises_value_error(tmp_path: Path, make_config: Callable[[Path, Path], Config]) -> None:
    """A camera name not in the configured list is fail-fast (operator typo or stale config)."""
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    config = make_config(internal_root, storage_root)
    _materialize_engine(internal_root)

    engine = _open_engine(internal_root)
    try:
        with pytest.raises(ValueError, match="not in the configured camera list"):
            _ = import_local(
                engine=engine,
                config=config,
                camera_name="nonsuch",
                source_dir=source_root,
                detector=None,
                limit=None,
                now=_NOW,
            )
    finally:
        engine.dispose()


def test_import_local_continues_after_per_clip_error(
    tmp_path: Path,
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure on one clip is logged + counted; the next clip still ingests."""
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    config = make_config(internal_root, storage_root)
    _materialize_engine(internal_root)

    payload = synthetic_clip_path.read_bytes()
    bad_clip, _ = _build_sd_tree(source_root, clip_payload=payload, hour=5, minute=50, start_sec=17, end_sec=20)
    _ = _build_sd_tree(source_root, clip_payload=payload, hour=6, minute=10, start_sec=0, end_sec=15)

    from cat_watcher import import_local as importer_module

    real_atomic_copy = importer_module._atomic_copy_with_fsync

    def fail_on_first_clip(source: Path, dest: Path) -> None:
        if source == bad_clip:
            msg = "simulated copy failure"
            raise OSError(msg)
        real_atomic_copy(source, dest)

    monkeypatch.setattr("cat_watcher.import_local._atomic_copy_with_fsync", fail_on_first_clip)

    engine = _open_engine(internal_root)
    try:
        report = import_local(
            engine=engine,
            config=config,
            camera_name="pantry",
            source_dir=source_root,
            detector=None,
            limit=None,
            now=_NOW,
        )
    finally:
        engine.dispose()

    assert report.errors == 1
    assert report.ingested == 1


def _patch_session_add_to_fail_on_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkey-patch :meth:`Session.add` to raise ``OSError`` on any ``Clip`` instance.

    ``OSError`` is in the per-clip ``except`` clause in
    :func:`cat_watcher.import_local._ingest_loop`, so the loop counts it as an error and continues;
    non-Clip instances pass through unchanged.
    """
    real_add = Session.add

    def failing_add(self: Session, instance: object, _warn: bool = True) -> None:  # noqa: FBT001, FBT002
        if isinstance(instance, Clip):
            msg = "simulated DB failure during clip insert"
            raise OSError(msg)
        real_add(self, instance, _warn=_warn)

    monkeypatch.setattr(Session, "add", failing_add)


def test_import_local_file_before_row_ordering_leaves_only_orphan_file_on_db_failure(
    tmp_path: Path,
    synthetic_clip_path: Path,
    make_config: Callable[[Path, Path], Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec §4.4 file-before-row invariant, on the import path.

    Mirror of the poller's ``test_full_tick_file_before_row_ordering_leaves_only_orphan_file_on_db_failure``.
    Both ingestion paths share :func:`cat_watcher.poller.materialize_and_persist_clip`, so the
    contract — files fsynced before the ``Clip`` row commits, leaving only orphan files on a DB
    failure — must hold for both. A regression that reordered the steps only on the import path
    would slip past the poller's test.
    """
    internal_root, storage_root, source_root = _setup_dirs(tmp_path)
    config = make_config(internal_root, storage_root)
    _materialize_engine(internal_root)
    _ = _build_sd_tree(source_root, clip_payload=synthetic_clip_path.read_bytes())
    _patch_session_add_to_fail_on_clip(monkeypatch)

    engine = _open_engine(internal_root)
    try:
        report = import_local(
            engine=engine,
            config=config,
            camera_name="pantry",
            source_dir=source_root,
            detector=None,
            limit=None,
            now=_NOW,
        )
    finally:
        engine.dispose()

    assert report.errors == 1
    assert report.ingested == 0

    # Files landed on disk before the failing DB insert — retention pass 2 will reap them later.
    expected_clip = storage_root / "clips/pantry/2026-04-27/055017.mp4"
    expected_thumb = storage_root / "thumbs/pantry/2026-04-27/055017.jpg"
    assert expected_clip.is_file(), "the clip file should have been written before the DB insert"
    assert expected_thumb.is_file(), "the thumbnail should have been written before the DB insert"

    engine = _open_engine(internal_root)
    try:
        with get_session(engine) as session:
            assert session.query(Clip).count() == 0
    finally:
        engine.dispose()

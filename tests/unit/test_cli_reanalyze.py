"""Tests for ``cat-watcher reanalyze``.

Most tests call ``_reanalyze`` with a fake ``detect`` function, so YOLO never runs. They cover the
row updates, the filter rules, the ``--limit`` and ``--camera`` flags, and the thumb files. They also
make sure that ``clip_frame_subjects`` rows and ``clips.reviewed_at`` survive re-detection.
"""

import sqlite3
from collections.abc import Callable  # noqa: TC003  # runtime: pytest evaluates fixture annotations during collection
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import pytest
from cli_test_helpers import (
    config_with_dirs,
    init_schema,
    make_handler_args,
    read_clip,
    seed_camera,
    seed_clip,
    set_clip_reviewed_at,
)
from db_helpers import scored_frames_with_boxes
from sqlalchemy import select

from cat_watcher.__main__ import (
    _EXIT_GENERIC_FAILURE,
    _EXIT_MISSING_DEPENDENCY,
    _build_parser,
    _reanalyze,
    _ReanalyzeRun,
    _run_reanalyze,
)
from cat_watcher.config import Config, ConfiguredSubject  # runtime: make_config callable annotation
from cat_watcher.db import Clip, ClipFrame, ClipFrameSubject, Subject, create_engine, get_session
from cat_watcher.detector import Detector, ScoredFrame
from cat_watcher.poller import DetectionFields
from cat_watcher.predictions import FrameCounts, LiveClassifier

if TYPE_CHECKING:
    from pathlib import Path

    from cat_watcher.classifier.benchmark import PredictFn

type _DetectResult = tuple[DetectionFields, tuple[ScoredFrame, ...]]


_TRIPWIRES = ("{_fmt", "{self.", "{cam.", "{cfg.", "NoneType")


_DETECTOR_VERSION = "yolov11n@new"


def _detector_mock() -> MagicMock:
    """Autospec'd ``Detector`` instance for ``Detector.from_weights``'s patched return value."""
    return cast("MagicMock", create_autospec(Detector, instance=True))


def _returning(fields: DetectionFields, frames: tuple[ScoredFrame, ...]) -> Callable[[Path], _DetectResult]:
    """Build a ``detect`` fake that gives the same result for each clip."""

    def detect(clip_full: Path) -> _DetectResult:
        _ = clip_full
        return fields, frames

    return detect


def _reanalyze_with(
    config: Config,
    detect: Callable[[Path], _DetectResult],
    *,
    classifier: LiveClassifier | None = None,
    **flags: object,
) -> int:
    """Run ``_reanalyze`` on the test database. ``flags`` replace the parsed argument defaults."""
    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        run = _ReanalyzeRun(engine=engine, config=config, detect=detect, detector_version=_DETECTOR_VERSION, classifier=classifier)
        args = make_handler_args(camera="", limit=None, all=False, predictions=False)
        for name, value in flags.items():
            setattr(args, name, value)
        return _reanalyze(run, args)
    finally:
        engine.dispose()


# --- helpers -------------------------------------------------------------------------------------


def _ensure_clip_file(config: Config, file_path: str = "clips/pantry/test.mp4") -> Path:
    """Materialize a non-empty clip file under ``storage_root`` so ``is_file()`` checks pass."""
    target = config.storage_root / file_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        _ = target.write_bytes(b"placeholder")
    return target


def _detection_fields(
    *,
    has_cat: bool = True,
    version: str = _DETECTOR_VERSION,
    analysis_error: str | None = None,
) -> DetectionFields:
    """Build a ``detection_for`` result that satisfies the reanalyze update path."""
    return DetectionFields(
        has_cat=has_cat,
        max_score=0.92,
        frames_sampled=5,
        frames_with_cat=4,
        best_box_xyxy=[10.0, 20.0, 30.0, 40.0],
        detector_version=version,
        analysis_error=analysis_error,
    )


def _set_weights_present(config: Config) -> None:
    """Drop a non-empty file at ``<internal_root>/models/<detector.model>``.

    The reanalyze handler refuses to start if the weights file is absent. Several tests don't care
    about the actual model — they patch ``Detector.from_weights`` and ``detection_for`` — but the
    existence-check still has to pass first.
    """
    models_dir = config.internal_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    _ = (models_dir / config.detector.model).write_bytes(b"placeholder weights")


def _read_clip_frames(config: Config, clip_id: int) -> list[ClipFrame]:
    """Read ``ClipFrame`` rows for ``clip_id`` ordered by ordinal, detached from any session."""
    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            return list(session.scalars(select(ClipFrame).where(ClipFrame.clip_id == clip_id).order_by(ClipFrame.ordinal)))
    finally:
        engine.dispose()


# --- tests ---------------------------------------------------------------------------------------


def test_reanalyze_refuses_without_weights(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing weights file -> non-zero exit + actionable message; no rows mutated."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    clip_id = seed_clip(config, camera_id=cam_id, analysis_error="skipped: --no-detect")

    with patch("cat_watcher.__main__.load_config", return_value=config):
        exit_code = _run_reanalyze(make_handler_args(camera="", limit=None, all=False))
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "fetch-models" in err
    # Row was not touched.
    clip = read_clip(config, clip_id)
    assert clip.analysis_error == "skipped: --no-detect"


def test_reanalyze_rescores_skip_marker_clips(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A clip with the ``--no-detect`` skip marker is picked up and rewritten with new fields."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=cam_id, analysis_error="skipped: --no-detect")

    exit_code = _reanalyze_with(config, _returning(_detection_fields(has_cat=True), ()))
    assert exit_code == 0
    clip = read_clip(config, clip_id)
    assert clip.has_cat is True
    assert clip.max_score == 0.92
    assert clip.frames_with_cat == 4
    assert clip.detector_version == "yolov11n@new"
    assert clip.analysis_error is None


def test_reanalyze_default_skips_clean_clips(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A clip with ``analysis_error=None`` is left alone unless ``--all`` is set."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=cam_id, analysis_error=None, detector_version="yolov11n@stale")

    _ = _reanalyze_with(config, _returning(_detection_fields(), ()))
    clip = read_clip(config, clip_id)
    assert clip.detector_version == "yolov11n@stale"  # untouched


def test_reanalyze_all_rescores_every_clip(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``--all`` widens the filter to clean clips too; ``detector_version`` advances."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=cam_id, analysis_error=None, detector_version="yolov11n@stale")

    _ = _reanalyze_with(config, _returning(_detection_fields(version="yolov11n@new"), ()), all=True)
    clip = read_clip(config, clip_id)
    assert clip.detector_version == "yolov11n@new"


def test_reanalyze_skips_missing_files_with_warning(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A clip whose file is missing on disk increments ``skipped_missing`` and the loop continues."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    # Two clips: one with a real file, one without. Both have skip markers so both qualify.
    _ = _ensure_clip_file(config, file_path="clips/pantry/present.mp4")
    present_id = seed_clip(
        config,
        camera_id=cam_id,
        analysis_error="skipped: --no-detect",
        file_path="clips/pantry/present.mp4",
        source_filename="present.mp4",
    )
    missing_id = seed_clip(
        config,
        camera_id=cam_id,
        analysis_error="skipped: --no-detect",
        file_path="clips/pantry/missing.mp4",
        source_filename="missing.mp4",
    )

    exit_code = _reanalyze_with(config, _returning(_detection_fields(), ()))
    assert exit_code == 0
    assert read_clip(config, present_id).analysis_error is None
    assert read_clip(config, missing_id).analysis_error == "skipped: --no-detect"
    out = capsys.readouterr().out
    assert "rescored=1 skipped_missing=1 errored=0" in out
    for token in _TRIPWIRES:
        assert token not in out


def test_reanalyze_limit_caps_count_in_start_ts_order(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``--limit N`` processes exactly the N earliest qualifying clips."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    clip_ids = [
        seed_clip(
            config,
            camera_id=cam_id,
            start_ts=base + timedelta(hours=i),
            analysis_error="skipped: --no-detect",
            source_filename=f"clip-{i:02d}.mp4",
        )
        for i in range(3)
    ]

    _ = _reanalyze_with(config, _returning(_detection_fields(), ()), limit=2)
    assert read_clip(config, clip_ids[0]).analysis_error is None
    assert read_clip(config, clip_ids[1]).analysis_error is None
    assert read_clip(config, clip_ids[2]).analysis_error == "skipped: --no-detect"


def test_reanalyze_camera_filter_restricts_scope(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``--camera <name>`` only re-scores clips from that camera."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    pantry_id = seed_camera(config, name="pantry", display_name="Pantry")
    bath_id = seed_camera(config, name="bath", display_name="Bath", host="cam2.example.com")
    _ = _ensure_clip_file(config)
    pantry_clip = seed_clip(config, camera_id=pantry_id, analysis_error="skipped: --no-detect", source_filename="pantry.mp4")
    bath_clip = seed_clip(config, camera_id=bath_id, analysis_error="skipped: --no-detect", source_filename="bath.mp4")

    _ = _reanalyze_with(config, _returning(_detection_fields(), ()), camera="pantry")
    assert read_clip(config, pantry_clip).analysis_error is None
    assert read_clip(config, bath_clip).analysis_error == "skipped: --no-detect"


def test_reanalyze_emits_one_summary_line_per_camera(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two cameras with qualifying clips produce two ``reanalyze [<display_name>]:`` lines."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    pantry_id = seed_camera(config, name="pantry", display_name="Pantry")
    bath_id = seed_camera(config, name="bath", display_name="Bath", host="cam2.example.com")
    _ = _ensure_clip_file(config)
    _ = seed_clip(config, camera_id=pantry_id, analysis_error="skipped: --no-detect", source_filename="pantry.mp4")
    _ = seed_clip(config, camera_id=bath_id, analysis_error="skipped: --no-detect", source_filename="bath.mp4")

    _ = _reanalyze_with(config, _returning(_detection_fields(), ()))
    out = capsys.readouterr().out
    assert "reanalyze [Pantry]:" in out
    assert "reanalyze [Bath]:" in out


def test_reanalyze_returns_one_when_any_clip_errors(
    tmp_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """A detection error during the run flips exit code to 1 and lands in the ``errored`` bucket only.

    Disjoint-counter contract: errored does NOT also bump rescored.
    """
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    _ = seed_clip(config, camera_id=cam_id, analysis_error="skipped: --no-detect")

    exit_code = _reanalyze_with(config, _returning(_detection_fields(analysis_error="detect failed: synthetic"), ()))
    assert exit_code == 1


def test_reanalyze_no_qualifying_clips_prints_message_and_exits_zero(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty DB (or filter that matches nothing) reports the no-op explicitly so the run isn't silent."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = seed_camera(config)

    exit_code = _reanalyze_with(config, _returning(_detection_fields(), ()), all=True)
    assert exit_code == 0
    assert "no qualifying clips" in capsys.readouterr().out


def _build_scored_frames(scores: tuple[float, ...]) -> tuple[ScoredFrame, ...]:
    """Build ``ScoredFrame``s with stub ndarrays for the backfill test."""
    stub_frame = np.zeros((180, 320, 3), dtype=np.uint8)
    return tuple(ScoredFrame(ordinal=i, t_offset_seconds=float(i + 1), score=score, frame=stub_frame) for i, score in enumerate(scores))


def test_reanalyze_all_backfills_clip_frames(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``--all`` over a clip with no per-frame thumbs encodes them, replaces the row, repoints thumb_path.

    Clips ingested without per-frame thumbs get a fresh set of ``clip_frames`` rows + a per-frame
    ``thumb_path``, and the legacy single-file thumb is cleaned up so ``thumbs/`` doesn't accumulate
    orphans.
    """
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = _ensure_clip_file(config)
    legacy_thumb_full = config.storage_root / "thumbs/pantry/legacy.jpg"
    legacy_thumb_full.parent.mkdir(parents=True, exist_ok=True)
    _ = legacy_thumb_full.write_bytes(b"\xff\xd8\xff\xe0placeholder")
    clip_id = seed_clip(
        config,
        camera_id=seed_camera(config),
        start_ts=datetime(2026, 5, 1, 12, 30, 45, tzinfo=UTC),
        analysis_error=None,
        thumb_path="thumbs/pantry/legacy.jpg",
    )
    scored_frames = _build_scored_frames((0.1, 0.85, 0.3, 0.6))

    exit_code = _reanalyze_with(config, _returning(_detection_fields(has_cat=True), scored_frames), all=True)
    assert exit_code == 0

    per_clip_dir = config.storage_root / "thumbs" / "pantry" / "2026-05-01" / "123045"
    for ordinal in range(4):
        assert (per_clip_dir / f"{ordinal:02d}.jpg").is_file()

    clip = read_clip(config, clip_id)
    assert clip.thumb_path.endswith("/01.jpg")
    assert legacy_thumb_full.is_file() is False
    frames = _read_clip_frames(config, clip_id)
    assert len(frames) == 4
    assert [frame.score for frame in frames] == [0.1, 0.85, 0.3, 0.6]


def _scored_frames(*ordinals: int, score: float = 0.7) -> tuple[ScoredFrame, ...]:
    """Build real ``ScoredFrame``s so re-detection reaches the frame-backfill path.

    ``_reanalyze_one_clip`` skips the backfill on an empty tuple, so a test that passes ``()``
    never exercises it.
    """
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    return tuple(ScoredFrame(ordinal=o, t_offset_seconds=float(o) / 2, score=score, frame=frame, box=None) for o in ordinals)


def _seed_tagged_frames(config: Config, clip_id: int, *ordinals: int, tagged: int) -> tuple[int, int]:
    """Attach a frame per ordinal and tag the one at ``tagged``. Return its ``(frame_id, subject_id)``."""
    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            subj = Subject(slug="rufus", display_name="Rufus", kind="cat", display_order=1)
            session.add(subj)
            session.flush()
            frame_ids: dict[int, int] = {}
            for ordinal in ordinals:
                frame = ClipFrame(
                    clip_id=clip_id,
                    ordinal=ordinal,
                    t_offset_seconds=float(ordinal),
                    score=0.1,
                    thumb_path=f"thumbs/legacy/{ordinal}.jpg",
                )
                session.add(frame)
                session.flush()
                frame_ids[ordinal] = frame.id
            session.add(ClipFrameSubject(clip_frame_id=frame_ids[tagged], subject_id=subj.id))
            return frame_ids[tagged], subj.id
    finally:
        engine.dispose()


def _read_frames(config: Config, clip_id: int) -> list[tuple[int, int, float, str]]:
    """Return ``(ordinal, id, score, thumb_path)`` per surviving frame, ordered by ordinal."""
    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            rows = session.execute(
                select(ClipFrame.ordinal, ClipFrame.id, ClipFrame.score, ClipFrame.thumb_path)
                .where(ClipFrame.clip_id == clip_id)
                .order_by(ClipFrame.ordinal),
            ).all()
            return [(r[0], r[1], r[2], r[3]) for r in rows]
    finally:
        engine.dispose()


def _read_tagged_frame_ids(config: Config, subject_id: int) -> list[int]:
    """Return every ``clip_frame_id`` still tagged with ``subject_id``."""
    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            rows = session.execute(
                select(ClipFrameSubject.clip_frame_id).where(ClipFrameSubject.subject_id == subject_id),
            ).all()
            return [r[0] for r in rows]
    finally:
        engine.dispose()


def test_reanalyze_preserves_clip_frame_subjects(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """An operator tag survives re-detection that rewrites its frame's detector columns."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=cam_id, analysis_error="skipped: --no-detect")
    tagged_frame_id, subject_id = _seed_tagged_frames(config, clip_id, 0, 1, tagged=0)

    exit_code = _reanalyze_with(config, _returning(_detection_fields(has_cat=True), _scored_frames(0, 1)))
    assert exit_code == 0

    # The tag must still point at the same row: (clip_id, ordinal) is the frame's identity, so a
    # re-score updates that row rather than replacing it.
    assert _read_tagged_frame_ids(config, subject_id) == [tagged_frame_id]
    frames = _read_frames(config, clip_id)
    assert [f[0] for f in frames] == [0, 1]
    assert frames[0][1] == tagged_frame_id
    assert frames[0][2] == 0.7  # detector columns refreshed in place
    assert frames[0][3].endswith("/00.jpg")


def test_reanalyze_drops_frames_the_new_detection_does_not_produce(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A shorter frame batch removes the surplus ordinals, and their tags go with them."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=cam_id, analysis_error="skipped: --no-detect")
    _, subject_id = _seed_tagged_frames(config, clip_id, 0, 1, 2, tagged=2)

    exit_code = _reanalyze_with(config, _returning(_detection_fields(has_cat=True), _scored_frames(0, 1)))
    assert exit_code == 0

    assert [f[0] for f in _read_frames(config, clip_id)] == [0, 1]
    assert _read_tagged_frame_ids(config, subject_id) == []


def test_reanalyze_preserves_reviewed_at(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``reviewed_at`` on the clip row is untouched by reanalyze."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=cam_id, analysis_error="skipped: --no-detect")
    reviewed_ts = datetime(2026, 5, 20, 9, 0, 0, tzinfo=UTC)

    set_clip_reviewed_at(config, clip_id, reviewed_ts)

    exit_code = _reanalyze_with(config, _returning(_detection_fields(has_cat=True), ()))
    assert exit_code == 0

    assert read_clip(config, clip_id).reviewed_at == reviewed_ts


# --- classification ------------------------------------------------------------------------------

_CLASSIFIER_VERSION = "cat-classifier@cafe"


def _fixed_predict(image_path: Path) -> tuple[str, float]:
    _ = image_path
    return "rufus", 0.91


def _failing_predict(image_path: Path) -> tuple[str, float]:
    msg = f"cannot classify {image_path}"
    raise RuntimeError(msg)


def _seed_rufus(config: Config) -> int:
    """Insert the ``rufus`` cat subject and return its id."""
    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            subj = Subject(slug="rufus", display_name="Rufus", kind="cat", display_order=1)
            session.add(subj)
            session.flush()
            return subj.id
    finally:
        engine.dispose()


def _classifier(subject_id: int, predict: PredictFn = _fixed_predict) -> LiveClassifier:
    return LiveClassifier(
        predict=predict,
        version=_CLASSIFIER_VERSION,
        subject_id_by_slug={"rufus": subject_id},
        totals=FrameCounts(),
    )


def _boxed_frames(*ordinals: int) -> tuple[ScoredFrame, ...]:
    """Build frames large enough to crop, each with a box, so the classifier predicts on every one."""
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    return tuple(
        ScoredFrame(ordinal=o, t_offset_seconds=float(o) / 2, score=0.7, frame=frame, box=(10.0, 20.0, 60.0, 90.0)) for o in ordinals
    )


def _read_predictions(config: Config, clip_id: int) -> list[tuple[int | None, float | None, str | None]]:
    """Return ``(predicted_subject_id, prediction_confidence, classifier_version)`` per frame, by ordinal."""
    return [(f.predicted_subject_id, f.prediction_confidence, f.classifier_version) for f in _read_clip_frames(config, clip_id)]


def test_reanalyze_all_with_classifier_stores_predictions_on_every_frame(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``--all`` with a loaded classifier writes all three prediction columns on each frame."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=seed_camera(config), analysis_error=None)
    subject_id = _seed_rufus(config)

    exit_code = _reanalyze_with(
        config,
        _returning(_detection_fields(has_cat=True), _boxed_frames(0, 1, 2)),
        classifier=_classifier(subject_id),
        all=True,
    )
    assert exit_code == 0
    assert _read_predictions(config, clip_id) == [(subject_id, 0.91, _CLASSIFIER_VERSION)] * 3


def test_reanalyze_updated_frame_keeps_tag_and_takes_new_prediction(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A re-run updates the tagged frame in place, so its tag stays and its prediction is new."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=seed_camera(config), analysis_error="skipped: --no-detect")
    tagged_frame_id, subject_id = _seed_tagged_frames(config, clip_id, 0, 1, tagged=0)

    exit_code = _reanalyze_with(
        config,
        _returning(_detection_fields(has_cat=True), _boxed_frames(0, 1)),
        classifier=_classifier(subject_id),
    )
    assert exit_code == 0
    assert _read_tagged_frame_ids(config, subject_id) == [tagged_frame_id]
    frames = _read_clip_frames(config, clip_id)
    assert frames[0].id == tagged_frame_id
    assert (frames[0].predicted_subject_id, frames[0].prediction_confidence, frames[0].classifier_version) == (
        subject_id,
        0.91,
        _CLASSIFIER_VERSION,
    )


def _can_take_write_lock(db_path: Path) -> bool:
    """Return whether a second connection can start a write transaction now."""
    conn = sqlite3.connect(db_path, timeout=0)
    try:
        _ = conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError:
        return False
    else:
        conn.rollback()
        return True
    finally:
        conn.close()


def test_reanalyze_holds_no_write_lock_during_detection_or_inference(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """Another writer can take the SQLite write lock while each clip runs detection and inference.

    The second clip shows that the writes of the first clip are committed before the next detection.
    """
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    db_path = config.internal_root / "cat_watcher.sqlite"
    cam_id = seed_camera(config)
    for name in ("a", "b"):
        _ = _ensure_clip_file(config, f"clips/pantry/{name}.mp4")
        _ = seed_clip(
            config,
            camera_id=cam_id,
            analysis_error="skipped: --no-detect",
            file_path=f"clips/pantry/{name}.mp4",
            source_filename=f"{name}.mp4",
        )
    subject_id = _seed_rufus(config)
    at_detection: list[bool] = []
    at_inference: list[bool] = []

    def detect(clip_full: Path) -> _DetectResult:
        _ = clip_full
        at_detection.append(_can_take_write_lock(db_path))
        return _detection_fields(has_cat=True), _boxed_frames(0)

    def predict(image_path: Path) -> tuple[str, float]:
        at_inference.append(_can_take_write_lock(db_path))
        return _fixed_predict(image_path)

    exit_code = _reanalyze_with(config, detect, classifier=_classifier(subject_id, predict))
    assert exit_code == 0
    assert at_detection == [True, True]
    assert at_inference == [True, True]


def test_reanalyze_summary_reports_frame_counts(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The per-camera line adds the classified, no-box, and errored frame counts."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = _ensure_clip_file(config)
    _ = seed_clip(config, camera_id=seed_camera(config, name="pantry", display_name="Pantry"), analysis_error="skipped: --no-detect")
    subject_id = _seed_rufus(config)
    scored = scored_frames_with_boxes(np.zeros((180, 320, 3), dtype=np.uint8))

    exit_code = _reanalyze_with(config, _returning(_detection_fields(has_cat=True), scored), classifier=_classifier(subject_id))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "reanalyze [Pantry]: rescored=1 skipped_missing=0 errored=0 frames_classified=2 frames_no_box=1 frames_errored=0" in out
    for token in _TRIPWIRES:
        assert token not in out


# --- --predictions -------------------------------------------------------------------------------


def _seed_frames_with_version(config: Config, clip_id: int, *, version: str | None, ordinals: tuple[int, ...] = (0, 1)) -> None:
    """Attach one frame per ordinal, all stamped with ``version``."""
    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            session.add_all(
                ClipFrame(
                    classifier_version=version,
                    thumb_path=f"thumbs/legacy/{clip_id}-{ordinal}.jpg",
                    clip_id=clip_id,
                    ordinal=ordinal,
                    score=0.5,
                    t_offset_seconds=0.0,
                )
                for ordinal in ordinals
            )
    finally:
        engine.dispose()


def _run_predictions(config: Config, classifier: LiveClassifier | None, frames: tuple[ScoredFrame, ...] = ()) -> int:
    return _reanalyze_with(config, _returning(_detection_fields(has_cat=True), frames), classifier=classifier, predictions=True)


def test_reanalyze_predictions_selects_only_clips_with_a_stale_frame(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A NULL or different version makes a clip a candidate. A clip at the active version is not."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    stale, null, current, no_frames = (
        seed_clip(
            config,
            camera_id=cam_id,
            file_path=f"clips/pantry/{name}.mp4",
            source_filename=f"{name}.mp4",
            analysis_error=None,
            detector_version=_DETECTOR_VERSION,
        )
        for name in ("stale", "null", "current", "none")
    )
    for path in ("stale", "null", "current", "none"):
        _ = _ensure_clip_file(config, f"clips/pantry/{path}.mp4")
    _seed_frames_with_version(config, stale, version="cat-classifier@old")
    _seed_frames_with_version(config, null, version=None)
    _seed_frames_with_version(config, current, version=_CLASSIFIER_VERSION)
    subject_id = _seed_rufus(config)
    seen: list[Path] = []

    def detect(path: Path) -> _DetectResult:
        seen.append(path)
        return _detection_fields(has_cat=True), _boxed_frames(0, 1)

    exit_code = _reanalyze_with(config, detect, classifier=_classifier(subject_id), predictions=True)
    assert exit_code == 0
    assert sorted(p.name for p in seen) == ["null.mp4", "stale.mp4"]
    assert _read_predictions(config, stale) == [(subject_id, 0.91, _CLASSIFIER_VERSION)] * 2
    assert not _read_clip_frames(config, no_frames)


def test_reanalyze_predictions_second_run_selects_no_clip(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """After a pass at the active version, the next pass finds nothing to do."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=seed_camera(config), analysis_error=None)
    _seed_frames_with_version(config, clip_id, version=None)
    subject_id = _seed_rufus(config)
    classifier = _classifier(subject_id)

    assert _run_predictions(config, classifier, _boxed_frames(0, 1)) == 0
    _ = capsys.readouterr()
    assert _run_predictions(config, classifier, _boxed_frames(0, 1)) == 0
    out = capsys.readouterr().out
    assert "no qualifying clips" in out
    assert not any(t in out for t in _TRIPWIRES)


def test_reanalyze_predictions_without_classifier_exits_missing_dependency(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No loaded classifier -> exit 5 with an error on stderr, and no row changes."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    clip_id = seed_clip(config, camera_id=seed_camera(config), analysis_error=None)
    _seed_frames_with_version(config, clip_id, version=None)

    assert _run_predictions(config, None) == _EXIT_MISSING_DEPENDENCY
    captured = capsys.readouterr()
    assert "--predictions" in captured.err
    assert "[classifier].model" in captured.err
    assert "fetch-models" not in captured.err
    assert not any(t in captured.err + captured.out for t in _TRIPWIRES)
    assert [f.classifier_version for f in _read_clip_frames(config, clip_id)] == [None, None]


def test_reanalyze_predictions_exits_failure_when_a_frame_errors(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A frame the classifier fails on makes ``--predictions`` exit non-zero."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=seed_camera(config), analysis_error=None)
    _seed_frames_with_version(config, clip_id, version=None)
    classifier = _classifier(_seed_rufus(config), predict=_failing_predict)

    assert _run_predictions(config, classifier, _boxed_frames(0, 1)) == _EXIT_GENERIC_FAILURE


def test_reanalyze_predictions_cannot_combine_with_all() -> None:
    """The two flags are mutually exclusive at argument parse."""
    with pytest.raises(SystemExit) as excinfo:
        _ = _build_parser().parse_args(["reanalyze", "--predictions", "--all"])
    assert excinfo.value.code == 2


# --- thumb files ---------------------------------------------------------------------------------

_PER_CLIP_DIR = "thumbs/pantry/2026-05-01/123045"
_CLIP_START = datetime(2026, 5, 1, 12, 30, 45, tzinfo=UTC)


def test_reanalyze_keeps_old_best_thumb_when_another_frame_becomes_best(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """The old ``thumb_path`` is the thumb of a frame, so the file stays when the best frame moves."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(
        config,
        camera_id=seed_camera(config),
        start_ts=_CLIP_START,
        analysis_error=None,
        thumb_path=f"{_PER_CLIP_DIR}/01.jpg",
    )

    exit_code = _reanalyze_with(config, _returning(_detection_fields(), _build_scored_frames((0.1, 0.3, 0.85))), all=True)

    assert exit_code == 0
    assert read_clip(config, clip_id).thumb_path == f"{_PER_CLIP_DIR}/02.jpg"
    assert (config.storage_root / _PER_CLIP_DIR / "01.jpg").is_file()
    assert [frame[3] for frame in _read_frames(config, clip_id)] == [f"{_PER_CLIP_DIR}/{n:02d}.jpg" for n in range(3)]


def test_reanalyze_removes_thumbs_of_clip_deleted_during_detection(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A clip that leaves the database during the run leaves no thumbs and counts as skipped."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=seed_camera(config), start_ts=_CLIP_START, analysis_error="skipped: --no-detect")

    def detect(clip_full: Path) -> _DetectResult:
        _ = clip_full
        engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
        try:
            with get_session(engine) as session:
                session.delete(session.get_one(Clip, clip_id))
        finally:
            engine.dispose()
        return _detection_fields(), _build_scored_frames((0.4, 0.8))

    exit_code = _reanalyze_with(config, detect)

    assert exit_code == 0
    assert not (config.storage_root / _PER_CLIP_DIR).exists()
    out = capsys.readouterr().out
    assert "rescored=0 skipped_missing=1 errored=0" in out
    assert not any(t in out for t in _TRIPWIRES)


# --- --predictions selection ---------------------------------------------------------------------


def test_reanalyze_predictions_selects_clip_scored_by_an_older_detector(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A clip whose frames are at the active checkpoint is a candidate when its detector version is old."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=seed_camera(config), analysis_error=None, detector_version="yolov11n@old")
    _seed_frames_with_version(config, clip_id, version=_CLASSIFIER_VERSION)
    subject_id = _seed_rufus(config)

    assert _run_predictions(config, _classifier(subject_id), _boxed_frames(0, 1)) == 0
    assert read_clip(config, clip_id).detector_version == _DETECTOR_VERSION


def test_reanalyze_predictions_skips_clip_whose_detection_failed(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A clip with ``analysis_error`` belongs to the default mode, so it never takes a ``--limit`` slot."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=seed_camera(config), analysis_error="detect failed: synthetic")
    _seed_frames_with_version(config, clip_id, version=None)

    assert _run_predictions(config, _classifier(_seed_rufus(config)), _boxed_frames(0, 1)) == 0
    assert "no qualifying clips" in capsys.readouterr().out
    assert read_clip(config, clip_id).analysis_error == "detect failed: synthetic"


# --- _run_reanalyze wiring -----------------------------------------------------------------------


def test_run_reanalyze_syncs_subjects_from_config(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """The classifier maps slugs to subject ids, so the run first syncs the configured subjects."""
    config = config_with_dirs(tmp_path, make_config).model_copy(
        update={"subjects": [ConfiguredSubject(slug="marcel", display_name="Marcel", kind="cat", display_order=1)]},
    )
    init_schema(config.internal_root)
    _set_weights_present(config)

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
    ):
        exit_code = _run_reanalyze(make_handler_args(camera="", limit=None, all=False, predictions=False))

    assert exit_code == 0
    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        with get_session(engine) as session:
            assert list(session.scalars(select(Subject.slug))) == ["marcel"]
    finally:
        engine.dispose()

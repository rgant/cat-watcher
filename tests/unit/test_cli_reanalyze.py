"""Tests for ``cat-watcher reanalyze``.

Reanalyze re-scores clips whose detection failed (default filter ``analysis_error IS NOT NULL``) or
every clip (``--all``). Each test patches the detector + the ``detection_for`` helper so the
CPU-heavy YOLO path is never touched; what's exercised here is the row-update logic, the filter
rules, the ``--limit`` / ``--camera`` flags, and the preserve-``manual_has_cat`` invariant.
"""

from collections.abc import Callable  # noqa: TC003  # runtime: pytest evaluates fixture annotations during collection
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
from cli_test_helpers import (
    config_with_dirs,
    init_schema,
    make_handler_args,
    read_clip,
    seed_camera,
    seed_clip,
)
from sqlalchemy import select

from cat_watcher.__main__ import _run_reanalyze
from cat_watcher.config import Config  # noqa: TC001  # runtime: make_config callable annotation
from cat_watcher.db import ClipFrame, create_engine, get_session
from cat_watcher.detector import Detector, ScoredFrame

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _detector_mock() -> MagicMock:
    """Autospec'd ``Detector`` instance for ``Detector.from_weights``'s patched return value."""
    return cast("MagicMock", create_autospec(Detector, instance=True))


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
    version: str = "yolov11n@new",
    analysis_error: str | None = None,
) -> dict[str, object]:
    """Build a fake ``detection_fields_for`` payload that satisfies the reanalyze update path."""
    return {
        "has_cat": has_cat,
        "max_score": 0.92,
        "frames_sampled": 5,
        "frames_with_cat": 4,
        "best_box_xyxy": [10.0, 20.0, 30.0, 40.0],
        "detector_version": version,
        "analysis_error": analysis_error,
    }


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
    _set_weights_present(config)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=cam_id, analysis_error="skipped: --no-detect")

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
        patch("cat_watcher.__main__.detection_for", return_value=(_detection_fields(has_cat=True), ())),
    ):
        exit_code = _run_reanalyze(make_handler_args(camera="", limit=None, all=False))
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
    _set_weights_present(config)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=cam_id, analysis_error=None, detector_version="yolov11n@stale")

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
        patch("cat_watcher.__main__.detection_for", return_value=(_detection_fields(), ())),
    ):
        _ = _run_reanalyze(make_handler_args(camera="", limit=None, all=False))
    clip = read_clip(config, clip_id)
    assert clip.detector_version == "yolov11n@stale"  # untouched


def test_reanalyze_all_rescores_every_clip(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``--all`` widens the filter to clean clips too; ``detector_version`` advances."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _set_weights_present(config)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(config, camera_id=cam_id, analysis_error=None, detector_version="yolov11n@stale")

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
        patch("cat_watcher.__main__.detection_for", return_value=(_detection_fields(version="yolov11n@new"), ())),
    ):
        _ = _run_reanalyze(make_handler_args(camera="", limit=None, all=True))
    clip = read_clip(config, clip_id)
    assert clip.detector_version == "yolov11n@new"


def test_reanalyze_preserves_manual_has_cat(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """Manual labels survive re-detection; ``has_cat`` reflects the new model output."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _set_weights_present(config)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    clip_id = seed_clip(
        config,
        camera_id=cam_id,
        analysis_error="skipped: --no-detect",
        manual_has_cat=True,
        has_cat=False,
    )

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
        patch("cat_watcher.__main__.detection_for", return_value=(_detection_fields(has_cat=False), ())),
    ):
        _ = _run_reanalyze(make_handler_args(camera="", limit=None, all=False))
    clip = read_clip(config, clip_id)
    assert clip.manual_has_cat is True
    assert clip.has_cat is False  # detector said no; manual override (True) is preserved separately


def test_reanalyze_skips_missing_files_with_warning(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A clip whose file is missing on disk increments ``skipped_missing`` and the loop continues."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _set_weights_present(config)
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

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
        patch("cat_watcher.__main__.detection_for", return_value=(_detection_fields(), ())),
    ):
        exit_code = _run_reanalyze(make_handler_args(camera="", limit=None, all=False))
    assert exit_code == 0
    assert read_clip(config, present_id).analysis_error is None
    assert read_clip(config, missing_id).analysis_error == "skipped: --no-detect"


def test_reanalyze_limit_caps_count_in_start_ts_order(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``--limit N`` processes exactly the N earliest qualifying clips."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _set_weights_present(config)
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

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
        patch("cat_watcher.__main__.detection_for", return_value=(_detection_fields(), ())),
    ):
        _ = _run_reanalyze(make_handler_args(camera="", limit=2, all=False))
    assert read_clip(config, clip_ids[0]).analysis_error is None
    assert read_clip(config, clip_ids[1]).analysis_error is None
    assert read_clip(config, clip_ids[2]).analysis_error == "skipped: --no-detect"


def test_reanalyze_camera_filter_restricts_scope(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``--camera <name>`` only re-scores clips from that camera."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _set_weights_present(config)
    pantry_id = seed_camera(config, name="pantry", display_name="Pantry")
    bath_id = seed_camera(config, name="bath", display_name="Bath", host="cam2.example.com")
    _ = _ensure_clip_file(config)
    pantry_clip = seed_clip(config, camera_id=pantry_id, analysis_error="skipped: --no-detect", source_filename="pantry.mp4")
    bath_clip = seed_clip(config, camera_id=bath_id, analysis_error="skipped: --no-detect", source_filename="bath.mp4")

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
        patch("cat_watcher.__main__.detection_for", return_value=(_detection_fields(), ())),
    ):
        _ = _run_reanalyze(make_handler_args(camera="pantry", limit=None, all=False))
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
    _set_weights_present(config)
    pantry_id = seed_camera(config, name="pantry", display_name="Pantry")
    bath_id = seed_camera(config, name="bath", display_name="Bath", host="cam2.example.com")
    _ = _ensure_clip_file(config)
    _ = seed_clip(config, camera_id=pantry_id, analysis_error="skipped: --no-detect", source_filename="pantry.mp4")
    _ = seed_clip(config, camera_id=bath_id, analysis_error="skipped: --no-detect", source_filename="bath.mp4")

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
        patch("cat_watcher.__main__.detection_for", return_value=(_detection_fields(), ())),
    ):
        _ = _run_reanalyze(make_handler_args(camera="", limit=None, all=False))
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
    _set_weights_present(config)
    cam_id = seed_camera(config)
    _ = _ensure_clip_file(config)
    _ = seed_clip(config, camera_id=cam_id, analysis_error="skipped: --no-detect")

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
        patch(
            "cat_watcher.__main__.detection_for",
            return_value=(_detection_fields(analysis_error="detect failed: synthetic"), ()),
        ),
    ):
        exit_code = _run_reanalyze(make_handler_args(camera="", limit=None, all=False))
    assert exit_code == 1


def test_reanalyze_no_qualifying_clips_prints_message_and_exits_zero(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty DB (or filter that matches nothing) reports the no-op explicitly so the run isn't silent."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _set_weights_present(config)
    _ = seed_camera(config)

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
    ):
        exit_code = _run_reanalyze(make_handler_args(camera="", limit=None, all=True))
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
    _set_weights_present(config)
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

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.Detector.from_weights", return_value=_detector_mock()),
        patch("cat_watcher.__main__.detection_for", return_value=(_detection_fields(has_cat=True), scored_frames)),
    ):
        exit_code = _run_reanalyze(make_handler_args(camera="", limit=None, all=True))
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

"""Tests for cat_watcher.detector.

YOLO model is mocked at the boundary (we don't own ultralytics). ffmpeg / ffprobe run for real
against the synthetic_clip_path fixture (they're external binaries we always control via the pixi
env, and mocking them would leave the actual command lines untested).
"""

import hashlib
import subprocess
from pathlib import Path  # noqa: TC003  # runtime: pytest fixture annotations are evaluated by collectors
from unittest.mock import MagicMock

import numpy as np
import pytest
from ultralytics import YOLO  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateImportUsage]  # ultralytics lazily loads models
from ultralytics.engine.results import Boxes, Results

from cat_watcher.detector import DetectionResult, Detector, DetectorError

_COCO_CAT = 15
_COCO_DOG = 16
_DEFAULT_THRESHOLD = 0.35


def _fake_results(*, cls_ids: list[float], confidences: list[float], boxes: list[list[float]]) -> list[MagicMock]:
    """Build a one-element list mimicking ``ultralytics.engine.results.Results``."""
    fake_boxes = MagicMock(spec=Boxes)
    fake_boxes.cls = np.asarray(cls_ids)
    fake_boxes.conf = np.asarray(confidences)
    fake_boxes.xyxy = np.asarray(boxes).reshape(-1, 4) if boxes else np.empty((0, 4))
    fake = MagicMock(spec=Results)
    fake.boxes = fake_boxes
    return [fake]


def _make_detector(
    *,
    side_effect: list[list[MagicMock]] | None = None,
    return_value: list[MagicMock] | None = None,
    frames_to_sample: int = 3,
    confidence_threshold: float = _DEFAULT_THRESHOLD,
) -> tuple[Detector, MagicMock]:
    """Build a Detector with an injected, autospec'd YOLO mock and return both."""
    mock_model = MagicMock(spec=YOLO)
    if side_effect is not None:
        mock_model.side_effect = side_effect
    elif return_value is not None:
        mock_model.return_value = return_value
    detector = Detector(
        model=mock_model,
        version="test-detector@deadbeef",
        frames_to_sample=frames_to_sample,
        confidence_threshold=confidence_threshold,
    )
    return detector, mock_model


def test_detect_has_cat_when_all_frames_match(synthetic_clip_path: Path) -> None:
    """All frames return a cat above the threshold -> has_cat=True, frames_with_cat==N."""
    per_frame = _fake_results(cls_ids=[_COCO_CAT], confidences=[0.9], boxes=[[10.0, 20.0, 30.0, 40.0]])
    detector, mock_model = _make_detector(return_value=per_frame, frames_to_sample=3)

    result = detector.detect(synthetic_clip_path)

    assert mock_model.call_count == 3
    assert isinstance(result, DetectionResult)
    assert result.has_cat is True
    assert result.frames_sampled == 3
    assert result.frames_with_cat == 3
    assert result.max_score == 0.9
    assert result.best_box_xyxy == (10.0, 20.0, 30.0, 40.0)
    assert result.detector_version == "test-detector@deadbeef"


def test_detect_no_cat_when_no_frames_match(synthetic_clip_path: Path) -> None:
    """Empty boxes from every frame -> has_cat=False, no best box."""
    per_frame = _fake_results(cls_ids=[], confidences=[], boxes=[])
    detector, _ = _make_detector(return_value=per_frame, frames_to_sample=3)

    result = detector.detect(synthetic_clip_path)

    assert result.has_cat is False
    assert result.frames_sampled == 3
    assert result.frames_with_cat == 0
    assert result.max_score == 0.0
    assert result.best_box_xyxy is None


def test_detect_filters_below_confidence_threshold(synthetic_clip_path: Path) -> None:
    """A cat detection below the threshold doesn't count."""
    below_threshold = _fake_results(cls_ids=[_COCO_CAT], confidences=[0.20], boxes=[[1.0, 2.0, 3.0, 4.0]])
    detector, _ = _make_detector(return_value=below_threshold, frames_to_sample=3, confidence_threshold=0.50)

    result = detector.detect(synthetic_clip_path)

    assert result.has_cat is False
    assert result.frames_with_cat == 0
    assert result.best_box_xyxy is None


def test_detect_filters_non_cat_classes(synthetic_clip_path: Path) -> None:
    """A high-confidence non-cat (e.g. COCO class 16, dog) is ignored."""
    dog_only = _fake_results(cls_ids=[_COCO_DOG], confidences=[0.99], boxes=[[1.0, 2.0, 3.0, 4.0]])
    detector, _ = _make_detector(return_value=dog_only, frames_to_sample=3)

    result = detector.detect(synthetic_clip_path)

    assert result.has_cat is False
    assert result.frames_with_cat == 0
    assert result.max_score == 0.0


def test_detect_picks_highest_scoring_frame_for_best_box(synthetic_clip_path: Path) -> None:
    """Across frames with different scores, the highest-confidence box wins ``best_box_xyxy``."""
    side_effect = [
        _fake_results(cls_ids=[_COCO_CAT], confidences=[0.50], boxes=[[1.0, 1.0, 2.0, 2.0]]),
        _fake_results(cls_ids=[_COCO_CAT], confidences=[0.95], boxes=[[100.0, 100.0, 200.0, 200.0]]),
        _fake_results(cls_ids=[_COCO_CAT], confidences=[0.75], boxes=[[10.0, 10.0, 20.0, 20.0]]),
    ]
    detector, _ = _make_detector(side_effect=side_effect, frames_to_sample=3)

    result = detector.detect(synthetic_clip_path)

    assert result.has_cat is True
    assert result.frames_with_cat == 3
    assert result.max_score == 0.95
    assert result.best_box_xyxy == (100.0, 100.0, 200.0, 200.0)


def test_detect_picks_highest_scoring_box_within_a_frame(synthetic_clip_path: Path) -> None:
    """Multiple cat detections in one frame: the highest-confidence one wins for that frame."""
    multi_box = _fake_results(
        cls_ids=[_COCO_CAT, _COCO_CAT, _COCO_DOG],
        confidences=[0.40, 0.92, 0.99],
        boxes=[[1.0, 1.0, 2.0, 2.0], [50.0, 60.0, 70.0, 80.0], [0.0, 0.0, 1.0, 1.0]],
    )
    detector, _ = _make_detector(return_value=multi_box, frames_to_sample=2)

    result = detector.detect(synthetic_clip_path)

    assert result.has_cat is True
    assert result.frames_with_cat == 2
    assert result.max_score == 0.92
    # The dog box (highest overall) must NOT win — it's not a cat.
    assert result.best_box_xyxy == (50.0, 60.0, 70.0, 80.0)


def test_detect_counts_only_matching_frames(synthetic_clip_path: Path) -> None:
    """Mixed: 3 frames, 2 with cat, 1 without -> frames_with_cat==2, frames_sampled==3."""
    side_effect = [
        _fake_results(cls_ids=[_COCO_CAT], confidences=[0.80], boxes=[[1.0, 2.0, 3.0, 4.0]]),
        _fake_results(cls_ids=[], confidences=[], boxes=[]),
        _fake_results(cls_ids=[_COCO_CAT], confidences=[0.60], boxes=[[5.0, 6.0, 7.0, 8.0]]),
    ]
    detector, _ = _make_detector(side_effect=side_effect, frames_to_sample=3)

    result = detector.detect(synthetic_clip_path)

    assert result.has_cat is True
    assert result.frames_sampled == 3
    assert result.frames_with_cat == 2


def test_from_weights_computes_version_from_filename_and_sha256(tmp_path: Path) -> None:
    """``Detector.from_weights`` builds the version string ``<filename>@<sha256-of-bytes>``."""
    fake_weights = tmp_path / "yolo11n.pt"
    fake_bytes = b"not real model bytes, but enough to hash"
    _ = fake_weights.write_bytes(fake_bytes)
    expected_sha = hashlib.sha256(fake_bytes).hexdigest()

    sentinel_model = MagicMock(spec=YOLO)

    def fake_factory(_model_path: Path) -> MagicMock:
        return sentinel_model

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("cat_watcher.detector._yolo_factory", fake_factory)

        detector = Detector.from_weights(
            model_path=fake_weights,
            frames_to_sample=3,
            confidence_threshold=_DEFAULT_THRESHOLD,
        )

    assert detector.version == f"yolo11n.pt@{expected_sha}"


def test_from_weights_missing_file_raises(tmp_path: Path) -> None:
    """A missing weights file raises FileNotFoundError before attempting to load YOLO."""
    missing = tmp_path / "no-weights-here.pt"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        _ = Detector.from_weights(
            model_path=missing,
            frames_to_sample=3,
            confidence_threshold=_DEFAULT_THRESHOLD,
        )


def test_detect_calls_model_with_numpy_arrays(synthetic_clip_path: Path) -> None:
    """Frames are numpy uint8 RGB24 arrays; the model is called with ``verbose=False``."""
    per_frame = _fake_results(cls_ids=[], confidences=[], boxes=[])
    detector, mock_model = _make_detector(return_value=per_frame, frames_to_sample=2)

    _ = detector.detect(synthetic_clip_path)

    assert mock_model.call_count == 2
    for call in mock_model.call_args_list:
        frame = call.args[0]
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB24 -> 3 channels
        # ``verbose=False`` matters in production: a poller running every 5 minutes with N frames
        # per clip would otherwise spam the log file with one ultralytics banner per inference.
        assert call.kwargs.get("verbose") is False


def test_detect_skips_results_without_boxes_attribute(synthetic_clip_path: Path) -> None:
    """A Results object with ``boxes=None`` (defensive against future ultralytics versions) is skipped."""
    no_boxes = MagicMock(spec=Results)
    no_boxes.boxes = None
    detector, _ = _make_detector(return_value=[no_boxes], frames_to_sample=2)

    result = detector.detect(synthetic_clip_path)

    assert result.has_cat is False
    assert result.frames_with_cat == 0


@pytest.mark.parametrize("missing_binary", ["ffprobe", "ffmpeg"])
def test_detect_raises_when_required_binary_missing(
    missing_binary: str,
    synthetic_clip_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each required binary, when absent from PATH, surfaces as ``DetectorError`` naming it."""
    import shutil

    detector, _ = _make_detector(return_value=_fake_results(cls_ids=[], confidences=[], boxes=[]))
    real_which = shutil.which

    def selective_which(name: str) -> str | None:
        if name == missing_binary:
            return None
        return real_which(name)

    monkeypatch.setattr("cat_watcher.detector.shutil.which", selective_which)

    with pytest.raises(DetectorError, match=f"{missing_binary} not on PATH"):
        _ = detector.detect(synthetic_clip_path)


def test_detect_raises_on_unparseable_ffprobe_output(synthetic_clip_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed ffprobe response surfaces as ``DetectorError``, not a raw KeyError/JSONError."""
    detector, _ = _make_detector(return_value=_fake_results(cls_ids=[], confidences=[], boxes=[]))

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="this is not json", stderr="")

    monkeypatch.setattr("cat_watcher.detector.subprocess.run", fake_run)

    with pytest.raises(DetectorError, match="unparseable"):
        _ = detector.detect(synthetic_clip_path)


def test_detect_raises_on_non_positive_duration(synthetic_clip_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero / negative duration from ffprobe is rejected (would otherwise produce empty frame list)."""
    detector, _ = _make_detector(return_value=_fake_results(cls_ids=[], confidences=[], boxes=[]))
    bad_payload = '{"streams":[{"width":400,"height":266}],"format":{"duration":"0.000000"}}'

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=bad_payload, stderr="")

    monkeypatch.setattr("cat_watcher.detector.subprocess.run", fake_run)

    with pytest.raises(DetectorError, match="non-positive duration"):
        _ = detector.detect(synthetic_clip_path)


def test_detect_raises_on_short_ffmpeg_output(synthetic_clip_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ffmpeg returning fewer bytes than ``width*height*3`` is treated as a decode failure."""
    detector, _ = _make_detector(return_value=_fake_results(cls_ids=[], confidences=[], boxes=[]))
    real_run = subprocess.run

    def truncating_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
        if cmd[0].endswith("ffmpeg"):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"\x00\x01\x02", stderr=b"")
        # ``**kwargs: object`` is the right boundary for an opaque pass-through, but subprocess.run
        # has fully typed params; pyright/mypy can't prove the types line up across the splat.
        return real_run(cmd, **kwargs)  # type: ignore[call-overload, no-any-return]  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]

    monkeypatch.setattr("cat_watcher.detector.subprocess.run", truncating_run)

    with pytest.raises(DetectorError, match="expected"):
        _ = detector.detect(synthetic_clip_path)


def test_detect_samples_frames_at_distinct_timestamps(synthetic_clip_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ffmpeg is invoked with N distinct ``-ss <timestamp>`` values for ``frames_to_sample=N``.

    Guards against a refactor that calls ``_extract_frame`` N times with the same timestamp
    (which would yield N copies of one frame; a passing test that proves nothing).
    """
    per_frame = _fake_results(cls_ids=[], confidences=[], boxes=[])
    detector, _ = _make_detector(return_value=per_frame, frames_to_sample=3)
    real_run = subprocess.run
    ffmpeg_timestamps: list[str] = []

    def recording_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes] | subprocess.CompletedProcess[str]:
        if cmd[0].endswith("ffmpeg") and "-ss" in cmd:
            ffmpeg_timestamps.append(cmd[cmd.index("-ss") + 1])
        return real_run(cmd, **kwargs)  # type: ignore[call-overload, no-any-return]  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]

    monkeypatch.setattr("cat_watcher.detector.subprocess.run", recording_run)

    _ = detector.detect(synthetic_clip_path)

    assert len(ffmpeg_timestamps) == 3
    assert len(set(ffmpeg_timestamps)) == 3, f"expected 3 distinct timestamps, got {ffmpeg_timestamps}"

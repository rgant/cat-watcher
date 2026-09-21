"""Tests for cat_watcher.classifier.sources.

``extract_frame_at`` and ``load_yolo`` call their private helpers (``_probe_video``,
``_extract_frame``, ``_yolo_factory``) by name inside ``cat_watcher.detector``. A monkeypatch of
those private names still takes effect here, even though this file imports the public names
directly.
"""

from pathlib import Path  # noqa: TC003  # runtime: pytest fixture annotations are evaluated by collectors
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
from image_helpers import gradient_rgb
from PIL import Image
from ultralytics import YOLO  # type: ignore[attr-defined]  # ultralytics lazily loads models
from ultralytics.engine.results import Boxes, Results

from cat_watcher.classifier.dataset import LocalizedBox
from cat_watcher.classifier.labels_query import CatFrameRow
from cat_watcher.classifier.sources import make_frame_source, make_localizer
from cat_watcher.detector import DetectorError

if TYPE_CHECKING:
    import pytest

_COCO_CAT = 15.0
_COCO_DOG = 16.0


def _make_row(*, clip_file_path: str, frame_thumb_path: str, t_offset_seconds: float = 1.5) -> CatFrameRow:
    return CatFrameRow(
        clip_id=1,
        frame_id=1,
        ordinal=0,
        t_offset_seconds=t_offset_seconds,
        cat_slug="rufus",
        clip_file_path=clip_file_path,
        frame_thumb_path=frame_thumb_path,
    )


def _fake_results(*, cls_ids: list[float], confidences: list[float], boxes: list[list[float]]) -> list[MagicMock]:
    """Build a one-element list mimicking ``ultralytics.engine.results.Results``."""
    xyxy = np.asarray(boxes).reshape(-1, 4) if boxes else np.empty((0, 4))
    box_attrs = {"cls": np.asarray(cls_ids), "conf": np.asarray(confidences), "xyxy": xyxy}
    fake_boxes = MagicMock(spec=Boxes)
    for attr, value in box_attrs.items():
        setattr(fake_boxes, attr, value)
    return [MagicMock(spec=Results, boxes=fake_boxes)]


def _fake_probe(_clip_path: Path) -> tuple[float, int, int]:
    return (5.0, 6, 4)


def _make_fake_extract(image: np.ndarray, calls: list[tuple[Path, float]]) -> object:
    """Build a fake ``_extract_frame`` that records the ``(clip_path, timestamp)`` it receives."""

    def fake_extract(clip_path: Path, timestamp: float, *, width: int, height: int) -> np.ndarray:
        assert (width, height) == (6, 4)
        calls.append((clip_path, timestamp))
        return image

    return fake_extract


def _raising_extract(_clip_path: Path, _timestamp: float, *, width: int, height: int) -> np.ndarray:
    assert (width, height) == (6, 4)
    msg = "ffmpeg exploded"
    raise DetectorError(msg)


def _patch_yolo_factory(monkeypatch: pytest.MonkeyPatch, model: MagicMock) -> None:
    def fake_factory(_model_path: Path) -> MagicMock:
        return model

    monkeypatch.setattr("cat_watcher.detector._yolo_factory", fake_factory)


# --- make_frame_source ------------------------------------------------------------------------


def test_frame_source_reads_the_clip_when_it_decodes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip that decodes cleanly wins over the thumbnail, and its source tag is ``clip``.

    If ``make_frame_source`` passes the wrong field as the timestamp, the call assertion below
    fails. ``t_offset_seconds=7.5`` differs from both ``0.0`` and ``row.ordinal`` (``0``).
    """
    known_array = gradient_rgb(4, 6)
    calls: list[tuple[Path, float]] = []
    monkeypatch.setattr("cat_watcher.detector._probe_video", _fake_probe)
    monkeypatch.setattr("cat_watcher.detector._extract_frame", _make_fake_extract(known_array, calls))
    clip_path = tmp_path / "clip.mp4"
    _ = clip_path.write_bytes(b"not a real clip; extraction is faked")
    row = _make_row(clip_file_path="clip.mp4", frame_thumb_path="thumb.jpg", t_offset_seconds=7.5)

    frame_source = make_frame_source(storage_root=tmp_path)
    loaded = frame_source(row)

    assert loaded is not None
    assert loaded.source == "clip"
    assert np.array_equal(loaded.image, known_array)
    assert calls == [(clip_path, 7.5)]


def test_frame_source_falls_back_to_thumb_when_clip_is_absent(tmp_path: Path) -> None:
    """A missing clip file falls back to the frame thumbnail, tagged ``thumb``."""
    thumb_path = tmp_path / "thumb.jpg"
    Image.fromarray(gradient_rgb(8, 8)).save(thumb_path, format="JPEG")
    row = _make_row(clip_file_path="clip.mp4", frame_thumb_path="thumb.jpg")

    frame_source = make_frame_source(storage_root=tmp_path)
    loaded = frame_source(row)

    assert loaded is not None
    assert loaded.source == "thumb"
    assert loaded.image.shape == (8, 8, 3)


def test_frame_source_falls_back_to_thumb_on_detector_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip that exists but fails to decode falls back to the frame thumbnail."""
    monkeypatch.setattr("cat_watcher.detector._probe_video", _fake_probe)
    monkeypatch.setattr("cat_watcher.detector._extract_frame", _raising_extract)
    clip_path = tmp_path / "clip.mp4"
    _ = clip_path.write_bytes(b"present but undecodable")
    thumb_path = tmp_path / "thumb.jpg"
    Image.fromarray(gradient_rgb(5, 5)).save(thumb_path, format="JPEG")
    row = _make_row(clip_file_path="clip.mp4", frame_thumb_path="thumb.jpg")

    frame_source = make_frame_source(storage_root=tmp_path)
    loaded = frame_source(row)

    assert loaded is not None
    assert loaded.source == "thumb"
    assert loaded.image.shape == (5, 5, 3)


def test_frame_source_returns_none_when_clip_and_thumb_are_absent(tmp_path: Path) -> None:
    """Neither the clip nor the thumbnail exists, so no frame is loadable."""
    row = _make_row(clip_file_path="clip.mp4", frame_thumb_path="thumb.jpg")

    frame_source = make_frame_source(storage_root=tmp_path)

    assert frame_source(row) is None


def test_frame_source_returns_none_when_clip_fails_and_thumb_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed clip decode with no thumbnail on disk leaves nothing to load."""
    monkeypatch.setattr("cat_watcher.detector._probe_video", _fake_probe)
    monkeypatch.setattr("cat_watcher.detector._extract_frame", _raising_extract)
    clip_path = tmp_path / "clip.mp4"
    _ = clip_path.write_bytes(b"present but undecodable")
    row = _make_row(clip_file_path="clip.mp4", frame_thumb_path="thumb.jpg")

    frame_source = make_frame_source(storage_root=tmp_path)

    assert frame_source(row) is None


def test_frame_source_returns_none_when_thumb_is_corrupt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed clip decode plus an unreadable (corrupt) thumbnail file yields ``None``."""
    monkeypatch.setattr("cat_watcher.detector._probe_video", _fake_probe)
    monkeypatch.setattr("cat_watcher.detector._extract_frame", _raising_extract)
    clip_path = tmp_path / "clip.mp4"
    _ = clip_path.write_bytes(b"present but undecodable")
    thumb_path = tmp_path / "thumb.jpg"
    _ = thumb_path.write_bytes(b"this is not a jpeg file at all")
    row = _make_row(clip_file_path="clip.mp4", frame_thumb_path="thumb.jpg")

    frame_source = make_frame_source(storage_root=tmp_path)

    assert frame_source(row) is None


# --- make_localizer ----------------------------------------------------------------------------


def test_localizer_returns_the_cat_box(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single cat-class detection becomes the returned ``LocalizedBox``."""
    model = MagicMock(spec=YOLO)
    model.return_value = _fake_results(cls_ids=[_COCO_CAT], confidences=[0.42], boxes=[[1.0, 2.0, 3.0, 4.0]])
    _patch_yolo_factory(monkeypatch, model)
    localizer = make_localizer(model_path=tmp_path / "weights.pt", conf=0.10)

    result = localizer(gradient_rgb(4, 4))

    assert result == LocalizedBox(box=(1.0, 2.0, 3.0, 4.0), conf=0.42)


def test_localizer_returns_none_with_no_boxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No detections at all yields ``None``."""
    model = MagicMock(spec=YOLO)
    model.return_value = _fake_results(cls_ids=[], confidences=[], boxes=[])
    _patch_yolo_factory(monkeypatch, model)
    localizer = make_localizer(model_path=tmp_path / "weights.pt", conf=0.10)

    assert localizer(gradient_rgb(4, 4)) is None


def test_localizer_returns_none_with_only_non_cat_boxes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A high-confidence non-cat detection (COCO class 16, dog) does not count as a cat box."""
    model = MagicMock(spec=YOLO)
    model.return_value = _fake_results(cls_ids=[_COCO_DOG], confidences=[0.95], boxes=[[0.0, 0.0, 1.0, 1.0]])
    _patch_yolo_factory(monkeypatch, model)
    localizer = make_localizer(model_path=tmp_path / "weights.pt", conf=0.10)

    assert localizer(gradient_rgb(4, 4)) is None


def test_localizer_picks_the_highest_scoring_cat_box(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Of several cat-class detections, the highest-confidence one wins."""
    model = MagicMock(spec=YOLO)
    model.return_value = _fake_results(
        cls_ids=[_COCO_CAT, _COCO_CAT],
        confidences=[0.30, 0.77],
        boxes=[[0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 9.0, 9.0]],
    )
    _patch_yolo_factory(monkeypatch, model)
    localizer = make_localizer(model_path=tmp_path / "weights.pt", conf=0.10)

    result = localizer(gradient_rgb(4, 4))

    assert result == LocalizedBox(box=(5.0, 5.0, 9.0, 9.0), conf=0.77)


def test_localizer_passes_conf_into_the_model_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``make_localizer`` passes ``conf`` into the model call. It must not use ultralytics' 0.25 default."""
    model = MagicMock(spec=YOLO)
    model.return_value = _fake_results(cls_ids=[_COCO_CAT], confidences=[0.50], boxes=[[0.0, 0.0, 1.0, 1.0]])
    _patch_yolo_factory(monkeypatch, model)
    localizer = make_localizer(model_path=tmp_path / "weights.pt", conf=0.10)

    _ = localizer(gradient_rgb(4, 4))

    assert model.call_args is not None
    assert model.call_args.kwargs.get("conf") == 0.10

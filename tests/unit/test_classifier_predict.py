"""Tests for :mod:`cat_watcher.classifier.predict`, its ``labels_query`` addition, and the CLI action.

Every ``predict_clips`` test fakes ``FrameSource``, ``Localizer``, and ``PredictFn``, keyed by
frame id / image shape / crop filename, the same pattern ``test_classifier_dataset.py`` and
``test_classifier_benchmark.py`` use. Every CLI test patches ``sources.make_frame_source``,
``sources.make_localizer``, and ``benchmark.make_predict_fn``, so no real ffmpeg or YOLO call runs.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from classifier_helpers import write_checkpoint_with_sidecar
from cli_test_helpers import assert_missing_dependency, config_with_dirs, init_schema, make_classifier_args
from cli_test_helpers import seed_camera as seed_camera_for_config
from db_helpers import add_clip, add_clip_frame, add_event_subject, seed_cat_subject, tag_frame
from image_helpers import gradient_rgb
from PIL import Image
from sqlalchemy import func, select

from cat_watcher.classifier.cli import run
from cat_watcher.classifier.dataset import CROP_MAX_WIDTH, CROP_QUALITY, ExportSources, LoadedFrame, LocalizedBox
from cat_watcher.classifier.geometry import PAD_FRAC, square_pad_box
from cat_watcher.classifier.labels_query import CatFrameRow, ClipFrameCandidate, query_untagged_cat_clip_frames
from cat_watcher.classifier.predict import ClipVerdict, FramePrediction, PredictOptions, predict_clips, render_rows
from cat_watcher.db import ClipFrameSubject, create_engine, get_session
from cat_watcher.thumbnails import encode_frame

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from sqlalchemy.engine import Engine

    from cat_watcher.classifier.benchmark import PredictFn
    from cat_watcher.config import Config

_START = datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)


# --- shared fakes and helpers ------------------------------------------------------------------------


@dataclass
class _FakeFrameSource:
    """Fake ``FrameSource``: a canned frame keyed by ``frame_id``, or ``None`` for a miss."""

    missing_frame_ids: frozenset[int] = frozenset()
    image_by_frame_id: dict[int, np.ndarray] = field(default_factory=dict)
    default_shape: tuple[int, int] = (40, 40)

    def __call__(self, row: CatFrameRow) -> LoadedFrame | None:
        if row.frame_id in self.missing_frame_ids:
            return None
        image = self.image_by_frame_id.get(row.frame_id)
        if image is None:
            image = gradient_rgb(*self.default_shape)
        return LoadedFrame(image=image, source="clip")


@dataclass
class _FakeLocalizer:
    """Fake ``Localizer``: a canned box, or ``None`` for a frame of ``miss_shapes``."""

    box: tuple[float, float, float, float] = (4.0, 4.0, 30.0, 30.0)
    conf: float = 0.8
    miss_shapes: frozenset[tuple[int, int]] = frozenset()

    def __call__(self, image: np.ndarray) -> LocalizedBox | None:
        if image.shape[:2] in self.miss_shapes:
            return None
        return LocalizedBox(box=self.box, conf=self.conf)


def _fake_predict_by_filename(mapping: dict[str, tuple[str, float]]) -> PredictFn:
    """Build a fixed-output ``PredictFn`` keyed by crop filename."""

    def predict(image_path: Path) -> tuple[str, float]:
        return mapping[image_path.name]

    return predict


def _candidate(  # noqa: PLR0913  # test builder: one kwarg per ClipFrameCandidate field, no natural grouping
    clip_id: int,
    *,
    frame_id: int,
    ordinal: int = 0,
    camera_name: str = "pantry",
    start_ts: datetime = _START,
    clip_file_path: str = "clips/pantry/a.mp4",
    frame_thumb_path: str = "thumbs/pantry/a.jpg",
) -> ClipFrameCandidate:
    """Build one ``ClipFrameCandidate``. ``t_offset_seconds`` derives from ``ordinal``."""
    return ClipFrameCandidate(
        clip_id=clip_id,
        camera_name=camera_name,
        start_ts=start_ts,
        clip_file_path=clip_file_path,
        frame_id=frame_id,
        ordinal=ordinal,
        t_offset_seconds=float(ordinal),
        frame_thumb_path=frame_thumb_path,
    )


def _assert_no_placeholder_leak(text: str) -> None:
    """CLI output tripwire (CLAUDE.md): a missing ``f`` prefix prints the placeholder text unformatted."""
    for token in ("{_fmt", "{self.", "{cam.", "{cfg.", "NoneType"):
        assert token not in text


# --- query_untagged_cat_clip_frames ------------------------------------------------------------------------


def test_query_untagged_cat_clip_frames_on_an_empty_database_returns_an_empty_list(alembic_engine: Engine) -> None:
    """No clip rows at all still returns a valid, empty result."""
    assert query_untagged_cat_clip_frames(alembic_engine) == []


def test_query_untagged_cat_clip_frames_returns_a_has_cat_clip_with_no_cat_tag(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """A ``has_cat`` clip with no tag on any frame is a candidate. It carries its best frame."""
    cam_id = seed_camera(alembic_engine)
    clip_id = add_clip(alembic_engine, cam_id, start_ts=_START, name="a.mp4")
    frame_id = add_clip_frame(alembic_engine, clip_id, 0)

    candidates = query_untagged_cat_clip_frames(alembic_engine)

    assert [c.clip_id for c in candidates] == [clip_id]
    assert candidates[0].frame_id == frame_id
    assert candidates[0].camera_name == "pantry"


def test_query_untagged_cat_clip_frames_excludes_a_clip_with_has_cat_false(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """A clip the detector never flagged is not a candidate, tagged or not."""
    cam_id = seed_camera(alembic_engine)
    clip_id = add_clip(alembic_engine, cam_id, start_ts=_START, name="a.mp4", has_cat=False)
    _ = add_clip_frame(alembic_engine, clip_id, 0)

    assert query_untagged_cat_clip_frames(alembic_engine) == []


def test_query_untagged_cat_clip_frames_excludes_a_clip_whose_frame_has_a_cat_tag(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """A clip an operator already judged drops out."""
    cam_id = seed_camera(alembic_engine)
    marcel_id = seed_cat_subject(alembic_engine, slug="marcel", display_order=1)
    clip_id = add_clip(alembic_engine, cam_id, start_ts=_START, name="a.mp4")
    frame_id = add_clip_frame(alembic_engine, clip_id, 0)
    tag_frame(alembic_engine, frame_id, marcel_id)

    assert query_untagged_cat_clip_frames(alembic_engine) == []


def test_query_untagged_cat_clip_frames_keeps_a_clip_with_only_an_event_tag(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """An event tag does not count as a cat judgment, so the clip stays a candidate."""
    cam_id = seed_camera(alembic_engine)
    cleaning_id = add_event_subject(alembic_engine, slug="cleaning", display_order=1)
    clip_id = add_clip(alembic_engine, cam_id, start_ts=_START, name="a.mp4")
    frame_id = add_clip_frame(alembic_engine, clip_id, 0)
    tag_frame(alembic_engine, frame_id, cleaning_id)

    assert [c.clip_id for c in query_untagged_cat_clip_frames(alembic_engine)] == [clip_id]


def test_query_untagged_cat_clip_frames_returns_every_frame_of_a_clip_in_ordinal_order(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """Every frame comes back, whatever its detector score. One clip can hold two cats."""
    cam_id = seed_camera(alembic_engine)
    clip_id = add_clip(alembic_engine, cam_id, start_ts=_START, name="a.mp4")
    _ = add_clip_frame(alembic_engine, clip_id, 0, score=0.5)
    _ = add_clip_frame(alembic_engine, clip_id, 1, score=0.9)
    _ = add_clip_frame(alembic_engine, clip_id, 2, score=0.9)

    candidates = query_untagged_cat_clip_frames(alembic_engine)

    assert [c.ordinal for c in candidates] == [0, 1, 2]
    assert {c.clip_id for c in candidates} == {clip_id}


def test_query_untagged_cat_clip_frames_camera_filter_narrows_the_result(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """``camera`` restricts the result to that one configured camera."""
    pantry_id = seed_camera(alembic_engine, name="pantry", display_name="Pantry")
    kitchen_id = seed_camera(alembic_engine, name="kitchen", display_name="Kitchen")
    pantry_clip = add_clip(alembic_engine, pantry_id, start_ts=_START, name="a.mp4")
    _ = add_clip_frame(alembic_engine, pantry_clip, 0)
    kitchen_clip = add_clip(alembic_engine, kitchen_id, start_ts=_START, name="b.mp4")
    _ = add_clip_frame(alembic_engine, kitchen_clip, 0)

    candidates = query_untagged_cat_clip_frames(alembic_engine, camera="kitchen")

    assert [c.clip_id for c in candidates] == [kitchen_clip]


def test_query_untagged_cat_clip_frames_since_and_until_narrow_the_result(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """``since`` and ``until`` each cut the result to one side of a split point."""
    cam_id = seed_camera(alembic_engine)
    old_clip = add_clip(alembic_engine, cam_id, start_ts=_START, name="a.mp4")
    _ = add_clip_frame(alembic_engine, old_clip, 0)
    new_clip = add_clip(alembic_engine, cam_id, start_ts=_START + timedelta(days=10), name="b.mp4")
    _ = add_clip_frame(alembic_engine, new_clip, 0)
    split = _START + timedelta(days=1)

    since_result = query_untagged_cat_clip_frames(alembic_engine, since=split)
    until_result = query_untagged_cat_clip_frames(alembic_engine, until=split)

    assert [c.clip_id for c in since_result] == [new_clip]
    assert [c.clip_id for c in until_result] == [old_clip]


def test_query_untagged_cat_clip_frames_limit_counts_clips_not_frames(
    alembic_engine: Engine,
    seed_camera: Callable[..., int],
) -> None:
    """``limit`` caps clips. Each kept clip still returns every frame it holds, newest clip first.

    A ``limit`` on the outer select caps frames instead. This test then returns one clip.
    """
    cam_id = seed_camera(alembic_engine)
    clip_ids: list[int] = []
    for i in range(3):
        clip_id = add_clip(alembic_engine, cam_id, start_ts=_START + timedelta(minutes=i), name=f"{i}.mp4")
        _ = add_clip_frame(alembic_engine, clip_id, 0)
        _ = add_clip_frame(alembic_engine, clip_id, 1)
        clip_ids.append(clip_id)

    candidates = query_untagged_cat_clip_frames(alembic_engine, limit=2)

    assert [(c.clip_id, c.ordinal) for c in candidates] == [
        (clip_ids[2], 0),
        (clip_ids[2], 1),
        (clip_ids[1], 0),
        (clip_ids[1], 1),
    ]


# --- predict_clips -------------------------------------------------------------------------------


def test_predict_clips_returns_one_verdict_per_clip_in_order(tmp_path: Path) -> None:
    """One ``ClipVerdict`` per clip, in input order. Each carries the fake predictor's output."""
    candidates = [_candidate(1, frame_id=1), _candidate(2, frame_id=2, ordinal=3)]
    predict = _fake_predict_by_filename({"1_0.jpg": ("marcel", 0.91), "2_3.jpg": ("rufus", 0.62)})
    options = PredictOptions(crops_dir=tmp_path / "spotcheck", threshold=0.5)

    predictions = predict_clips(
        candidates,
        sources=ExportSources(frame_source=_FakeFrameSource(), localizer=_FakeLocalizer()),
        predict=predict,
        options=options,
    )

    assert [p.clip_id for p in predictions] == [1, 2]
    assert predictions[0].frames[0].cat_slug == "marcel"
    assert predictions[0].frames[0].confidence == pytest.approx(0.91)
    assert predictions[1].frames[0].cat_slug == "rufus"
    assert predictions[1].frames[0].confidence == pytest.approx(0.62)


def test_predict_clips_with_a_frame_load_failure_yields_unsure_and_continues(tmp_path: Path) -> None:
    """A candidate whose frame fails to load is unsure with no slug. The batch still scores the rest."""
    candidates = [_candidate(1, frame_id=1), _candidate(2, frame_id=2)]
    frame_source = _FakeFrameSource(missing_frame_ids=frozenset({1}))
    predict = _fake_predict_by_filename({"2_0.jpg": ("marcel", 0.9)})
    options = PredictOptions(crops_dir=tmp_path / "spotcheck", threshold=0.5)

    predictions = predict_clips(
        candidates,
        sources=ExportSources(frame_source=frame_source, localizer=_FakeLocalizer()),
        predict=predict,
        options=options,
    )

    assert predictions[0].frames[0].cat_slug is None
    assert predictions[0].frames[0].confidence == 0.0
    assert predictions[0].frames[0].unsure is True
    assert predictions[1].frames[0].cat_slug == "marcel"
    assert predictions[1].frames[0].unsure is False


def test_predict_clips_with_no_localized_box_yields_unsure_and_continues(tmp_path: Path) -> None:
    """A candidate whose localizer finds no cat is unsure with no slug. The batch still scores the rest."""
    candidates = [_candidate(1, frame_id=1), _candidate(2, frame_id=2)]
    frame_source = _FakeFrameSource(image_by_frame_id={1: gradient_rgb(50, 50), 2: gradient_rgb(40, 40)})
    localizer = _FakeLocalizer(miss_shapes=frozenset({(50, 50)}))
    predict = _fake_predict_by_filename({"2_0.jpg": ("rufus", 0.77)})
    options = PredictOptions(crops_dir=tmp_path / "spotcheck", threshold=0.5)

    predictions = predict_clips(
        candidates,
        sources=ExportSources(frame_source=frame_source, localizer=localizer),
        predict=predict,
        options=options,
    )

    assert predictions[0].frames[0].cat_slug is None
    assert predictions[0].frames[0].unsure is True
    assert predictions[1].frames[0].cat_slug == "rufus"
    assert predictions[1].frames[0].unsure is False


def test_predict_clips_confidence_below_threshold_is_unsure(tmp_path: Path) -> None:
    """A confidence below the threshold marks the prediction unsure."""
    candidates = [_candidate(1, frame_id=1)]
    predict = _fake_predict_by_filename({"1_0.jpg": ("marcel", 0.49)})
    options = PredictOptions(crops_dir=tmp_path / "spotcheck", threshold=0.5)

    predictions = predict_clips(
        candidates,
        sources=ExportSources(frame_source=_FakeFrameSource(), localizer=_FakeLocalizer()),
        predict=predict,
        options=options,
    )

    assert predictions[0].frames[0].unsure is True


def test_predict_clips_confidence_exactly_at_threshold_is_not_unsure(tmp_path: Path) -> None:
    """A confidence exactly at the threshold does not count as unsure."""
    candidates = [_candidate(1, frame_id=1)]
    predict = _fake_predict_by_filename({"1_0.jpg": ("marcel", 0.5)})
    options = PredictOptions(crops_dir=tmp_path / "spotcheck", threshold=0.5)

    predictions = predict_clips(
        candidates,
        sources=ExportSources(frame_source=_FakeFrameSource(), localizer=_FakeLocalizer()),
        predict=predict,
        options=options,
    )

    assert predictions[0].frames[0].unsure is False


def test_predict_clips_writes_a_crop_file_under_crops_dir(tmp_path: Path) -> None:
    """A successful prediction names a crop file that exists under ``crops_dir``."""
    candidates = [_candidate(7, frame_id=7, ordinal=2)]
    predict = _fake_predict_by_filename({"7_2.jpg": ("marcel", 0.9)})
    crops_dir = tmp_path / "spotcheck"
    options = PredictOptions(crops_dir=crops_dir, threshold=0.5)

    predictions = predict_clips(
        candidates,
        sources=ExportSources(frame_source=_FakeFrameSource(), localizer=_FakeLocalizer()),
        predict=predict,
        options=options,
    )

    assert (crops_dir / predictions[0].frames[0].thumb_relpath).is_file()


def _export_reference_crop_size(frame: np.ndarray, box: tuple[float, float, float, float], dest: Path) -> tuple[int, int]:
    """Build the crop ``export_dataset`` produces for ``frame``/``box``, and return its pixel size."""
    frame_h, frame_w = cast("tuple[int, int]", frame.shape[:2])
    x1, y1, x2, y2 = square_pad_box(box, frame_w=frame_w, frame_h=frame_h, pad_frac=PAD_FRAC)
    crop_array = frame[y1:y2, x1:x2].copy()
    encode_frame(crop_array, dest, max_width=CROP_MAX_WIDTH, quality=CROP_QUALITY)
    with Image.open(dest) as img:
        return img.size


def test_predict_crop_matches_the_export_crop_dimensions(tmp_path: Path) -> None:
    """predict.py's crop must match export_dataset's pipeline, or a spot-check score means nothing.

    Both must use the same ``square_pad_box`` call and the same ``encode_frame`` constants. A
    predict.py that pads, slices, or encodes differently fails this comparison.
    """
    frame = gradient_rgb(120, 160)
    box = (10.0, 20.0, 90.0, 110.0)
    crops_dir = tmp_path / "spotcheck"

    predictions = predict_clips(
        [_candidate(1, frame_id=1)],
        sources=ExportSources(frame_source=_FakeFrameSource(image_by_frame_id={1: frame}), localizer=_FakeLocalizer(box=box)),
        predict=_fake_predict_by_filename({"1_0.jpg": ("marcel", 0.9)}),
        options=PredictOptions(crops_dir=crops_dir, threshold=0.5),
    )
    with Image.open(crops_dir / predictions[0].frames[0].thumb_relpath) as predict_crop:
        predict_size = predict_crop.size

    export_size = _export_reference_crop_size(frame, box, tmp_path / "export_crop.jpg")

    assert predict_size == export_size


# --- grouping and render_rows ------------------------------------------------------------------------


def _frame(ordinal: int, slug: str | None, confidence: float, *, threshold: float = 0.5) -> FramePrediction:
    """Build one ``FramePrediction``. ``unsure`` derives from ``confidence`` against ``threshold``."""
    return FramePrediction(
        ordinal=ordinal,
        cat_slug=slug,
        confidence=confidence,
        unsure=slug is None or confidence < threshold,
        thumb_relpath=f"1_{ordinal}.jpg",
    )


def _verdict(*frames: FramePrediction, clip_id: int = 42) -> ClipVerdict:
    """Build one ``ClipVerdict`` over ``frames``, at a fixed camera and start time."""
    return ClipVerdict(
        clip_id=clip_id,
        camera_name="pantry",
        start_ts=datetime(2026, 7, 2, 16, 19, 5, tzinfo=UTC),
        frames=frames,
    )


def test_predict_clips_groups_every_frame_of_one_clip_into_one_verdict(tmp_path: Path) -> None:
    """Every frame of one clip lands in a single verdict, in candidate order."""
    candidates = [_candidate(1, frame_id=10, ordinal=0), _candidate(1, frame_id=11, ordinal=1), _candidate(1, frame_id=12, ordinal=2)]
    predict = _fake_predict_by_filename(
        {"1_0.jpg": ("marcel", 0.99), "1_1.jpg": ("marcel", 0.95), "1_2.jpg": ("rufus", 0.88)},
    )

    verdicts = predict_clips(
        candidates,
        sources=ExportSources(frame_source=_FakeFrameSource(), localizer=_FakeLocalizer()),
        predict=predict,
        options=PredictOptions(crops_dir=tmp_path / "spotcheck", threshold=0.5),
    )

    assert len(verdicts) == 1
    assert [f.ordinal for f in verdicts[0].frames] == [0, 1, 2]
    assert [f.cat_slug for f in verdicts[0].frames] == ["marcel", "marcel", "rufus"]


def test_clip_verdict_reports_two_cats_as_mixed() -> None:
    """A clip whose frames name two cats is mixed."""
    verdict = _verdict(_frame(0, "marcel", 1.0), _frame(1, "rufus", 1.0))

    assert verdict.is_mixed is True


def test_clip_verdict_with_one_cat_and_a_miss_is_not_mixed() -> None:
    """The poller samples frames the cat is absent from. Such a frame is not a second opinion."""
    verdict = _verdict(_frame(0, "marcel", 0.99), _frame(1, None, 0.0), _frame(2, "marcel", 0.98))

    assert verdict.is_mixed is False
    assert verdict.unsure is False
    assert verdict.min_confidence == pytest.approx(0.98)


def test_clip_verdict_min_confidence_is_the_weakest_frame() -> None:
    """The clip's confidence is the lowest of its frames, never the first or the best."""
    verdict = _verdict(_frame(0, "marcel", 0.99), _frame(1, "marcel", 0.72), _frame(2, "marcel", 0.90))

    assert verdict.min_confidence == pytest.approx(0.72)


def test_render_rows_prints_one_line_for_a_clip_whose_frames_agree() -> None:
    """Frames that agree collapse to one line. That line names the cat, the count, and the lowest confidence."""
    tz = ZoneInfo("America/New_York")
    verdict = _verdict(_frame(0, "marcel", 0.99), _frame(1, "marcel", 0.87))

    text = render_rows([verdict], tz=tz)

    assert len(text.splitlines()) == 1
    assert "clip=42" in text
    assert "marcel" in text
    assert "2/2" in text
    assert "conf=0.87" in text
    assert "2026-07-02" in text
    _assert_no_placeholder_leak(text)


def test_render_rows_expands_a_mixed_clip_into_one_line_per_frame() -> None:
    """Two cats in one clip print a ``MIXED`` header with the tally, then every frame."""
    tz = ZoneInfo("America/New_York")
    verdict = _verdict(_frame(0, "marcel", 1.0), _frame(1, "marcel", 0.98), _frame(2, "rufus", 0.91))

    lines = render_rows([verdict], tz=tz).splitlines()

    assert len(lines) == 4
    assert "MIXED" in lines[0]
    assert "marcel 2" in lines[0]
    assert "rufus 1" in lines[0]
    assert "of 3" in lines[0]
    assert [line.strip().split()[0] for line in lines[1:]] == ["ord=0", "ord=1", "ord=2"]
    assert "rufus" in lines[3]
    _assert_no_placeholder_leak("\n".join(lines))


def test_render_rows_keeps_a_clip_with_one_cat_and_a_miss_on_one_line() -> None:
    """One cat plus a frame holding no cat prints ``1/2`` on one line, and never expands.

    A cat absent from a sampled frame is the normal case. An expansion on it buries the mixed
    clips that need the operator's eyes.
    """
    tz = ZoneInfo("America/New_York")
    verdict = _verdict(_frame(0, "marcel", 0.95), _frame(1, None, 0.0))

    lines = render_rows([verdict], tz=tz).splitlines()

    assert len(lines) == 1
    assert "MIXED" not in lines[0]
    assert "marcel" in lines[0]
    assert "1/2" in lines[0]
    assert "UNSURE" not in lines[0]


def test_render_rows_prints_one_line_when_every_frame_missed() -> None:
    """A clip no frame classified prints ``0/2``, because zero frames produced a cat."""
    tz = ZoneInfo("America/New_York")
    verdict = _verdict(_frame(0, None, 0.0), _frame(1, None, 0.0))

    text = render_rows([verdict], tz=tz)

    assert len(text.splitlines()) == 1
    assert "0/2" in text
    assert "UNSURE" in text
    _assert_no_placeholder_leak(text)


def test_render_rows_marks_an_agreeing_clip_unsure_below_the_threshold() -> None:
    """A frame under the threshold marks the whole clip unsure, although every frame names one cat."""
    tz = ZoneInfo("America/New_York")
    verdict = _verdict(_frame(0, "marcel", 0.99), _frame(1, "marcel", 0.31))

    text = render_rows([verdict], tz=tz)

    assert len(text.splitlines()) == 1
    assert "UNSURE" in text


# --- CLI: classifier predict -----------------------------------------------------------------------


def _write_checkpoint(models_dir: Path, *, classes: tuple[str, ...] = ("marcel", "rufus")) -> Path:
    """Write a stub checkpoint plus a matching sidecar under ``models_dir``."""
    model_path = models_dir / "cat-classifier-aaaaaaaa.pt"
    write_checkpoint_with_sidecar(model_path, classes=classes, manifest_path=models_dir / "unused-manifest.json")
    return model_path


def _fake_predict_fn(_image_path: Path) -> tuple[str, float]:
    return "marcel", 0.9


def test_classifier_predict_with_no_checkpoint_exits_missing_dependency(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty (or absent) models directory exits 5. Stderr names ``classifier train`` as the fix."""
    config = config_with_dirs(tmp_path, make_config)

    exit_code = run(make_classifier_args("predict"), config=config)

    err = capsys.readouterr().err
    assert_missing_dependency(exit_code, err, names="classifier train")


def test_classifier_predict_with_a_missing_model_flag_exits_missing_dependency(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicit ``--model`` naming a file that does not exist exits 5."""
    config = config_with_dirs(tmp_path, make_config)
    missing = config.internal_root / "models" / "cat-classifier-ffffffff.pt"

    exit_code = run(make_classifier_args("predict", model=missing), config=config)

    err = capsys.readouterr().err
    assert_missing_dependency(exit_code, err, names="classifier train")


def test_classifier_predict_happy_path_prints_rows(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A patched ``predict_clips`` exits 0, and its rows appear in stdout."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    models_dir = config.internal_root / "models"
    models_dir.mkdir(parents=True)
    _ = _write_checkpoint(models_dir)
    weights = models_dir / config.detector.model
    _ = weights.write_bytes(b"stub-detector-weights")

    fake_verdicts = [
        ClipVerdict(
            clip_id=1,
            camera_name="pantry",
            start_ts=datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC),
            frames=(FramePrediction(ordinal=0, cat_slug="marcel", confidence=0.9, unsure=False, thumb_relpath="1_0.jpg"),),
        ),
    ]
    with (
        patch("cat_watcher.classifier.sources.make_frame_source", return_value=_FakeFrameSource()),
        patch("cat_watcher.classifier.sources.make_localizer", return_value=_FakeLocalizer()),
        patch("cat_watcher.classifier.benchmark.make_predict_fn", return_value=_fake_predict_fn),
        patch("cat_watcher.classifier.predict.predict_clips", return_value=fake_verdicts) as predict_mock,
    ):
        exit_code = run(make_classifier_args("predict"), config=config)

    assert exit_code == 0
    predict_mock.assert_called_once()
    out = capsys.readouterr().out
    assert "marcel" in out
    assert "1" in out
    _assert_no_placeholder_leak(out)


def test_classifier_predict_writes_no_db_row(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A real (unpatched) ``predict_clips`` run leaves ``clip_frame_subjects`` untouched."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera_for_config(config)
    engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        clip_id = add_clip(engine, cam_id, start_ts=_START, name="a.mp4")
        _ = add_clip_frame(engine, clip_id, 0)
        before = _count_clip_frame_subjects(engine)
    finally:
        engine.dispose()

    models_dir = config.internal_root / "models"
    models_dir.mkdir(parents=True)
    _ = _write_checkpoint(models_dir)
    weights = models_dir / config.detector.model
    _ = weights.write_bytes(b"stub-detector-weights")

    with (
        patch("cat_watcher.classifier.sources.make_frame_source", return_value=_FakeFrameSource()),
        patch("cat_watcher.classifier.sources.make_localizer", return_value=_FakeLocalizer()),
        patch("cat_watcher.classifier.benchmark.make_predict_fn", return_value=_fake_predict_fn),
    ):
        exit_code = run(make_classifier_args("predict"), config=config)
    assert exit_code == 0

    engine2 = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
    try:
        after = _count_clip_frame_subjects(engine2)
    finally:
        engine2.dispose()
    assert after == before


def _count_clip_frame_subjects(engine: Engine) -> int:
    """Count every ``clip_frame_subjects`` row, to prove a run added or removed none."""
    with get_session(engine) as session:
        return session.execute(
            select(func.count()).select_from(ClipFrameSubject),  # pylint: disable=not-callable  # sqlalchemy func.count() is a generative construct, not the builtin; pylint false positive
        ).scalar_one()

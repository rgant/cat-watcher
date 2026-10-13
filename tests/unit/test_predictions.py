"""Tests for :mod:`cat_watcher.predictions`. A fake ``PredictFn`` stands in for the model."""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from db_helpers import add_event_subject, make_clip_frame, scored_frames_with_boxes, seed_cat_subject
from image_helpers import gradient_rgb

from cat_watcher.classifier.crop import crop_frame, write_crop_jpeg
from cat_watcher.config import ClassifierConfig
from cat_watcher.db import ClipFrame, Subject, get_session
from cat_watcher.predictions import FrameCounts, LiveClassifier, classify_frames, load_live_classifier

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from sqlalchemy.engine import Engine

    from cat_watcher.classifier.benchmark import PredictFn
    from cat_watcher.config import Config

_VERSION = "m.pt@abc"
_LOGGER = "cat_watcher.predictions"
_LABEL = "garage/clip-001.mp4"


@pytest.fixture(autouse=True)
def _enable_predictions_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo a ``disabled`` flag on this logger.

    ``migrations/env.py`` calls ``logging.config.fileConfig``, which disables existing loggers. A
    test that runs Alembic first can then leave this logger disabled.
    """
    monkeypatch.setattr(logging.getLogger(_LOGGER), "disabled", False)


@dataclass
class _FakePredict:
    """Return a fixed answer. Raise on the Nth call (0-based) when ``raise_on_call`` is set."""

    slug: str = "marcel"
    confidence: float = 0.91
    raise_on_call: int | None = None
    seen_bytes: list[bytes] = field(default_factory=list)

    def __call__(self, image_path: Path) -> tuple[str, float]:
        call = len(self.seen_bytes)
        self.seen_bytes.append(image_path.read_bytes())
        if call == self.raise_on_call:
            msg = "boom"
            raise RuntimeError(msg)
        return self.slug, self.confidence


def _classifier(predict: PredictFn, subject_ids: dict[str, int] | None = None) -> LiveClassifier:
    return LiveClassifier(
        predict=predict,
        version=_VERSION,
        subject_id_by_slug={"marcel": 1} if subject_ids is None else subject_ids,
        totals=FrameCounts(),
    )


def _frames() -> list[ClipFrame]:
    return [make_clip_frame(1, ordinal) for ordinal in range(3)]


def _classifier_config(make_config: Callable[..., Config], tmp_path: Path) -> Config:
    return make_config(tmp_path / "internal", tmp_path / "storage").model_copy(
        update={"classifier": ClassifierConfig(model="m.pt")},
    )


# --- load_live_classifier ----------------------------------------------------------------------------


def test_load_returns_none_without_classifier_table(
    make_config: Callable[..., Config],
    db_engine: Engine,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without a ``[classifier]`` table, the loader returns ``None`` and logs nothing."""
    config = make_config(tmp_path / "internal", tmp_path / "storage")
    with caplog.at_level(logging.DEBUG):
        assert load_live_classifier(config, db_engine) is None
    assert not caplog.records


def test_load_warns_and_returns_none_when_checkpoint_absent(
    make_config: Callable[..., Config],
    db_engine: Engine,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An absent checkpoint file gives ``None`` and one warning."""
    config = _classifier_config(make_config, tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert load_live_classifier(config, db_engine) is None
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_load_warns_and_returns_none_when_model_load_raises(
    make_config: Callable[..., Config],
    db_engine: Engine,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model load fault gives ``None`` and one warning."""
    config = _classifier_config(make_config, tmp_path)
    models = config.internal_root / "models"
    models.mkdir(parents=True)
    _ = (models / "m.pt").write_bytes(b"stub")
    with (
        patch("cat_watcher.predictions.make_predict_fn", side_effect=RuntimeError("bad weights")),
        caplog.at_level(logging.DEBUG),
    ):
        assert load_live_classifier(config, db_engine) is None
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_load_builds_classifier_with_every_cat_subject(
    make_config: Callable[..., Config],
    db_engine: Engine,
    tmp_path: Path,
) -> None:
    """The loader maps every cat subject, archived ones too, and versions the weights file.

    A checkpoint can still name a cat that the operator archived. That is a real prediction, not a fault.
    """
    config = _classifier_config(make_config, tmp_path)
    models = config.internal_root / "models"
    models.mkdir(parents=True)
    _ = (models / "m.pt").write_bytes(b"stub")
    marcel_id = seed_cat_subject(db_engine, slug="marcel", display_order=1)
    rufus_id = seed_cat_subject(db_engine, slug="rufus", display_name="Rufus", display_order=2)
    _ = add_event_subject(db_engine, slug="visitor", display_order=3)
    gone_id = seed_cat_subject(db_engine, slug="gone", display_name="Gone", display_order=4)
    with get_session(db_engine) as session:
        gone = session.get(Subject, gone_id)
        assert gone is not None
        gone.archived_at = datetime(2026, 1, 1, tzinfo=UTC)
    fake = _FakePredict()
    with patch("cat_watcher.predictions.make_predict_fn", return_value=fake) as make:
        live = load_live_classifier(config, db_engine)
    assert live is not None
    make.assert_called_once_with(models / "m.pt")
    assert live.predict is fake
    assert live.version.startswith("m.pt@")
    assert live.subject_id_by_slug == {"marcel": marcel_id, "rufus": rufus_id, "gone": gone_id}
    assert live.totals == FrameCounts()
    assert live.load_seconds > 0.0


# --- classify_frames ---------------------------------------------------------------------------------


def test_boxed_frame_gets_subject_confidence_and_version() -> None:
    """A boxed frame gets the subject id, the confidence, and the version."""
    frames = _frames()
    counts = classify_frames(scored_frames_with_boxes(gradient_rgb(60, 80)), frames, _classifier(_FakePredict()), clip_label=_LABEL)
    assert (frames[0].predicted_subject_id, frames[0].prediction_confidence, frames[0].classifier_version) == (
        1,
        0.91,
        _VERSION,
    )
    assert (counts.classified, counts.no_box, counts.errored) == (2, 1, 0)


def test_frame_without_box_gets_no_cat_box_state() -> None:
    """A frame with no box keeps the version, and both prediction columns stay ``NULL``."""
    frames = _frames()
    _ = classify_frames(scored_frames_with_boxes(gradient_rgb(60, 80)), frames, _classifier(_FakePredict()), clip_label=_LABEL)
    assert (frames[1].predicted_subject_id, frames[1].prediction_confidence, frames[1].classifier_version) == (
        None,
        None,
        _VERSION,
    )


def test_unknown_slug_stamps_version_without_prediction_and_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    """A slug with no subject gets the version and no prediction, and logs an error.

    The version marks the frame as tried, so ``reanalyze --predictions`` does not select it again.
    """
    frames = _frames()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        counts = classify_frames(
            scored_frames_with_boxes(gradient_rgb(60, 80)),
            frames,
            _classifier(_FakePredict(slug="ghost")),
            clip_label=_LABEL,
        )
    assert (frames[0].predicted_subject_id, frames[0].prediction_confidence, frames[0].classifier_version) == (
        None,
        None,
        _VERSION,
    )
    assert [r.levelno for r in caplog.records] == [logging.ERROR, logging.ERROR]
    assert (counts.classified, counts.no_box, counts.errored) == (0, 1, 2)


def test_predict_failure_on_one_frame_keeps_the_others(caplog: pytest.LogCaptureFixture) -> None:
    """A fault on one frame does not stop the other frames."""
    frames = _frames()
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        counts = classify_frames(
            scored_frames_with_boxes(gradient_rgb(60, 80)),
            frames,
            _classifier(_FakePredict(raise_on_call=0)),
            clip_label=_LABEL,
        )
    assert (frames[0].predicted_subject_id, frames[0].classifier_version) == (None, _VERSION)
    assert frames[2].classifier_version == _VERSION
    assert frames[2].predicted_subject_id == 1
    assert [r.levelno for r in caplog.records] == [logging.ERROR]
    assert caplog.records[0].exc_info is not None
    assert _LABEL in caplog.records[0].getMessage()
    assert "None" not in caplog.records[0].getMessage()
    assert (counts.classified, counts.errored) == (1, 1)


def test_predict_receives_the_shared_crop_bytes(tmp_path: Path) -> None:
    """The predict function reads the same bytes as the shared crop pipeline writes."""
    image = gradient_rgb(60, 80)
    scored = scored_frames_with_boxes(image)
    box = scored[0].box
    assert box is not None
    expected_path = tmp_path / "expected.jpg"
    write_crop_jpeg(crop_frame(image, box)[0], expected_path)
    fake = _FakePredict()
    _ = classify_frames(scored, _frames(), _classifier(fake), clip_label=_LABEL)
    assert fake.seen_bytes[0] == expected_path.read_bytes()


def test_two_calls_add_up_in_totals() -> None:
    """Each call adds its counts and wall time to ``classifier.totals``."""
    classifier = _classifier(_FakePredict())
    scored = scored_frames_with_boxes(gradient_rgb(60, 80))
    first = classify_frames(scored, _frames(), classifier, clip_label=_LABEL)
    second = classify_frames(scored, _frames(), classifier, clip_label=_LABEL)
    assert classifier.totals.classified == first.classified + second.classified == 4
    assert classifier.totals.no_box == 2
    assert classifier.totals.elapsed_seconds == pytest.approx(first.elapsed_seconds + second.elapsed_seconds)


def test_temp_directory_failure_does_not_escape_and_counts_frames() -> None:
    """A temp directory fault stamps the version on every frame without a prediction, and totals still get the counts."""
    classifier = _classifier(_FakePredict())
    frames = _frames()
    with patch("cat_watcher.predictions.tempfile.TemporaryDirectory", side_effect=OSError("disk full")):
        counts = classify_frames(scored_frames_with_boxes(gradient_rgb(60, 80)), frames, classifier, clip_label=_LABEL)
    assert (counts.classified, counts.no_box, counts.errored) == (0, 0, 3)
    assert classifier.totals.errored == 3
    assert [(frame.predicted_subject_id, frame.classifier_version) for frame in frames] == [(None, _VERSION)] * 3


def test_crop_write_failure_on_one_frame_keeps_the_others() -> None:
    """A fault in the crop write for one frame does not stop the others, and totals get the counts."""
    classifier = _classifier(_FakePredict())
    frames = _frames()
    real_write = write_crop_jpeg
    calls: list[int] = []

    def flaky_write(crop_rgb: np.ndarray, dest: Path) -> None:
        calls.append(1)
        if len(calls) == 1:
            msg = "encode failed"
            raise OSError(msg)
        real_write(crop_rgb, dest)

    with patch("cat_watcher.predictions.write_crop_jpeg", flaky_write):
        counts = classify_frames(scored_frames_with_boxes(gradient_rgb(60, 80)), frames, classifier, clip_label=_LABEL)
    assert (frames[0].predicted_subject_id, frames[0].classifier_version) == (None, _VERSION)
    assert frames[2].classifier_version == _VERSION
    assert (counts.classified, counts.no_box, counts.errored) == (1, 1, 1)
    assert classifier.totals.classified == 1
    assert classifier.totals.elapsed_seconds == counts.elapsed_seconds

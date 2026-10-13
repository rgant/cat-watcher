"""Classify the frames of a clip onto their ``ClipFrame`` rows.

The poller loads one :class:`LiveClassifier` at start. :func:`classify_frames` then fills the
prediction columns of each frame. A classifier fault never escapes :func:`classify_frames`, because
a prediction is optional and ingest must still store the clip.
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from sqlalchemy import select

from cat_watcher.classifier.benchmark import make_predict_fn
from cat_watcher.classifier.crop import crop_frame, write_crop_jpeg
from cat_watcher.db import Subject, get_session
from cat_watcher.detector import weights_version

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.engine import Engine

    from cat_watcher.classifier.benchmark import PredictFn
    from cat_watcher.config import Config
    from cat_watcher.db import ClipFrame
    from cat_watcher.detector import ScoredFrame

__all__ = ("FrameCounts", "LiveClassifier", "classify_frames", "load_live_classifier")

logger = logging.getLogger(__name__)

_MODELS_SUBDIR = "models"
_CROP_FILENAME = "crop.jpg"


@dataclass
class FrameCounts:
    """Frame outcomes and wall time. ``errored`` counts frames that stay not classified."""

    classified: int = 0
    no_box: int = 0
    errored: int = 0
    elapsed_seconds: float = 0.0

    def add(self, other: FrameCounts) -> None:
        """Add ``other`` to this count."""
        self.classified += other.classified
        self.no_box += other.no_box
        self.errored += other.errored
        self.elapsed_seconds += other.elapsed_seconds


@dataclass
class LiveClassifier:
    """A loaded checkpoint, its version string, and the cat subjects its slugs map to."""

    predict: PredictFn
    version: str
    subject_id_by_slug: dict[str, int]
    totals: FrameCounts
    load_seconds: float = 0.0


def load_live_classifier(config: Config, engine: Engine) -> LiveClassifier | None:
    """Load the configured checkpoint, or return ``None`` when the poller must not classify.

    ``None`` means no ``[classifier]`` table (silent), a missing checkpoint, or a load fault (one
    warning each). Ingest then stores frames without a prediction.
    """
    if config.classifier is None:
        return None
    model_path = config.internal_root / _MODELS_SUBDIR / config.classifier.model
    if not model_path.is_file():
        logger.warning("classifier checkpoint %s is absent. Frames will not be classified.", model_path)
        return None
    load_started = time.perf_counter()
    try:
        predict = make_predict_fn(model_path)
        version = weights_version(model_path)
        load_seconds = time.perf_counter() - load_started
    except Exception:
        logger.warning("classifier checkpoint %s did not load. Frames will not be classified.", model_path, exc_info=True)
        return None
    with get_session(engine) as session:
        rows = session.execute(
            select(Subject.slug, Subject.id).where(Subject.kind == "cat", Subject.archived_at.is_(None)),
        ).all()
    return LiveClassifier(
        predict=predict,
        version=version,
        subject_id_by_slug=dict(rows),
        totals=FrameCounts(),
        load_seconds=load_seconds,
    )


def classify_frames(
    scored_frames: Sequence[ScoredFrame],
    clip_frames: Sequence[ClipFrame],
    classifier: LiveClassifier,
    *,
    clip_label: str,
) -> FrameCounts:
    """Set the prediction columns of each ``ClipFrame``, matched to its scored frame by ``ordinal``.

    A frame with no box gets the version only. A frame that fails keeps all three columns ``NULL``.
    Counts for this call also go into ``classifier.totals``. ``clip_label`` names the clip in error
    logs, because the clip can lack an id at this point.
    """
    counts = FrameCounts()
    started = time.perf_counter()
    scored_by_ordinal = {scored.ordinal: scored for scored in scored_frames}
    pairs = [(scored_by_ordinal[frame.ordinal], frame) for frame in clip_frames if frame.ordinal in scored_by_ordinal]
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            crop_path = Path(tmp) / _CROP_FILENAME
            for scored, clip_frame in pairs:
                try:
                    outcome = _classify_one(scored, clip_frame, classifier, crop_path, clip_label)
                except Exception:
                    logger.exception("classifier failed on clip %s frame %s", clip_label, clip_frame.ordinal)
                    outcome = "errored"
                if outcome == "classified":
                    counts.classified += 1
                elif outcome == "no_box":
                    counts.no_box += 1
                else:
                    counts.errored += 1
    except Exception:
        # The temp directory failed. Every frame not yet counted stays not classified.
        logger.exception("classifier could not use its temp directory")
        handled = counts.classified + counts.no_box + counts.errored
        counts.errored += max(len(pairs) - handled, 0)
    counts.elapsed_seconds = time.perf_counter() - started
    classifier.totals.add(counts)
    return counts


def _classify_one(
    scored: ScoredFrame,
    clip_frame: ClipFrame,
    classifier: LiveClassifier,
    crop_path: Path,
    clip_label: str,
) -> Literal["classified", "no_box", "errored"]:
    if scored.box is None:
        clip_frame.classifier_version = classifier.version
        return "no_box"
    write_crop_jpeg(crop_frame(scored.frame, scored.box)[0], crop_path)
    slug, confidence = classifier.predict(crop_path)
    subject_id = classifier.subject_id_by_slug.get(slug)
    if subject_id is None:
        logger.error("classifier predicted unknown slug %r for clip %s frame %s", slug, clip_label, clip_frame.ordinal)
        return "errored"
    clip_frame.predicted_subject_id = subject_id
    clip_frame.prediction_confidence = confidence
    clip_frame.classifier_version = classifier.version
    return "classified"

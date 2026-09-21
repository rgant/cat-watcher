"""Score untagged clips with a trained classifier, for a by-eye spot check.

This module writes no database row. :func:`predict_clips` loads each candidate's best frame
through an injected ``FrameSource``. It localizes and crops the frame through an injected
``Localizer``, and classifies the crop through an injected ``PredictFn``. A frame or box miss on
one candidate never stops the batch. That candidate yields ``cat_slug=None`` and ``unsure=True``,
and the run moves to the next one.

The crop pipeline matches :func:`cat_watcher.classifier.dataset.export_dataset` exactly: the same
:func:`~cat_watcher.classifier.geometry.square_pad_box`, and the same ``CROP_MAX_WIDTH`` and
``CROP_QUALITY`` through :func:`~cat_watcher.thumbnails.encode_frame`. A crop that differs from
the training crop makes its prediction meaningless.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from cat_watcher.classifier.dataset import CROP_MAX_WIDTH, CROP_QUALITY
from cat_watcher.classifier.geometry import PAD_FRAC, square_pad_box
from cat_watcher.classifier.labels_query import CatFrameRow
from cat_watcher.thumbnails import encode_frame
from cat_watcher.timefmt import local_stamp

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    import numpy as np

    from cat_watcher.classifier.benchmark import PredictFn
    from cat_watcher.classifier.dataset import ExportSources
    from cat_watcher.classifier.labels_query import ClipCandidate

SPOTCHECK_SUBDIR: str = "classifier/spotcheck"
# A 2-class softmax confidence above chance (0.5) means the model picked one cat over the other
# with some conviction. Used only when neither ``--threshold`` nor a benchmark report is present.
DEFAULT_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class ClipPrediction:
    """One clip's spot-check verdict: the predicted class, its confidence, and where to look."""

    clip_id: int
    camera_name: str
    start_ts: datetime
    cat_slug: str | None  # None when the frame fails to load, or the localizer finds no cat.
    confidence: float  # 0.0 when cat_slug is None.
    unsure: bool  # confidence < threshold, or cat_slug is None.
    # The crop file under PredictOptions.crops_dir, on a successful prediction. On a miss, this
    # field falls back to the clip's own recorded thumbnail, so the operator still has a file to
    # open.
    thumb_relpath: str


@dataclass(frozen=True)
class PredictOptions:
    """The tunable knobs of one predict run."""

    crops_dir: Path
    threshold: float
    pad_frac: float = PAD_FRAC


def predict_clips(
    candidates: list[ClipCandidate],
    *,
    sources: ExportSources,
    predict: PredictFn,
    options: PredictOptions,
) -> list[ClipPrediction]:
    """Score every candidate, in input order. One bad candidate never stops the batch.

    Writes a crop under ``options.crops_dir`` for every candidate whose frame loads and whose box
    localizes. A candidate whose frame fails to load, or whose localizer finds no box, writes no
    crop and yields ``cat_slug=None`` with ``unsure=True``.
    """
    options.crops_dir.mkdir(parents=True, exist_ok=True)
    return [_predict_one(candidate, sources=sources, predict=predict, options=options) for candidate in candidates]


def _predict_one(
    candidate: ClipCandidate,
    *,
    sources: ExportSources,
    predict: PredictFn,
    options: PredictOptions,
) -> ClipPrediction:
    """Load, localize, crop, and classify one candidate. Falls back to a miss on any failed step."""
    loaded = sources.frame_source(_as_frame_row(candidate))
    if loaded is None:
        return _miss_prediction(candidate)
    located = sources.localizer(loaded.image)
    if located is None:
        return _miss_prediction(candidate)

    frame_h, frame_w = cast("tuple[int, int]", loaded.image.shape[:2])
    box_xyxy = square_pad_box(located.box, frame_w=frame_w, frame_h=frame_h, pad_frac=options.pad_frac)
    crop = _crop_to_box(loaded.image, box_xyxy)
    crop_relpath = f"{candidate.clip_id}_{candidate.ordinal}.jpg"
    crop_path = options.crops_dir / crop_relpath
    _encode_crop(crop, crop_path)

    cat_slug, confidence = predict(crop_path)
    return ClipPrediction(
        clip_id=candidate.clip_id,
        camera_name=candidate.camera_name,
        start_ts=candidate.start_ts,
        cat_slug=cat_slug,
        confidence=confidence,
        unsure=confidence < options.threshold,
        thumb_relpath=crop_relpath,
    )


def _as_frame_row(candidate: ClipCandidate) -> CatFrameRow:
    """Adapt a ``ClipCandidate`` into the ``CatFrameRow`` shape a ``FrameSource`` reads.

    ``cat_slug`` plays no part in loading a frame, so this carries an unused placeholder.
    """
    return CatFrameRow(
        clip_id=candidate.clip_id,
        frame_id=candidate.frame_id,
        ordinal=candidate.ordinal,
        t_offset_seconds=candidate.t_offset_seconds,
        cat_slug="",
        clip_file_path=candidate.clip_file_path,
        frame_thumb_path=candidate.frame_thumb_path,
    )


def _miss_prediction(candidate: ClipCandidate) -> ClipPrediction:
    """Build the fallback ``ClipPrediction`` for a candidate with no frame or no box."""
    return ClipPrediction(
        clip_id=candidate.clip_id,
        camera_name=candidate.camera_name,
        start_ts=candidate.start_ts,
        cat_slug=None,
        confidence=0.0,
        unsure=True,
        thumb_relpath=candidate.frame_thumb_path,
    )


def _crop_to_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Slice ``image`` to ``box`` (x1, y1, x2, y2), and copy it clear of the source frame."""
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].copy()


def _encode_crop(crop: np.ndarray, dest: Path) -> None:
    """Write ``crop`` as a JPEG at ``dest``, at the same width and quality the export uses."""
    encode_frame(crop, dest, max_width=CROP_MAX_WIDTH, quality=CROP_QUALITY)


def render_rows(predictions: list[ClipPrediction], *, tz: ZoneInfo) -> str:
    """Render one aligned text row per prediction: clip id, local start time, camera, slug, confidence, and an unsure marker."""
    return "\n".join(_render_one_row(prediction, tz=tz) for prediction in predictions)


def _render_one_row(prediction: ClipPrediction, *, tz: ZoneInfo) -> str:
    slug = prediction.cat_slug if prediction.cat_slug is not None else "?"
    marker = "UNSURE" if prediction.unsure else ""
    return (
        f"clip={prediction.clip_id:<6} {local_stamp(prediction.start_ts, tz=tz)}  "
        f"camera={prediction.camera_name:<12} slug={slug:<8} confidence={prediction.confidence:.2f} {marker}"
    )


__all__ = ["DEFAULT_THRESHOLD", "SPOTCHECK_SUBDIR", "ClipPrediction", "PredictOptions", "predict_clips", "render_rows"]

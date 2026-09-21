"""Score untagged clips with a trained classifier, for a by-eye spot check.

This module writes no database row. :func:`predict_clips` scores every frame of each candidate
clip, then groups the frames back under their clip. It loads each frame through an injected
``FrameSource``, localizes and crops it through an injected ``Localizer``, and classifies the crop
through an injected ``PredictFn``. A frame or box miss on one frame never stops the batch. That
frame yields ``cat_slug=None`` and ``unsure=True``, and the run moves to the next one.

Every frame is scored because one clip can hold two cats at different times. A clip scored from a
single frame reports one cat and hides the other.

The crop pipeline matches :func:`cat_watcher.classifier.dataset.export_dataset` exactly: the same
:func:`~cat_watcher.classifier.geometry.square_pad_box`, and the same ``CROP_MAX_WIDTH`` and
``CROP_QUALITY`` through :func:`~cat_watcher.thumbnails.encode_frame`. A crop that differs from
the training crop makes its prediction meaningless.
"""

from collections import Counter
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
    from cat_watcher.classifier.labels_query import ClipFrameCandidate

SPOTCHECK_SUBDIR: str = "classifier/spotcheck"
# A 2-class softmax confidence above chance (0.5) means the model picked one cat over the other
# with some conviction. Used only when neither ``--threshold`` nor a benchmark report is present.
DEFAULT_THRESHOLD: float = 0.5
# Printed in place of a cat slug when a frame fails to load, or the localizer finds no cat.
MISS_SLUG: str = "?"


@dataclass(frozen=True)
class FramePrediction:
    """One frame's verdict. Its clip identity lives on the owning :class:`ClipVerdict`."""

    ordinal: int
    cat_slug: str | None  # None when the frame fails to load, or the localizer finds no cat.
    confidence: float  # 0.0 when cat_slug is None.
    unsure: bool  # confidence < threshold, or cat_slug is None.
    # The crop file under PredictOptions.crops_dir, on a successful prediction. On a miss, this
    # field falls back to the clip's own recorded thumbnail, so the operator still has a file to
    # open.
    thumb_relpath: str


@dataclass(frozen=True)
class ClipVerdict:
    """One clip's spot-check result, built from every frame the clip holds."""

    clip_id: int
    camera_name: str
    start_ts: datetime
    frames: tuple[FramePrediction, ...]

    @property
    def named_frames(self) -> tuple[FramePrediction, ...]:
        """The frames that produced a cat.

        The poller samples frames across the whole clip, so a cat is absent from some of them.
        Such a frame carries no opinion. It must not count as a vote, and it must not count as
        disagreement.
        """
        return tuple(frame for frame in self.frames if frame.cat_slug is not None)

    @property
    def cat_counts(self) -> tuple[tuple[str, int], ...]:
        """Frame count for each cat named, most frequent first.

        A tie sorts by slug, so the order stays stable between runs.
        """
        counts = Counter(cast("str", frame.cat_slug) for frame in self.named_frames)
        return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    @property
    def is_mixed(self) -> bool:
        """The frames name two or more different cats.

        This is the case a per-clip verdict cannot express. Both cats used the box in one clip.
        """
        return len(self.cat_counts) > 1

    @property
    def top_slug(self) -> str | None:
        """The cat the most frames name. If no frame named one, this is ``None``."""
        return self.cat_counts[0][0] if self.cat_counts else None

    @property
    def min_confidence(self) -> float:
        """The lowest confidence among the frames that named a cat. ``0.0`` when none did."""
        return min((frame.confidence for frame in self.named_frames), default=0.0)

    @property
    def unsure(self) -> bool:
        """No frame named a cat, or a frame that named one scored below the threshold."""
        return not self.named_frames or any(frame.unsure for frame in self.named_frames)


@dataclass(frozen=True)
class PredictOptions:
    """The tunable knobs of one predict run."""

    crops_dir: Path
    threshold: float
    pad_frac: float = PAD_FRAC


def predict_clips(
    candidates: list[ClipFrameCandidate],
    *,
    sources: ExportSources,
    predict: PredictFn,
    options: PredictOptions,
) -> list[ClipVerdict]:
    """Score every candidate frame, then group the frames under their clip.

    Clips come back in the order their first frame appears in ``candidates``. Frames keep their
    input order inside each clip. One bad frame never stops the batch.

    Every frame that loads and localizes gets a crop under ``options.crops_dir``. A frame that
    fails either step writes no crop and yields ``cat_slug=None`` with ``unsure=True``.
    """
    options.crops_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[ClipFrameCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.clip_id, []).append(candidate)
    return [
        ClipVerdict(
            clip_id=clip_id,
            camera_name=group[0].camera_name,
            start_ts=group[0].start_ts,
            frames=tuple(_predict_one(candidate, sources=sources, predict=predict, options=options) for candidate in group),
        )
        for clip_id, group in grouped.items()
    ]


def _predict_one(
    candidate: ClipFrameCandidate,
    *,
    sources: ExportSources,
    predict: PredictFn,
    options: PredictOptions,
) -> FramePrediction:
    """Load, localize, crop, and classify one frame. Falls back to a miss on any failed step."""
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
    return FramePrediction(
        ordinal=candidate.ordinal,
        cat_slug=cat_slug,
        confidence=confidence,
        unsure=confidence < options.threshold,
        thumb_relpath=crop_relpath,
    )


def _as_frame_row(candidate: ClipFrameCandidate) -> CatFrameRow:
    """Adapt a ``ClipFrameCandidate`` into the ``CatFrameRow`` shape a ``FrameSource`` reads.

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


def _miss_prediction(candidate: ClipFrameCandidate) -> FramePrediction:
    """Build the fallback ``FramePrediction`` for a frame with no image or no box."""
    return FramePrediction(
        ordinal=candidate.ordinal,
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


def render_rows(verdicts: list[ClipVerdict], *, tz: ZoneInfo) -> str:
    """Render each clip as one summary line, and expand a clip that names two cats.

    Only a mixed clip expands. Its frames disagree, so the operator must see which frame held
    which cat. Every other clip prints one line, because a single cat and a count carry the whole
    verdict.
    """
    return "\n".join(_render_verdict(verdict, tz=tz) for verdict in verdicts)


def _render_verdict(verdict: ClipVerdict, *, tz: ZoneInfo) -> str:
    """Render one clip's summary line. A clip that names two cats also gets a line per frame."""
    head = _render_head(verdict, tz=tz)
    if not verdict.is_mixed:
        return head
    return "\n".join([head, *(_render_frame(frame) for frame in verdict.frames)])


def _render_head(verdict: ClipVerdict, *, tz: ZoneInfo) -> str:
    """Render one clip's summary line.

    The shapes, in the order this function tests them:

    * Two or more cats: ``MIXED`` and the per-cat frame counts. This is the case to look at.
    * No frame named a cat: ``?`` and ``0/<total>``.
    * One cat: the slug, ``<named>/<total>``, and the lowest confidence of the frames that named
      it. ``<total>`` counts every sampled frame, so ``3/5`` means two frames held no cat.
    """
    prefix = f"clip={verdict.clip_id:<6} {local_stamp(verdict.start_ts, tz=tz)}  camera={verdict.camera_name:<12}"
    marker = " UNSURE" if verdict.unsure else ""
    total = len(verdict.frames)
    if verdict.is_mixed:
        tally = ", ".join(f"{slug} {count}" for slug, count in verdict.cat_counts)
        return f"{prefix} {'MIXED':<8} {tally} of {total}{marker}"
    slug = verdict.top_slug
    if slug is None:
        return f"{prefix} {MISS_SLUG:<8} 0/{total}{marker}"
    return f"{prefix} {slug:<8} {verdict.cat_counts[0][1]}/{total}  conf={verdict.min_confidence:.2f}{marker}"


def _render_frame(frame: FramePrediction) -> str:
    """Render one frame of an expanded clip, indented under its clip's summary line."""
    slug = frame.cat_slug if frame.cat_slug is not None else MISS_SLUG
    marker = " UNSURE" if frame.unsure else ""
    return f"    ord={frame.ordinal:<3} {slug:<8} conf={frame.confidence:.2f}{marker}"


__all__ = [
    "DEFAULT_THRESHOLD",
    "MISS_SLUG",
    "SPOTCHECK_SUBDIR",
    "ClipVerdict",
    "FramePrediction",
    "PredictOptions",
    "predict_clips",
    "render_rows",
]

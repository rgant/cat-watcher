"""JPEG encoding and per-frame storage-path helpers for detector-sampled clip frames.

The detector produces one ndarray per sampled frame plus a YOLO max-cat score; this module
turns each ndarray into a downsized JPEG thumbnail on disk and emits parallel
:class:`FrameRecord` entries that the caller persists as ``ClipFrame`` rows. ``Clip.thumb_path``
is then set to whichever frame wins :func:`best_frame_relpath`.

The ``from PIL import Image`` import is intentionally module-level: both :func:`encode_frame`
and :func:`write_clip_frames` need it, and Pillow has no heavy-payload concern (unlike
``ultralytics`` in :mod:`cat_watcher.detector`, which is lazily imported to keep torch off
agents that don't run inference).
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

    import numpy as np

    from cat_watcher.detector import ScoredFrame


THUMB_QUALITY: int = 80
THUMB_MAX_WIDTH: int = 320
_ORDINAL_WIDTH: int = 2


@dataclass(frozen=True)
class FrameRecord:
    """Per-frame data ready for a ``ClipFrame`` insert."""

    ordinal: int
    t_offset_seconds: float
    score: float
    thumb_relpath: str


def per_clip_thumb_dir(camera_name: str, start_ts_local: datetime) -> str:
    """Return ``thumbs/<camera>/<YYYY-MM-DD>/<HHMMSS>`` (POSIX separators, no trailing slash)."""
    date_dir = start_ts_local.strftime("%Y-%m-%d")
    hhmmss = start_ts_local.strftime("%H%M%S")
    return f"thumbs/{camera_name}/{date_dir}/{hhmmss}"


def per_frame_thumb_relpath(per_clip_dir: str, ordinal: int) -> str:
    """Compose ``<per_clip_dir>/<NN>.jpg`` with ``ordinal`` zero-padded to ``_ORDINAL_WIDTH``."""
    return f"{per_clip_dir}/{ordinal:0{_ORDINAL_WIDTH}d}.jpg"


def encode_frame(
    frame: np.ndarray,
    dest: Path,
    *,
    quality: int = THUMB_QUALITY,
    max_width: int = THUMB_MAX_WIDTH,
) -> None:
    """Encode an RGB24 ndarray to a JPEG at ``dest``, fsynced before return.

    Resizes so the long edge equals ``max_width`` (preserving aspect ratio); never upscales.
    The fsync mirrors :func:`cat_watcher.poller.extract_thumbnail` so a later DB row pointing
    at this file never references partial bytes after a crash. Caller must have created
    ``dest.parent``.
    """
    img = Image.fromarray(frame, mode="RGB")
    # ``Image.thumbnail`` is in-place, preserves aspect ratio, and is a no-op when the source
    # is already smaller than the box — exactly the contract we want.
    img.thumbnail((max_width, max_width))
    img.save(str(dest), format="JPEG", quality=quality, optimize=False)
    fd = os.open(dest, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_clip_frames(
    scored_frames: Sequence[ScoredFrame],
    *,
    storage_root: Path,
    per_clip_dir: str,
) -> list[FrameRecord]:
    """Encode every :class:`ScoredFrame` to its computed relpath; return ``FrameRecord``s ordered by ``ordinal``."""
    ordered = sorted(scored_frames, key=_scored_frame_ordinal)
    records: list[FrameRecord] = []
    for scored in ordered:
        relpath = per_frame_thumb_relpath(per_clip_dir, scored.ordinal)
        dest = storage_root / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        encode_frame(scored.frame, dest)
        records.append(
            FrameRecord(
                ordinal=scored.ordinal,
                t_offset_seconds=scored.t_offset_seconds,
                score=scored.score,
                thumb_relpath=relpath,
            ),
        )
    return records


def best_frame_relpath(records: Sequence[FrameRecord]) -> str:
    """Return the ``thumb_relpath`` of the highest-scoring record; ties go to lowest ``ordinal``.

    Raises ``ValueError`` on empty input — the caller must use the no-detect fallback there.
    """
    if not records:
        msg = "best_frame_relpath called on empty records sequence"
        raise ValueError(msg)
    return max(records, key=_score_then_neg_ordinal).thumb_relpath


def _scored_frame_ordinal(scored: ScoredFrame) -> int:
    return scored.ordinal


def _score_then_neg_ordinal(record: FrameRecord) -> tuple[float, int]:
    # Higher score wins; on ties, the negation flips ``max`` into preferring the smaller ordinal.
    return (record.score, -record.ordinal)

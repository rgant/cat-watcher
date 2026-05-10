"""Tests for cat_watcher.thumbnails."""

from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from PIL import Image

from cat_watcher.detector import ScoredFrame
from cat_watcher.thumbnails import (
    THUMB_MAX_WIDTH,
    FrameRecord,
    best_frame_relpath,
    encode_frame,
    per_clip_thumb_dir,
    per_frame_thumb_relpath,
    write_clip_frames,
)

if TYPE_CHECKING:
    from pathlib import Path


def _gradient_rgb(height: int, width: int) -> np.ndarray:
    """Deterministic RGB24 ndarray with a horizontal gradient — gives JPEG something to compress."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[np.newaxis, :]
    arr[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, np.newaxis]
    arr[..., 2] = 128
    return arr


def test_encode_frame_writes_valid_jpeg(tmp_path: Path) -> None:
    """An over-width frame produces a JPEG and gets resized down to ``THUMB_MAX_WIDTH``."""
    arr = _gradient_rgb(270, 480)
    dest = tmp_path / "out.jpg"

    encode_frame(arr, dest)

    assert dest.exists()
    assert dest.read_bytes()[:2] == b"\xff\xd8"
    with Image.open(dest) as img:
        assert img.width <= THUMB_MAX_WIDTH


def test_encode_frame_preserves_aspect_ratio(tmp_path: Path) -> None:
    """A frame under ``THUMB_MAX_WIDTH`` must not be upscaled — original dimensions preserved."""
    arr = _gradient_rgb(100, 200)
    dest = tmp_path / "small.jpg"

    encode_frame(arr, dest)

    with Image.open(dest) as img:
        assert img.width == 200
        assert img.height == 100


def test_per_clip_thumb_dir_uses_local_date_and_time() -> None:
    """Per-clip thumb dir uses camera-local date + HHMMSS with POSIX separators."""
    when = datetime(2026, 5, 8, 10, 30, 45, tzinfo=ZoneInfo("America/New_York"))
    assert per_clip_thumb_dir("pantry", when) == "thumbs/pantry/2026-05-08/103045"


def test_per_frame_thumb_relpath_zero_pads_ordinal() -> None:
    """Single-digit ordinals are zero-padded to width 2 so lexicographic order matches numeric order."""
    assert per_frame_thumb_relpath("thumbs/pantry/2026-05-08/103045", 3) == "thumbs/pantry/2026-05-08/103045/03.jpg"


def test_best_frame_relpath_picks_max_score() -> None:
    """The chosen thumbnail is the frame with the highest score, regardless of position in the input list."""
    records = [
        FrameRecord(ordinal=0, t_offset_seconds=0.0, score=0.1, thumb_relpath="a/00.jpg"),
        FrameRecord(ordinal=1, t_offset_seconds=1.0, score=0.9, thumb_relpath="a/01.jpg"),
        FrameRecord(ordinal=2, t_offset_seconds=2.0, score=0.4, thumb_relpath="a/02.jpg"),
    ]
    assert best_frame_relpath(records) == "a/01.jpg"


def test_best_frame_relpath_breaks_ties_by_ordinal() -> None:
    """Ties on score break to the lowest ordinal so ranked output is reproducible."""
    records = [
        FrameRecord(ordinal=5, t_offset_seconds=5.0, score=0.5, thumb_relpath="a/05.jpg"),
        FrameRecord(ordinal=2, t_offset_seconds=2.0, score=0.5, thumb_relpath="a/02.jpg"),
    ]
    assert best_frame_relpath(records) == "a/02.jpg"


def test_best_frame_relpath_raises_on_empty() -> None:
    """An empty input is a programmer error — the caller must apply the no-detect fallback first."""
    with pytest.raises(ValueError, match="empty"):
        _ = best_frame_relpath([])


def test_write_clip_frames_emits_records_in_ordinal_order(tmp_path: Path) -> None:
    """Out-of-order input is sorted; one JPEG per frame lands at ``<per_clip_dir>/<NN>.jpg``."""
    per_clip_dir = "thumbs/pantry/2026-05-08/103045"
    frame_a = _gradient_rgb(120, 200)
    frame_b = _gradient_rgb(120, 200)
    frame_c = _gradient_rgb(120, 200)
    # Non-sorted insertion order exercises the sort.
    scored = [
        ScoredFrame(ordinal=2, t_offset_seconds=2.0, score=0.40, frame=frame_c),
        ScoredFrame(ordinal=0, t_offset_seconds=0.0, score=0.10, frame=frame_a),
        ScoredFrame(ordinal=1, t_offset_seconds=1.0, score=0.90, frame=frame_b),
    ]
    (tmp_path / per_clip_dir).mkdir(parents=True)

    records = write_clip_frames(scored, storage_root=tmp_path, per_clip_dir=per_clip_dir)

    assert [r.ordinal for r in records] == [0, 1, 2]
    for record in records:
        expected = f"{per_clip_dir}/{record.ordinal:02d}.jpg"
        assert record.thumb_relpath == expected
        assert (tmp_path / record.thumb_relpath).exists()

"""Unit tests for :mod:`cat_watcher.classifier.crop`."""

from typing import TYPE_CHECKING

import numpy as np
from image_helpers import gradient_rgb
from PIL import Image

from cat_watcher.classifier.crop import CROP_MAX_WIDTH, CROP_QUALITY, crop_frame, write_crop_jpeg
from cat_watcher.classifier.geometry import square_pad_box
from cat_watcher.thumbnails import encode_frame

if TYPE_CHECKING:
    from pathlib import Path


def test_crop_frame_returns_the_slice_and_the_square_pad_box() -> None:
    """The crop is the frame slice at the box that ``square_pad_box`` names."""
    frame = gradient_rgb(80, 100)
    box = (60.0, 40.0, 90.0, 70.0)
    crop, padded = crop_frame(frame, box)
    assert padded == square_pad_box(box, frame_w=100, frame_h=80)
    x1, y1, x2, y2 = padded
    assert np.array_equal(crop, frame[y1:y2, x1:x2])


def test_crop_frame_copies_the_slice_clear_of_the_frame() -> None:
    """A write to the crop must not change the source frame."""
    frame = gradient_rgb(80, 100)
    before = frame.copy()
    crop, _ = crop_frame(frame, (10.0, 10.0, 40.0, 40.0))
    crop[:] = 0
    assert np.array_equal(frame, before)


def test_crop_frame_applies_pad_frac() -> None:
    """A larger ``pad_frac`` gives a larger padded box."""
    frame = gradient_rgb(200, 200)
    box = (80.0, 80.0, 120.0, 120.0)
    _, tight = crop_frame(frame, box, pad_frac=0.0)
    _, loose = crop_frame(frame, box, pad_frac=0.5)
    assert tight == square_pad_box(box, frame_w=200, frame_h=200, pad_frac=0.0)
    assert loose[2] - loose[0] > tight[2] - tight[0]


def test_write_crop_jpeg_uses_crop_width_and_quality(tmp_path: Path) -> None:
    """The file equals ``encode_frame`` at ``CROP_MAX_WIDTH`` and ``CROP_QUALITY``."""
    crop = gradient_rgb(500, 500)
    dest = tmp_path / "crop.jpg"
    write_crop_jpeg(crop, dest)
    reference = tmp_path / "reference.jpg"
    encode_frame(crop, reference, max_width=CROP_MAX_WIDTH, quality=CROP_QUALITY)
    assert dest.read_bytes() == reference.read_bytes()
    with Image.open(dest) as saved:
        assert saved.size[0] == CROP_MAX_WIDTH


def test_crop_constants_hold_the_required_values() -> None:
    """The long edge is 256 px and the JPEG quality is 92."""
    assert CROP_MAX_WIDTH == 256
    assert CROP_QUALITY == 92

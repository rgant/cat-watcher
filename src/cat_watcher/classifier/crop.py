"""One crop pipeline for dataset export, spot-check predict, and the poller.

Training crops and scored crops must be byte-identical for the same frame and box. A crop that
differs from the training crop makes its prediction meaningless.
"""

from typing import TYPE_CHECKING, cast

from cat_watcher.classifier.geometry import PAD_FRAC, square_pad_box
from cat_watcher.thumbnails import encode_frame

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

CROP_MAX_WIDTH: int = 256
CROP_QUALITY: int = 92


def crop_frame(
    image_rgb: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    *,
    pad_frac: float = PAD_FRAC,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return the square padded crop of ``box_xyxy`` and the padded box.

    The crop is a copy, so a write to it cannot change ``image_rgb``.
    """
    frame_h, frame_w = cast("tuple[int, int]", image_rgb.shape[:2])
    padded = square_pad_box(box_xyxy, frame_w=frame_w, frame_h=frame_h, pad_frac=pad_frac)
    x1, y1, x2, y2 = padded
    return image_rgb[y1:y2, x1:x2].copy(), padded


def write_crop_jpeg(crop_rgb: np.ndarray, dest: Path) -> None:
    """Write ``crop_rgb`` as a JPEG at ``dest``, at ``CROP_MAX_WIDTH`` and ``CROP_QUALITY``."""
    encode_frame(crop_rgb, dest, max_width=CROP_MAX_WIDTH, quality=CROP_QUALITY)

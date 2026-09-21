"""Convert a YOLO detection box into a square crop rectangle.

No I/O and no third-party imports. Pure geometry, shared by dataset export and inference.
"""

PAD_FRAC = 0.12


def square_pad_box(
    box: tuple[float, float, float, float],
    *,
    frame_w: int,
    frame_h: int,
    pad_frac: float = PAD_FRAC,
) -> tuple[int, int, int, int]:
    """Return an integer square crop centered on ``box``, clamped to the frame bounds.

    The crop side equals the longer edge of ``box`` times ``(1 + pad_frac)``. A clamp at a frame
    edge can shrink one side, so the result is not always a perfect square.
    """
    x1, y1, x2, y2 = box
    half = max(x2 - x1, y2 - y1) * (1 + pad_frac) / 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    crop_x1, crop_x2 = _clamp_axis(cx - half, cx + half, frame_w)
    crop_y1, crop_y2 = _clamp_axis(cy - half, cy + half, frame_h)
    return crop_x1, crop_y1, crop_x2, crop_y2


def _clamp_axis(low: float, high: float, bound: int) -> tuple[int, int]:
    """Clamp both ends of one axis into ``[0, bound]``, then widen a collapsed axis by 1px.

    A box that sits outside the frame, or is narrower than 1px, clamps both ends to the
    same edge. A caller slicing that as a crop needs an ordered, non-empty range instead.
    """
    lo = max(0, min(bound, round(low)))
    hi = max(0, min(bound, round(high)))
    if lo == hi:
        if lo == bound:
            lo -= 1
        else:
            hi += 1
    return lo, hi

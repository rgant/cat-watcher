"""Unit tests for :mod:`cat_watcher.classifier.geometry`."""

from cat_watcher.classifier.geometry import square_pad_box


def test_square_pad_box_keeps_center_and_grows_a_square_box_by_pad_frac() -> None:
    """A centered square box in a large frame keeps its center. The side grows by ``pad_frac``."""
    assert square_pad_box((400.0, 400.0, 600.0, 600.0), frame_w=1000, frame_h=1000) == (388, 388, 612, 612)


def test_square_pad_box_squares_a_wide_box_to_its_longer_edge() -> None:
    """A wide box returns a square crop whose side equals the longer edge times ``(1 + pad_frac)``."""
    result = square_pad_box((300.0, 450.0, 700.0, 550.0), frame_w=1000, frame_h=1000)
    assert result == (276, 276, 724, 724)
    x1, y1, x2, y2 = result
    assert x2 - x1 == y2 - y1


def test_square_pad_box_clamps_top_left_to_zero() -> None:
    """A box near the top-left corner clamps ``x1``/``y1`` to 0, never negative."""
    x1, y1, _x2, _y2 = square_pad_box((0.0, 0.0, 50.0, 50.0), frame_w=1000, frame_h=1000)
    assert (x1, y1) == (0, 0)


def test_square_pad_box_clamps_bottom_right_to_frame_bounds() -> None:
    """A box near the bottom-right clamps ``x2``/``y2`` to ``frame_w``/``frame_h`` exactly."""
    _x1, _y1, x2, y2 = square_pad_box((950.0, 950.0, 1000.0, 1000.0), frame_w=1000, frame_h=1000)
    assert (x2, y2) == (1000, 1000)


def test_square_pad_box_larger_than_frame_returns_the_full_frame() -> None:
    """A box larger than the frame returns exactly ``(0, 0, frame_w, frame_h)``."""
    assert square_pad_box((-100.0, -100.0, 1100.0, 1100.0), frame_w=1000, frame_h=1000) == (0, 0, 1000, 1000)


def test_square_pad_box_returns_ordered_ints() -> None:
    """Every returned coordinate is an ``int``, and each crop edge is ordered low-to-high."""
    result = square_pad_box((300.0, 450.0, 700.0, 550.0), frame_w=1000, frame_h=1000)
    x1, y1, x2, y2 = result
    assert all(isinstance(v, int) for v in result)
    assert x1 < x2
    assert y1 < y2


def test_square_pad_box_widens_a_sub_pixel_box_to_a_non_empty_crop() -> None:
    """A box under 1px wide still returns an ordered, non-empty crop."""
    x1, y1, x2, y2 = square_pad_box((500.0, 500.0, 500.4, 500.4), frame_w=1920, frame_h=1080)
    assert x1 < x2
    assert y1 < y2


def test_square_pad_box_centered_outside_the_frame_stays_inside_and_ordered() -> None:
    """A box centered entirely outside the frame still returns a crop inside the frame."""
    x1, y1, x2, y2 = square_pad_box((-300.0, -300.0, -250.0, -250.0), frame_w=1920, frame_h=1080)
    assert x1 < x2
    assert y1 < y2
    assert x1 >= 0
    assert x2 <= 1920
    assert y1 >= 0
    assert y2 <= 1080

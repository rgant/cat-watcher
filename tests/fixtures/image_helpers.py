"""Synthetic RGB24 ndarrays shared across the test suite."""

import numpy as np


def gradient_rgb(height: int, width: int) -> np.ndarray:
    """Return a deterministic RGB24 ndarray with a horizontal gradient.

    The gradient gives JPEG something to compress, so an encoded file is not trivially small.
    """
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[np.newaxis, :]
    arr[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, np.newaxis]
    arr[..., 2] = 128
    return arr

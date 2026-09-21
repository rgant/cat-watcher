"""Deterministic stratified train/val/test split for classifier clips.

No I/O and no third-party imports. A fixed seed keeps the split reproducible across a rebuild of
the dataset.
"""

import hashlib
import math
from functools import partial

RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
SEED: int = 1729

_MIN_CLIPS_PER_CLASS = 3  # one clip each for train, val, and test


def split_by_clip(
    clip_class: dict[int, str],
    *,
    ratios: tuple[float, float, float] = RATIOS,
    seed: int = SEED,
) -> dict[int, str]:
    """Return clip_id -> split ('train'|'val'|'test'), stratified per cat slug.

    Within each class the clip ids sort by a seeded hash key, then cut into
    splits by ``ratios``. Every ratio must be non-negative. The tuple must
    also sum to 1.0. Either violation raises ``ValueError``.
    """
    if any(r < 0 for r in ratios):
        msg = f"ratios must not hold a negative member, got {ratios!r}"
        raise ValueError(msg)
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        msg = f"ratios must sum to 1.0, got {ratios!r}"
        raise ValueError(msg)

    by_class: dict[str, list[int]] = {}
    for clip_id, cat in clip_class.items():
        by_class.setdefault(cat, []).append(clip_id)

    assignment: dict[int, str] = {}
    for cat, members in by_class.items():
        ordered = sorted(members, key=partial(_hash_key, seed))
        splits = _splits_for(len(ordered), ratios, cat=cat)
        assignment |= dict(zip(ordered, splits, strict=True))
    return assignment


def _hash_key(seed: int, clip_id: int) -> str:
    """Return the sha256 hex digest of ``seed`` and ``clip_id``.

    A hash key holds forever. A CPython ``shuffle`` result is an interpreter detail, not a
    stable contract, so a Python upgrade can silently move a clip between splits.
    """
    return hashlib.sha256(f"{seed}:{clip_id}".encode()).hexdigest()


def _splits_for(count: int, ratios: tuple[float, float, float], *, cat: str) -> list[str]:
    """Return ``count`` split labels: one clip seeded into each split, then the rest by ``ratios``.

    A floored 15% share is 0 for a class of 6 clips or fewer, so val and test each get one clip
    first. The remaining ``count - 3`` clips split by ``ratios``: floor for val, floor for test,
    everything left over to train. A class of fewer than 3 clips cannot fill all three splits, so
    this raises ``ValueError``. A non-negative ``ratios`` always returns exactly ``count`` labels.
    """
    if count < _MIN_CLIPS_PER_CLASS:
        msg = f"class {cat!r} has {count} clip(s), below the minimum of {_MIN_CLIPS_PER_CLASS} needed to fill train, val, and test"
        raise ValueError(msg)

    remaining = count - _MIN_CLIPS_PER_CLASS
    val_n = 1 + math.floor(remaining * ratios[1])
    test_n = 1 + math.floor(remaining * ratios[2])
    train_n = count - val_n - test_n
    return ["train"] * train_n + ["val"] * val_n + ["test"] * test_n

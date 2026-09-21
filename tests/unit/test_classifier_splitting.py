"""Unit tests for :mod:`cat_watcher.classifier.splitting`."""

import pytest

from cat_watcher.classifier.splitting import split_by_clip


def _balanced_clip_class(per_class: int) -> dict[int, str]:
    """Return ``per_class`` marcel clips and ``per_class`` rufus clips with disjoint ids."""
    marcel = dict.fromkeys(range(per_class), "marcel")
    rufus = {i + per_class: "rufus" for i in range(per_class)}
    return marcel | rufus


def test_split_by_clip_is_deterministic_for_the_same_seed() -> None:
    """Two calls with the same input and seed return identical mappings."""
    clip_class = _balanced_clip_class(50)
    assert split_by_clip(clip_class, seed=7) == split_by_clip(clip_class, seed=7)


def test_split_by_clip_differs_across_seeds() -> None:
    """A different seed produces a different assignment on a large input."""
    clip_class = _balanced_clip_class(100)
    assert split_by_clip(clip_class, seed=1) != split_by_clip(clip_class, seed=2)


def test_split_by_clip_assigns_every_input_clip_exactly_once() -> None:
    """Every clip_id in the input appears exactly once in the output."""
    clip_class = _balanced_clip_class(30)
    result = split_by_clip(clip_class)
    assert len(result) == len(clip_class)
    assert set(result) == set(clip_class)


def test_split_by_clip_stratifies_each_class_across_splits() -> None:
    """For a balanced input, each split holds both classes near the target ratio."""
    clip_class = _balanced_clip_class(100)
    result = split_by_clip(clip_class)
    for cat in ("marcel", "rufus"):
        counts: dict[str, int] = dict.fromkeys(("train", "val", "test"), 0)
        for clip_id, split in result.items():
            if clip_class[clip_id] == cat:
                counts[split] += 1
        assert counts["train"] == pytest.approx(70, abs=1)
        assert counts["val"] == pytest.approx(15, abs=1)
        assert counts["test"] == pytest.approx(15, abs=1)


def test_split_by_clip_produces_disjoint_splits_that_cover_the_input() -> None:
    """The train, val, and test id sets are disjoint and their union covers every clip."""
    clip_class = _balanced_clip_class(40)
    result = split_by_clip(clip_class)
    train_ids = {clip_id for clip_id, split in result.items() if split == "train"}
    val_ids = {clip_id for clip_id, split in result.items() if split == "val"}
    test_ids = {clip_id for clip_id, split in result.items() if split == "test"}
    assert train_ids & val_ids == set()
    assert train_ids & test_ids == set()
    assert val_ids & test_ids == set()
    assert train_ids | val_ids | test_ids == set(clip_class)


def test_split_by_clip_rejects_ratios_that_do_not_sum_to_one() -> None:
    """A ratios tuple that does not sum to 1.0 raises ValueError."""
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        _ = split_by_clip({1: "marcel"}, ratios=(0.5, 0.3, 0.3))


def test_split_by_clip_rejects_a_negative_ratio_with_train_taking_the_loss() -> None:
    """A ratios tuple with a negative train share raises ValueError naming the tuple."""
    with pytest.raises(ValueError, match=r"\(-1\.0, 1\.5, 0\.5\)"):
        _ = split_by_clip(dict.fromkeys(range(10), "marcel"), ratios=(-1.0, 1.5, 0.5))


def test_split_by_clip_rejects_a_negative_ratio_with_val_and_test_taking_the_loss() -> None:
    """A ratios tuple with negative val and test shares raises ValueError."""
    with pytest.raises(ValueError, match="negative"):
        _ = split_by_clip(dict.fromkeys(range(10), "marcel"), ratios=(1.5, -0.25, -0.25))


@pytest.mark.parametrize("ratios", [(0.70, 0.15, 0.15), (0.0, 0.0, 1.0), (0.34, 0.33, 0.33)])
def test_split_by_clip_never_drops_or_duplicates_a_clip_for_any_valid_ratios(ratios: tuple[float, float, float]) -> None:
    """Any ratios tuple that passes validation returns exactly one split per input clip."""
    clip_class = dict.fromkeys(range(50), "marcel")
    result = split_by_clip(clip_class, ratios=ratios)
    assert len(result) == len(clip_class)


@pytest.mark.parametrize("class_size", [3, 4, 6])
def test_split_by_clip_gives_a_small_class_at_least_one_val_and_test_clip(class_size: int) -> None:
    """A class of 3 to 6 clips still puts at least one clip in val and one in test."""
    clip_class = dict.fromkeys(range(class_size), "marcel")
    result = split_by_clip(clip_class)
    counts: dict[str, int] = dict.fromkeys(("train", "val", "test"), 0)
    for split in result.values():
        counts[split] += 1
    assert counts["val"] >= 1
    assert counts["test"] >= 1
    assert sum(counts.values()) == class_size


@pytest.mark.parametrize("class_size", [1, 2])
def test_split_by_clip_rejects_a_class_too_small_to_fill_all_splits(class_size: int) -> None:
    """A class of 1 or 2 clips raises ValueError naming the class slug and the count."""
    clip_class = dict.fromkeys(range(class_size), "marcel")
    with pytest.raises(ValueError, match="marcel"):
        _ = split_by_clip(clip_class)


@pytest.mark.parametrize(
    ("class_size", "expected"),
    [(194, {"train": 136, "val": 29, "test": 29}), (294, {"train": 206, "val": 44, "test": 44})],
)
def test_split_by_clip_leaves_a_large_class_at_its_floor_partition(class_size: int, expected: dict[str, int]) -> None:
    """A large class matches the pre-fix floor partition, unaffected by the seed-then-distribute rule."""
    clip_class = dict.fromkeys(range(class_size), "marcel")
    result = split_by_clip(clip_class)
    counts: dict[str, int] = dict.fromkeys(("train", "val", "test"), 0)
    for split in result.values():
        counts[split] += 1
    assert counts == expected


@pytest.mark.parametrize(
    ("class_size", "ratios", "expected"),
    [
        (3, (0.0, 0.0, 1.0), {"train": 1, "val": 1, "test": 1}),
        (10, (0.0, 0.0, 1.0), {"train": 1, "val": 1, "test": 8}),
    ],
)
def test_split_by_clip_handles_a_zero_ratio_without_crashing(
    class_size: int,
    ratios: tuple[float, float, float],
    expected: dict[str, int],
) -> None:
    """A ratios tuple with a 0.0 share still returns a full split, not a `zip` length crash."""
    clip_class = dict.fromkeys(range(class_size), "marcel")
    result = split_by_clip(clip_class, ratios=ratios)
    counts: dict[str, int] = dict.fromkeys(("train", "val", "test"), 0)
    for split in result.values():
        counts[split] += 1
    assert counts == expected


def test_split_by_clip_matches_a_hand_computed_seeded_assignment() -> None:
    """A fixed seed and a fixed 10-clip input produce this exact known split."""
    clip_class = dict.fromkeys(range(10), "marcel")
    result = split_by_clip(clip_class, seed=1729)
    assert result == {
        3: "train",
        5: "train",
        8: "train",
        9: "train",
        7: "train",
        4: "train",
        0: "val",
        2: "val",
        1: "test",
        6: "test",
    }

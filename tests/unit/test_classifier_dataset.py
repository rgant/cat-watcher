"""Unit tests for :mod:`cat_watcher.classifier.dataset`.

``FrameSource`` and ``Localizer`` are fakes keyed by ``frame_id`` or by image shape, never a
``MagicMock``. This keeps every test independent of call order, so ``pytest-randomly`` cannot
break it.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from image_helpers import gradient_rgb
from PIL import Image

from cat_watcher.classifier.dataset import (
    CROP_MAX_WIDTH,
    CROP_QUALITY,
    DATASET_SUBDIR,
    LOCALIZE_CONF,
    MIN_CROPS_PER_CLASS,
    CropRecord,
    ExportError,
    ExportManifest,
    ExportParams,
    ExportSources,
    ExportSummary,
    LoadedFrame,
    LocalizedBox,
    export_dataset,
    read_manifest,
    write_manifest,
)
from cat_watcher.classifier.geometry import PAD_FRAC, square_pad_box
from cat_watcher.classifier.labels_query import CatFrameRow
from cat_watcher.classifier.splitting import RATIOS, SEED, split_by_clip
from cat_watcher.thumbnails import encode_frame

if TYPE_CHECKING:
    from pathlib import Path

_DEFAULT_SHAPE = (100, 100)
_DEFAULT_BOX = (10.0, 10.0, 40.0, 40.0)


@dataclass
class _FrameSourceStub:
    """Fake ``FrameSource``: returns a canned frame, or ``None``, keyed by ``frame_id``."""

    missing_frame_ids: frozenset[int] = frozenset()
    source_by_frame_id: dict[int, str] = field(default_factory=dict)
    shape_by_frame_id: dict[int, tuple[int, int]] = field(default_factory=dict)
    image_by_frame_id: dict[int, np.ndarray] = field(default_factory=dict)

    def __call__(self, row: CatFrameRow) -> LoadedFrame | None:
        if row.frame_id in self.missing_frame_ids:
            return None
        source = self.source_by_frame_id.get(row.frame_id, "clip")
        if row.frame_id in self.image_by_frame_id:
            return LoadedFrame(image=self.image_by_frame_id[row.frame_id], source=source)
        shape = self.shape_by_frame_id.get(row.frame_id, _DEFAULT_SHAPE)
        image = np.zeros((*shape, 3), dtype=np.uint8)
        return LoadedFrame(image=image, source=source)


@dataclass
class _LocalizerStub:
    """Fake ``Localizer``: returns a canned box, or ``None`` for a frame of ``miss_shapes``."""

    box: tuple[float, float, float, float] = _DEFAULT_BOX
    conf: float = 0.9
    miss_shapes: frozenset[tuple[int, int]] = frozenset()

    def __call__(self, image: np.ndarray) -> LocalizedBox | None:
        if image.shape[:2] in self.miss_shapes:
            return None
        return LocalizedBox(box=self.box, conf=self.conf)


def _row(clip_id: int, ordinal: int, cat_slug: str) -> CatFrameRow:
    """Build one ``CatFrameRow``. ``frame_id`` derives from ``clip_id``/``ordinal``, so it stays unique."""
    return CatFrameRow(
        clip_id=clip_id,
        frame_id=clip_id * 1000 + ordinal,
        ordinal=ordinal,
        t_offset_seconds=float(ordinal),
        cat_slug=cat_slug,
        clip_file_path=f"clips/{clip_id}.mp4",
        frame_thumb_path=f"thumbs/{clip_id}/{ordinal:02d}.jpg",
    )


def _rows_for_clips(cat_slug: str, clip_ids: list[int], *, frames_per_clip: int = 1) -> list[CatFrameRow]:
    """Return one row per ``(clip_id, ordinal)`` pair, all tagged ``cat_slug``."""
    return [_row(clip_id, ordinal, cat_slug) for clip_id in clip_ids for ordinal in range(frames_per_clip)]


def _sources(
    frame_source: _FrameSourceStub | None = None,
    localizer: _LocalizerStub | None = None,
) -> ExportSources:
    """Build an ``ExportSources`` from the two fakes. Each defaults to a plain stub."""
    return ExportSources(
        frame_source=frame_source if frame_source is not None else _FrameSourceStub(),
        localizer=localizer if localizer is not None else _LocalizerStub(),
    )


def test_module_constants_match_the_task_brief() -> None:
    """Every module constant holds the value the brief specifies verbatim."""
    assert DATASET_SUBDIR == "classifier/dataset"
    assert LOCALIZE_CONF == 0.10
    assert MIN_CROPS_PER_CLASS == 20
    assert CROP_MAX_WIDTH == 256
    assert CROP_QUALITY == 92


def test_export_dataset_default_params_matches_the_module_constants() -> None:
    """The ``params`` keyword default is an ``ExportParams()`` built from the module constants.

    This reads ``__kwdefaults__`` directly. ``inspect.signature`` evaluates every annotation,
    including ``CatFrameRow``, which the module imports only under ``TYPE_CHECKING``.
    """
    assert export_dataset.__kwdefaults__ is not None
    default_params = cast("ExportParams", export_dataset.__kwdefaults__["params"])
    assert default_params == ExportParams()
    assert default_params.min_crops_per_class == MIN_CROPS_PER_CLASS


def test_export_dataset_writes_one_crop_file_per_surviving_row(tmp_path: Path) -> None:
    """N clean rows across both cats produce N crop files, and the manifest holds N records."""
    rows = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("rufus", [4, 5, 6])
    manifest = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=tmp_path / "dataset",
        sources=_sources(),
        params=ExportParams(min_crops_per_class=3),
    )
    assert len(manifest.records) == len(rows)
    for record in manifest.records:
        crop_path = tmp_path / "dataset" / record.crop_relpath
        assert crop_path.is_file()
        assert record.crop_relpath == f"{record.split}/{record.cat_slug}/{record.clip_id}_{record.ordinal}.jpg"


def test_export_dataset_excludes_a_row_whose_frame_fails_to_load(tmp_path: Path) -> None:
    """A row whose ``frame_source`` returns ``None`` is excluded, counted as a frame-load failure, and writes no file."""
    rows = _rows_for_clips("marcel", [1, 2, 3, 4]) + _rows_for_clips("rufus", [5, 6, 7])
    missing_row = rows[0]
    manifest = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=tmp_path / "dataset",
        sources=_sources(frame_source=_FrameSourceStub(missing_frame_ids=frozenset({missing_row.frame_id}))),
        params=ExportParams(min_crops_per_class=3),
    )
    assert manifest.summary.frame_load_failures == 1
    assert manifest.summary.localization_misses == 0
    assert manifest.summary.crops_per_class["marcel"] == 3
    assert not any(r.frame_id == missing_row.frame_id for r in manifest.records)
    assert not list((tmp_path / "dataset").rglob(f"{missing_row.clip_id}_*.jpg"))
    # candidates counts every row that entered the localization pass. This is input, not output.
    # It stays 7 (== len(rows)) even though one row misses and only 6 crops get written.
    assert manifest.summary.candidates == len(rows)
    assert manifest.summary.candidates == (
        sum(manifest.summary.crops_per_class.values()) + manifest.summary.localization_misses + manifest.summary.frame_load_failures
    )


def test_export_dataset_excludes_a_row_whose_localizer_finds_nothing(tmp_path: Path) -> None:
    """A row whose ``localizer`` returns ``None`` is excluded and counted as a localization miss, not a frame-load failure."""
    rows = _rows_for_clips("marcel", [1, 2, 3, 4]) + _rows_for_clips("rufus", [5, 6, 7])
    missing_row = rows[0]
    manifest = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=tmp_path / "dataset",
        sources=_sources(
            frame_source=_FrameSourceStub(shape_by_frame_id={missing_row.frame_id: (50, 50)}),
            localizer=_LocalizerStub(miss_shapes=frozenset({(50, 50)})),
        ),
        params=ExportParams(min_crops_per_class=3),
    )
    assert manifest.summary.localization_misses == 1
    assert manifest.summary.frame_load_failures == 0
    assert manifest.summary.crops_per_class["marcel"] == 3


def test_export_dataset_drops_a_clip_whose_rows_name_two_cats(tmp_path: Path) -> None:
    """A clip tagged with both cats contributes no crop. It counts once in ``mixed_class_clips``."""
    mixed_clip_id = 99
    rows = (
        _rows_for_clips("marcel", [1, 2, 3])
        + _rows_for_clips("rufus", [4, 5, 6])
        + [_row(mixed_clip_id, 0, "marcel"), _row(mixed_clip_id, 1, "rufus")]
    )
    manifest = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=tmp_path / "dataset",
        sources=_sources(),
        params=ExportParams(min_crops_per_class=3),
    )
    assert manifest.summary.mixed_class_clips == 1
    assert not any(r.clip_id == mixed_clip_id for r in manifest.records)
    assert not list((tmp_path / "dataset").rglob(f"{mixed_clip_id}_*.jpg"))
    assert manifest.summary.crops_per_class == {"marcel": 3, "rufus": 3}


def test_export_dataset_raises_on_a_slug_not_in_classes(tmp_path: Path) -> None:
    """A surviving row whose ``cat_slug`` is absent from ``classes`` raises ``ExportError``.

    Without this guard, the row writes a fourth ImageFolder directory. ultralytics indexes
    classes by sorted directory name, so that directory shifts every class index.
    """
    rows = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("ghost", [4, 5, 6])
    with pytest.raises(ExportError, match="ghost"):
        _ = export_dataset(
            rows,
            classes=("marcel", "rufus"),
            dataset_root=tmp_path / "dataset",
            sources=_sources(),
            params=ExportParams(min_crops_per_class=3),
        )
    assert not (tmp_path / "dataset").exists()


def test_export_dataset_raises_on_unknown_slug_even_when_its_frame_fails_to_load(tmp_path: Path) -> None:
    """The unknown-class guard runs before the frame load, so a row that fails both still raises.

    Before this ordering fix, a row whose ``frame_source`` returned ``None`` counted as a
    ``frame_load_failures`` miss. The unknown-slug check below it never ran.
    """
    rows = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("ghost", [4, 5, 6])
    ghost_row = next(r for r in rows if r.cat_slug == "ghost")
    with pytest.raises(ExportError, match="ghost"):
        _ = export_dataset(
            rows,
            classes=("marcel", "rufus"),
            dataset_root=tmp_path / "dataset",
            sources=_sources(frame_source=_FrameSourceStub(missing_frame_ids=frozenset({ghost_row.frame_id}))),
            params=ExportParams(min_crops_per_class=3),
        )


def test_export_dataset_keeps_every_clip_in_a_single_split(tmp_path: Path) -> None:
    """No crop's on-disk split diverges from its record.

    Every split gets used, and the assignment matches a direct ``split_by_clip`` call on the
    same clips.
    """
    marcel_ids = list(range(1, 11))
    rufus_ids = list(range(11, 21))
    rows = _rows_for_clips("marcel", marcel_ids, frames_per_clip=2) + _rows_for_clips(
        "rufus",
        rufus_ids,
        frames_per_clip=2,
    )
    manifest = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=tmp_path / "dataset",
        sources=_sources(),
        params=ExportParams(min_crops_per_class=1),
    )
    splits_by_clip: dict[int, set[str]] = {}
    for record in manifest.records:
        assert record.crop_relpath.startswith(f"{record.split}/")
        splits_by_clip.setdefault(record.clip_id, set()).add(record.split)
    assert all(len(splits) == 1 for splits in splits_by_clip.values())
    assert {r.split for r in manifest.records} == {"train", "val", "test"}

    clip_class = dict.fromkeys(marcel_ids, "marcel") | dict.fromkeys(rufus_ids, "rufus")
    expected_assignment = split_by_clip(clip_class, ratios=RATIOS, seed=SEED)
    actual_assignment = {clip_id: next(iter(splits)) for clip_id, splits in splits_by_clip.items()}
    assert actual_assignment == expected_assignment


def test_export_dataset_counts_crops_by_source(tmp_path: Path) -> None:
    """``crops_by_source`` matches the ``clip``/``thumb`` mix of the surviving rows' loaded frames."""
    rows = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("rufus", [4, 5, 6])
    thumb_row = rows[0]
    manifest = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=tmp_path / "dataset",
        sources=_sources(frame_source=_FrameSourceStub(source_by_frame_id={thumb_row.frame_id: "thumb"})),
        params=ExportParams(min_crops_per_class=3),
    )
    assert manifest.summary.crops_by_source == {"clip": len(rows) - 1, "thumb": 1}


def test_export_dataset_summary_classes_matches_the_argument_verbatim(tmp_path: Path) -> None:
    """``summary.classes`` equals the ``classes`` argument, in the same order."""
    rows = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("rufus", [4, 5, 6])
    manifest = export_dataset(
        rows,
        classes=("rufus", "marcel"),
        dataset_root=tmp_path / "dataset",
        sources=_sources(),
        params=ExportParams(min_crops_per_class=3),
    )
    assert manifest.summary.classes == ("rufus", "marcel")


def test_export_dataset_raises_export_error_below_the_floor(tmp_path: Path) -> None:
    """A class with fewer crops than ``min_crops_per_class`` raises ``ExportError`` naming it."""
    rows = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("rufus", [4, 5, 6, 7, 8])
    with pytest.raises(ExportError, match="marcel"):
        _ = export_dataset(
            rows,
            classes=("marcel", "rufus"),
            dataset_root=tmp_path / "dataset",
            sources=_sources(),
            params=ExportParams(min_crops_per_class=5),
        )


def test_export_dataset_raises_export_error_on_empty_rows(tmp_path: Path) -> None:
    """An empty row list raises ``ExportError`` instead of silently writing an empty dataset."""
    with pytest.raises(ExportError, match="marcel"):
        _ = export_dataset(
            [],
            classes=("marcel", "rufus"),
            dataset_root=tmp_path / "dataset",
            sources=_sources(),
            params=ExportParams(min_crops_per_class=1),
        )


def test_export_dataset_raises_when_a_split_lacks_a_class(tmp_path: Path) -> None:
    """An empty export with the floor at 0 clears the class-count check, but no split gets a crop.

    ``min_crops_per_class=0`` reaches the split-coverage check with no test double. This is the
    branch a prior version of this report wrongly called unreachable.
    """
    with pytest.raises(ExportError, match="split"):
        _ = export_dataset(
            [],
            classes=("marcel",),
            dataset_root=tmp_path / "dataset",
            sources=_sources(),
            params=ExportParams(min_crops_per_class=0),
        )


def test_export_dataset_wraps_split_by_clip_value_error_for_a_two_clip_class(tmp_path: Path) -> None:
    """A class with only 2 clips fails in ``split_by_clip``. ``export_dataset`` re-raises as ``ExportError``."""
    rows = _rows_for_clips("marcel", [1, 2])
    with pytest.raises(ExportError, match=r"has 2 clip\(s\)"):
        _ = export_dataset(
            rows,
            classes=("marcel",),
            dataset_root=tmp_path / "dataset",
            sources=_sources(),
            params=ExportParams(min_crops_per_class=1),
        )


def test_export_dataset_box_xyxy_matches_square_pad_box_for_the_frame(tmp_path: Path) -> None:
    """``box_xyxy`` equals ``square_pad_box`` applied to the localizer's box for that frame's size."""
    rows = _rows_for_clips("marcel", [1, 2, 3])
    frame_w, frame_h = 100, 80
    box = (60.0, 40.0, 90.0, 70.0)
    manifest = export_dataset(
        rows,
        classes=("marcel",),
        dataset_root=tmp_path / "dataset",
        sources=_sources(
            frame_source=_FrameSourceStub(shape_by_frame_id=dict.fromkeys((r.frame_id for r in rows), (frame_h, frame_w))),
            localizer=_LocalizerStub(box=box),
        ),
        params=ExportParams(min_crops_per_class=3),
    )
    expected = square_pad_box(box, frame_w=frame_w, frame_h=frame_h, pad_frac=PAD_FRAC)
    record = manifest.records[0]
    assert record.box_xyxy == expected
    with Image.open(tmp_path / "dataset" / record.crop_relpath) as saved:
        assert saved.size == (expected[2] - expected[0], expected[3] - expected[1])


def test_export_dataset_resizes_a_large_crop_to_crop_max_width(tmp_path: Path) -> None:
    """A crop wider than ``CROP_MAX_WIDTH`` is downsized to it, not the thumbnail default of 320."""
    rows = _rows_for_clips("marcel", [1, 2, 3])
    manifest = export_dataset(
        rows,
        classes=("marcel",),
        dataset_root=tmp_path / "dataset",
        sources=_sources(
            frame_source=_FrameSourceStub(shape_by_frame_id=dict.fromkeys((r.frame_id for r in rows), (1000, 1000))),
            localizer=_LocalizerStub(box=(100.0, 100.0, 500.0, 500.0)),
        ),
        params=ExportParams(min_crops_per_class=3),
    )
    with Image.open(tmp_path / "dataset" / manifest.records[0].crop_relpath) as saved:
        assert saved.size[0] == CROP_MAX_WIDTH


def test_export_dataset_writes_crops_at_crop_quality(tmp_path: Path) -> None:
    """A crop is encoded at ``CROP_QUALITY`` (92), not the thumbnail default of 80.

    A flat-color crop compresses to the same bytes at any quality, so this uses a gradient. The
    written file is compared byte-for-byte against ``encode_frame`` run directly on the same
    crop pixels at each quality.
    """
    rows = _rows_for_clips("marcel", [1, 2, 3])
    gradient = gradient_rgb(200, 200)
    manifest = export_dataset(
        rows,
        classes=("marcel",),
        dataset_root=tmp_path / "dataset",
        sources=_sources(
            frame_source=_FrameSourceStub(image_by_frame_id=dict.fromkeys((r.frame_id for r in rows), gradient)),
        ),
        params=ExportParams(min_crops_per_class=3),
    )
    written = (tmp_path / "dataset" / manifest.records[0].crop_relpath).read_bytes()

    x1, y1, x2, y2 = manifest.records[0].box_xyxy
    crop_pixels = gradient[y1:y2, x1:x2]
    at_crop_quality = tmp_path / "at_crop_quality.jpg"
    at_thumb_quality = tmp_path / "at_thumb_quality.jpg"
    encode_frame(crop_pixels, at_crop_quality, max_width=CROP_MAX_WIDTH, quality=CROP_QUALITY)
    encode_frame(crop_pixels, at_thumb_quality, max_width=CROP_MAX_WIDTH, quality=80)

    assert written == at_crop_quality.read_bytes()
    assert written != at_thumb_quality.read_bytes()


def test_write_manifest_then_read_manifest_round_trips(tmp_path: Path) -> None:
    """``write_manifest`` then ``read_manifest`` returns a manifest equal to the original."""
    rows = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("rufus", [4, 5, 6])
    manifest = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=tmp_path / "dataset",
        sources=_sources(),
        params=ExportParams(min_crops_per_class=3),
    )
    dest = tmp_path / "manifest.json"
    write_manifest(manifest, dest)
    assert read_manifest(dest) == manifest


def test_crop_record_carries_frame_id_source_and_yolo_conf(tmp_path: Path) -> None:
    """``CropRecord`` carries ``frame_id``, ``source``, and ``yolo_conf`` from the fakes, unchanged."""
    rows = _rows_for_clips("marcel", [1, 2, 3])
    target_row = rows[0]
    manifest = export_dataset(
        rows,
        classes=("marcel",),
        dataset_root=tmp_path / "dataset",
        sources=_sources(
            frame_source=_FrameSourceStub(source_by_frame_id={target_row.frame_id: "thumb"}),
            localizer=_LocalizerStub(conf=0.77),
        ),
        params=ExportParams(min_crops_per_class=3),
    )
    record = next(r for r in manifest.records if r.frame_id == target_row.frame_id)
    assert record.frame_id == target_row.frame_id
    assert record.source == "thumb"
    assert record.yolo_conf == 0.77


def test_export_dataset_removes_a_stale_crop_on_reexport(tmp_path: Path) -> None:
    """A re-export to the same root drops the crop for a row the new input no longer includes.

    Without this fix, a label correction plus a re-export leaves the old crop on disk.
    ultralytics trains on the directory tree, not the manifest, so it trains on the crop meant
    for removal.
    """
    dataset_root = tmp_path / "dataset"
    rows = _rows_for_clips("marcel", [1, 2, 3, 4]) + _rows_for_clips("rufus", [5, 6, 7])
    first = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=dataset_root,
        sources=_sources(),
        params=ExportParams(min_crops_per_class=3),
    )
    removed_clip_id = 1
    stale_relpath = next(r.crop_relpath for r in first.records if r.clip_id == removed_clip_id)
    assert (dataset_root / stale_relpath).is_file()

    remaining_rows = [r for r in rows if r.clip_id != removed_clip_id]
    second = export_dataset(
        remaining_rows,
        classes=("marcel", "rufus"),
        dataset_root=dataset_root,
        sources=_sources(),
        params=ExportParams(min_crops_per_class=3),
    )
    assert not (dataset_root / stale_relpath).exists()
    assert len(second.records) == len(remaining_rows)


def test_export_dataset_leaves_no_manifest_when_a_write_fails_partway(tmp_path: Path) -> None:
    """A raise partway through the write pass removes the previous run's manifest too.

    Without this, a partial crop tree sits beside the previous run's manifest. A later stage
    reads the tree for images and the manifest for the class order and the hash. It then trains
    on the partial tree, and stamps it with the old hash.
    """
    dataset_root = tmp_path / "dataset"
    rows = _rows_for_clips("marcel", [1, 2, 3, 4, 5])
    _ = export_dataset(
        rows,
        classes=("marcel",),
        dataset_root=dataset_root,
        sources=_sources(),
        params=ExportParams(min_crops_per_class=3),
    )
    assert (dataset_root / "manifest.json").is_file()

    bad_row = rows[2]  # third row: _write_crops fails on the third crop
    bad_image = np.zeros((*_DEFAULT_SHAPE, 3), dtype=np.float64)  # encode_frame requires uint8
    with pytest.raises(TypeError):
        _ = export_dataset(
            rows,
            classes=("marcel",),
            dataset_root=dataset_root,
            sources=_sources(frame_source=_FrameSourceStub(image_by_frame_id={bad_row.frame_id: bad_image})),
            params=ExportParams(min_crops_per_class=3),
        )
    assert not (dataset_root / "manifest.json").exists()


def _export(rows: list[CatFrameRow], dataset_root: Path, *, pad_frac: float = PAD_FRAC) -> ExportManifest:
    """Export ``rows`` with fixed test classes, fakes, and floor. Only ``pad_frac`` varies."""
    return export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=dataset_root,
        sources=_sources(),
        params=ExportParams(min_crops_per_class=3, pad_frac=pad_frac),
    )


def test_export_dataset_is_deterministic_across_repeat_exports(tmp_path: Path) -> None:
    """Two exports of the same rows and seed produce the same split assignment and the same hash."""
    rows = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("rufus", [4, 5, 6])
    manifest_a = _export(rows, tmp_path / "a")
    manifest_b = _export(rows, tmp_path / "b")
    assert manifest_a.summary.dataset_hash == manifest_b.summary.dataset_hash
    assignment_a = {r.clip_id: r.split for r in manifest_a.records}
    assignment_b = {r.clip_id: r.split for r in manifest_b.records}
    assert assignment_a == assignment_b


@pytest.mark.parametrize(
    ("variant_params", "variant_classes"),
    [
        pytest.param(ExportParams(min_crops_per_class=3, pad_frac=0.50), ("marcel", "rufus"), id="pad_frac"),
        pytest.param(ExportParams(min_crops_per_class=3, localize_conf=0.90), ("marcel", "rufus"), id="localize_conf"),
        pytest.param(ExportParams(min_crops_per_class=3, ratios=(0.34, 0.33, 0.33)), ("marcel", "rufus"), id="ratios"),
        pytest.param(ExportParams(min_crops_per_class=3, seed=42), ("marcel", "rufus"), id="seed"),
        pytest.param(ExportParams(min_crops_per_class=3), ("rufus", "marcel"), id="classes_order"),
    ],
)
def test_dataset_hash_differs_when_any_export_parameter_changes(
    tmp_path: Path,
    variant_params: ExportParams,
    variant_classes: tuple[str, str],
) -> None:
    """A change to ``pad_frac``, ``localize_conf``, ``ratios``, ``seed``, or the ``classes`` order each changes the hash."""
    rows = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("rufus", [4, 5, 6])
    baseline = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=tmp_path / "baseline",
        sources=_sources(),
        params=ExportParams(min_crops_per_class=3),
    )
    variant = export_dataset(
        rows,
        classes=variant_classes,
        dataset_root=tmp_path / "variant",
        sources=_sources(),
        params=variant_params,
    )
    assert baseline.summary.dataset_hash != variant.summary.dataset_hash


def test_dataset_hash_differs_when_a_clips_class_and_split_change(tmp_path: Path) -> None:
    """A relabel of which clips belong to which class changes the crop paths, so the hash changes."""
    rows_a = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("rufus", [4, 5, 6])
    rows_b = _rows_for_clips("marcel", [1, 2, 4]) + _rows_for_clips("rufus", [3, 5, 6])
    manifest_a = _export(rows_a, tmp_path / "a")
    manifest_b = _export(rows_b, tmp_path / "b")
    assert manifest_a.summary.dataset_hash != manifest_b.summary.dataset_hash


def test_dataset_hash_differs_when_source_or_box_changes_but_crop_relpath_does_not(tmp_path: Path) -> None:
    """A clip-vs-thumbnail fallback, or a moved localizer box, changes the pixels but not ``crop_relpath``.

    Before this fix, ``_dataset_hash`` read only ``(crop_relpath, cat_slug, split)``. A same-path
    export through a smaller thumbnail and a different box then hashed identically to the
    full-resolution clip export. Same rows, same labels, different training data, one hash.
    """
    rows = _rows_for_clips("marcel", [1, 2, 3]) + _rows_for_clips("rufus", [4, 5, 6])
    frame_ids = [row.frame_id for row in rows]

    clip_manifest = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=tmp_path / "clip",
        sources=_sources(
            frame_source=_FrameSourceStub(shape_by_frame_id=dict.fromkeys(frame_ids, (156, 156))),
            localizer=_LocalizerStub(box=(10.0, 10.0, 40.0, 40.0)),
        ),
        params=ExportParams(min_crops_per_class=3),
    )
    thumb_manifest = export_dataset(
        rows,
        classes=("marcel", "rufus"),
        dataset_root=tmp_path / "thumb",
        sources=_sources(
            frame_source=_FrameSourceStub(
                source_by_frame_id=dict.fromkeys(frame_ids, "thumb"),
                shape_by_frame_id=dict.fromkeys(frame_ids, (90, 90)),
            ),
            localizer=_LocalizerStub(box=(5.0, 5.0, 20.0, 20.0)),
        ),
        params=ExportParams(min_crops_per_class=3),
    )

    assert clip_manifest.summary.dataset_hash != thumb_manifest.summary.dataset_hash


def test_export_summary_records_the_export_parameters() -> None:
    """The dataclass fields exist and hold the values ``export_dataset`` is told to use."""
    summary = ExportSummary(
        classes=("marcel", "rufus"),
        candidates=10,
        crops_per_class={"marcel": 5, "rufus": 5},
        crops_by_source={"clip": 10, "thumb": 0},
        localization_misses=0,
        frame_load_failures=0,
        mixed_class_clips=0,
        seed=1729,
        ratios=(0.7, 0.15, 0.15),
        pad_frac=0.12,
        localize_conf=LOCALIZE_CONF,
        dataset_hash="deadbeef",
    )
    assert summary.localize_conf == LOCALIZE_CONF


def test_crop_record_holds_the_documented_fields() -> None:
    """``CropRecord`` exposes every field the manifest schema documents."""
    record = CropRecord(
        clip_id=1,
        frame_id=2,
        ordinal=0,
        cat_slug="marcel",
        split="train",
        source="clip",
        crop_relpath="train/marcel/1_0.jpg",
        box_xyxy=(0, 0, 10, 10),
        yolo_conf=0.9,
    )
    assert record.crop_relpath == "train/marcel/1_0.jpg"


def test_export_manifest_holds_summary_and_records() -> None:
    """``ExportManifest`` pairs one ``ExportSummary`` with its list of ``CropRecord`` entries."""
    summary = ExportSummary(
        classes=("marcel",),
        candidates=1,
        crops_per_class={"marcel": 1},
        crops_by_source={"clip": 1, "thumb": 0},
        localization_misses=0,
        frame_load_failures=0,
        mixed_class_clips=0,
        seed=1729,
        ratios=(0.7, 0.15, 0.15),
        pad_frac=0.12,
        localize_conf=LOCALIZE_CONF,
        dataset_hash="deadbeef",
    )
    record = CropRecord(
        clip_id=1,
        frame_id=2,
        ordinal=0,
        cat_slug="marcel",
        split="train",
        source="clip",
        crop_relpath="train/marcel/1_0.jpg",
        box_xyxy=(0, 0, 10, 10),
        yolo_conf=0.9,
    )
    manifest = ExportManifest(summary=summary, records=[record])
    assert manifest.records == [record]

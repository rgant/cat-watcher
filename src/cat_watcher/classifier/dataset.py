"""Turn operator-labeled frames into an ImageFolder crop dataset, plus its manifest.

:class:`FrameSource` and :class:`Localizer` inject the frame load and the cat localization. This
lets :func:`export_dataset` run with no ffmpeg and no YOLO.

The export runs in two passes. The first pass loads and localizes every surviving row. The
second pass writes crops only after the floor guard passes. A bad export then leaves no partial
dataset on disk.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import TypeAdapter

from cat_watcher.classifier.geometry import PAD_FRAC, square_pad_box
from cat_watcher.classifier.splitting import RATIOS, SEED, split_by_clip
from cat_watcher.thumbnails import encode_frame

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from cat_watcher.classifier.labels_query import CatFrameRow

DATASET_SUBDIR: str = "classifier/dataset"
LOCALIZE_CONF: float = 0.10
MIN_CROPS_PER_CLASS: int = 20
CROP_MAX_WIDTH: int = 256
CROP_QUALITY: int = 92

_SPLITS: tuple[str, ...] = ("train", "val", "test")
_MANIFEST_FILENAME: str = "manifest.json"


class ExportError(RuntimeError):
    """Raised when the input rows cannot produce a usable dataset."""


@dataclass(frozen=True)
class LoadedFrame:
    """One decoded frame, ready for localization."""

    image: np.ndarray
    source: str  # "clip" | "thumb"


@dataclass(frozen=True)
class LocalizedBox:
    """One YOLO cat detection on a loaded frame."""

    box: tuple[float, float, float, float]
    conf: float


class FrameSource(Protocol):
    """Loads the frame a :class:`CatFrameRow` points at.

    If the frame is unavailable, this returns ``None``.
    """

    def __call__(self, row: CatFrameRow) -> LoadedFrame | None:
        """Load the frame ``row`` points at, or return ``None``."""
        ...


class Localizer(Protocol):
    """Finds a cat box in a loaded frame.

    If it finds no cat, this returns ``None``.
    """

    def __call__(self, image: np.ndarray) -> LocalizedBox | None:
        """Find a cat box in ``image``, or return ``None``."""
        ...


@dataclass(frozen=True)
class CropRecord:  # pylint: disable=too-many-instance-attributes  # flat crop record; the rule targets behavior-rich classes, not data containers
    """One written crop file, ready for a manifest row."""

    clip_id: int
    frame_id: int
    ordinal: int
    cat_slug: str
    split: str
    source: str
    crop_relpath: str
    box_xyxy: tuple[int, int, int, int]
    yolo_conf: float


@dataclass(frozen=True)
class ExportSummary:  # pylint: disable=too-many-instance-attributes  # flat manifest summary; the rule targets behavior-rich classes, not data containers
    """Export-run totals and the parameters that produced them."""

    classes: tuple[str, ...]
    candidates: int  # rows into the localization pass (== sum(crops_per_class) + localization_misses + frame_load_failures)
    crops_per_class: dict[str, int]
    crops_by_source: dict[str, int]
    localization_misses: int  # YOLO at localize_conf found no box
    frame_load_failures: int  # frame_source failed to load the row. It never reached YOLO.
    mixed_class_clips: int
    seed: int
    ratios: tuple[float, float, float]
    pad_frac: float
    localize_conf: float
    dataset_hash: str


@dataclass(frozen=True)
class ExportManifest:
    """A finished export: its summary plus every crop record."""

    summary: ExportSummary
    records: list[CropRecord]


@dataclass(frozen=True)
class _Candidate:
    """One row that survived load and localization, ready for the split and write passes."""

    row: CatFrameRow
    crop: np.ndarray
    box_xyxy: tuple[int, int, int, int]
    conf: float
    source: str


@dataclass(frozen=True)
class _LocalizeCounts:
    """The miss counts and source mix from one localization pass."""

    frame_load_failures: int
    localization_misses: int
    crops_by_source: dict[str, int]


@dataclass(frozen=True)
class _ExportCounts:
    """Every summary count except ``dataset_hash``, which needs the write pass."""

    crops_per_class: dict[str, int]
    crops_by_source: dict[str, int]
    localization_misses: int
    frame_load_failures: int
    mixed_class_clips: int
    candidates: int


@dataclass(frozen=True)
class _PreparedExport:
    """The candidates and split assignment a dataset export is ready to write."""

    candidates: list[_Candidate]
    assignment: dict[int, str]
    counts: _ExportCounts


@dataclass(frozen=True)
class ExportParams:
    """The tunable knobs of one dataset export.

    ``dataset_hash`` covers every field here, plus ``classes``. ``classes`` is hashed
    separately, since class order is not a tuning knob. Two exports with the same rows,
    classes, and params always hash the same.
    """

    ratios: tuple[float, float, float] = RATIOS
    seed: int = SEED
    pad_frac: float = PAD_FRAC
    localize_conf: float = LOCALIZE_CONF
    min_crops_per_class: int = MIN_CROPS_PER_CLASS


@dataclass(frozen=True)
class ExportSources:
    """The injected image-load and cat-localization boundaries one dataset export needs."""

    frame_source: FrameSource
    localizer: Localizer


_MANIFEST_ADAPTER: TypeAdapter[ExportManifest] = TypeAdapter(ExportManifest)

# A frozen-dataclass instance is safe to share across every call, but ruff (B008) still bans a
# call in an argument default. A module-level singleton is its own suggested fix.
_DEFAULT_EXPORT_PARAMS: ExportParams = ExportParams()


def export_dataset(
    rows: list[CatFrameRow],
    *,
    classes: tuple[str, ...],
    dataset_root: Path,
    sources: ExportSources,
    params: ExportParams = _DEFAULT_EXPORT_PARAMS,
) -> ExportManifest:
    """Turn ``rows`` into a crop dataset under ``dataset_root``, and return its manifest.

    A row whose ``cat_slug`` is not in ``classes`` raises ``ExportError``, checked before its
    frame loads. A clip whose rows name more than one cat is dropped whole. It counts once in
    ``mixed_class_clips``. A frame-load failure excludes any other row and counts in
    ``frame_load_failures``. A localization miss excludes a row too, and counts in
    ``localization_misses``. ``split_by_clip`` then assigns each surviving clip a split.

    Each class must reach ``params.min_crops_per_class`` crops. Each split must hold at least
    one crop of every class. This function raises ``ExportError`` otherwise.

    This function clears the previous export under ``dataset_root`` (its split directories and
    its manifest), so a re-export leaves no stale crop and no stale manifest. It then writes
    every crop and ``manifest.json`` under ``dataset_root``.
    """
    prepared = _prepare_export(rows, classes=classes, sources=sources, params=params)
    return _finalize_export(prepared, dataset_root=dataset_root, classes=classes, params=params)


def _prepare_export(
    rows: list[CatFrameRow],
    *,
    classes: tuple[str, ...],
    sources: ExportSources,
    params: ExportParams,
) -> _PreparedExport:
    """Load, localize, and split every row. Enforce the floor guard before any crop is written."""
    clean_rows, mixed_class_clips = _drop_mixed_class_clips(rows)
    candidates, localize_counts = _localize_rows(
        clean_rows,
        classes=classes,
        frame_source=sources.frame_source,
        localizer=sources.localizer,
        pad_frac=params.pad_frac,
    )
    assignment = _split_or_raise(candidates, ratios=params.ratios, seed=params.seed)
    crops_per_class = _count_crops_per_class(candidates, classes)
    _check_floor(crops_per_class, assignment, candidates, classes, params.min_crops_per_class)
    counts = _ExportCounts(
        crops_per_class=crops_per_class,
        crops_by_source=localize_counts.crops_by_source,
        localization_misses=localize_counts.localization_misses,
        frame_load_failures=localize_counts.frame_load_failures,
        mixed_class_clips=mixed_class_clips,
        candidates=len(clean_rows),
    )
    return _PreparedExport(candidates=candidates, assignment=assignment, counts=counts)


def _finalize_export(
    prepared: _PreparedExport,
    *,
    dataset_root: Path,
    classes: tuple[str, ...],
    params: ExportParams,
) -> ExportManifest:
    """Write every candidate's crop, build the manifest, and save it under ``dataset_root``."""
    _clear_stale_splits(dataset_root)
    records = _write_crops(prepared.candidates, prepared.assignment, dataset_root)
    dataset_hash = _dataset_hash(records, classes, params)
    summary = ExportSummary(
        classes=classes,
        candidates=prepared.counts.candidates,
        crops_per_class=prepared.counts.crops_per_class,
        crops_by_source=prepared.counts.crops_by_source,
        localization_misses=prepared.counts.localization_misses,
        frame_load_failures=prepared.counts.frame_load_failures,
        mixed_class_clips=prepared.counts.mixed_class_clips,
        seed=params.seed,
        ratios=params.ratios,
        pad_frac=params.pad_frac,
        localize_conf=params.localize_conf,
        dataset_hash=dataset_hash,
    )
    manifest = ExportManifest(summary=summary, records=records)
    write_manifest(manifest, dataset_root / _MANIFEST_FILENAME)
    return manifest


def _clear_stale_splits(dataset_root: Path) -> None:
    """Remove the previous export under ``dataset_root``: its split directories and its manifest.

    A re-export must leave no stale crop from a row the input no longer includes, and no
    manifest from a run that did not finish. A raise partway through ``_write_crops`` otherwise
    leaves a partial crop tree beside the previous run's manifest. A later stage then trains on
    the partial tree, and stamps it with the old hash. This removes only ``train``, ``val``,
    ``test``, and ``manifest.json``, and never ``dataset_root`` itself.
    """
    for split in _SPLITS:
        split_dir = dataset_root / split
        if split_dir.is_dir():
            shutil.rmtree(split_dir)
    (dataset_root / _MANIFEST_FILENAME).unlink(missing_ok=True)


def _split_or_raise(candidates: list[_Candidate], *, ratios: tuple[float, float, float], seed: int) -> dict[int, str]:
    """Call ``split_by_clip`` on the candidates' clips. Re-raise its ``ValueError`` as ``ExportError``."""
    clip_class = {c.row.clip_id: c.row.cat_slug for c in candidates}
    try:
        return split_by_clip(clip_class, ratios=ratios, seed=seed)
    except ValueError as exc:
        raise ExportError(str(exc)) from exc


def _count_crops_per_class(candidates: list[_Candidate], classes: tuple[str, ...]) -> dict[str, int]:
    """Count surviving crops per ``cat_slug``, seeded at zero for every class in ``classes``."""
    crops_per_class = dict.fromkeys(classes, 0)
    for candidate in candidates:
        crops_per_class[candidate.row.cat_slug] = crops_per_class.get(candidate.row.cat_slug, 0) + 1
    return crops_per_class


def write_manifest(manifest: ExportManifest, dest: Path) -> None:
    """Write ``manifest`` to ``dest`` as JSON."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _ = dest.write_bytes(_MANIFEST_ADAPTER.dump_json(manifest, indent=2))


def read_manifest(src: Path) -> ExportManifest:
    """Read a manifest written by :func:`write_manifest`."""
    return _MANIFEST_ADAPTER.validate_json(src.read_bytes())


def _drop_mixed_class_clips(rows: list[CatFrameRow]) -> tuple[list[CatFrameRow], int]:
    """Return the rows of every single-cat clip, plus a count of the clips dropped for mixing cats."""
    by_clip: dict[int, list[CatFrameRow]] = {}
    for row in rows:
        by_clip.setdefault(row.clip_id, []).append(row)
    clean_rows: list[CatFrameRow] = []
    mixed_class_clips = 0
    for clip_rows in by_clip.values():
        if len({r.cat_slug for r in clip_rows}) > 1:
            mixed_class_clips += 1
        else:
            clean_rows.extend(clip_rows)
    return clean_rows, mixed_class_clips


def _localize_rows(
    rows: list[CatFrameRow],
    *,
    classes: tuple[str, ...],
    frame_source: FrameSource,
    localizer: Localizer,
    pad_frac: float,
) -> tuple[list[_Candidate], _LocalizeCounts]:
    """Load and localize every row. Return the surviving candidates and the miss/source counts."""
    candidates: list[_Candidate] = []
    frame_load_failures = 0
    localization_misses = 0
    crops_by_source: dict[str, int] = dict.fromkeys(("clip", "thumb"), 0)
    for row in rows:
        candidate, miss_kind = _localize_one(
            row,
            classes=classes,
            frame_source=frame_source,
            localizer=localizer,
            pad_frac=pad_frac,
        )
        if candidate is None:
            if miss_kind == "frame_load":
                frame_load_failures += 1
            else:
                localization_misses += 1
            continue
        crops_by_source[candidate.source] = crops_by_source.get(candidate.source, 0) + 1
        candidates.append(candidate)
    return candidates, _LocalizeCounts(
        frame_load_failures=frame_load_failures,
        localization_misses=localization_misses,
        crops_by_source=crops_by_source,
    )


def _localize_one(
    row: CatFrameRow,
    *,
    classes: tuple[str, ...],
    frame_source: FrameSource,
    localizer: Localizer,
    pad_frac: float,
) -> tuple[_Candidate, None] | tuple[None, Literal["frame_load", "localization"]]:
    """Load and localize one row.

    Raises ``ExportError`` first, if ``row``'s ``cat_slug`` is not in ``classes``. This check
    needs only ``row.cat_slug``, so it runs before any load or localization attempt. Otherwise
    returns ``(candidate, None)`` on success, or ``(None, "frame_load")``/``(None,
    "localization")`` naming which pass excluded the row.
    """
    _require_known_class(row.cat_slug, classes)
    loaded = frame_source(row)
    if loaded is None:
        return None, "frame_load"
    located = localizer(loaded.image)
    if located is None:
        return None, "localization"
    frame_h, frame_w = cast("tuple[int, int]", loaded.image.shape[:2])
    box_xyxy = square_pad_box(located.box, frame_w=frame_w, frame_h=frame_h, pad_frac=pad_frac)
    crop = _crop_to_box(loaded.image, box_xyxy)
    return _Candidate(row=row, crop=crop, box_xyxy=box_xyxy, conf=located.conf, source=loaded.source), None


def _require_known_class(cat_slug: str, classes: tuple[str, ...]) -> None:
    """If ``cat_slug`` is not one of ``classes``, this raises ``ExportError``.

    ultralytics indexes classes by sorted directory name, so an unlisted slug shifts every
    class index away from ``summary.classes``.
    """
    if cat_slug not in classes:
        msg = f"class {cat_slug!r} is not one of {classes!r}"
        raise ExportError(msg)


def _crop_to_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Slice ``image`` to ``box`` (x1, y1, x2, y2), and copy it clear of the source frame."""
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].copy()


def _check_floor(
    crops_per_class: dict[str, int],
    assignment: dict[int, str],
    candidates: list[_Candidate],
    classes: tuple[str, ...],
    min_crops_per_class: int,
) -> None:
    """Enforce the class and split floors for a dataset export.

    Each class must reach ``min_crops_per_class`` total crops. Each split must hold at least one
    crop of every class. If either check fails, this raises ``ExportError``.
    """
    for cat_slug in classes:
        count = crops_per_class.get(cat_slug, 0)
        if count < min_crops_per_class:
            msg = f"class {cat_slug!r} has {count} crop(s), below the minimum of {min_crops_per_class}"
            raise ExportError(msg)

    clip_class = {c.row.clip_id: c.row.cat_slug for c in candidates}
    present: dict[str, set[str]] = {split: set() for split in _SPLITS}
    for clip_id, split in assignment.items():
        present[split].add(clip_class[clip_id])
    for split in _SPLITS:
        for cat_slug in classes:
            if cat_slug not in present[split]:
                msg = f"split {split!r} has no crop for class {cat_slug!r}"
                raise ExportError(msg)


def _write_crops(candidates: list[_Candidate], assignment: dict[int, str], dataset_root: Path) -> list[CropRecord]:
    """Write every candidate's crop to its assigned split directory, and return the crop records."""
    records: list[CropRecord] = []
    for candidate in candidates:
        row = candidate.row
        split = assignment[row.clip_id]
        crop_relpath = f"{split}/{row.cat_slug}/{row.clip_id}_{row.ordinal}.jpg"
        dest = dataset_root / crop_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        encode_frame(candidate.crop, dest, max_width=CROP_MAX_WIDTH, quality=CROP_QUALITY)
        records.append(
            CropRecord(
                clip_id=row.clip_id,
                frame_id=row.frame_id,
                ordinal=row.ordinal,
                cat_slug=row.cat_slug,
                split=split,
                source=candidate.source,
                crop_relpath=crop_relpath,
                box_xyxy=candidate.box_xyxy,
                yolo_conf=candidate.conf,
            ),
        )
    return records


def _dataset_hash(records: list[CropRecord], classes: tuple[str, ...], params: ExportParams) -> str:
    """Hash the sorted crop rows, ``classes``, and the export parameters.

    The model filename carries this hash. Two parameter sets must not collide on one filename. A
    ``display_order`` edit reorders ``classes``, so ``classes`` order must move the hash too.
    ``source`` and ``box_xyxy`` cover the pixels: a clip-vs-thumbnail fallback, or a moved box
    from different detector weights, changes the crop without changing ``crop_relpath``.
    ``CROP_MAX_WIDTH`` and ``CROP_QUALITY`` cover the encode step the same way.
    """
    rows_component = sorted((r.crop_relpath, r.cat_slug, r.split, r.source, r.box_xyxy) for r in records)
    payload = {
        "rows": rows_component,
        "classes": list(classes),
        "pad_frac": params.pad_frac,
        "localize_conf": params.localize_conf,
        "ratios": list(params.ratios),
        "seed": params.seed,
        "crop_max_width": CROP_MAX_WIDTH,
        "crop_quality": CROP_QUALITY,
    }
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

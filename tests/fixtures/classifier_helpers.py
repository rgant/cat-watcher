"""Shared test data builders for the classifier package (``dataset``, ``train``, ``benchmark``, ``predict``)."""

import json
from typing import TYPE_CHECKING

from cat_watcher.classifier.dataset import ExportManifest, ExportSummary, write_manifest
from cat_watcher.classifier.geometry import PAD_FRAC
from cat_watcher.classifier.splitting import RATIOS, SEED

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_DATASET_HASH: str = "deadbeef" + "0" * 56


def write_export_manifest(  # noqa: PLR0913  # constructor wrapper; flat kwargs map 1:1 to ExportSummary fields
    dest: Path,
    *,
    classes: tuple[str, ...],
    dataset_hash: str = DEFAULT_DATASET_HASH,
    candidates: int = 0,
    localization_misses: int = 0,
    frame_load_failures: int = 0,
    mixed_class_clips: int = 0,
) -> None:
    """Write a manifest of the shape Task 5's ``export_dataset`` produces, with no crop records.

    A test that only needs the manifest's summary fields, never its per-crop records, writes one
    with this.
    """
    summary = ExportSummary(
        classes=classes,
        candidates=candidates,
        crops_per_class=dict.fromkeys(classes, 0),
        crops_by_source={"clip": 0, "thumb": 0},
        localization_misses=localization_misses,
        frame_load_failures=frame_load_failures,
        mixed_class_clips=mixed_class_clips,
        seed=SEED,
        ratios=RATIOS,
        pad_frac=PAD_FRAC,
        localize_conf=0.10,
        dataset_hash=dataset_hash,
    )
    write_manifest(ExportManifest(summary=summary, records=[]), dest)


def write_checkpoint_with_sidecar(
    path: Path,
    *,
    classes: tuple[str, ...],
    manifest_path: Path,
    dataset_hash: str = DEFAULT_DATASET_HASH,
) -> None:
    """Write a dummy checkpoint plus a sidecar of the shape Task 7's ``train_classifier`` writes."""
    _ = path.write_bytes(b"stub-checkpoint")
    payload: dict[str, object] = {
        "classes": list(classes),
        "model_names": {str(i): slug for i, slug in enumerate(classes)},
        "epochs": 1,
        "imgsz": 64,
        "seed": 1,
        "base_weights": str(path),
        "dataset_hash": dataset_hash,
        "manifest_path": str(manifest_path),
    }
    _ = path.with_suffix(".json").write_text(json.dumps(payload))

"""Train ``yolo11n-cls`` on an exported crop dataset, and save a traceability sidecar.

:class:`YoloClsFactory` injects the model load, so :func:`train_classifier` runs with no real
training. It reads ``classes`` and ``dataset_hash`` from the manifest :mod:`.dataset` wrote, so
the saved checkpoint always names the dataset that produced it. :mod:`.benchmark` reads the
sidecar this module writes, to recover the trained model's own class index map.
"""

import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from cat_watcher.classifier.dataset import read_manifest
from cat_watcher.classifier.splitting import SEED
from cat_watcher.detector import load_yolo

if TYPE_CHECKING:
    from pathlib import Path

BASE_WEIGHTS: str = "yolo11n-cls.pt"
EPOCHS: int = 40
IMGSZ: int = 224

_MODEL_STEM_PREFIX: str = "cat-classifier-"
_HASH_PREFIX_LEN: int = 8
_WEIGHTS_SUBDIR: str = "weights"
_BEST_WEIGHTS_FILENAME: str = "best.pt"


@dataclass(frozen=True)
class TrainResult:
    """One finished training run: where its checkpoint lives, and the fields its sidecar holds."""

    model_path: Path
    sidecar_path: Path
    classes: tuple[str, ...]
    model_names: dict[int, str]
    epochs: int
    imgsz: int
    dataset_hash: str


@dataclass(frozen=True)
class TrainPaths:
    """The filesystem locations one training run reads from and writes to."""

    dataset_root: Path
    manifest_path: Path
    models_dir: Path
    run_dir: Path
    base_weights: Path


@dataclass(frozen=True)
class TrainParams:
    """The tunable knobs of one training run."""

    epochs: int = EPOCHS
    imgsz: int = IMGSZ
    seed: int = SEED


class YoloClsFactory(Protocol):
    """Loads a YOLO classification model from a weights file.

    A bare filename makes ultralytics auto-download into the working directory. The factory
    takes a ``Path``, so a caller keeps every weights file under ``models_dir``.
    """

    def __call__(self, base_weights: Path) -> object:
        """Load the model at ``base_weights``, and return it."""
        ...


class _Trainer(Protocol):
    """The ultralytics trainer field this module reads once ``model.train()`` runs."""

    save_dir: Path


class _TrainingRun(Protocol):
    """The model attributes and methods :func:`train_classifier` reads.

    A local, structural protocol, not the real ``YOLO`` class. :class:`YoloClsFactory` returns
    ``object`` so this module never imports ``ultralytics`` at load time. ``cast`` narrows the
    factory's return value to this protocol at the boundary.
    """

    trainer: _Trainer
    names: dict[int, str]

    def train(self, **kwargs: object) -> object:
        """Train the model, and populate ``self.trainer`` and ``self.names``."""
        ...


def _default_yolo_factory(base_weights: Path) -> object:  # pragma: no cover  # boundary: a test always injects a fake
    """Load the real classification model at ``base_weights`` via :func:`cat_watcher.detector.load_yolo`."""
    return load_yolo(base_weights)


_DEFAULT_TRAIN_PARAMS: TrainParams = TrainParams()


def train_classifier(
    *,
    paths: TrainPaths,
    params: TrainParams = _DEFAULT_TRAIN_PARAMS,
    yolo_factory: YoloClsFactory = _default_yolo_factory,
) -> TrainResult:
    """Train a classifier on ``paths.dataset_root``, and save its best checkpoint.

    Reads ``classes`` and ``dataset_hash`` from ``paths.manifest_path``. Loads the base model
    from ``paths.base_weights``, never from a bare filename, so ultralytics never auto-downloads
    into the working directory. Copies the trainer's best checkpoint to
    ``paths.models_dir / "cat-classifier-<hash8>.pt"``, and writes a JSON sidecar of the same
    stem. When the model's own class names disagree with the manifest's classes, this raises
    ``ValueError``. That mismatch means the dataset on disk changed since the manifest was
    written.
    """
    manifest = read_manifest(paths.manifest_path)
    if not paths.dataset_root.is_dir():
        msg = f"dataset directory not found: {paths.dataset_root}"
        raise FileNotFoundError(msg)

    classes = manifest.summary.classes
    dataset_hash = manifest.summary.dataset_hash

    model = cast("_TrainingRun", yolo_factory(paths.base_weights))
    _ = model.train(
        data=paths.dataset_root,
        epochs=params.epochs,
        imgsz=params.imgsz,
        seed=params.seed,
        project=paths.run_dir,
    )
    model_names = dict(model.names)
    _ensure_names_match(model_names, classes)

    best_weights = model.trainer.save_dir / _WEIGHTS_SUBDIR / _BEST_WEIGHTS_FILENAME
    if not best_weights.is_file():
        msg = f"training did not produce a checkpoint at {best_weights}"
        raise FileNotFoundError(msg)

    paths.models_dir.mkdir(parents=True, exist_ok=True)
    model_stem = f"{_MODEL_STEM_PREFIX}{dataset_hash[:_HASH_PREFIX_LEN]}"
    model_path = paths.models_dir / f"{model_stem}.pt"
    sidecar_path = paths.models_dir / f"{model_stem}.json"

    sidecar_payload: dict[str, object] = {
        "classes": list(classes),
        "model_names": model_names,
        "epochs": params.epochs,
        "imgsz": params.imgsz,
        "seed": params.seed,
        "base_weights": str(paths.base_weights.resolve()),
        "dataset_hash": dataset_hash,
        "manifest_path": str(paths.manifest_path.resolve()),
    }
    _write_checkpoint_and_sidecar(
        model_path,
        sidecar_path,
        best_weights=best_weights,
        sidecar_payload=sidecar_payload,
    )

    return TrainResult(
        model_path=model_path,
        sidecar_path=sidecar_path,
        classes=classes,
        model_names=model_names,
        epochs=params.epochs,
        imgsz=params.imgsz,
        dataset_hash=dataset_hash,
    )


def _ensure_names_match(model_names: dict[int, str], classes: tuple[str, ...]) -> None:
    """If ``model_names``' values do not name the same class set as ``classes``, this raises ``ValueError``.

    Ultralytics indexes its classes by sorted dataset directory name, so ``model_names`` can
    order its classes differently than ``classes``. Only the two sets must agree.
    """
    model_class_set = set(model_names.values())
    manifest_class_set = set(classes)
    if model_class_set != manifest_class_set:
        msg = f"trained model classes {sorted(model_class_set)!r} do not match manifest classes {sorted(manifest_class_set)!r}"
        raise ValueError(msg)


def _write_checkpoint_and_sidecar(
    model_path: Path,
    sidecar_path: Path,
    *,
    best_weights: Path,
    sidecar_payload: dict[str, object],
) -> None:
    """Write the checkpoint and its sidecar. The sidecar replaces its live target first.

    Each file writes to a ``.part`` name first. The sidecar-first order avoids the dangerous
    case. A new checkpoint paired with an old sidecar silently misnames the model's class index
    map. A failure between the two renames instead leaves a new sidecar beside the old
    checkpoint. This removes a leftover ``.part`` file on any failure, including a failed
    ``replace``.
    """
    model_tmp = model_path.with_name(model_path.name + ".part")
    sidecar_tmp = sidecar_path.with_name(sidecar_path.name + ".part")
    try:
        _ = shutil.copyfile(best_weights, model_tmp)
        _ = sidecar_tmp.write_text(json.dumps(sidecar_payload, indent=2))
        _ = sidecar_tmp.replace(sidecar_path)
        _ = model_tmp.replace(model_path)
    finally:
        model_tmp.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)

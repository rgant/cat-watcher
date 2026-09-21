"""Unit tests for :mod:`cat_watcher.classifier.train`.

The YOLO model is a ``MagicMock(spec=YOLO)``, the pattern ``test_detector.py`` uses, since
ultralytics is a third-party class this module does not own. Its ``.train()`` writes a stub
``best.pt`` into ultralytics' own output location, so the copy step runs for real.
"""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from classifier_helpers import DEFAULT_DATASET_HASH, write_export_manifest
from ultralytics import YOLO  # type: ignore[attr-defined]  # ultralytics lazily loads models

from cat_watcher.classifier.splitting import SEED
from cat_watcher.classifier.train import (
    BASE_WEIGHTS,
    EPOCHS,
    IMGSZ,
    TrainParams,
    TrainPaths,
    TrainResult,
    YoloClsFactory,
    train_classifier,
)

if TYPE_CHECKING:
    from pathlib import Path

_CLASSES = ("marcel", "rufus")
_DATASET_HASH = DEFAULT_DATASET_HASH


def _write_test_manifest(dest: Path, *, classes: tuple[str, ...] = _CLASSES, dataset_hash: str = _DATASET_HASH) -> None:
    """Write a manifest of the shape Task 5's ``export_dataset`` produces."""
    write_export_manifest(dest, classes=classes, dataset_hash=dataset_hash)


def _fake_yolo_factory(
    *,
    names: dict[int, str],
    write_best_pt: bool = True,
    best_pt_bytes: bytes = b"stub-weights",
) -> tuple[YoloClsFactory, MagicMock]:
    """Build a fake ``YoloClsFactory``. Its ``.train()`` mimics ultralytics' own output layout.

    Returns the factory and the ``train`` sub-mock, so a test can assert on its call. A mock
    built with ``spec=YOLO`` types ``mock_model.train`` as ``Any`` on a later read, so the
    caller keeps this handle instead. A test that must prove ``.train()`` never ran checks
    ``run_dir`` for the ``"train"`` subdirectory this fake creates, since ``run_dir`` itself
    stays untouched otherwise.
    """
    mock_model = MagicMock(spec=YOLO)
    mock_model.names = names

    def _train_side_effect(*, project: Path, **_kwargs: object) -> None:
        save_dir = project / "train"
        mock_model.trainer = SimpleNamespace(save_dir=save_dir)
        if write_best_pt:
            weights_dir = save_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            _ = (weights_dir / "best.pt").write_bytes(best_pt_bytes)

    train_mock: MagicMock = MagicMock(side_effect=_train_side_effect)
    mock_model.train = train_mock

    def factory(base_weights: Path) -> object:
        _ = base_weights  # the fake never reads the weights file
        return mock_model

    return factory, train_mock


def _paths(tmp_path: Path, *, manifest_path: Path | None = None) -> TrainPaths:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    return TrainPaths(
        dataset_root=dataset_root,
        manifest_path=manifest_path if manifest_path is not None else tmp_path / "dataset" / "manifest.json",
        models_dir=tmp_path / "models",
        run_dir=tmp_path / "runs",
        base_weights=tmp_path / "models" / BASE_WEIGHTS,
    )


def test_module_constants_match_the_task_brief() -> None:
    """Every module constant holds the value the brief specifies verbatim."""
    assert BASE_WEIGHTS == "yolo11n-cls.pt"
    assert EPOCHS == 40
    assert IMGSZ == 224


def test_train_classifier_writes_model_and_sidecar(tmp_path: Path) -> None:
    """A wiring run copies the stub checkpoint and writes a sidecar matching the inputs and the manifest."""
    paths = _paths(tmp_path)
    _write_test_manifest(paths.manifest_path)
    factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"})

    result = train_classifier(paths=paths, params=TrainParams(epochs=2, imgsz=64, seed=SEED), yolo_factory=factory)

    assert result.model_path.is_file()
    assert result.sidecar_path.is_file()
    assert result.model_path.read_bytes() == b"stub-weights"
    assert result.classes == _CLASSES
    assert result.model_names == {0: "marcel", 1: "rufus"}
    assert result.epochs == 2
    assert result.imgsz == 64
    assert result.dataset_hash == _DATASET_HASH

    sidecar = cast("dict[str, object]", json.loads(result.sidecar_path.read_text(encoding="utf-8")))
    assert sidecar == {
        "classes": list(_CLASSES),
        "model_names": {"0": "marcel", "1": "rufus"},
        "epochs": 2,
        "imgsz": 64,
        "seed": SEED,
        "base_weights": str(paths.base_weights.resolve()),
        "dataset_hash": _DATASET_HASH,
        "manifest_path": str(paths.manifest_path.resolve()),
    }


def test_model_filename_embeds_the_dataset_hash_prefix(tmp_path: Path) -> None:
    """The saved checkpoint's stem carries the manifest's ``dataset_hash``, truncated to 8 characters."""
    paths = _paths(tmp_path)
    _write_test_manifest(paths.manifest_path)
    factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"})

    result = train_classifier(paths=paths, yolo_factory=factory)

    assert result.model_path.stem == "cat-classifier-deadbeef"
    assert result.sidecar_path.stem == "cat-classifier-deadbeef"


def test_train_classifier_raises_on_class_name_mismatch(tmp_path: Path) -> None:
    """A model whose ``names`` names a class the manifest does not raises ``ValueError``.

    This is the regression guard for a dataset that grew a third class after the manifest
    was written. Training on it silently mislabels every prediction downstream.
    """
    paths = _paths(tmp_path)
    _write_test_manifest(paths.manifest_path)
    factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "unknown_slug"})

    with pytest.raises(ValueError, match="unknown_slug"):
        _ = train_classifier(paths=paths, yolo_factory=factory)

    assert not paths.models_dir.exists()


def test_train_classifier_raises_when_manifest_is_absent(tmp_path: Path) -> None:
    """A ``manifest_path`` that does not exist raises before any model loads."""
    paths = _paths(tmp_path)
    factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"})

    with pytest.raises(FileNotFoundError):
        _ = train_classifier(paths=paths, yolo_factory=factory)

    assert not paths.run_dir.exists()


def test_train_classifier_raises_when_dataset_root_is_absent(tmp_path: Path) -> None:
    """A manifest that exists beside a missing dataset directory raises ``FileNotFoundError``."""
    paths = _paths(tmp_path, manifest_path=tmp_path / "manifest.json")
    _write_test_manifest(paths.manifest_path)
    paths.dataset_root.rmdir()
    factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"})

    with pytest.raises(FileNotFoundError, match=str(paths.dataset_root)):
        _ = train_classifier(paths=paths, yolo_factory=factory)

    assert not paths.run_dir.exists()


def test_train_classifier_raises_when_best_pt_never_appears(tmp_path: Path) -> None:
    """A training run whose trainer never produces ``best.pt`` raises ``FileNotFoundError``."""
    paths = _paths(tmp_path)
    _write_test_manifest(paths.manifest_path)
    factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"}, write_best_pt=False)

    with pytest.raises(FileNotFoundError, match="did not produce a checkpoint"):
        _ = train_classifier(paths=paths, yolo_factory=factory)


def test_train_classifier_creates_the_models_dir_when_absent(tmp_path: Path) -> None:
    """``models_dir`` need not exist before the run. ``train_classifier`` creates it."""
    paths = _paths(tmp_path)
    _write_test_manifest(paths.manifest_path)
    assert not paths.models_dir.exists()
    factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"})

    result = train_classifier(paths=paths, yolo_factory=factory)

    assert paths.models_dir.is_dir()
    assert result.model_path.parent == paths.models_dir


def test_train_classifier_second_run_overwrites_the_same_files(tmp_path: Path) -> None:
    """A second run over the same dataset hash replaces the checkpoint and sidecar in place."""
    paths = _paths(tmp_path)
    _write_test_manifest(paths.manifest_path)

    first_factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"}, best_pt_bytes=b"first-run")
    first = train_classifier(paths=paths, yolo_factory=first_factory)

    second_factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"}, best_pt_bytes=b"second-run")
    second = train_classifier(paths=paths, yolo_factory=second_factory)

    assert first.model_path == second.model_path
    assert second.model_path.read_bytes() == b"second-run"


def test_train_classifier_result_is_a_train_result(tmp_path: Path) -> None:
    """``train_classifier`` returns a :class:`TrainResult`, not a bare tuple or dict."""
    paths = _paths(tmp_path)
    _write_test_manifest(paths.manifest_path)
    factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"})

    result = train_classifier(paths=paths, yolo_factory=factory)

    assert isinstance(result, TrainResult)


def test_train_classifier_calls_train_with_the_expected_kwargs(tmp_path: Path) -> None:
    """The wiring passes the dataset root, the hyperparameters, and ``run_dir`` to ``model.train()``.

    ``project`` keeps ultralytics' ``runs/`` output under gitignored storage, not the repo root.
    """
    paths = _paths(tmp_path)
    _write_test_manifest(paths.manifest_path)
    factory, train_mock = _fake_yolo_factory(names={0: "marcel", 1: "rufus"})

    _ = train_classifier(paths=paths, params=TrainParams(epochs=5, imgsz=128, seed=SEED), yolo_factory=factory)

    assert train_mock.call_args is not None
    assert train_mock.call_args.kwargs["data"] == paths.dataset_root
    assert train_mock.call_args.kwargs["epochs"] == 5
    assert train_mock.call_args.kwargs["imgsz"] == 128
    assert train_mock.call_args.kwargs["seed"] == SEED
    assert train_mock.call_args.kwargs["project"] == paths.run_dir


def test_sidecar_classes_come_from_the_manifest_not_the_model(tmp_path: Path) -> None:
    """The sidecar's ``classes`` keep the manifest's order, even when it disagrees with the model's.

    ``marcel`` and ``rufus`` sort alphabetically, so a manifest order of ``rufus``, ``marcel``
    disagrees with a model's sorted-by-directory ``names``. That disagreement is what proves the
    sidecar's classes come from the manifest, not from the model.
    """
    paths = _paths(tmp_path)
    _write_test_manifest(paths.manifest_path, classes=("rufus", "marcel"))
    factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"})

    result = train_classifier(paths=paths, yolo_factory=factory)

    assert result.classes == ("rufus", "marcel")
    sidecar = cast("dict[str, object]", json.loads(result.sidecar_path.read_text(encoding="utf-8")))
    assert sidecar["classes"] == ["rufus", "marcel"]


def test_atomic_write_leaves_no_new_checkpoint_beside_an_old_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sidecar write that fails after the checkpoint copy leaves the previous pair untouched.

    Task 8 reads the sidecar for the class order and the dataset hash. A new checkpoint beside
    an old sidecar silently benchmarks a real new model against the previous run's labels.
    """
    paths = _paths(tmp_path)
    _write_test_manifest(paths.manifest_path)
    first_factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"}, best_pt_bytes=b"first-run")
    first = train_classifier(paths=paths, yolo_factory=first_factory)
    original_model_bytes = first.model_path.read_bytes()
    original_sidecar_text = first.sidecar_path.read_text(encoding="utf-8")

    def _raising_dumps(*_args: object, **_kwargs: object) -> str:
        msg = "simulated sidecar-write failure"
        raise RuntimeError(msg)

    monkeypatch.setattr("cat_watcher.classifier.train.json.dumps", _raising_dumps)
    second_factory, _ = _fake_yolo_factory(names={0: "marcel", 1: "rufus"}, best_pt_bytes=b"second-run")

    with pytest.raises(RuntimeError, match="simulated sidecar-write failure"):
        _ = train_classifier(paths=paths, yolo_factory=second_factory)

    assert first.model_path.read_bytes() == original_model_bytes
    assert first.sidecar_path.read_text(encoding="utf-8") == original_sidecar_text
    assert not list(paths.models_dir.glob("*.part"))

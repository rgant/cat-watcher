"""Unit tests for :mod:`cat_watcher.classifier.benchmark`.

Every hand-computed metric here is worked out from the fake ``predict`` functions' fixed
outputs, not by calling the code under test a second way. This lets a wrong sklearn ``labels``
order, or a wrong zero-division default, fail a test.
"""

import json
from dataclasses import replace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest
from classifier_helpers import DEFAULT_DATASET_HASH, write_export_manifest
from ultralytics import YOLO  # type: ignore[attr-defined]  # ultralytics lazily loads models
from ultralytics.engine.results import Results

from cat_watcher.classifier.benchmark import (
    REPORTS_SUBDIR,
    BenchmarkReport,
    ClassMetrics,
    PredictFn,
    SidecarMeta,
    benchmark_model,
    make_predict_fn,
    read_sidecar,
    render_json,
    render_markdown,
    write_reports,
)
from cat_watcher.classifier.metrics import AbstainPoint
from cat_watcher.classifier.splitting import SEED

if TYPE_CHECKING:
    from pathlib import Path

_CLASSES = ("marcel", "rufus")


def _write_manifest(  # noqa: PLR0913  # constructor wrapper; flat kwargs map 1:1 to ExportSummary fields
    path: Path,
    *,
    classes: tuple[str, ...] = _CLASSES,
    candidates: int = 0,
    localization_misses: int = 0,
    frame_load_failures: int = 0,
    mixed_class_clips: int = 0,
) -> None:
    """Write a manifest of the shape Task 5's ``export_dataset`` produces."""
    write_export_manifest(
        path,
        classes=classes,
        candidates=candidates,
        localization_misses=localization_misses,
        frame_load_failures=frame_load_failures,
        mixed_class_clips=mixed_class_clips,
    )


def _write_class_images(class_dir: Path, names: list[str]) -> None:
    """Write ``names`` as placeholder files under ``class_dir``. ``benchmark_model`` never reads their bytes."""
    class_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        _ = (class_dir / name).write_bytes(b"placeholder crop")


def _fake_predict(mapping: dict[str, tuple[str, float]]) -> PredictFn:
    """Build a fixed-output ``PredictFn`` keyed by filename, per the no-double-over-mock preference."""

    def predict(image_path: Path) -> tuple[str, float]:
        return mapping[image_path.name]

    return predict


def _refusing_predict(image_path: Path) -> tuple[str, float]:
    """If the benchmark ever calls this ``PredictFn``, it fails the test."""
    msg = f"predict must not run, got {image_path}"
    raise AssertionError(msg)


def _assert_class_metrics(actual: ClassMetrics, *, precision: float, recall: float, f1: float, support: int) -> None:
    """Assert each field of ``actual`` against its expected value. Float fields compare by ``approx``."""
    assert actual.precision == pytest.approx(precision)
    assert actual.recall == pytest.approx(recall)
    assert actual.f1 == pytest.approx(f1)
    assert actual.support == support


# --- module constants -------------------------------------------------------------------------


def test_reports_subdir_matches_the_task_brief() -> None:
    """``REPORTS_SUBDIR`` holds the value the brief specifies verbatim."""
    assert REPORTS_SUBDIR == "classifier/reports"


# --- read_sidecar -------------------------------------------------------------------------------


def test_read_sidecar_round_trips_an_identical_payload(tmp_path: Path) -> None:
    """``read_sidecar`` rebuilds ``classes`` as a tuple and ``model_names`` as ``dict[int, str]``.

    The sidecar's class order (``rufus``, ``marcel``) differs from alphabetical order. This is
    the regression guard. A caller that sorts ``classes`` after the read still matches the class
    set. It does not match this exact tuple order.
    """
    model_path = tmp_path / "cat-classifier-deadbeef.pt"
    sidecar_path = model_path.with_suffix(".json")
    payload = {
        "classes": ["rufus", "marcel"],
        "model_names": {"0": "marcel", "1": "rufus"},
        "epochs": 40,
        "imgsz": 224,
        "seed": SEED,
        "base_weights": "/models/yolo11n-cls.pt",
        "dataset_hash": DEFAULT_DATASET_HASH,
        "manifest_path": "/data/manifest.json",
    }
    _ = sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    meta = read_sidecar(model_path)

    assert meta == SidecarMeta(
        classes=("rufus", "marcel"),
        model_names={0: "marcel", 1: "rufus"},
        dataset_hash=DEFAULT_DATASET_HASH,
        manifest_path="/data/manifest.json",
    )


def test_read_sidecar_raises_when_absent(tmp_path: Path) -> None:
    """A model path with no ``.json`` sidecar next to it raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        _ = read_sidecar(tmp_path / "cat-classifier-deadbeef.pt")


def test_read_sidecar_raises_when_malformed(tmp_path: Path) -> None:
    """A sidecar that is not valid JSON raises ``JSONDecodeError``."""
    model_path = tmp_path / "cat-classifier-deadbeef.pt"
    _ = model_path.with_suffix(".json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        _ = read_sidecar(model_path)


def test_read_sidecar_raises_when_a_required_key_is_missing(tmp_path: Path) -> None:
    """Valid JSON that lacks the ``classes`` key raises ``KeyError``, not a silent empty tuple."""
    model_path = tmp_path / "cat-classifier-deadbeef.pt"
    payload = {"model_names": {"0": "marcel"}, "dataset_hash": DEFAULT_DATASET_HASH, "manifest_path": "/data/manifest.json"}
    _ = model_path.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KeyError):
        _ = read_sidecar(model_path)


# --- benchmark_model: metrics ---------------------------------------------------------------------


def test_benchmark_model_matches_hand_computed_metrics(tmp_path: Path) -> None:
    """Confusion, accuracy, and per-class metrics match values worked out by hand from the fake's outputs."""
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["m1.jpg", "m2.jpg"])
    _write_class_images(dataset_root / "test" / "rufus", ["r1.jpg", "r2.jpg", "r3.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, candidates=9, localization_misses=4, frame_load_failures=1, mixed_class_clips=2)
    # marcel: m1 correct, m2 predicted as rufus (wrong). rufus: r1, r2 correct, r3 predicted as marcel (wrong).
    predict = _fake_predict(
        {
            "m1.jpg": ("marcel", 0.90),
            "m2.jpg": ("rufus", 0.60),
            "r1.jpg": ("rufus", 0.95),
            "r2.jpg": ("rufus", 0.55),
            "r3.jpg": ("marcel", 0.51),
        },
    )

    report = benchmark_model(dataset_root=dataset_root, manifest_path=manifest_path, classes=_CLASSES, predict=predict)

    assert report.classes == _CLASSES
    assert report.test_count == 5
    assert report.accuracy == pytest.approx(3 / 5)
    assert report.per_class.keys() == {"marcel", "rufus"}
    _assert_class_metrics(report.per_class["marcel"], precision=1 / 2, recall=1 / 2, f1=1 / 2, support=2)
    _assert_class_metrics(report.per_class["rufus"], precision=2 / 3, recall=2 / 3, f1=2 / 3, support=3)
    assert report.confusion == {
        ("marcel", "marcel"): 1,
        ("marcel", "rufus"): 1,
        ("rufus", "marcel"): 1,
        ("rufus", "rufus"): 2,
    }
    assert report.localization_misses == 4
    assert report.candidates == 9
    assert report.frame_load_failures == 1
    assert report.mixed_class_clips == 2


def test_benchmark_model_labels_follow_sidecar_class_order_not_alphabetical(tmp_path: Path) -> None:
    """``classes=("rufus", "marcel")`` (non-alphabetical) still labels precision and recall correctly.

    This is the regression test for the train and benchmark skew bug. A bug can sort ``classes``
    for scikit-learn's call, then zip results against the original order. That swaps which class
    shows a 0.0 recall.
    """
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["one.jpg"])
    _write_class_images(dataset_root / "test" / "rufus", ["two.jpg", "three.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, classes=("rufus", "marcel"))
    # Every image is predicted "marcel": one.jpg is correct, two.jpg and three.jpg are wrong.
    predict = _fake_predict(
        {"one.jpg": ("marcel", 0.99), "two.jpg": ("marcel", 0.80), "three.jpg": ("marcel", 0.85)},
    )

    report = benchmark_model(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        classes=("rufus", "marcel"),
        predict=predict,
    )

    assert report.per_class["rufus"] == ClassMetrics(precision=0.0, recall=0.0, f1=0.0, support=2)
    _assert_class_metrics(report.per_class["marcel"], precision=1 / 3, recall=1.0, f1=0.5, support=1)
    assert report.confusion == {
        ("rufus", "rufus"): 0,
        ("rufus", "marcel"): 2,
        ("marcel", "rufus"): 0,
        ("marcel", "marcel"): 1,
    }
    assert report.accuracy == pytest.approx(1 / 3)


def test_benchmark_model_one_predicted_class_for_every_image(tmp_path: Path) -> None:
    """A model that predicts ``marcel`` for every image still yields correct per-class metrics."""
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["m1.jpg", "m2.jpg"])
    _write_class_images(dataset_root / "test" / "rufus", ["r1.jpg", "r2.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    def predict(image_path: Path) -> tuple[str, float]:
        _ = image_path  # every image gets the same fixed prediction
        return "marcel", 0.65

    report = benchmark_model(dataset_root=dataset_root, manifest_path=manifest_path, classes=_CLASSES, predict=predict)

    assert report.accuracy == pytest.approx(0.5)
    assert report.per_class["rufus"] == ClassMetrics(precision=0.0, recall=0.0, f1=0.0, support=2)
    _assert_class_metrics(report.per_class["marcel"], precision=0.5, recall=1.0, f1=2 / 3, support=2)
    assert report.confusion == {
        ("marcel", "marcel"): 2,
        ("marcel", "rufus"): 0,
        ("rufus", "marcel"): 2,
        ("rufus", "rufus"): 0,
    }


def test_benchmark_model_every_prediction_is_wrong(tmp_path: Path) -> None:
    """A model that swaps every label yields accuracy 0.0, not a division error."""
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["m1.jpg", "m2.jpg"])
    _write_class_images(dataset_root / "test" / "rufus", ["r1.jpg", "r2.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    predict = _fake_predict(
        {"m1.jpg": ("rufus", 0.7), "m2.jpg": ("rufus", 0.8), "r1.jpg": ("marcel", 0.6), "r2.jpg": ("marcel", 0.9)},
    )

    report = benchmark_model(dataset_root=dataset_root, manifest_path=manifest_path, classes=_CLASSES, predict=predict)

    assert report.accuracy == 0.0
    assert report.per_class == {
        "marcel": ClassMetrics(precision=0.0, recall=0.0, f1=0.0, support=2),
        "rufus": ClassMetrics(precision=0.0, recall=0.0, f1=0.0, support=2),
    }
    assert report.confusion == {
        ("marcel", "marcel"): 0,
        ("marcel", "rufus"): 2,
        ("rufus", "marcel"): 2,
        ("rufus", "rufus"): 0,
    }


def test_benchmark_model_empty_class_directory_reports_zero_support_for_it(tmp_path: Path) -> None:
    """A ``test/marcel`` directory that exists but holds no files contributes zero support, not a crash."""
    dataset_root = tmp_path / "dataset"
    (dataset_root / "test" / "marcel").mkdir(parents=True)
    _write_class_images(dataset_root / "test" / "rufus", ["r1.jpg", "r2.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    predict = _fake_predict({"r1.jpg": ("rufus", 0.7), "r2.jpg": ("rufus", 0.8)})

    report = benchmark_model(dataset_root=dataset_root, manifest_path=manifest_path, classes=_CLASSES, predict=predict)

    assert report.test_count == 2
    assert report.per_class["marcel"] == ClassMetrics(precision=0.0, recall=0.0, f1=0.0, support=0)
    assert report.per_class["rufus"] == ClassMetrics(precision=1.0, recall=1.0, f1=1.0, support=2)


def test_benchmark_model_ignores_non_image_files_in_the_test_split(tmp_path: Path) -> None:
    """A ``.DS_Store`` file in a class directory does not inflate ``test_count`` or reach ``predict``."""
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["m1.jpg"])
    _ = (dataset_root / "test" / "marcel" / ".DS_Store").write_bytes(b"not an image")
    _write_class_images(dataset_root / "test" / "rufus", ["r1.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    predict = _fake_predict({"m1.jpg": ("marcel", 0.9), "r1.jpg": ("rufus", 0.9)})

    report = benchmark_model(dataset_root=dataset_root, manifest_path=manifest_path, classes=_CLASSES, predict=predict)

    assert report.test_count == 2


# --- benchmark_model: raises -----------------------------------------------------------------------


def test_benchmark_model_raises_when_manifest_is_absent(tmp_path: Path) -> None:
    """A missing ``manifest_path`` raises before any prediction runs."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()

    with pytest.raises(FileNotFoundError):
        _ = benchmark_model(
            dataset_root=dataset_root,
            manifest_path=tmp_path / "missing-manifest.json",
            classes=_CLASSES,
            predict=_refusing_predict,
        )


def test_benchmark_model_raises_when_manifest_classes_disagree_with_the_given_classes(tmp_path: Path) -> None:
    """A manifest whose classes differ from ``classes`` raises before any prediction runs.

    This is the guard for a stray class directory in the exported dataset, or a stale sidecar.
    """
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["m1.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, classes=("marcel", "ghost"))

    with pytest.raises(ValueError, match="ghost"):
        _ = benchmark_model(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            classes=_CLASSES,
            predict=_refusing_predict,
        )


def test_benchmark_model_raises_when_predict_returns_an_unknown_class(tmp_path: Path) -> None:
    """A prediction naming a class outside ``classes`` raises, instead of scikit-learn silently dropping it."""
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["m1.jpg"])
    _write_class_images(dataset_root / "test" / "rufus", ["r1.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    predict = _fake_predict({"m1.jpg": ("marcel", 0.9), "r1.jpg": ("ghost", 0.9)})

    with pytest.raises(ValueError, match="ghost"):
        _ = benchmark_model(dataset_root=dataset_root, manifest_path=manifest_path, classes=_CLASSES, predict=predict)


def test_benchmark_model_raises_when_the_test_split_directory_is_missing(tmp_path: Path) -> None:
    """A ``dataset_root`` with no ``test`` directory raises, instead of reporting an empty benchmark."""
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    with pytest.raises(ValueError, match="no test-split images"):
        _ = benchmark_model(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            classes=_CLASSES,
            predict=_refusing_predict,
        )


def test_benchmark_model_raises_when_every_class_directory_is_empty(tmp_path: Path) -> None:
    """Every class directory present but empty raises the same way a missing ``test/`` directory does."""
    dataset_root = tmp_path / "dataset"
    (dataset_root / "test" / "marcel").mkdir(parents=True)
    (dataset_root / "test" / "rufus").mkdir(parents=True)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)

    with pytest.raises(ValueError, match="no test-split images"):
        _ = benchmark_model(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            classes=_CLASSES,
            predict=_refusing_predict,
        )


# --- benchmark_model: abstain sweep and the recommended threshold ------------------------------


def test_benchmark_model_abstain_sweep_at_exact_and_uncleared_thresholds(tmp_path: Path) -> None:
    """A confidence exactly at a threshold counts as covered. No prediction clears the top threshold.

    Thresholds are ``(0.50, 0.60, 0.70, 0.80, 0.90, 0.95)``. One prediction's confidence is
    exactly ``0.50``. The highest confidence in the split is ``0.94``, so 0.95 covers nothing.
    """
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["a.jpg", "b.jpg"])
    _write_class_images(dataset_root / "test" / "rufus", ["c.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    predict = _fake_predict(
        {"a.jpg": ("marcel", 0.50), "b.jpg": ("rufus", 0.61), "c.jpg": ("rufus", 0.94)},
    )

    report = benchmark_model(dataset_root=dataset_root, manifest_path=manifest_path, classes=_CLASSES, predict=predict)

    expected = [
        (0.50, 1.0, 2 / 3),
        (0.60, 2 / 3, 0.5),
        (0.70, 1 / 3, 1.0),
        (0.80, 1 / 3, 1.0),
        (0.90, 1 / 3, 1.0),
        (0.95, 0.0, 0.0),
    ]
    assert [point.threshold for point in report.abstain] == [threshold for threshold, _, _ in expected]
    for point, (_, coverage, accuracy_on_covered) in zip(report.abstain, expected, strict=True):
        assert point.coverage == pytest.approx(coverage)
        assert point.accuracy_on_covered == pytest.approx(accuracy_on_covered)


def test_benchmark_model_recommends_the_lowest_threshold_already_meeting_the_target(tmp_path: Path) -> None:
    """When the lowest threshold already clears the target, ``recommended_threshold`` is that threshold."""
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["a.jpg", "b.jpg"])
    _write_class_images(dataset_root / "test" / "rufus", ["c.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    # Same split as the abstain-sweep test above: accuracy-on-covered is 2/3 at threshold 0.50,
    # which already clears a 0.60 target.
    predict = _fake_predict(
        {"a.jpg": ("marcel", 0.50), "b.jpg": ("rufus", 0.61), "c.jpg": ("rufus", 0.94)},
    )

    report = benchmark_model(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        classes=_CLASSES,
        predict=predict,
        target_accuracy=0.60,
    )

    assert report.target_accuracy == pytest.approx(0.60)
    assert report.recommended_threshold == pytest.approx(0.50)


def test_benchmark_model_recommends_a_middle_threshold_when_a_lower_one_falls_short(tmp_path: Path) -> None:
    """When only a middle threshold first clears the target, that threshold is recommended, not the lowest."""
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["a.jpg", "b.jpg"])
    _write_class_images(dataset_root / "test" / "rufus", ["c.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    # Same split again: accuracy-on-covered is 2/3 at 0.50, 0.5 at 0.60, then 1.0 from 0.70 up.
    # A 1.0 target first clears at 0.70, the third of six thresholds.
    predict = _fake_predict(
        {"a.jpg": ("marcel", 0.50), "b.jpg": ("rufus", 0.61), "c.jpg": ("rufus", 0.94)},
    )

    report = benchmark_model(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        classes=_CLASSES,
        predict=predict,
        target_accuracy=1.0,
    )

    assert report.recommended_threshold == pytest.approx(0.70)


def test_benchmark_model_recommends_none_when_no_threshold_meets_the_target(tmp_path: Path) -> None:
    """When no threshold reaches the target, ``recommended_threshold`` is ``None``."""
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    # a, b, c are correctly predicted "marcel". d, the highest-confidence image, is wrongly
    # predicted "rufus". Accuracy-on-covered then falls as the threshold rises: 0.75 at 0.50,
    # 2/3 at 0.60, 0.5 at 0.70, 0.0 at 0.80, then no coverage. The best is 0.75, below a 0.90 target.
    predict = _fake_predict(
        {
            "a.jpg": ("marcel", 0.55),
            "b.jpg": ("marcel", 0.65),
            "c.jpg": ("marcel", 0.75),
            "d.jpg": ("rufus", 0.85),
        },
    )

    report = benchmark_model(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        classes=_CLASSES,
        predict=predict,
        target_accuracy=0.90,
    )

    assert report.recommended_threshold is None


def test_benchmark_model_recommends_the_lowest_threshold_when_target_accuracy_is_zero(tmp_path: Path) -> None:
    """A ``target_accuracy`` of ``0.0`` always recommends the lowest threshold, even a wrong one."""
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["a.jpg"])
    _write_class_images(dataset_root / "test" / "rufus", ["b.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    # Every prediction is wrong, so accuracy-on-covered is 0.0 everywhere. A 0.0 target still
    # clears 0.0 >= 0.0 at the lowest threshold.
    predict = _fake_predict({"a.jpg": ("rufus", 0.9), "b.jpg": ("marcel", 0.9)})

    report = benchmark_model(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        classes=_CLASSES,
        predict=predict,
        target_accuracy=0.0,
    )

    assert report.recommended_threshold == pytest.approx(0.50)


# --- render_json ----------------------------------------------------------------------------------


def _sample_report() -> BenchmarkReport:
    return BenchmarkReport(
        classes=("marcel", "rufus"),
        accuracy=0.75,
        per_class={
            "marcel": ClassMetrics(precision=0.5, recall=1.0, f1=0.6667, support=2),
            "rufus": ClassMetrics(precision=1.0, recall=0.5, f1=0.6667, support=3),
        },
        confusion={("marcel", "marcel"): 2, ("marcel", "rufus"): 0, ("rufus", "marcel"): 1, ("rufus", "rufus"): 2},
        abstain=[
            AbstainPoint(threshold=0.50, coverage=1.0, accuracy_on_covered=0.75),
            AbstainPoint(threshold=0.95, coverage=0.0, accuracy_on_covered=0.0),
        ],
        recommended_threshold=0.50,
        target_accuracy=0.75,
        localization_misses=6,
        localize_conf=0.10,
        frame_load_failures=2,
        mixed_class_clips=1,
        candidates=8,
        dataset_hash=DEFAULT_DATASET_HASH,
        test_count=5,
    )


def test_render_json_round_trips() -> None:
    """A parse of ``render_json``'s output yields the same numbers, with ``"true>pred"`` confusion keys."""
    report = _sample_report()

    parsed = cast("dict[str, object]", json.loads(render_json(report)))

    assert parsed["accuracy"] == pytest.approx(0.75)
    assert parsed["test_count"] == 5
    assert parsed["recommended_threshold"] == pytest.approx(0.50)
    assert parsed["target_accuracy"] == pytest.approx(0.75)
    assert parsed["localization_misses"] == 6
    assert parsed["localize_conf"] == pytest.approx(0.10)
    assert parsed["frame_load_failures"] == 2
    assert parsed["mixed_class_clips"] == 1
    assert parsed["candidates"] == 8
    assert parsed["dataset_hash"] == DEFAULT_DATASET_HASH
    assert parsed["classes"] == ["marcel", "rufus"]
    assert parsed["confusion"] == {"marcel>marcel": 2, "marcel>rufus": 0, "rufus>marcel": 1, "rufus>rufus": 2}
    per_class = cast("dict[str, dict[str, float]]", parsed["per_class"])
    assert per_class["marcel"]["precision"] == pytest.approx(0.5)
    assert per_class["rufus"]["support"] == pytest.approx(3)
    abstain = cast("list[dict[str, float]]", parsed["abstain"])
    assert abstain[0]["threshold"] == pytest.approx(0.50)
    assert abstain[1]["coverage"] == pytest.approx(0.0)


def test_render_json_recommended_threshold_none_round_trips_to_null() -> None:
    """``recommended_threshold=None`` round-trips through JSON as ``null``, not a sentinel string."""
    report = replace(_sample_report(), recommended_threshold=None)

    parsed = cast("dict[str, object]", json.loads(render_json(report)))

    assert parsed["recommended_threshold"] is None


# --- render_markdown --------------------------------------------------------------------------


def test_render_markdown_contains_the_required_sections_and_distinctive_rows() -> None:
    """The rendered Markdown names every section, plus one full, distinctive row from each table.

    A substring check alone does not discriminate. ``"0.50"`` matches both an abstain threshold
    and part of ``"0.5000"``. ``"0.7500"`` matches both a header value and a table cell. Each
    assertion below checks a full row instead, so deleting a renderer call breaks this test.
    """
    report = _sample_report()

    markdown = render_markdown(report)

    assert "## Confusion matrix" in markdown
    assert "| marcel | 2 | 0 |" in markdown
    assert "## Per-class metrics" in markdown
    assert "| marcel | 0.5000 | 1.0000 | 0.6667 | 2 |" in markdown
    assert "## Abstain sweep" in markdown
    assert "Coverage is the fraction the model classifies itself." in markdown
    assert "| 0.50 | 1.0000 | 0.7500 |" in markdown
    assert "| 0.95 | 0.0000 | n/a |" in markdown
    assert "## Recommended threshold" in markdown
    assert "Recommended threshold: 0.50" in markdown
    assert "coverage 1.00, accuracy-on-covered 0.75" in markdown
    assert "## Export diagnostics" in markdown
    assert "Localization misses (conf 0.10): 6/8" in markdown
    assert "Frame load failures: 2" in markdown
    assert "Mixed-class clips: 1" in markdown
    assert "Dataset hash:" in markdown
    assert DEFAULT_DATASET_HASH in markdown
    assert "Check the support column." in markdown


def test_render_markdown_header_names_the_model_path() -> None:
    """The header names which checkpoint produced the numbers, or an operator cannot attribute them."""
    report = replace(_sample_report(), model_path="/data/models/cat-classifier-deadbeef.pt")

    markdown = render_markdown(report)

    assert "Model: /data/models/cat-classifier-deadbeef.pt" in markdown


def test_render_markdown_names_the_best_available_row_when_no_threshold_qualifies() -> None:
    """When ``recommended_threshold`` is ``None``, the markdown names the best available row instead."""
    report = replace(_sample_report(), recommended_threshold=None, target_accuracy=0.999)

    markdown = render_markdown(report)

    assert "Recommended threshold: none" in markdown
    assert "No threshold reaches the 0.999 target." in markdown
    assert "Best available: 0.50 -> coverage 1.00, accuracy 0.75" in markdown


def test_render_markdown_best_available_skips_a_zero_coverage_row(tmp_path: Path) -> None:
    """Even when a zero-coverage row ties every other row on accuracy, it never reads as "best".

    Every prediction is wrong and lands in ``[0.50, 0.60)``. Coverage is ``1.0`` at threshold
    ``0.50`` and ``0.0`` at every higher threshold, so every row ties at accuracy ``0.0``. The
    best available row must still be ``0.50``, the only one that covers a prediction.
    """
    dataset_root = tmp_path / "dataset"
    _write_class_images(dataset_root / "test" / "marcel", ["a.jpg", "b.jpg"])
    _write_class_images(dataset_root / "test" / "rufus", ["c.jpg", "d.jpg"])
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path)
    predict = _fake_predict(
        {
            "a.jpg": ("rufus", 0.55),
            "b.jpg": ("rufus", 0.55),
            "c.jpg": ("marcel", 0.55),
            "d.jpg": ("marcel", 0.55),
        },
    )

    report = benchmark_model(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        classes=_CLASSES,
        predict=predict,
        target_accuracy=0.90,
    )
    markdown = render_markdown(report)

    assert report.recommended_threshold is None
    recommendation = markdown.split("## Recommended threshold")[1].split("## Export diagnostics")[0]
    assert "Best available: 0.50 -> coverage 1.00, accuracy 0.00" in recommendation
    assert "0.95" not in recommendation


def test_render_markdown_best_available_tie_break_prefers_the_lower_threshold() -> None:
    """When two rows tie on accuracy, the best available row is the one at the lower threshold.

    ``0.60`` has more coverage than ``0.50`` here. A coverage-based tie-break picks it wrongly.
    The fix breaks the tie on the threshold itself, toward the lower one.
    """
    report = replace(
        _sample_report(),
        abstain=[
            AbstainPoint(threshold=0.50, coverage=0.20, accuracy_on_covered=0.60),
            AbstainPoint(threshold=0.60, coverage=0.80, accuracy_on_covered=0.60),
            AbstainPoint(threshold=0.95, coverage=0.0, accuracy_on_covered=0.0),
        ],
        recommended_threshold=None,
        target_accuracy=0.99,
    )

    markdown = render_markdown(report)

    assert "Best available: 0.50 -> coverage 0.20, accuracy 0.60" in markdown


# --- write_reports ----------------------------------------------------------------------------


def test_write_reports_writes_both_files(tmp_path: Path) -> None:
    """``write_reports`` writes the exact ``render_markdown``/``render_json`` output under ``reports_dir``."""
    report = _sample_report()
    reports_dir = tmp_path / "reports"

    md_path, json_path = write_reports(report, reports_dir=reports_dir)

    assert md_path.read_text(encoding="utf-8") == render_markdown(report)
    assert json_path.read_text(encoding="utf-8") == render_json(report)


def test_write_reports_names_files_after_the_model_stem_so_a_second_run_does_not_erase_the_first(tmp_path: Path) -> None:
    """Two models' reports land beside each other, not on top of each other."""
    reports_dir = tmp_path / "reports"
    report_a = replace(_sample_report(), model_path="/data/models/cat-classifier-aaaaaaaa.pt")
    report_b = replace(_sample_report(), model_path="/data/models/cat-classifier-bbbbbbbb.pt", accuracy=0.42)

    md_path_a, json_path_a = write_reports(report_a, reports_dir=reports_dir)
    md_path_b, json_path_b = write_reports(report_b, reports_dir=reports_dir)

    assert md_path_a != md_path_b
    assert json_path_a != json_path_b
    assert md_path_a.name == "cat-classifier-aaaaaaaa.md"
    assert md_path_b.name == "cat-classifier-bbbbbbbb.md"
    assert md_path_a.is_file()
    assert md_path_b.is_file()
    assert "Accuracy: 0.7500" in md_path_a.read_text(encoding="utf-8")
    assert "Accuracy: 0.4200" in md_path_b.read_text(encoding="utf-8")


# --- make_predict_fn --------------------------------------------------------------------------


def _fake_yolo_results(*, top1: int, top1conf: float) -> list[MagicMock]:
    """Build a one-element list mimicking a classification ``Results``, with a fake ``Probs``."""
    fake_probs = MagicMock(top1=top1, top1conf=top1conf)
    return [MagicMock(spec=Results, probs=fake_probs)]


def test_make_predict_fn_reads_the_class_name_through_model_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``make_predict_fn`` looks up the predicted index in ``model.names``, never in a bare index.

    ``model.names`` is deliberately not alphabetical: index 1 maps to "marcel". A bare ``top1``
    index into a sidecar ``classes`` tuple instead mislabels the prediction silently.
    """
    model = MagicMock(spec=YOLO)
    model.names = {0: "rufus", 1: "marcel"}
    model.return_value = _fake_yolo_results(top1=1, top1conf=0.87)

    def fake_factory(_model_path: Path) -> MagicMock:
        return model

    monkeypatch.setattr("cat_watcher.detector._yolo_factory", fake_factory)
    predict = make_predict_fn(tmp_path / "weights.pt")

    label, conf = predict(tmp_path / "crop.jpg")

    assert label == "marcel"
    assert conf == pytest.approx(0.87)


def test_make_predict_fn_raises_when_probs_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``Results`` with no ``probs`` (a non-classification model) raises, instead of a silent ``None`` label."""
    model = MagicMock(spec=YOLO)
    model.names = {0: "rufus", 1: "marcel"}
    model.return_value = [MagicMock(spec=Results, probs=None)]

    def fake_factory(_model_path: Path) -> MagicMock:
        return model

    monkeypatch.setattr("cat_watcher.detector._yolo_factory", fake_factory)
    predict = make_predict_fn(tmp_path / "weights.pt")

    with pytest.raises(RuntimeError, match="no classification"):
        _ = predict(tmp_path / "crop.jpg")

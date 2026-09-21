"""Evaluate a trained classifier on its held-out test split, and render Markdown + JSON reports.

:class:`PredictFn` injects the model prediction step. :func:`benchmark_model` then runs with a
fake, and no real YOLO inference. ``classes`` always comes from the model's traceability sidecar
(:func:`read_sidecar`), never from the database and never from directory names. This keeps the
confusion matrix rows and columns in the order the model trained on.

:func:`benchmark_model` also recommends an abstain threshold: the lowest threshold whose
accuracy-on-covered reaches a target. This is the hand-off value for the live-integration
project that follows this one.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from cat_watcher.classifier.dataset import read_manifest
from cat_watcher.classifier.metrics import AbstainPoint, Prediction, abstain_sweep
from cat_watcher.detector import load_yolo

if TYPE_CHECKING:
    import numpy as np
    from ultralytics.engine.results import Results

REPORTS_SUBDIR: str = "classifier/reports"
DEFAULT_TARGET_ACCURACY: float = 0.95

_TEST_SPLIT: str = "test"
_SIDECAR_SUFFIX: str = ".json"
# A report built directly, not through the CLI, carries no model_path. Its report filenames
# fall back to this stem.
_DEFAULT_REPORT_STEM: str = "benchmark"
_IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png")


class PredictFn(Protocol):
    """Classifies one crop image, and returns its predicted class slug and confidence."""

    def __call__(self, image_path: Path) -> tuple[str, float]:
        """Classify the image at ``image_path``, and return its predicted slug and confidence."""
        ...


@dataclass(frozen=True)
class ClassMetrics:
    """One class's precision, recall, F1, and test-split support."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class BenchmarkReport:  # pylint: disable=too-many-instance-attributes  # flat benchmark result; the rule targets behavior-rich classes, not data containers
    """One benchmark run: its metrics, its confusion matrix, and the export diagnostics it carries forward."""

    classes: tuple[str, ...]
    accuracy: float
    per_class: dict[str, ClassMetrics]
    confusion: dict[tuple[str, str], int]
    abstain: list[AbstainPoint]
    recommended_threshold: float | None
    target_accuracy: float
    localization_misses: int
    localize_conf: float
    frame_load_failures: int
    mixed_class_clips: int
    candidates: int
    dataset_hash: str
    test_count: int
    # The checkpoint this report scored. benchmark_model never sees a checkpoint path, only a
    # PredictFn, so this defaults empty. The CLI fills it in before writing the report.
    model_path: str = ""


@dataclass(frozen=True)
class SidecarMeta:
    """A trained model's traceability sidecar, as :mod:`cat_watcher.classifier.train` writes it."""

    classes: tuple[str, ...]
    model_names: dict[int, str]
    dataset_hash: str
    manifest_path: str


def read_sidecar(model_path: Path) -> SidecarMeta:
    """Read the JSON sidecar next to ``model_path``, and rebuild its typed fields.

    JSON stores ``classes`` as a list, and turns ``model_names``' integer keys into strings.
    This rebuilds ``classes`` as a tuple and ``model_names`` as ``dict[int, str]``. A caller
    never re-derives the class order from the database or from directory names.
    """
    sidecar_path = model_path.with_suffix(_SIDECAR_SUFFIX)
    payload = cast("dict[str, object]", json.loads(sidecar_path.read_text(encoding="utf-8")))
    raw_model_names = cast("dict[str, str]", payload["model_names"])
    return SidecarMeta(
        classes=tuple(cast("list[str]", payload["classes"])),
        model_names={int(index): name for index, name in raw_model_names.items()},
        dataset_hash=cast("str", payload["dataset_hash"]),
        manifest_path=cast("str", payload["manifest_path"]),
    )


def benchmark_model(
    *,
    dataset_root: Path,
    manifest_path: Path,
    classes: tuple[str, ...],
    predict: PredictFn,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
) -> BenchmarkReport:
    """Score ``predict`` over every image under ``dataset_root/test/<class>/``, against ``classes``.

    The true label for each image is its parent directory name. ``localization_misses``,
    ``frame_load_failures``, and ``mixed_class_clips`` come from the export manifest at
    ``manifest_path``. This raises ``ValueError`` in three cases:

    - The manifest's classes disagree with ``classes``.
    - A prediction names a class outside ``classes``.
    - The test split holds no image.
    """
    manifest = read_manifest(manifest_path)
    _ensure_classes_match(manifest.summary.classes, classes)
    predictions = _run_predictions(dataset_root, classes, predict)
    if not predictions:
        msg = f"no test-split images found under {dataset_root / _TEST_SPLIT}"
        raise ValueError(msg)
    accuracy, per_class, confusion = _score_predictions(predictions, classes)
    abstain = abstain_sweep(predictions)
    return BenchmarkReport(
        classes=classes,
        accuracy=accuracy,
        per_class=per_class,
        confusion=confusion,
        abstain=abstain,
        recommended_threshold=_recommend_threshold(abstain, target_accuracy),
        target_accuracy=target_accuracy,
        localization_misses=manifest.summary.localization_misses,
        localize_conf=manifest.summary.localize_conf,
        frame_load_failures=manifest.summary.frame_load_failures,
        mixed_class_clips=manifest.summary.mixed_class_clips,
        candidates=manifest.summary.candidates,
        dataset_hash=manifest.summary.dataset_hash,
        test_count=len(predictions),
    )


def _ensure_classes_match(manifest_classes: tuple[str, ...], classes: tuple[str, ...]) -> None:
    """When ``manifest_classes`` and ``classes`` name different class sets, this raises ``ValueError``.

    A stray class directory in the exported dataset, or a stale sidecar, shows up here as a set
    mismatch, before any prediction runs.
    """
    manifest_set = set(manifest_classes)
    classes_set = set(classes)
    if manifest_set != classes_set:
        msg = f"manifest classes {sorted(manifest_set)!r} do not match the given classes {sorted(classes_set)!r}"
        raise ValueError(msg)


def _run_predictions(dataset_root: Path, classes: tuple[str, ...], predict: PredictFn) -> list[Prediction]:
    """Run ``predict`` over every test-split image. The true label is its parent directory name.

    When a prediction names a class outside ``classes``, this raises ``ValueError``. scikit-learn
    otherwise drops that prediction from the confusion matrix silently, which inflates precision
    for the class it lands on by mistake.
    """
    class_set = set(classes)
    predictions: list[Prediction] = []
    for true_label, image_path in _collect_test_files(dataset_root, classes):
        pred_label, conf = predict(image_path)
        if pred_label not in class_set:
            msg = f"predict() returned {pred_label!r} for {image_path}, which is not one of {classes!r}"
            raise ValueError(msg)
        predictions.append(Prediction(true=true_label, pred=pred_label, conf=conf))
    return predictions


def _collect_test_files(dataset_root: Path, classes: tuple[str, ...]) -> list[tuple[str, Path]]:
    """List every test-split image under ``dataset_root``, paired with its true class.

    A missing test directory, or a missing or empty class directory, contributes no rows for
    that class. A file whose suffix is not in ``_IMAGE_SUFFIXES`` (for example ``.DS_Store``) is
    skipped.
    """
    test_dir = dataset_root / _TEST_SPLIT
    if not test_dir.is_dir():
        return []
    pairs: list[tuple[str, Path]] = []
    for cat_slug in classes:
        class_dir = test_dir / cat_slug
        if not class_dir.is_dir():
            continue
        pairs.extend(
            (cat_slug, image_path)
            for image_path in sorted(class_dir.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in _IMAGE_SUFFIXES
        )
    return pairs


def _recommend_threshold(abstain: list[AbstainPoint], target_accuracy: float) -> float | None:
    """Return the lowest abstain threshold whose accuracy-on-covered reaches ``target_accuracy``.

    ``abstain`` is in ascending threshold order, so the first qualifying point keeps the most
    coverage among the thresholds that clear the bar. When no threshold reaches
    ``target_accuracy``, this returns ``None``.
    """
    for point in abstain:
        if point.accuracy_on_covered >= target_accuracy:
            return point.threshold
    return None


class _AccuracyScoreFn(Protocol):
    """The one ``sklearn.metrics.accuracy_score`` call shape this module uses."""

    def __call__(self, y_true: list[str], y_pred: list[str]) -> float:
        """Return the fraction of ``y_pred`` entries that equal their ``y_true`` entry."""
        ...


class _ConfusionMatrixFn(Protocol):
    """The one ``sklearn.metrics.confusion_matrix`` call shape this module uses."""

    def __call__(self, y_true: list[str], y_pred: list[str], *, labels: list[str]) -> np.ndarray:
        """Return the confusion matrix of ``y_true`` against ``y_pred``, in ``labels`` order."""
        ...


class _PrecisionRecallFn(Protocol):
    """The one ``sklearn.metrics.precision_recall_fscore_support`` call shape this module uses."""

    def __call__(
        self,
        y_true: list[str],
        y_pred: list[str],
        *,
        labels: list[str],
        zero_division: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return per-label precision, recall, F1, and support arrays, in ``labels`` order."""
        ...


@dataclass(frozen=True)
class _RawScores:
    """One scikit-learn scoring call's results, as plain Python numbers and lists."""

    accuracy: float
    confusion: list[list[int]]
    precision: list[float]
    recall: list[float]
    f1: list[float]
    support: list[int]


def _call_sklearn(y_true: list[str], y_pred: list[str], label_list: list[str]) -> _RawScores:
    """Call scikit-learn once for ``y_true``/``y_pred``, in ``label_list`` order.

    ``labels=label_list`` keeps sklearn's row and column order aligned with ``label_list``, so a
    result never drifts to alphabetical order. This imports scikit-learn here, not at module
    load, so a caller that only reads a report never pulls in a dev-group dependency (deptry
    DEP004).
    """
    # scikit-learn ships no ``py.typed`` marker, and its stub package (``scikit-learn-stubs``)
    # still leaves ``reportUnknownVariableType`` since numpy types its arguments as partially
    # unknown. Each import below is cast to a local protocol just after.
    from sklearn.metrics import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]  # noqa: PLC0415
        accuracy_score,  # pyright: ignore[reportUnknownVariableType]
        confusion_matrix,  # pyright: ignore[reportUnknownVariableType]
        precision_recall_fscore_support,  # pyright: ignore[reportUnknownVariableType]
    )

    accuracy_fn = cast("_AccuracyScoreFn", accuracy_score)
    confusion_fn = cast("_ConfusionMatrixFn", confusion_matrix)
    precision_recall_fn = cast("_PrecisionRecallFn", precision_recall_fscore_support)
    precision, recall, f1, support = precision_recall_fn(y_true, y_pred, labels=label_list, zero_division=0)
    return _RawScores(
        accuracy=accuracy_fn(y_true, y_pred),
        confusion=cast("list[list[int]]", confusion_fn(y_true, y_pred, labels=label_list).tolist()),
        precision=cast("list[float]", precision.tolist()),
        recall=cast("list[float]", recall.tolist()),
        f1=cast("list[float]", f1.tolist()),
        support=cast("list[int]", support.tolist()),
    )


def _score_predictions(
    predictions: list[Prediction],
    classes: tuple[str, ...],
) -> tuple[float, dict[str, ClassMetrics], dict[tuple[str, str], int]]:
    """Score ``predictions`` against ``classes`` with scikit-learn, in ``classes`` order."""
    label_list = list(classes)
    y_true = [p.true for p in predictions]
    y_pred = [p.pred for p in predictions]
    raw = _call_sklearn(y_true, y_pred, label_list)
    per_class = {
        cat_slug: ClassMetrics(
            precision=raw.precision[i],
            recall=raw.recall[i],
            f1=raw.f1[i],
            support=raw.support[i],
        )
        for i, cat_slug in enumerate(label_list)
    }
    confusion = {
        (true_slug, pred_slug): raw.confusion[i][j] for i, true_slug in enumerate(label_list) for j, pred_slug in enumerate(label_list)
    }
    return raw.accuracy, per_class, confusion


def make_predict_fn(model_path: Path) -> PredictFn:
    """Return a ``PredictFn`` that runs the model at ``model_path`` on one image at a time.

    Reads the predicted class name through the loaded model's own ``names`` map, never through
    the sidecar's ``classes`` tuple. Ultralytics indexes ``names`` by sorted training-directory
    name, so its index order need not match the sidecar's recorded order.
    """
    model = load_yolo(model_path)

    def predict(image_path: Path) -> tuple[str, float]:
        results = cast("list[Results]", model(image_path, verbose=False))
        probs = results[0].probs
        if probs is None:
            msg = f"model produced no classification probabilities for {image_path}"
            raise RuntimeError(msg)
        return model.names[probs.top1], float(probs.top1conf)

    return predict


def render_json(report: BenchmarkReport) -> str:
    """Render ``report`` as indented JSON. Confusion keys serialize as ``"true>pred"``."""
    payload = {
        "model_path": report.model_path,
        "classes": list(report.classes),
        "accuracy": report.accuracy,
        "test_count": report.test_count,
        "recommended_threshold": report.recommended_threshold,
        "target_accuracy": report.target_accuracy,
        "localization_misses": report.localization_misses,
        "localize_conf": report.localize_conf,
        "frame_load_failures": report.frame_load_failures,
        "mixed_class_clips": report.mixed_class_clips,
        "candidates": report.candidates,
        "dataset_hash": report.dataset_hash,
        "per_class": {
            cat_slug: {
                "precision": class_metrics.precision,
                "recall": class_metrics.recall,
                "f1": class_metrics.f1,
                "support": class_metrics.support,
            }
            for cat_slug, class_metrics in report.per_class.items()
        },
        "confusion": {f"{true_slug}>{pred_slug}": count for (true_slug, pred_slug), count in report.confusion.items()},
        "abstain": [
            {
                "threshold": point.threshold,
                "coverage": point.coverage,
                "accuracy_on_covered": point.accuracy_on_covered,
            }
            for point in report.abstain
        ],
    }
    return json.dumps(payload, indent=2)


def render_markdown(report: BenchmarkReport) -> str:
    """Render ``report`` as Markdown.

    Includes the confusion table, per-class metrics, the abstain sweep, the recommended
    threshold, and the export diagnostics.
    """
    sections = [
        _render_header(report),
        _render_confusion_table(report),
        _render_per_class_table(report),
        _render_abstain_table(report),
        _render_recommendation(report),
        _render_diagnostics(report),
    ]
    return "\n\n".join(sections) + "\n"


def _render_header(report: BenchmarkReport) -> str:
    """Render the title, model path, class list, dataset hash, test-image count, and overall accuracy."""
    classes_line = ", ".join(report.classes)
    model_line = report.model_path or "(unknown)"
    return (
        f"# Classifier benchmark\n\n"
        f"Model: {model_line}\n"
        f"Classes: {classes_line}\n"
        f"Dataset hash: {report.dataset_hash}\n"
        f"Test images: {report.test_count}\n"
        f"Accuracy: {report.accuracy:.4f}"
    )


def _render_confusion_table(report: BenchmarkReport) -> str:
    """Render the confusion matrix as a Markdown table, rows and columns in ``report.classes`` order."""
    header = "| true \\ pred | " + " | ".join(report.classes) + " |"
    divider = "| --- | " + " | ".join("---" for _ in report.classes) + " |"
    rows = [
        "| " + true_slug + " | " + " | ".join(str(report.confusion[true_slug, pred_slug]) for pred_slug in report.classes) + " |"
        for true_slug in report.classes
    ]
    return "\n".join(["## Confusion matrix", "", header, divider, *rows])


def _render_per_class_table(report: BenchmarkReport) -> str:
    """Render one precision, recall, F1, and support row per class, plus a zero-precision footnote."""
    header = "| class | precision | recall | f1 | support |"
    divider = "| --- | --- | --- | --- | --- |"
    rows = [
        f"| {cat_slug} | {class_metrics.precision:.4f} | {class_metrics.recall:.4f} | {class_metrics.f1:.4f} | {class_metrics.support} |"
        for cat_slug, class_metrics in report.per_class.items()
    ]
    footnote = "_An undefined precision (no prediction landed on a class) prints as `0.0000`. Check the support column._"
    return "\n".join(["## Per-class metrics", "", header, divider, *rows, "", footnote])


def _render_abstain_table(report: BenchmarkReport) -> str:
    """Render one coverage and accuracy-on-covered row per abstain threshold."""
    coverage_note = "Coverage is the fraction the model classifies itself. The rest goes to the unsure bucket."
    header = "| threshold | coverage | accuracy on covered |"
    divider = "| --- | --- | --- |"
    rows = [_render_abstain_row(point) for point in report.abstain]
    return "\n".join(["## Abstain sweep", "", coverage_note, "", header, divider, *rows])


def _render_abstain_row(point: AbstainPoint) -> str:
    """Render one abstain-sweep row. Zero coverage prints its accuracy as ``n/a``, not ``0.0000``.

    Zero coverage means no prediction cleared this threshold, not that the model was wrong every
    time. ``0.0000`` reads as the second claim.
    """
    accuracy_cell = "n/a" if point.coverage == 0.0 else f"{point.accuracy_on_covered:.4f}"
    return f"| {point.threshold:.2f} | {point.coverage:.4f} | {accuracy_cell} |"


def _render_recommendation(report: BenchmarkReport) -> str:
    """Render the recommended abstain threshold. If none qualifies, render the best available row instead."""
    target_text = _format_target_accuracy(report.target_accuracy)
    if report.recommended_threshold is None:
        best = _best_available_point(report.abstain)
        best_line = (
            f"  Best available: {best.threshold:.2f} -> coverage {best.coverage:.2f}, accuracy {best.accuracy_on_covered:.2f}"
            if best is not None
            else "  No threshold covers any prediction."
        )
        return "\n".join(
            [
                "## Recommended threshold",
                "",
                "Recommended threshold: none",
                f"  No threshold reaches the {target_text} target.",
                best_line,
            ],
        )
    point = _abstain_point_at(report.abstain, report.recommended_threshold)
    return "\n".join(
        [
            "## Recommended threshold",
            "",
            f"Recommended threshold: {report.recommended_threshold:.2f}",
            f"  coverage {point.coverage:.2f}, accuracy-on-covered {point.accuracy_on_covered:.2f}",
            f"  (lowest threshold meeting the {target_text} target)",
        ],
    )


def _format_target_accuracy(target_accuracy: float) -> str:
    """Format ``target_accuracy`` without rounding it into a different number.

    ``.2f`` prints ``0.999`` as ``1.00``, which reads as a different target. ``g`` keeps every
    significant digit up to its default precision instead.
    """
    return f"{target_accuracy:g}"


def _abstain_point_at(abstain: list[AbstainPoint], threshold: float) -> AbstainPoint:
    """Return the abstain-sweep point at exactly ``threshold``."""
    for point in abstain:
        if point.threshold == threshold:
            return point
    msg = f"no abstain point at threshold {threshold!r}"
    raise ValueError(msg)


def _best_available_point(abstain: list[AbstainPoint]) -> AbstainPoint | None:
    """Return the covered point with the highest accuracy-on-covered, ties toward the lower threshold.

    Only a point with ``coverage > 0.0`` qualifies, so a threshold that covers nothing never
    reads as "best". When no point covers any prediction, this returns ``None``.
    """
    covered = [point for point in abstain if point.coverage > 0.0]
    if not covered:
        return None
    return max(covered, key=lambda point: (point.accuracy_on_covered, -point.threshold))


def _render_diagnostics(report: BenchmarkReport) -> str:
    """Render the export-manifest diagnostics. The miss count names its source confidence, not recall."""
    return "\n".join(
        [
            "## Export diagnostics",
            "",
            f"- Localization misses (conf {report.localize_conf:.2f}): {report.localization_misses}/{report.candidates}",
            f"- Frame load failures: {report.frame_load_failures}",
            f"- Mixed-class clips: {report.mixed_class_clips}",
        ],
    )


def write_reports(report: BenchmarkReport, *, reports_dir: Path) -> tuple[Path, Path]:
    """Write ``report`` as Markdown and JSON under ``reports_dir``, and return both paths.

    Named after ``report.model_path``'s stem, so a second model's report does not overwrite the
    first. A report with no ``model_path`` set falls back to the fixed ``benchmark`` stem.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(report.model_path).stem if report.model_path else _DEFAULT_REPORT_STEM
    md_path = reports_dir / f"{stem}.md"
    json_path = reports_dir / f"{stem}.json"
    _ = md_path.write_text(render_markdown(report), encoding="utf-8")
    _ = json_path.write_text(render_json(report), encoding="utf-8")
    return md_path, json_path

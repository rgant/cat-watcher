"""CLI wiring for ``cat-watcher classifier export|train|benchmark|predict``, over production adapters.

Each action resolves its paths from :class:`~cat_watcher.config.Config`, builds the ``sources``
adapters or the injected stage function, and maps the result to a process exit code. Every path
this module reads or writes lives under ``config.storage_root / "classifier"`` (dataset, reports,
training runs, spot-check crops) or ``config.internal_root / "models"`` (weights).
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

from cat_watcher.classifier import benchmark, predict, sources, train
from cat_watcher.classifier.dataset import DATASET_SUBDIR, ExportError, ExportSources, export_dataset, read_manifest
from cat_watcher.classifier.labels_query import query_single_cat_frames, query_untagged_cat_clip_frames, resolve_cat_classes
from cat_watcher.classifier.splitting import SEED
from cat_watcher.db import engine_for
from cat_watcher.logs_viewer import parse_since

if TYPE_CHECKING:
    from datetime import datetime

    from cat_watcher.classifier.dataset import ExportSummary
    from cat_watcher.config import Config

_MODELS_SUBDIR: str = "models"
_RUNS_SUBDIR: str = "classifier/runs"
_MANIFEST_FILENAME: str = "manifest.json"
_CHECKPOINT_GLOB: str = "cat-classifier-*.pt"
_DEFAULT_PREDICT_LIMIT: int = 25

_EXIT_OK: int = 0
_EXIT_GENERIC_FAILURE: int = 1
_EXIT_MISSING_DEPENDENCY: int = 5


def _parse_since_arg(value: str) -> datetime:
    """Parse ``--since``/``--until`` via ``parse_since``, with an argparse-friendly error on a bad value.

    A bare ``ValueError`` from ``parse_since`` prints as ``invalid parse_since value: ...``,
    which names a function, not an accepted form. ``ArgumentTypeError`` lets argparse print this
    message instead, so an operator learns what to type.
    """
    try:
        return parse_since(value)
    except ValueError as exc:
        msg = f"not a valid time: {value!r} (use a duration like 7d, 1h, 30m, or an ISO 8601 date like 2026-07-01)"
        raise argparse.ArgumentTypeError(msg) from exc


class ClassifierNamespace(argparse.Namespace):
    """Typed namespace for ``cat-watcher classifier`` (matches the dests in :func:`configure_classifier_parser`).

    Subclasses (notably ``_ParsedArgs`` in :mod:`cat_watcher.__main__`) inherit these fields so the
    umbrella's parser keeps a single namespace covering every sub-command's flags.
    """

    action: str = ""
    epochs: int = train.EPOCHS
    imgsz: int = train.IMGSZ
    seed: int = SEED
    model: Path | None = None
    # ``predict_*`` dests keep ``classifier predict``'s flags apart from same-named flags on other
    # sub-commands (``import-local --camera``, ``reanalyze --limit``), the same trick
    # ``LogsNamespace`` uses for ``logs --camera``.
    predict_camera: str | None = None
    predict_since: datetime | None = None
    predict_until: datetime | None = None
    predict_limit: int = _DEFAULT_PREDICT_LIMIT
    threshold: float | None = None
    target_accuracy: float = benchmark.DEFAULT_TARGET_ACCURACY


def configure_classifier_parser(subparser: argparse.ArgumentParser) -> None:
    """Attach the ``export`` | ``train`` | ``benchmark`` | ``predict`` classifier actions onto ``subparser``."""
    actions = subparser.add_subparsers(dest="action", required=True)

    _ = actions.add_parser("export", help="Export a Marcel-vs-Rufus crop dataset from operator labels")

    train_parser = actions.add_parser("train", help="Train yolo11n-cls on the exported dataset")
    _ = train_parser.add_argument("--epochs", type=int, default=train.EPOCHS, help=f"Training epochs (default {train.EPOCHS})")
    _ = train_parser.add_argument("--imgsz", type=int, default=train.IMGSZ, help=f"Training image size (default {train.IMGSZ})")
    _ = train_parser.add_argument("--seed", type=int, default=SEED, help=f"Split and train seed (default {SEED})")

    benchmark_parser = actions.add_parser("benchmark", help="Benchmark a trained checkpoint on its held-out test split")
    _ = benchmark_parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Checkpoint to benchmark; default is the newest cat-classifier-*.pt in <internal_root>/models",
    )
    _ = benchmark_parser.add_argument(
        "--target-accuracy",
        type=float,
        default=benchmark.DEFAULT_TARGET_ACCURACY,
        help=f"Accuracy target for the recommended abstain threshold (default {benchmark.DEFAULT_TARGET_ACCURACY})",
    )

    predict_parser = actions.add_parser(
        "predict",
        help="Spot-check untagged clips with a trained checkpoint (read-only, writes no DB row)",
    )
    _ = predict_parser.add_argument(
        "--camera",
        dest="predict_camera",
        default=None,
        metavar="CAMERA",
        help="Restrict to one configured camera name",
    )
    _ = predict_parser.add_argument(
        "--since",
        dest="predict_since",
        type=_parse_since_arg,
        default=None,
        metavar="SINCE",
        help="Only clips at or after this time (duration shorthand like 7d, or ISO 8601)",
    )
    _ = predict_parser.add_argument(
        "--until",
        dest="predict_until",
        type=_parse_since_arg,
        default=None,
        metavar="UNTIL",
        help="Only clips at or before this time (duration shorthand like 7d, or ISO 8601)",
    )
    _ = predict_parser.add_argument(
        "--limit",
        dest="predict_limit",
        type=int,
        default=_DEFAULT_PREDICT_LIMIT,
        metavar="N",
        help=f"Clips per batch, small enough to check by eye (default {_DEFAULT_PREDICT_LIMIT}). Every frame of each clip is scored.",
    )
    _ = predict_parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Checkpoint to score with; default is the newest cat-classifier-*.pt in <internal_root>/models",
    )
    _ = predict_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence floor for 'unsure'; default is the model's recommended threshold, else 0.5",
    )


def run(args: ClassifierNamespace, *, config: Config) -> int:
    """Dispatch to the chosen classifier action. Returns a process exit code.

    Exit 0 means success. Exit 5 means a required input is absent: the detector weights, the
    exported dataset, the classifier base weights, or a trained checkpoint.
    """
    if args.action == "export":
        return _run_export(config)
    if args.action == "train":
        return _run_train(config, args)
    if args.action == "benchmark":
        return _run_benchmark(config, args)
    if args.action == "predict":
        return _run_predict(config, args)
    msg = f"unknown classifier action: {args.action!r}"  # pragma: no cover  # unreachable: argparse required=True
    raise ValueError(msg)  # pragma: no cover


# --- export ----------------------------------------------------------------------------------------


def _run_export(config: Config) -> int:
    """Export a crop dataset from operator-labeled frames. Returns an exit code."""
    weights = _models_dir(config) / config.detector.model
    if not weights.is_file():
        _ = sys.stderr.write(
            f"classifier export: detector weights not found at {weights}; run `cat-watcher fetch-models` first\n",
        )
        return _EXIT_MISSING_DEPENDENCY

    engine = engine_for(config.internal_root)
    try:
        classes = resolve_cat_classes(engine)
        rows = query_single_cat_frames(engine)
    finally:
        engine.dispose()

    export_sources = ExportSources(
        frame_source=sources.make_frame_source(storage_root=config.storage_root),
        localizer=sources.make_localizer(model_path=weights),
    )
    try:
        manifest = export_dataset(rows, classes=classes, dataset_root=_dataset_root(config), sources=export_sources)
    except ExportError as exc:
        _ = sys.stderr.write(f"classifier export: {exc}\n")
        return _EXIT_GENERIC_FAILURE
    _print_export_summary(manifest.summary)
    return _EXIT_OK


def _print_export_summary(summary: ExportSummary) -> None:
    """Write the export totals as ``key=value`` tokens, one line per statistic."""
    per_class = " ".join(f"{slug}={count}" for slug, count in summary.crops_per_class.items())
    per_source = " ".join(f"{source}={count}" for source, count in summary.crops_by_source.items())
    _ = sys.stdout.write(f"classifier export: candidates={summary.candidates} classes: {per_class}\n")
    _ = sys.stdout.write(f"  crops by source: {per_source}\n")
    _ = sys.stdout.write(f"  localization_misses={summary.localization_misses} frame_load_failures={summary.frame_load_failures}\n")
    _ = sys.stdout.write(f"  mixed_class_clips={summary.mixed_class_clips}\n")
    _ = sys.stdout.write(f"  dataset_hash={summary.dataset_hash}\n")


# --- train -----------------------------------------------------------------------------------------


def _run_train(config: Config, args: ClassifierNamespace) -> int:
    """Train a classifier checkpoint from the exported dataset. Returns an exit code."""
    manifest_path = _manifest_path(config)
    if not manifest_path.is_file():
        _ = sys.stderr.write(
            f"classifier train: no manifest at {manifest_path}; run `cat-watcher classifier export` first\n",
        )
        return _EXIT_MISSING_DEPENDENCY

    models_dir = _models_dir(config)
    base_weights = models_dir / train.BASE_WEIGHTS
    if not base_weights.is_file():
        _ = sys.stderr.write(
            f"classifier train: base weights not found at {base_weights}; "
            f"run `cat-watcher fetch-models --model {train.BASE_WEIGHTS}` first\n",
        )
        return _EXIT_MISSING_DEPENDENCY

    paths = train.TrainPaths(
        dataset_root=_dataset_root(config),
        manifest_path=manifest_path,
        models_dir=models_dir,
        run_dir=config.storage_root / _RUNS_SUBDIR,
        base_weights=base_weights,
    )
    params = train.TrainParams(epochs=args.epochs, imgsz=args.imgsz, seed=args.seed)
    result = train.train_classifier(paths=paths, params=params)
    _ = sys.stdout.write(f"classifier train: wrote {result.model_path} (epochs={result.epochs} imgsz={result.imgsz})\n")
    return _EXIT_OK


# --- benchmark ---------------------------------------------------------------------------------------


def _run_benchmark(config: Config, args: ClassifierNamespace) -> int:
    """Benchmark a trained checkpoint against its held-out test split. Returns an exit code."""
    models_dir = _models_dir(config)
    model_path = args.model if args.model is not None else _newest_checkpoint(models_dir)
    if model_path is None or not model_path.is_file():
        _ = sys.stderr.write(
            f"classifier benchmark: no checkpoint found at {model_path or models_dir}; run `cat-watcher classifier train` first\n",
        )
        return _EXIT_MISSING_DEPENDENCY

    sidecar_path = model_path.with_suffix(".json")
    if not sidecar_path.is_file():
        _ = sys.stderr.write(f"classifier benchmark: no sidecar at {sidecar_path}; the checkpoint is incomplete\n")
        return _EXIT_MISSING_DEPENDENCY

    sidecar = benchmark.read_sidecar(model_path)
    manifest_path = _manifest_path(config)
    manifest = read_manifest(manifest_path)
    if sidecar.dataset_hash != manifest.summary.dataset_hash:
        _ = sys.stderr.write(
            f"classifier benchmark: stale model. {model_path} trained on dataset_hash="
            f"{sidecar.dataset_hash}, but {manifest_path} now holds dataset_hash="
            f"{manifest.summary.dataset_hash}. Re-train the model or re-export the dataset so they match.\n",
        )
        return _EXIT_MISSING_DEPENDENCY

    predict_fn = benchmark.make_predict_fn(model_path)
    report = benchmark.benchmark_model(
        dataset_root=_dataset_root(config),
        manifest_path=manifest_path,
        classes=sidecar.classes,
        predict=predict_fn,
        target_accuracy=args.target_accuracy,
    )
    report = dataclasses.replace(report, model_path=str(model_path))
    md_path, json_path = benchmark.write_reports(report, reports_dir=_reports_dir(config))
    _ = sys.stdout.write(f"classifier benchmark: accuracy={report.accuracy:.4f} test_count={report.test_count}\n")
    _ = sys.stdout.write(f"  wrote {md_path}\n  wrote {json_path}\n")
    return _EXIT_OK


# --- predict -----------------------------------------------------------------------------------------


def _run_predict(config: Config, args: ClassifierNamespace) -> int:
    """Spot-check untagged clips with a trained checkpoint. Writes no database row."""
    models_dir = _models_dir(config)
    model_path = args.model if args.model is not None else _newest_checkpoint(models_dir)
    if model_path is None or not model_path.is_file():
        _ = sys.stderr.write(
            f"classifier predict: no checkpoint found at {model_path or models_dir}; run `cat-watcher classifier train` first\n",
        )
        return _EXIT_MISSING_DEPENDENCY

    sidecar_path = model_path.with_suffix(".json")
    if not sidecar_path.is_file():
        _ = sys.stderr.write(f"classifier predict: no sidecar at {sidecar_path}; the checkpoint is incomplete\n")
        return _EXIT_MISSING_DEPENDENCY

    weights = models_dir / config.detector.model
    if not weights.is_file():
        _ = sys.stderr.write(
            f"classifier predict: detector weights not found at {weights}; run `cat-watcher fetch-models` first\n",
        )
        return _EXIT_MISSING_DEPENDENCY

    threshold, threshold_source = _resolve_threshold(args.threshold, _reports_dir(config), model_path)

    engine = engine_for(config.internal_root)
    try:
        candidates = query_untagged_cat_clip_frames(
            engine,
            camera=args.predict_camera,
            since=args.predict_since,
            until=args.predict_until,
            limit=args.predict_limit,
        )
    finally:
        engine.dispose()

    predict_sources = ExportSources(
        frame_source=sources.make_frame_source(storage_root=config.storage_root),
        localizer=sources.make_localizer(model_path=weights),
    )
    predict_fn = benchmark.make_predict_fn(model_path)
    options = predict.PredictOptions(crops_dir=config.storage_root / predict.SPOTCHECK_SUBDIR, threshold=threshold)
    verdicts = predict.predict_clips(candidates, sources=predict_sources, predict=predict_fn, options=options)

    _ = sys.stdout.write(
        f"classifier predict: {len(verdicts)} clip(s), {len(candidates)} frame(s), threshold={threshold:.2f} (source={threshold_source})\n",
    )
    if verdicts:
        _ = sys.stdout.write(predict.render_rows(verdicts, tz=ZoneInfo(config.web.display_timezone)) + "\n")
    return _EXIT_OK


def _resolve_threshold(explicit: float | None, reports_dir: Path, model_path: Path) -> tuple[float, str]:
    """Resolve the unsure-confidence threshold. ``--threshold`` wins, then this model's own benchmark report, then the default.

    A report is named after its model's stem (:func:`benchmark.write_reports`), so this reads
    only the report that scored ``model_path``, never an unrelated model's report. Returns the
    threshold and a short label that names its source. The printed output shows the label, so the
    operator knows the basis for each row.
    """
    if explicit is not None:
        return explicit, "--threshold"
    report_path = reports_dir / f"{model_path.stem}.json"
    if report_path.is_file():
        payload = cast("dict[str, object]", json.loads(report_path.read_text(encoding="utf-8")))
        recommended = payload.get("recommended_threshold")
        if isinstance(recommended, int | float):
            return float(recommended), "benchmark report"
    return predict.DEFAULT_THRESHOLD, "default"


def _newest_checkpoint(models_dir: Path) -> Path | None:
    """Return the newest ``cat-classifier-*.pt`` under ``models_dir``. If none exist, this returns ``None``."""
    if not models_dir.is_dir():
        return None
    checkpoints = sorted(models_dir.glob(_CHECKPOINT_GLOB), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return checkpoints[0] if checkpoints else None


# --- shared path resolution --------------------------------------------------------------------------


def _dataset_root(config: Config) -> Path:
    return config.storage_root / DATASET_SUBDIR


def _manifest_path(config: Config) -> Path:
    return _dataset_root(config) / _MANIFEST_FILENAME


def _models_dir(config: Config) -> Path:
    return config.internal_root / _MODELS_SUBDIR


def _reports_dir(config: Config) -> Path:
    return config.storage_root / benchmark.REPORTS_SUBDIR


__all__ = ["ClassifierNamespace", "configure_classifier_parser", "run"]

"""Unit tests for :mod:`cat_watcher.classifier.cli` and its ``__main__.py`` wiring.

Every export test patches ``sources.make_frame_source`` and ``make_localizer``. Every
train/benchmark happy-path test patches ``train.train_classifier`` or
``benchmark.benchmark_model``, plus ``benchmark.make_predict_fn``, since the on-disk checkpoint
holds placeholder bytes, not a real model. No real ffmpeg or YOLO call happens in this file.
"""

import argparse
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest
from classifier_helpers import DEFAULT_DATASET_HASH, write_checkpoint_with_sidecar, write_export_manifest
from cli_test_helpers import assert_missing_dependency, config_with_dirs, init_schema, make_classifier_args
from db_helpers import build_test_clip, make_clip_frame, seed_cat_subject
from image_helpers import gradient_rgb

from cat_watcher.__main__ import _build_parser, _ParsedArgs, main
from cat_watcher.classifier.benchmark import REPORTS_SUBDIR as CLASSIFIER_REPORTS_SUBDIR
from cat_watcher.classifier.benchmark import BenchmarkReport, ClassMetrics
from cat_watcher.classifier.cli import ClassifierNamespace, run
from cat_watcher.classifier.dataset import DATASET_SUBDIR, LoadedFrame, LocalizedBox
from cat_watcher.classifier.metrics import AbstainPoint
from cat_watcher.classifier.train import BASE_WEIGHTS, TrainParams, TrainPaths, TrainResult
from cat_watcher.db import Camera, ClipFrameSubject, PollStatus, create_engine, get_session

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from sqlalchemy.engine import Engine

    from cat_watcher.classifier.labels_query import CatFrameRow
    from cat_watcher.config import Config

_START = datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC)
_CLIPS_PER_CLASS = 3  # one clip lands in each of train/val/test (splitting._MIN_CLIPS_PER_CLASS)
_FRAMES_PER_CLIP = 8  # 3 clips * 8 frames = 24 crops per class, above dataset.MIN_CROPS_PER_CLASS (20)


def _db_engine_for(config: Config) -> Engine:
    return create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")


def _seed_export_rows(config: Config) -> None:
    """Seed 2 cats x 3 clips x 8 frames, enough to clear the export floor guard in every split."""
    engine = _db_engine_for(config)
    try:
        with get_session(engine) as session:
            cam = Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)
            session.add(cam)
            session.flush()
            cam_id = cam.id
        for class_index, slug in enumerate(("marcel", "rufus")):
            subject_id = seed_cat_subject(engine, slug=slug, display_name=slug.title(), display_order=class_index + 1)
            for clip_index in range(_CLIPS_PER_CLASS):
                start = _START + timedelta(minutes=100 * class_index + clip_index)
                with get_session(engine) as session:
                    clip = build_test_clip(cam_id, start_ts=start, source_filename=f"{slug}-{clip_index}.mp4")
                    session.add(clip)
                    session.flush()
                    clip_id = clip.id
                    for ordinal in range(_FRAMES_PER_CLIP):
                        frame = make_clip_frame(clip_id, ordinal)
                        session.add(frame)
                        session.flush()
                        session.add(ClipFrameSubject(clip_frame_id=frame.id, subject_id=subject_id))
    finally:
        engine.dispose()


def _write_weights(config: Config) -> Path:
    """Write a stub detector weights file at the path ``_run_export`` checks for."""
    weights = config.internal_root / "models" / config.detector.model
    weights.parent.mkdir(parents=True, exist_ok=True)
    _ = weights.write_bytes(b"stub-detector-weights")
    return weights


def _fake_frame_source(_row: CatFrameRow) -> LoadedFrame | None:
    return LoadedFrame(image=gradient_rgb(32, 32), source="clip")


def _fake_localizer(_image: np.ndarray) -> LocalizedBox | None:
    return LocalizedBox(box=(2.0, 2.0, 20.0, 20.0), conf=0.8)


def _fake_predict(_image_path: Path) -> tuple[str, float]:
    """Stand in for ``PredictFn`` in a benchmark test. The checkpoint on disk holds no real weights."""
    return "marcel", 0.9


def _fake_benchmark_report(classes: tuple[str, ...]) -> BenchmarkReport:
    """Build a minimal ``BenchmarkReport`` that renders. ``write_reports`` then runs for real on it."""
    per_class = {slug: ClassMetrics(precision=1.0, recall=1.0, f1=1.0, support=1) for slug in classes}
    confusion = {(true_slug, pred_slug): 1 if true_slug == pred_slug else 0 for true_slug in classes for pred_slug in classes}
    return BenchmarkReport(
        classes=classes,
        accuracy=1.0,
        per_class=per_class,
        confusion=confusion,
        abstain=[AbstainPoint(threshold=0.0, coverage=1.0, accuracy_on_covered=1.0)],
        recommended_threshold=0.0,
        target_accuracy=0.95,
        localization_misses=0,
        localize_conf=0.10,
        frame_load_failures=0,
        mixed_class_clips=0,
        candidates=len(classes),
        dataset_hash=DEFAULT_DATASET_HASH,
        test_count=len(classes),
    )


def _assert_no_placeholder_leak(text: str) -> None:
    """CLI output tripwire (CLAUDE.md): a missing ``f`` prefix prints the placeholder text unformatted."""
    for token in ("{_fmt", "{self.", "{cam.", "{cfg.", "NoneType"):
        assert token not in text


def _find_subparsers_action(parser: argparse.ArgumentParser) -> argparse.Action:
    """Find ``parser``'s own nested sub-parsers action. argparse exposes no public API for this."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return cast("argparse.Action", action)
    msg = f"{parser.prog!r} defines no nested sub-parsers"
    raise ValueError(msg)


def _sub_parser(parser: argparse.ArgumentParser, *names: str) -> argparse.ArgumentParser:
    """Walk nested sub-parsers by name, such as ``("classifier", "predict")``."""
    current = parser
    for name in names:
        subparsers_action = cast("argparse._SubParsersAction[argparse.ArgumentParser]", _find_subparsers_action(current))
        current = subparsers_action.choices[name]
    return current


# --- argparse wiring -------------------------------------------------------------------------------


def test_build_parser_help_lists_classifier() -> None:
    """The umbrella parser registers ``classifier`` and surfaces it in help text."""
    parser = _build_parser()
    help_text = parser.format_help()
    assert "classifier" in help_text


def test_fetch_models_and_benchmark_model_flags_do_not_collide() -> None:
    """``fetch-models --model`` and ``classifier benchmark --model`` write to distinct dests."""
    parser = _build_parser()

    fetch_args = parser.parse_args(["fetch-models", "--model", "yolo11n-cls.pt"], namespace=_ParsedArgs())
    assert fetch_args.fetch_model == "yolo11n-cls.pt"
    assert fetch_args.model is None

    benchmark_args = parser.parse_args(["classifier", "benchmark", "--model", "x.pt"], namespace=_ParsedArgs())
    assert benchmark_args.model == Path("x.pt")
    assert benchmark_args.fetch_model is None


def test_classifier_train_flags_parse() -> None:
    """``classifier train --epochs --imgsz --seed`` parse into the typed namespace."""
    parser = _build_parser()
    args = parser.parse_args(
        ["classifier", "train", "--epochs", "5", "--imgsz", "128", "--seed", "7"],
        namespace=_ParsedArgs(),
    )
    assert args.command == "classifier"
    assert args.action == "train"
    assert args.epochs == 5
    assert args.imgsz == 128
    assert args.seed == 7


def test_classifier_subparser_requires_an_action() -> None:
    """``cat-watcher classifier`` with no action raises argparse's exit-2 error."""
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        _ = parser.parse_args(["classifier"], namespace=_ParsedArgs())
    assert exc.value.code != 0


def test_classifier_predict_help_does_not_leak_the_dest_names() -> None:
    """``classifier predict --help`` shows ``CAMERA``/``SINCE``/``UNTIL``/``N``, not the ``predict_*`` dest."""
    predict_parser = _sub_parser(_build_parser(), "classifier", "predict")
    help_text = predict_parser.format_help()
    assert "PREDICT_" not in help_text
    assert "--camera CAMERA" in help_text
    assert "--since SINCE" in help_text
    assert "--until UNTIL" in help_text
    assert "--limit N" in help_text


# --- dispatch through main() ------------------------------------------------------------------------


def test_main_classifier_export_dispatches_to_cli_run(
    tmp_path: Path,
    make_config: Callable[..., Config],
    restore_root_logger: object,
) -> None:
    """``main(["classifier", "export"])`` reaches ``classifier.cli.run``, not the parser alone."""
    _ = restore_root_logger
    config = config_with_dirs(tmp_path, make_config)
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.run_classifier", return_value=0) as run_mock,
    ):
        exit_code = main(["classifier", "export"])
    assert exit_code == 0
    run_mock.assert_called_once()
    passed_args = cast("ClassifierNamespace", run_mock.call_args.args[0])
    assert passed_args.action == "export"
    assert run_mock.call_args.kwargs["config"] is config


# --- export ------------------------------------------------------------------------------------------


def test_classifier_export_with_missing_detector_weights_exits_missing_dependency(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No weights at ``internal_root/models/<detector.model>`` -> exit 5, `fetch-models` named on stderr."""
    config = config_with_dirs(tmp_path, make_config)
    exit_code = run(make_classifier_args("export"), config=config)
    err = capsys.readouterr().err
    assert_missing_dependency(exit_code, err, names="fetch-models")


def test_classifier_export_writes_dataset_and_manifest(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A seeded, labeled DB produces crop files and ``manifest.json`` under the dataset root, exit 0."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _seed_export_rows(config)
    _ = _write_weights(config)

    with (
        patch("cat_watcher.classifier.sources.make_frame_source", return_value=_fake_frame_source),
        patch("cat_watcher.classifier.sources.make_localizer", return_value=_fake_localizer),
    ):
        exit_code = run(make_classifier_args("export"), config=config)

    assert exit_code == 0
    dataset_root = config.storage_root / DATASET_SUBDIR
    assert (dataset_root / "manifest.json").is_file()
    train_marcel = list((dataset_root / "train" / "marcel").glob("*.jpg"))
    assert train_marcel


def test_classifier_export_prints_summary_with_named_counts(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The export summary names candidate/class/source/miss counts by token, not by a bare digit."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _seed_export_rows(config)
    _ = _write_weights(config)

    with (
        patch("cat_watcher.classifier.sources.make_frame_source", return_value=_fake_frame_source),
        patch("cat_watcher.classifier.sources.make_localizer", return_value=_fake_localizer),
    ):
        exit_code = run(make_classifier_args("export"), config=config)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "candidates=48" in out
    assert "marcel=24" in out
    assert "rufus=24" in out
    assert "clip=48" in out
    assert "localization_misses=0" in out
    assert "mixed_class_clips=0" in out
    _assert_no_placeholder_leak(out)


def test_classifier_export_maps_export_error_to_nonzero_exit_with_stderr_message(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A class with zero labeled crops trips the floor guard: non-zero exit, ``ExportError`` text on stderr."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    engine = _db_engine_for(config)
    try:
        _ = seed_cat_subject(engine, slug="marcel", display_order=1)
        _ = seed_cat_subject(engine, slug="rufus", display_order=2)
    finally:
        engine.dispose()
    _ = _write_weights(config)

    with (
        patch("cat_watcher.classifier.sources.make_frame_source", return_value=_fake_frame_source),
        patch("cat_watcher.classifier.sources.make_localizer", return_value=_fake_localizer),
    ):
        exit_code = run(make_classifier_args("export"), config=config)

    assert exit_code != 0
    err = capsys.readouterr().err
    assert "below the minimum" in err
    _assert_no_placeholder_leak(err)


# --- train ---------------------------------------------------------------------------------------


def test_classifier_train_missing_manifest_exits_missing_dependency(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No exported dataset -> exit 5, ``classifier export`` named on stderr."""
    config = config_with_dirs(tmp_path, make_config)
    exit_code = run(make_classifier_args("train"), config=config)
    err = capsys.readouterr().err
    assert_missing_dependency(exit_code, err, names="classifier export")


def test_classifier_train_missing_base_weights_exits_missing_dependency(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A manifest with no ``yolo11n-cls.pt`` present -> exit 5, the exact fetch-models fix on stderr."""
    config = config_with_dirs(tmp_path, make_config)
    dataset_root = config.storage_root / DATASET_SUBDIR
    write_export_manifest(dataset_root / "manifest.json", classes=("marcel", "rufus"))

    exit_code = run(make_classifier_args("train"), config=config)

    err = capsys.readouterr().err
    assert_missing_dependency(exit_code, err, names=f"fetch-models --model {BASE_WEIGHTS}")


def test_classifier_train_happy_path_invokes_train_classifier_and_exits_ok(
    tmp_path: Path,
    make_config: Callable[..., Config],
) -> None:
    """With a manifest and base weights present, ``run`` calls ``train_classifier`` with the CLI's paths/params."""
    config = config_with_dirs(tmp_path, make_config)
    dataset_root = config.storage_root / DATASET_SUBDIR
    manifest_path = dataset_root / "manifest.json"
    write_export_manifest(manifest_path, classes=("marcel", "rufus"))
    base_weights = config.internal_root / "models" / BASE_WEIGHTS
    base_weights.parent.mkdir(parents=True, exist_ok=True)
    _ = base_weights.write_bytes(b"stub-base-weights")

    fake_result = TrainResult(
        model_path=config.internal_root / "models" / "cat-classifier-deadbeef.pt",
        sidecar_path=config.internal_root / "models" / "cat-classifier-deadbeef.json",
        classes=("marcel", "rufus"),
        model_names={0: "marcel", 1: "rufus"},
        epochs=3,
        imgsz=96,
        dataset_hash=DEFAULT_DATASET_HASH,
    )
    with patch("cat_watcher.classifier.train.train_classifier", return_value=fake_result) as train_mock:
        exit_code = run(make_classifier_args("train", epochs=3, imgsz=96, seed=42), config=config)

    assert exit_code == 0
    train_mock.assert_called_once()
    call_kwargs = train_mock.call_args.kwargs
    train_paths = cast("TrainPaths", call_kwargs["paths"])
    train_params = cast("TrainParams", call_kwargs["params"])
    assert train_paths.dataset_root == dataset_root
    assert train_paths.manifest_path == manifest_path
    assert train_paths.base_weights == base_weights
    assert train_params.epochs == 3
    assert train_params.imgsz == 96
    assert train_params.seed == 42


# --- benchmark ---------------------------------------------------------------------------------------


def test_classifier_benchmark_missing_model_exits_missing_dependency(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No checkpoint under ``internal_root/models`` -> exit 5, ``classifier train`` named on stderr."""
    config = config_with_dirs(tmp_path, make_config)
    exit_code = run(make_classifier_args("benchmark"), config=config)
    err = capsys.readouterr().err
    assert_missing_dependency(exit_code, err, names="classifier train")


def test_classifier_benchmark_checkpoint_without_sidecar_exits_missing_dependency(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``.pt`` file with no matching ``.json`` sidecar -> exit 5, not an uncaught ``FileNotFoundError``."""
    config = config_with_dirs(tmp_path, make_config)
    models_dir = config.internal_root / "models"
    models_dir.mkdir(parents=True)
    _ = (models_dir / "cat-classifier-deadbeef.pt").write_bytes(b"stub")

    exit_code = run(make_classifier_args("benchmark"), config=config)

    err = capsys.readouterr().err
    assert_missing_dependency(exit_code, err, names="sidecar")


def test_classifier_benchmark_defaults_to_the_newest_checkpoint(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """With no ``--model``, ``run`` reads the sidecar of the checkpoint with the newest mtime."""
    config = config_with_dirs(tmp_path, make_config)
    models_dir = config.internal_root / "models"
    models_dir.mkdir(parents=True)
    dataset_root = config.storage_root / DATASET_SUBDIR
    manifest_path = dataset_root / "manifest.json"
    write_export_manifest(manifest_path, classes=("marcel", "rufus"))

    older = models_dir / "cat-classifier-aaaaaaaa.pt"
    newer = models_dir / "cat-classifier-bbbbbbbb.pt"
    write_checkpoint_with_sidecar(older, classes=("marcel", "rufus"), manifest_path=manifest_path)
    write_checkpoint_with_sidecar(newer, classes=("rufus", "marcel"), manifest_path=manifest_path)
    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))

    fake_report = _fake_benchmark_report(("rufus", "marcel"))
    with (
        patch("cat_watcher.classifier.benchmark.make_predict_fn", return_value=_fake_predict),
        patch("cat_watcher.classifier.benchmark.benchmark_model", return_value=fake_report) as bench_mock,
    ):
        exit_code = run(make_classifier_args("benchmark"), config=config)

    assert exit_code == 0
    bench_mock.assert_called_once()
    call_kwargs = bench_mock.call_args.kwargs
    assert call_kwargs["classes"] == ("rufus", "marcel")
    assert call_kwargs["dataset_root"] == dataset_root
    assert call_kwargs["manifest_path"] == manifest_path
    reports_dir = config.storage_root / CLASSIFIER_REPORTS_SUBDIR
    assert (reports_dir / f"{newer.stem}.md").is_file()
    assert (reports_dir / f"{newer.stem}.json").is_file()


def test_classifier_benchmark_respects_an_explicit_model_flag(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``--model`` overrides the newest-checkpoint default and its own sidecar's classes are used."""
    config = config_with_dirs(tmp_path, make_config)
    models_dir = config.internal_root / "models"
    models_dir.mkdir(parents=True)
    dataset_root = config.storage_root / DATASET_SUBDIR
    manifest_path = dataset_root / "manifest.json"
    write_export_manifest(manifest_path, classes=("marcel", "rufus"))

    older = models_dir / "cat-classifier-aaaaaaaa.pt"
    newer = models_dir / "cat-classifier-bbbbbbbb.pt"
    write_checkpoint_with_sidecar(older, classes=("marcel", "rufus"), manifest_path=manifest_path)
    write_checkpoint_with_sidecar(newer, classes=("rufus", "marcel"), manifest_path=manifest_path)
    now = time.time()
    os.utime(older, (now - 100, now - 100))
    os.utime(newer, (now, now))

    fake_report = _fake_benchmark_report(("marcel", "rufus"))
    with (
        patch("cat_watcher.classifier.benchmark.make_predict_fn", return_value=_fake_predict),
        patch("cat_watcher.classifier.benchmark.benchmark_model", return_value=fake_report) as bench_mock,
    ):
        exit_code = run(make_classifier_args("benchmark", model=older), config=config)

    assert exit_code == 0
    call_kwargs = bench_mock.call_args.kwargs
    assert call_kwargs["classes"] == ("marcel", "rufus")


def test_classifier_benchmark_passes_target_accuracy_through(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """A non-default ``--target-accuracy`` reaches ``benchmark_model`` unchanged."""
    config = config_with_dirs(tmp_path, make_config)
    models_dir = config.internal_root / "models"
    models_dir.mkdir(parents=True)
    dataset_root = config.storage_root / DATASET_SUBDIR
    manifest_path = dataset_root / "manifest.json"
    write_export_manifest(manifest_path, classes=("marcel", "rufus"))
    model_path = models_dir / "cat-classifier-aaaaaaaa.pt"
    write_checkpoint_with_sidecar(model_path, classes=("marcel", "rufus"), manifest_path=manifest_path)

    fake_report = _fake_benchmark_report(("marcel", "rufus"))
    with (
        patch("cat_watcher.classifier.benchmark.make_predict_fn", return_value=_fake_predict),
        patch("cat_watcher.classifier.benchmark.benchmark_model", return_value=fake_report) as bench_mock,
    ):
        exit_code = run(make_classifier_args("benchmark", target_accuracy=0.80), config=config)

    assert exit_code == 0
    call_kwargs = bench_mock.call_args.kwargs
    assert call_kwargs["target_accuracy"] == 0.80


def test_classifier_benchmark_with_a_stale_sidecar_exits_nonzero_without_scoring(
    tmp_path: Path,
    make_config: Callable[..., Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A sidecar hash that disagrees with the current manifest names a stale model. It never scores."""
    config = config_with_dirs(tmp_path, make_config)
    models_dir = config.internal_root / "models"
    models_dir.mkdir(parents=True)
    dataset_root = config.storage_root / DATASET_SUBDIR
    manifest_path = dataset_root / "manifest.json"
    write_export_manifest(manifest_path, classes=("marcel", "rufus"), dataset_hash=DEFAULT_DATASET_HASH)

    model_path = models_dir / "cat-classifier-aaaaaaaa.pt"
    stale_hash = "b" * 64
    write_checkpoint_with_sidecar(model_path, classes=("marcel", "rufus"), manifest_path=manifest_path, dataset_hash=stale_hash)

    with patch("cat_watcher.classifier.benchmark.benchmark_model") as bench_mock:
        exit_code = run(make_classifier_args("benchmark"), config=config)

    assert exit_code == 5
    bench_mock.assert_not_called()
    err = capsys.readouterr().err
    assert stale_hash in err
    assert DEFAULT_DATASET_HASH in err
    assert str(model_path) in err
    assert str(manifest_path) in err
    _assert_no_placeholder_leak(err)


def test_newest_checkpoint_breaks_an_mtime_tie_deterministically(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """Two checkpoints with the same mtime resolve to the same pick, every time, not filesystem order."""
    config = config_with_dirs(tmp_path, make_config)
    models_dir = config.internal_root / "models"
    models_dir.mkdir(parents=True)
    dataset_root = config.storage_root / DATASET_SUBDIR
    manifest_path = dataset_root / "manifest.json"
    write_export_manifest(manifest_path, classes=("marcel", "rufus"))

    first = models_dir / "cat-classifier-aaaaaaaa.pt"
    second = models_dir / "cat-classifier-bbbbbbbb.pt"
    write_checkpoint_with_sidecar(first, classes=("marcel", "rufus"), manifest_path=manifest_path)
    write_checkpoint_with_sidecar(second, classes=("rufus", "marcel"), manifest_path=manifest_path)
    same_time = time.time()
    os.utime(first, (same_time, same_time))
    os.utime(second, (same_time, same_time))

    fake_report = _fake_benchmark_report(("rufus", "marcel"))
    picks: set[tuple[str, ...]] = set()
    for _ in range(5):
        with (
            patch("cat_watcher.classifier.benchmark.make_predict_fn", return_value=_fake_predict),
            patch("cat_watcher.classifier.benchmark.benchmark_model", return_value=fake_report) as bench_mock,
        ):
            exit_code = run(make_classifier_args("benchmark"), config=config)
        assert exit_code == 0
        picks.add(cast("tuple[str, ...]", bench_mock.call_args.kwargs["classes"]))

    assert len(picks) == 1

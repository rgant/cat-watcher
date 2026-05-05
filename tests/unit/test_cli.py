"""Tests for cat_watcher.__main__ umbrella CLI.

Focuses on argument parsing + dispatch to handler functions. End-to-end behavior of the
``import-local`` handler itself is covered by tests/integration/test_import_local_end_to_end.py.
Task 25 will extend this file with status/test-cameras/etc. handler tests.
"""

from collections.abc import Callable  # noqa: TC003  # runtime: pytest evaluates fixture annotations during collection
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.engine import Engine

from cat_watcher.__main__ import _build_parser, _ParsedArgs, _run_import_local, main
from cat_watcher.config import Config  # noqa: TC001  # runtime: make_config callable annotation
from cat_watcher.import_local import ImportReport
from cat_watcher.poller import PollerLockedError

# --- argparse wiring -----------------------------------------------------------------------------


def test_build_parser_help_lists_import_local() -> None:
    """The umbrella parser registers ``import-local`` and surfaces it in help text."""
    parser = _build_parser()
    help_text = parser.format_help()
    assert "import-local" in help_text


def test_main_with_no_args_exits_nonzero() -> None:
    """``cat-watcher`` with no subcommand is an argparse error (required=True)."""
    with pytest.raises(SystemExit) as exc:
        _ = main([])
    assert exc.value.code != 0


def test_main_with_unknown_subcommand_exits_nonzero() -> None:
    """An unknown subcommand triggers argparse error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        _ = main(["bogus"])
    assert exc.value.code != 0


def test_import_local_subparser_requires_camera_and_source(tmp_path: Path) -> None:
    """``import-local`` requires both ``--camera`` and the positional source directory."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        _ = parser.parse_args(["import-local"])  # missing --camera AND source_dir
    with pytest.raises(SystemExit):
        _ = parser.parse_args(["import-local", "--camera", "pantry"])  # missing source_dir
    with pytest.raises(SystemExit):
        _ = parser.parse_args(["import-local", str(tmp_path)])  # missing --camera


def test_import_local_subparser_parses_all_flags(tmp_path: Path) -> None:
    """All ``import-local`` flags parse into the typed namespace with the expected values."""
    parser = _build_parser()
    args = parser.parse_args(
        [
            "import-local",
            "--config",
            "/etc/cat-watcher.toml",
            "--camera",
            "pantry",
            "--no-detect",
            "--limit",
            "5",
            str(tmp_path),
        ],
        namespace=_ParsedArgs(),
    )
    assert args.command == "import-local"
    assert args.config == Path("/etc/cat-watcher.toml")
    assert args.camera == "pantry"
    assert args.no_detect is True
    assert args.limit == 5
    assert args.source_dir == tmp_path


# --- _run_import_local exit codes ----------------------------------------------------------------


def _make_args(tmp_path: Path, *, no_detect: bool = True) -> _ParsedArgs:
    args = _ParsedArgs()
    args.command = "import-local"
    args.config = None
    args.camera = "pantry"
    args.no_detect = no_detect
    args.limit = None
    args.source_dir = tmp_path
    return args


def _setup_handler_env(tmp_path: Path, make_config: Callable[[Path, Path], Config]) -> Config:
    """Create internal/storage roots and return a real Config wired to them."""
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    internal_root.mkdir()
    storage_root.mkdir()
    return make_config(internal_root, storage_root)


def test_run_import_local_returns_zero_on_clean_report(tmp_path: Path, make_config: Callable[[Path, Path], Config]) -> None:
    """A clean ImportReport (no errors) returns exit code 0."""
    args = _make_args(tmp_path)
    config = _setup_handler_env(tmp_path, make_config)
    clean_report = ImportReport(inspected=2, ingested=2, duplicates=0, skipped=0, errors=0)
    engine_mock = MagicMock(spec=Engine)

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.ensure_storage_layout"),
        patch("cat_watcher.__main__.create_engine", return_value=engine_mock),
        patch("cat_watcher.__main__.import_local", return_value=clean_report) as import_local_mock,
    ):
        exit_code = _run_import_local(args)

    assert exit_code == 0
    assert import_local_mock.call_count == 1
    engine_mock.dispose.assert_called_once()


def test_run_import_local_returns_one_when_errors_present(tmp_path: Path, make_config: Callable[[Path, Path], Config]) -> None:
    """An ImportReport with per-clip errors returns exit code 1 (partial-failure signal)."""
    args = _make_args(tmp_path)
    config = _setup_handler_env(tmp_path, make_config)
    failing_report = ImportReport(inspected=3, ingested=2, duplicates=0, skipped=0, errors=1)

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.ensure_storage_layout"),
        patch("cat_watcher.__main__.create_engine", return_value=MagicMock(spec=Engine)),
        patch("cat_watcher.__main__.import_local", return_value=failing_report),
    ):
        exit_code = _run_import_local(args)

    assert exit_code == 1


def test_run_import_local_returns_two_when_poller_lock_held(tmp_path: Path, make_config: Callable[[Path, Path], Config]) -> None:
    """A held PID lock surfaces as exit code 2 with an actionable error log."""
    args = _make_args(tmp_path)
    config = _setup_handler_env(tmp_path, make_config)

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.ensure_storage_layout"),
        patch("cat_watcher.__main__.create_engine", return_value=MagicMock(spec=Engine)),
        patch("cat_watcher.__main__.import_local", side_effect=PollerLockedError("lock held")),
    ):
        exit_code = _run_import_local(args)

    assert exit_code == 2


def test_run_import_local_disposes_engine_even_when_import_raises(tmp_path: Path, make_config: Callable[[Path, Path], Config]) -> None:
    """The engine.dispose() in the finally block runs even when the inner call raises."""
    args = _make_args(tmp_path)
    config = _setup_handler_env(tmp_path, make_config)
    engine_mock = MagicMock(spec=Engine)

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.ensure_storage_layout"),
        patch("cat_watcher.__main__.create_engine", return_value=engine_mock),
        patch("cat_watcher.__main__.import_local", side_effect=PollerLockedError("lock held")),
    ):
        _ = _run_import_local(args)

    engine_mock.dispose.assert_called_once()


def test_run_import_local_skips_detector_when_no_detect(tmp_path: Path, make_config: Callable[[Path, Path], Config]) -> None:
    """``--no-detect`` short-circuits ``Detector.from_weights`` (no model load on import-only)."""
    args = _make_args(tmp_path, no_detect=True)
    config = _setup_handler_env(tmp_path, make_config)
    clean_report = ImportReport(inspected=1, ingested=1, duplicates=0, skipped=0, errors=0)

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.ensure_storage_layout"),
        patch("cat_watcher.__main__.create_engine", return_value=MagicMock(spec=Engine)),
        patch("cat_watcher.__main__.import_local", return_value=clean_report) as import_local_mock,
        patch("cat_watcher.__main__.Detector", autospec=True) as detector_cls,
    ):
        _ = _run_import_local(args)

    detector_cls.from_weights.assert_not_called()
    # Verify import_local was called with detector=None.
    _, kwargs = import_local_mock.call_args
    assert kwargs["detector"] is None
    assert isinstance(kwargs["now"], datetime)
    assert kwargs["now"].tzinfo == UTC

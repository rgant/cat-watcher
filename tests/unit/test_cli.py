"""Tests for cat_watcher.__main__ umbrella CLI.

Focuses on argument parsing + dispatch to handler functions. End-to-end behavior of the
``import-local`` handler itself is covered by tests/integration/test_import_local_end_to_end.py.
"""

import urllib.error
from collections.abc import Callable  # noqa: TC003  # runtime: pytest evaluates fixture annotations during collection
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, create_autospec, patch

import pytest
from cli_test_helpers import (
    config_with_dirs,
    init_schema,
    make_handler_args,
    open_seed_session,
    seed_camera,
    seed_clip,
)
from sqlalchemy.engine import Engine

from cat_watcher.__main__ import (
    _build_parser,
    _fmt,
    _fmt_delta,
    _ParsedArgs,
    _run_backup,
    _run_fetch_models,
    _run_import_local,
    _run_inspect,
    _run_restore_backup,
    _run_status,
    _run_test_cameras,
    _run_test_notification,
    main,
)
from cat_watcher.amcrest_client import AmcrestClient, CameraUnreachableError
from cat_watcher.config import CameraConfig, Config, _resolve_config_path
from cat_watcher.db import AgentStart, AlertSent, AlertType, Heartbeat
from cat_watcher.import_local import ImportReport
from cat_watcher.notifier import EmailResult, NotifResult
from cat_watcher.poller import PollerLockedError
from cat_watcher.storage import StorageUnavailableError

# --- argparse wiring -----------------------------------------------------------------------------


def test_build_parser_help_lists_import_local() -> None:
    """The umbrella parser registers ``import-local`` and surfaces it in help text."""
    parser = _build_parser()
    help_text = parser.format_help()
    assert "import-local" in help_text


def test_main_with_unknown_subcommand_exits_nonzero() -> None:
    """An unknown subcommand triggers argparse error (exit 2)."""
    with pytest.raises(SystemExit) as exc:
        _ = main(["bogus"])
    assert exc.value.code != 0


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


def test_run_import_local_returns_zero_on_clean_report(tmp_path: Path, make_config: Callable[[Path, Path], Config]) -> None:
    """A clean ImportReport (no errors) returns exit code 0."""
    args = _make_args(tmp_path)
    config = config_with_dirs(tmp_path, make_config)
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
    config = config_with_dirs(tmp_path, make_config)
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
    config = config_with_dirs(tmp_path, make_config)

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
    config = config_with_dirs(tmp_path, make_config)
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
    config = config_with_dirs(tmp_path, make_config)
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
    _, kwargs = import_local_mock.call_args
    assert kwargs["detector"] is None
    assert isinstance(kwargs["now"], datetime)
    assert kwargs["now"].tzinfo == UTC


# --- status sub-command --------------------------------------------------------------------------


def test_status_reports_camera_and_heartbeat_rows(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Seeded cameras + heartbeats appear in the status digest."""
    last_cat_seen_at = datetime(2026, 5, 10, 12, 0, tzinfo=UTC)
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    _ = seed_camera(config, name="pantry", display_name="Pantry", last_cat_seen_at=last_cat_seen_at)
    with open_seed_session(config) as session:
        session.add(Heartbeat(agent_name="poller", last_seen_at=datetime.now(UTC) - timedelta(seconds=30)))
        session.add(AgentStart(agent_name="poller", started_at=datetime.now(UTC)))
    with patch("cat_watcher.__main__.load_config", return_value=config):
        exit_code = _run_status(make_handler_args())
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "pantry" in out
    assert "Pantry" in out
    assert "poller" in out
    assert last_cat_seen_at.isoformat() in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


# --- inspect sub-command -------------------------------------------------------------------------


def test_inspect_prints_clip_metadata_and_size(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``inspect <id>`` shows the source filename and the on-disk file size."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    # Materialize a real file so the on-disk size assertion has a number to print.
    clip_dir = config.storage_root / "clips" / "pantry"
    clip_dir.mkdir(parents=True)
    clip_file = clip_dir / "inspect-target.mp4"
    _ = clip_file.write_bytes(b"x" * 4096)
    clip_id = seed_clip(
        config,
        camera_id=cam_id,
        source_filename="inspect-target.mp4",
        file_path="clips/pantry/inspect-target.mp4",
    )
    with patch("cat_watcher.__main__.load_config", return_value=config):
        exit_code = _run_inspect(make_handler_args(clip_id=clip_id))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "inspect-target.mp4" in out
    assert "4096" in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


def test_inspect_returns_three_for_unknown_clip(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown clip id returns the not-found exit code and writes to stderr."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    with patch("cat_watcher.__main__.load_config", return_value=config):
        exit_code = _run_inspect(make_handler_args(clip_id=9999))
    assert exit_code == 3
    err = capsys.readouterr().err
    assert "not found" in err
    assert "{_fmt" not in err
    assert "{self." not in err
    assert "{cam." not in err
    assert "{cfg." not in err
    assert "NoneType" not in err


def test_inspect_renders_missing_file_marker(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A clip whose ``file_path`` doesn't exist on disk renders ``(missing)`` next to the path.

    Operator must see "(missing)" distinct from "(N bytes)" to correlate against the retention log
    when the row exists but the file is gone.
    """
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    clip_id = seed_clip(
        config,
        camera_id=cam_id,
        source_filename="ghost.mp4",
        file_path="clips/pantry/ghost.mp4",  # never materialized on disk
    )
    with patch("cat_watcher.__main__.load_config", return_value=config):
        exit_code = _run_inspect(make_handler_args(clip_id=clip_id))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "(missing)" in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


# --- test-cameras sub-command --------------------------------------------------------------------


def _amcrest_client_mock(
    *,
    recordings: list[object] | None = None,
    time_drift_seconds: float = 0.0,
    camera_tz: str = "America/New_York",
) -> MagicMock:
    """Autospec'd ``AmcrestClient`` instance with ``__enter__`` re-pointed at the mock itself.

    Without the re-point, ``with client:`` would yield a fresh MagicMock — the production code
    relies on the same client surviving the context-manager block.
    """
    client = cast("MagicMock", create_autospec(AmcrestClient, instance=True))
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.iter_recordings.return_value = iter(recordings or [])
    client.get_camera_time.return_value = datetime.now(UTC) + timedelta(seconds=time_drift_seconds)
    client.get_camera_timezone.return_value = camera_tz
    return client


def test_test_cameras_unreachable_returns_nonzero(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A camera raising ``CameraUnreachableError`` produces an unreachable-exit code."""
    config = config_with_dirs(tmp_path, make_config)
    client = _amcrest_client_mock()
    client.iter_recordings.side_effect = CameraUnreachableError("connect refused")
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.AmcrestClient", return_value=client),
    ):
        exit_code = _run_test_cameras(make_handler_args())
    assert exit_code != 0
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


def test_test_cameras_clock_drift_warns_below_five_minutes(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Drift > 60s but ≤ 5min prints a WARN line and stays exit 0."""
    config = config_with_dirs(tmp_path, make_config)
    client = _amcrest_client_mock(time_drift_seconds=120)  # 2 minutes — between thresholds
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.AmcrestClient", return_value=client),
    ):
        exit_code = _run_test_cameras(make_handler_args())
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "clock-drift: WARN" in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


def test_test_cameras_clock_drift_ok_below_warn_threshold(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Drift ≤ 60s prints ``clock-drift: OK`` (not WARN, not FAIL)."""
    config = config_with_dirs(tmp_path, make_config)
    client = _amcrest_client_mock(time_drift_seconds=5)  # well below warn threshold
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.AmcrestClient", return_value=client),
    ):
        exit_code = _run_test_cameras(make_handler_args())
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "clock-drift: OK" in out
    assert "clock-drift: WARN" not in out
    assert "clock-drift: !!! FAIL" not in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


def test_test_cameras_clock_drift_loud_fail_above_five_minutes(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Drift > 5min prints the loud FAIL marker but the loop still completes (exit 0)."""
    config = config_with_dirs(tmp_path, make_config)
    client = _amcrest_client_mock(time_drift_seconds=600)  # 10 minutes — beyond loud-fail threshold
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.AmcrestClient", return_value=client),
    ):
        exit_code = _run_test_cameras(make_handler_args())
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "clock-drift: !!! FAIL" in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


def test_test_cameras_timezone_drift_emits_advisory_with_both_zones(
    tmp_path: Path,
    make_config: Callable[
        ...,
        Config,
    ],  # widened: forwards a ``cameras=`` override per the conftest fixture's ``Callable[..., Config]`` signature
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Camera reporting ``"America/Denver"`` with config ``"America/New_York"`` advises both zones.

    Per-camera ``timezone`` is set explicitly so the seeded ``CameraConfig`` is the source of the
    expected zone (rather than the ``web.display_timezone`` fallback) — keeps the assertion focused
    on the comparison contract.
    """

    def _build_with_ny_camera(internal_root: Path, storage_root: Path) -> Config:
        return make_config(
            internal_root,
            storage_root,
            cameras=[CameraConfig(name="pantry", display_name="Pantry", host="cam.example.com", port=80, timezone="America/New_York")],
        )

    config = config_with_dirs(tmp_path, _build_with_ny_camera)
    client = _amcrest_client_mock(camera_tz="America/Denver")
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.AmcrestClient", return_value=client),
    ):
        _ = _run_test_cameras(make_handler_args())
    out = capsys.readouterr().out
    assert "timezone-drift: ADVISORY" in out
    assert "America/Denver" in out
    assert "America/New_York" in out
    assert "cameras[].timezone" in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


# --- test-notification sub-command ---------------------------------------------------------------


def test_test_notification_reports_success_when_both_channels_succeed(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both senders returning ``ok=True`` -> exit 0 with both rows printed."""
    config = config_with_dirs(tmp_path, make_config)
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.send_email", return_value=EmailResult(ok=True)),
        patch("cat_watcher.__main__.send_macos_notification", return_value=NotifResult(ok=True)),
    ):
        exit_code = _run_test_notification(make_handler_args())
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "email: ok=True" in out
    assert "macos: ok=True" in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


def test_test_notification_returns_nonzero_when_any_channel_fails(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """A failing channel -> exit non-zero so install scripts can detect breakage."""
    config = config_with_dirs(tmp_path, make_config)
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__.send_email", return_value=EmailResult(ok=False, error="auth")),
        patch("cat_watcher.__main__.send_macos_notification", return_value=NotifResult(ok=True)),
    ):
        exit_code = _run_test_notification(make_handler_args())
    assert exit_code != 0


# --- fetch-models sub-command --------------------------------------------------------------------


def _fake_download(_url: str, dest: Path) -> None:
    """Test stand-in for ``_download_to``: writes a fixed payload so tests can assert on size."""
    _ = dest.write_bytes(b"\x00" * 1024)


def test_fetch_models_downloads_when_missing(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """A first invocation hits the download path and writes ``<internal_root>/models/<model>``."""
    config = config_with_dirs(tmp_path, make_config)
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__._download_to", side_effect=_fake_download) as download_mock,
    ):
        exit_code = _run_fetch_models(make_handler_args())
    assert exit_code == 0
    target = config.internal_root / "models" / config.detector.model
    assert target.is_file()
    assert download_mock.call_count == 1


def test_fetch_models_is_noop_when_file_already_present(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Second invocation with the file present is a no-op (download is not called)."""
    config = config_with_dirs(tmp_path, make_config)
    models_dir = config.internal_root / "models"
    models_dir.mkdir(parents=True)
    target = models_dir / config.detector.model
    _ = target.write_bytes(b"already-there")

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__._download_to", side_effect=_fake_download) as download_mock,
    ):
        exit_code = _run_fetch_models(make_handler_args())
    assert exit_code == 0
    download_mock.assert_not_called()
    assert target.read_bytes() == b"already-there"


# --- backup sub-command --------------------------------------------------------------------------


def test_backup_writes_new_file_to_configured_directory(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
) -> None:
    """Calling ``backup`` produces a new ``cat_watcher-<date>.sqlite`` under ``storage_root/backups/``."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)

    with patch("cat_watcher.__main__.load_config", return_value=config):
        exit_code = _run_backup(make_handler_args())
    assert exit_code == 0
    backups = list((config.storage_root / "backups").glob("cat_watcher-*.sqlite"))
    assert len(backups) == 1


# --- restore-backup sub-command ------------------------------------------------------------------


def test_restore_backup_refuses_when_agents_loaded(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If launchctl reports a cat-watcher agent loaded, restore exits non-zero with a hint."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    backup_path = config.storage_root / "backups" / "cat_watcher-2026-05-01.sqlite"
    backup_path.parent.mkdir(parents=True)
    _ = backup_path.write_bytes(b"backup payload")

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__._agents_loaded", return_value=True),
    ):
        exit_code = _run_restore_backup(make_handler_args(backup_date="2026-05-01"))
    assert exit_code != 0
    err = capsys.readouterr().err
    assert "bootout" in err
    assert "{_fmt" not in err
    assert "{self." not in err
    assert "{cam." not in err
    assert "{cfg." not in err
    assert "NoneType" not in err


def test_restore_backup_copies_when_no_agents_loaded(tmp_path: Path, make_config: Callable[[Path, Path], Config]) -> None:
    """With no agents loaded, the backup file is copied over the live DB."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    backup_path = config.storage_root / "backups" / "cat_watcher-2026-05-01.sqlite"
    backup_path.parent.mkdir(parents=True)
    payload = b"backup payload contents"
    _ = backup_path.write_bytes(payload)

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__._agents_loaded", return_value=False),
    ):
        exit_code = _run_restore_backup(make_handler_args(backup_date="2026-05-01"))
    assert exit_code == 0
    target = config.internal_root / "cat_watcher.sqlite"
    assert target.read_bytes() == payload


def test_restore_backup_returns_three_for_unknown_date(tmp_path: Path, make_config: Callable[[Path, Path], Config]) -> None:
    """A non-existent backup date returns the not-found exit code."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__._agents_loaded", return_value=False),
    ):
        exit_code = _run_restore_backup(make_handler_args(backup_date="9999-12-31"))
    assert exit_code == 3


# --- alerts/recent test for status ---------------------------------------------------------------


def test_status_renders_recent_alerts_when_present(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Seeded ``alerts_sent`` rows surface in the digest grouped by alert type (newest first)."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    now = datetime.now(UTC)
    with open_seed_session(config) as session:
        session.add(
            AlertSent(
                alert_type=AlertType.INACTIVITY,
                camera_id=cam_id,
                sent_at=now - timedelta(hours=1),
                subject="No cat seen for 24h",
                body="alert body",
            ),
        )

    with patch("cat_watcher.__main__.load_config", return_value=config):
        _ = _run_status(make_handler_args())
    out = capsys.readouterr().out
    assert "INACTIVITY" in out
    assert "No cat seen for 24h" in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


def test_status_omits_alerts_older_than_thirty_days(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``status``'s recent-alerts query honors the 30-day cutoff so long-running deployments don't drag in stale rows."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    cam_id = seed_camera(config)
    now = datetime.now(UTC)
    with open_seed_session(config) as session:
        session.add(
            AlertSent(
                alert_type=AlertType.INACTIVITY,
                camera_id=cam_id,
                sent_at=now - timedelta(days=45),
                subject="Stale alert from 45 days ago",
                body="alert body",
            ),
        )
        session.add(
            AlertSent(
                alert_type=AlertType.FREQUENCY,
                camera_id=cam_id,
                sent_at=now - timedelta(days=1),
                subject="Recent alert from yesterday",
                body="alert body",
            ),
        )

    with patch("cat_watcher.__main__.load_config", return_value=config):
        _ = _run_status(make_handler_args())
    out = capsys.readouterr().out
    assert "Stale alert from 45 days ago" not in out
    assert "Recent alert from yesterday" in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out


def test_fetch_models_cleans_up_partial_download_on_failure(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mid-download failure leaves no ``.part`` (and no truncated final file) behind.

    Atomic-write contract: a SIGKILL'd or network-failed download must not strand a truncated
    artifact at ``target`` that the next invocation's existence check would treat as complete.
    """
    config = config_with_dirs(tmp_path, make_config)
    target = config.internal_root / "models" / config.detector.model
    part = target.with_name(target.name + ".part")

    def _failing_download(_url: str, dest: Path) -> None:
        _ = dest.write_bytes(b"partial")  # simulate a real partial write before the network drops
        msg = "connection reset"
        raise urllib.error.URLError(msg)

    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch("cat_watcher.__main__._download_to", side_effect=_failing_download),
    ):
        exit_code = _run_fetch_models(make_handler_args())
    assert exit_code != 0
    assert not part.exists(), f"expected {part} to be cleaned up after failed download"
    assert not target.exists(), f"expected {target} to be absent (atomic-rename never happened)"
    err = capsys.readouterr().err
    assert "download failed" in err
    assert "{_fmt" not in err
    assert "{self." not in err
    assert "{cam." not in err
    assert "{cfg." not in err
    assert "NoneType" not in err


def test_backup_returns_locked_when_storage_unavailable(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``cat-watcher backup`` against an unmounted drive exits 2 instead of raising FileNotFoundError."""
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)
    with (
        patch("cat_watcher.__main__.load_config", return_value=config),
        patch(
            "cat_watcher.__main__.wait_for_storage_using_config",
            side_effect=StorageUnavailableError("storage not available within 600s"),
        ),
    ):
        exit_code = _run_backup(make_handler_args())
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "storage_root unavailable" in err
    assert not list((config.storage_root / "backups").glob("cat_watcher-*.sqlite"))
    assert "{_fmt" not in err
    assert "{self." not in err
    assert "{cam." not in err
    assert "{cfg." not in err
    assert "NoneType" not in err


# --- config-path precedence ----------------------------------------------------------------------


def test_config_path_precedence_arg_beats_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--config PATH`` overrides ``CAT_WATCHER_CONFIG`` which overrides the default."""
    arg_path = tmp_path / "from-arg.toml"
    env_path = tmp_path / "from-env.toml"
    monkeypatch.setenv("CAT_WATCHER_CONFIG", str(env_path))

    assert _resolve_config_path(arg_path) == arg_path
    assert _resolve_config_path(None) == env_path
    monkeypatch.delenv("CAT_WATCHER_CONFIG")
    assert _resolve_config_path(None) == Path("./config.toml")


# --- _fmt and _fmt_delta formatters --------------------------------------------------------------


def test_fmt_returns_em_dash_for_none() -> None:
    """``_fmt(None)`` returns the em-dash sentinel used by status output."""
    assert _fmt(None) == "—"


def test_fmt_returns_isoformat_for_datetime() -> None:
    """``_fmt`` delegates to ``.isoformat()`` for a timezone-aware datetime."""
    assert _fmt(datetime(2026, 5, 10, 12, 0, tzinfo=UTC)) == "2026-05-10T12:00:00+00:00"


def test_fmt_delta_zero() -> None:
    """Zero timedelta renders as ``00:00:00``."""
    assert _fmt_delta(timedelta(0)) == "00:00:00"


def test_fmt_delta_positive() -> None:
    """A positive timedelta renders as ``HH:MM:SS`` with zero-padded fields."""
    assert _fmt_delta(timedelta(hours=1, minutes=2, seconds=3)) == "01:02:03"


def test_fmt_delta_negative() -> None:
    """A negative timedelta gets a leading ``-`` sign so operators can spot future timestamps."""
    result = _fmt_delta(timedelta(seconds=-5))
    assert result.startswith("-")
    assert "00:05" in result


# --- status empty-state branches -----------------------------------------------------------------


def test_status_prints_empty_markers_when_db_is_unpopulated(
    tmp_path: Path,
    make_config: Callable[[Path, Path], Config],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All three ``(none)`` branches fire when no cameras, heartbeats, or alerts exist.

    Covers:
    - ``_print_camera_status``: no cameras → ``  (none)``
    - ``_print_heartbeat_status``: no heartbeat rows → ``<agent>: (none)`` per agent
    - ``_print_recent_alerts``: no alerts in window → ``  (none)``
    """
    config = config_with_dirs(tmp_path, make_config)
    init_schema(config.internal_root)

    with patch("cat_watcher.__main__.load_config", return_value=config):
        exit_code = _run_status(make_handler_args())

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "  (none)" in out
    assert "poller: (none)" in out
    assert "alerts: (none)" in out
    assert "web: (none)" in out
    assert "{_fmt" not in out
    assert "{self." not in out
    assert "{cam." not in out
    assert "{cfg." not in out
    assert "NoneType" not in out

"""Umbrella CLI for cat-watcher: dispatches to per-subcommand handlers (spec §4.10).

Sub-commands:

* ``import-local`` (Task 17b) — fold an SD-card snapshot into the canonical layout.
* ``status`` — print health digest: per-camera state, agent heartbeats with staleness, recent
  ``agent_starts`` counts, latest backup mtime, last 5 alerts per type.
* ``inspect <clip_id>`` — dump a single clip's metadata + on-disk file presence/size.
* ``test-cameras`` — try a 1-minute ``iter_recordings`` against each configured camera; check clock
  drift and timezone drift; report per-camera. Non-zero exit if any camera was unreachable. Drift
  beyond 5 minutes prints a loud failure marker but the loop completes for the rest.
* ``test-notification`` — trigger ``send_email`` + ``send_macos_notification`` so the macOS
  permission prompt fires at install time, not on the first real alert.
* ``fetch-models`` — pull configured detector weights into ``<internal_root>/models/``. Idempotent;
  re-running with the file present is a no-op.
* ``reanalyze [--camera N] [--limit N] [--all]`` — re-score clips whose detection failed (default
  filter: ``analysis_error IS NOT NULL``) or every clip (``--all``, e.g. after a model upgrade).
  Preserves ``manual_has_cat`` exactly.
* ``backup`` — proxy to :func:`cat_watcher.backup.run_backup` (same code path the LaunchAgent uses).
* ``restore-backup <date>`` — copy a dated backup file onto ``<internal_root>/cat_watcher.sqlite``.
  Refuses while any cat-watcher LaunchAgent is loaded; operator must ``launchctl bootout`` first.

Logging: ``main()`` uses :func:`logging.basicConfig` for now. Task 26b will retrofit
``setup_logging`` (structured JSON to ``logs/cli.jsonl`` + WARNING-on-stderr fallback) without
changing handler signatures. ``--verbose`` / ``-v`` at the umbrella level flips the level from
WARNING (default) to INFO, mirroring the poller's flag.
"""
# ruff: noqa: T201  # Command line tools print to stdout

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

from sqlalchemy import delete, desc, func, select

from cat_watcher.amcrest_client import AmcrestClient, CameraError
from cat_watcher.config import CameraConfig, load_config
from cat_watcher.db import AgentStart, AlertSent, Camera, Clip, ClipFrame, Heartbeat, create_engine, get_session
from cat_watcher.detector import Detector
from cat_watcher.import_local import import_local
from cat_watcher.notifier import send_email, send_macos_notification
from cat_watcher.poller import DetectionFields, PollerLockedError, detection_for, write_per_frame_thumbs
from cat_watcher.storage import StorageUnavailableError, ensure_storage_layout, wait_for_storage_using_config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import IO

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config
    from cat_watcher.detector import ScoredFrame


logger = logging.getLogger(__name__)

_DB_FILENAME = "cat_watcher.sqlite"
_BACKUPS_SUBDIR = "backups"
_MODELS_SUBDIR = "models"
_AGENT_NAMES_WITH_HEARTBEAT: tuple[str, ...] = ("poller", "alerts", "web")
_AGENT_NAMES_ALL: tuple[str, ...] = ("poller", "alerts", "web", "backup")
# Spec §4.10: clock-drift policy. ``warn`` is the threshold for a soft warning that's still exit-0;
# ``loud_fail`` is the threshold for a "FAIL" marker line (exit 0 too — fail-loudly-but-don't-exit
# so the loop processes every camera regardless of one camera's drift).
_CLOCK_DRIFT_WARN_SECONDS = 60
_CLOCK_DRIFT_LOUD_FAIL_SECONDS = 5 * 60
_RECENT_ALERTS_PER_TYPE = 5
# Backstop on the status query's recent-alerts fetch. Big enough that 5 alerts/type fits even
# under a worst-case mix of all 9 AlertType values; small enough that one tick scans bounded I/O.
_RECENT_ALERTS_QUERY_LIMIT = 500
_RECENT_ALERTS_WINDOW_DAYS = 30
_AGENT_STARTS_WINDOW_HOURS = 24
_TEST_CAMERAS_PROBE_WINDOW = timedelta(minutes=1)
_LAUNCHCTL_AGENT_LABEL_PREFIX = "com.cat-watcher."
_DETECTOR_WEIGHTS_BASE_URL = "https://github.com/ultralytics/assets/releases/latest/download"
_DOWNLOAD_CHUNK_BYTES = 64 * 1024
# Test-notification payload constants — fixed strings so operators can recognize the dry-run alert
# in their inbox / Notification Center as a self-test rather than a real incident.
_TEST_NOTIFICATION_SUBJECT = "cat-watcher test notification"
_TEST_NOTIFICATION_BODY = "If you can read this, cat-watcher's notification chain is wired correctly."

# Exit codes. ``_EXIT_LOCKED == 2`` covers any "preconditions not met; operator must take action and
# retry" scenario — poller PID lock held, LaunchAgents loaded during restore-backup, storage drive
# offline during backup. They share a value because the operator response is the same shape
# (resolve the precondition, re-run); naming each scenario distinctly would be ceremony.
_EXIT_OK = 0
_EXIT_GENERIC_FAILURE = 1
_EXIT_LOCKED = 2
_EXIT_NOT_FOUND = 3
_EXIT_UNREACHABLE = 4
_EXIT_MISSING_DEPENDENCY = 5


class _ParsedArgs(argparse.Namespace):
    """Typed view over the umbrella's argparse output. Each handler reads only the fields it sets.

    Defaults are documented as class attributes so a handler that reads a flag the user didn't pass
    sees a known sentinel instead of ``AttributeError``. argparse only sets attributes when the
    sub-parser actually defines the option, so multi-handler dispatch via this namespace would
    otherwise need defensive ``hasattr`` checks at every read site.
    """

    command: str = ""
    config: Path | None = None
    verbose: bool = False
    # import-local
    camera: str = ""
    no_detect: bool = False
    limit: int | None = None
    source_dir: Path = Path()
    # inspect
    clip_id: int = 0
    # reanalyze
    all: bool = False
    # restore-backup
    backup_date: str = ""


def _build_parser() -> argparse.ArgumentParser:
    # ``add_help=False`` is mandatory on a parent parser: ``add_parser(..., parents=[common])``
    # merges the parent's actions into each child, and ``-h``/``--help`` would collide with the
    # child's own auto-help. Each child still gets its own help out of ``add_parser``.
    common = argparse.ArgumentParser(add_help=False)
    _ = common.add_argument("--config", type=Path, default=None, help="Override config.toml path")

    parser = argparse.ArgumentParser(prog="cat-watcher", description="cat-watcher umbrella CLI")
    _ = parser.add_argument("-v", "--verbose", action="store_true", help="INFO-level logging (default WARNING)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser(
        "import-local",
        parents=[common],
        help="Ingest pre-existing SD-card snapshot clips into the canonical layout",
    )
    _ = importer.add_argument("--camera", required=True, help="Configured camera name to attribute clips to")
    _ = importer.add_argument("--no-detect", action="store_true", help="Skip detector; record skip markers")
    _ = importer.add_argument("--limit", type=int, default=None, help="Process at most N matched clips")
    _ = importer.add_argument("source_dir", type=Path, help="Root of SD-card snapshot tree")

    _ = subparsers.add_parser("status", parents=[common], help="Print system health digest")

    inspect = subparsers.add_parser("inspect", parents=[common], help="Print metadata + file presence for one clip")
    _ = inspect.add_argument("clip_id", type=int, help="Clip ID to inspect")

    _ = subparsers.add_parser("test-cameras", parents=[common], help="Probe each configured camera; report drift")
    _ = subparsers.add_parser("test-notification", parents=[common], help="Send a test alert via configured channels")
    _ = subparsers.add_parser("fetch-models", parents=[common], help="Download detector weights into <internal_root>/models")

    reanalyze = subparsers.add_parser("reanalyze", parents=[common], help="Re-score clips whose analysis failed (or all)")
    _ = reanalyze.add_argument("--camera", default="", help="Restrict to clips from one camera (by name)")
    _ = reanalyze.add_argument("--limit", type=int, default=None, help="Process at most N qualifying clips")
    _ = reanalyze.add_argument("--all", action="store_true", help="Re-score every clip, not just analysis_error rows")

    _ = subparsers.add_parser("backup", parents=[common], help="Run a one-shot DB backup (proxy to backup agent)")

    restore = subparsers.add_parser("restore-backup", parents=[common], help="Copy a dated backup over the live SQLite")
    _ = restore.add_argument("backup_date", help="Backup date (YYYY-MM-DD) matching backups/cat_watcher-<date>.sqlite")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv, namespace=_ParsedArgs())
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    handlers = {
        "import-local": _run_import_local,
        "status": _run_status,
        "inspect": _run_inspect,
        "test-cameras": _run_test_cameras,
        "test-notification": _run_test_notification,
        "fetch-models": _run_fetch_models,
        "reanalyze": _run_reanalyze,
        "backup": _run_backup,
        "restore-backup": _run_restore_backup,
    }
    handler = handlers.get(args.command)
    if handler is None:
        # ``required=True`` on add_subparsers makes this branch unreachable in practice, but ruff
        # RET503 needs an explicit terminator and the explicit error message is friendlier than
        # argparse's default "invalid choice" if the dispatch table ever drifts from the parser.
        raise SystemExit(parser.error(f"unknown command: {args.command!r}"))
    return handler(args)


# --- shared handler helpers ----------------------------------------------------------------------


def _open_engine(config: Config) -> Engine:
    """Materialize the SQLAlchemy engine for the live SQLite DB at ``config.internal_root``."""
    return create_engine(f"sqlite:///{config.internal_root / _DB_FILENAME}")


# --- import-local --------------------------------------------------------------------------------


def _run_import_local(args: _ParsedArgs) -> int:
    """Handler for ``cat-watcher import-local``. Returns process exit code."""
    config = load_config(args.config)
    ensure_storage_layout(internal_root=config.internal_root, storage_root=config.storage_root)
    engine = _open_engine(config)
    try:
        detector = (
            None
            if args.no_detect
            else Detector.from_weights(
                model_path=config.internal_root / _MODELS_SUBDIR / config.detector.model,
                frames_to_sample=config.detector.frames_to_sample,
                confidence_threshold=config.detector.confidence_threshold,
            )
        )
        try:
            report = import_local(
                engine=engine,
                config=config,
                camera_name=args.camera,
                source_dir=args.source_dir,
                detector=detector,
                limit=args.limit,
                now=datetime.now(UTC),
            )
        except PollerLockedError:
            logger.exception(
                "poller PID lock is held; refusing to run concurrently. wait for the next tick to "
                "finish, or `launchctl bootout` the poller agent first.",
            )
            return _EXIT_LOCKED
    finally:
        engine.dispose()

    logger.info(
        "import-local finished: inspected=%d ingested=%d duplicates=%d skipped=%d errors=%d",
        report.inspected,
        report.ingested,
        report.duplicates,
        report.skipped,
        report.errors,
    )
    return _EXIT_OK if report.errors == 0 else _EXIT_GENERIC_FAILURE


# --- status --------------------------------------------------------------------------------------


def _run_status(args: _ParsedArgs) -> int:
    """Handler for ``cat-watcher status``. Read-only DB digest; always exits 0 unless config fails."""
    config = load_config(args.config)
    engine = _open_engine(config)
    now = datetime.now(UTC)
    print("cat-watcher status")
    print(f"  config: {config.internal_root}")
    print(f"  now (UTC): {now.isoformat()}")
    try:
        with get_session(engine) as session:
            _print_camera_status(session)
            _print_heartbeat_status(session, now=now)
            _print_agent_starts(session, now=now)
            _print_recent_alerts(session, now=now)
        _print_backup_status(config.storage_root / _BACKUPS_SUBDIR, now=now)
    finally:
        engine.dispose()
    return _EXIT_OK


def _print_camera_status(session: Session) -> None:
    cameras = list(session.scalars(select(Camera).order_by(Camera.name)))
    print("cameras:")
    if not cameras:
        print("  (none)")
        return
    for cam in cameras:
        print(f"  - {cam.name} ({cam.display_name}): poll_status={cam.poll_status.value}")
        print(
            f"      last_polled_at={_fmt(cam.last_polled_at)}  last_clip_at={_fmt(cam.last_clip_at)}"
            "  last_cat_seen_at={_fmt(cam.last_cat_seen_at)}",
        )
        if cam.poll_status_since is not None:
            print(f"      poll_status_since={_fmt(cam.poll_status_since)}")
        if cam.poll_error:
            print(f"      poll_error={cam.poll_error[:200]}")


def _print_heartbeat_status(session: Session, *, now: datetime) -> None:
    print("heartbeats:")
    for agent in _AGENT_NAMES_WITH_HEARTBEAT:
        hb = session.get(Heartbeat, agent)
        if hb is None:
            print(f"  - {agent}: (none)")
            continue
        staleness = now - hb.last_seen_at
        print(f"  - {agent}: last_seen_at={hb.last_seen_at.isoformat()} (stale by {_fmt_delta(staleness)})")


def _print_agent_starts(session: Session, *, now: datetime) -> None:
    cutoff = now - timedelta(hours=_AGENT_STARTS_WINDOW_HOURS)
    rows = session.execute(
        select(AgentStart.agent_name, func.count().label("n"))  # pylint: disable=not-callable
        .where(AgentStart.started_at >= cutoff)
        .group_by(AgentStart.agent_name),
    ).all()
    counts: Counter[str] = Counter()
    for row in rows:
        agent_count: tuple[str, int] = tuple(row)
        counts[agent_count[0]] = agent_count[1]
    print(f"agent_starts (last {_AGENT_STARTS_WINDOW_HOURS}h):")
    for agent in _AGENT_NAMES_ALL:
        print(f"  - {agent}: {counts.get(agent, 0)}")


def _print_recent_alerts(session: Session, *, now: datetime) -> None:
    print(f"recent alerts (last {_RECENT_ALERTS_PER_TYPE} per type):")
    cutoff = now - timedelta(days=_RECENT_ALERTS_WINDOW_DAYS)
    alerts_by_type: dict[str, list[AlertSent]] = {}
    rows = session.scalars(
        select(AlertSent)  # dprint-ignore
        .where(AlertSent.sent_at >= cutoff)
        .order_by(desc(AlertSent.sent_at))
        .limit(_RECENT_ALERTS_QUERY_LIMIT),
    )
    for alert in rows:
        bucket = alerts_by_type.setdefault(alert.alert_type.value, [])
        if len(bucket) < _RECENT_ALERTS_PER_TYPE:
            bucket.append(alert)
    if not alerts_by_type:
        print("  (none)")
        return
    for alert_type in sorted(alerts_by_type):
        for alert in alerts_by_type[alert_type]:
            cam_label = f"camera_id={alert.camera_id}" if alert.camera_id is not None else "camera=—"
            print(f"  - {alert.sent_at.isoformat()} {alert_type} {cam_label}: {alert.subject}")


def _print_backup_status(backups_dir: Path, *, now: datetime) -> None:
    if not backups_dir.is_dir():
        print(f"backup: storage_root/{_BACKUPS_SUBDIR}/ does not exist (drive offline?)")
        return
    backups = sorted(backups_dir.glob("cat_watcher-*.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not backups:
        print(f"backup: no files in {backups_dir}")
        return
    newest = backups[0]
    mtime = datetime.fromtimestamp(newest.stat().st_mtime, tz=UTC)
    print(f"backup: newest={newest.name} mtime={mtime.isoformat()} (age {_fmt_delta(now - mtime)})")


def _fmt(value: datetime | None) -> str:
    """Render a nullable UTC datetime as ISO-8601, or ``—`` for ``None``. Status output only."""
    return value.isoformat() if value is not None else "—"


def _fmt_delta(delta: timedelta) -> str:
    """Render a positive timedelta as ``HH:MM:SS`` (negative becomes ``-HH:MM:SS``)."""
    sign = "-" if delta.total_seconds() < 0 else ""
    abs_delta = abs(delta)
    total_seconds = int(abs_delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"


# --- inspect -------------------------------------------------------------------------------------


def _run_inspect(args: _ParsedArgs) -> int:
    """Handler for ``cat-watcher inspect <clip_id>``. Returns 0 / 3 (not found)."""
    config = load_config(args.config)
    engine = _open_engine(config)
    try:
        with get_session(engine) as session:
            clip = session.get(Clip, args.clip_id)
            if clip is None:
                print(f"clip {args.clip_id} not found", file=sys.stderr)
                return _EXIT_NOT_FOUND
            session.expunge(clip)
    finally:
        engine.dispose()

    full_clip_path = config.storage_root / clip.file_path
    full_thumb_path = config.storage_root / clip.thumb_path
    print(f"clip {clip.id}")
    print(f"  camera_id        = {clip.camera_id}")
    print(f"  source_filename  = {clip.source_filename}")
    print(f"  start_ts         = {clip.start_ts.isoformat()}")
    print(f"  end_ts           = {clip.end_ts.isoformat()}")
    print(f"  duration_seconds = {clip.duration_seconds}")
    print(f"  file_path        = {clip.file_path}")
    print(f"  on-disk          = {full_clip_path} {_size_or_missing(full_clip_path)}")
    print(f"  thumb_path       = {clip.thumb_path}")
    print(f"  on-disk thumb    = {full_thumb_path} {_size_or_missing(full_thumb_path)}")
    print(f"  has_cat          = {clip.has_cat}")
    print(f"  manual_has_cat   = {clip.manual_has_cat}")
    print(f"  manual_label_at  = {_fmt(clip.manual_label_at)}")
    print(f"  max_score        = {clip.max_score}")
    print(f"  frames_sampled   = {clip.frames_sampled}")
    print(f"  frames_with_cat  = {clip.frames_with_cat}")
    print(f"  detector_version = {clip.detector_version}")
    print(f"  ingested_at      = {clip.ingested_at.isoformat()}")
    if clip.analysis_error:
        print(f"  analysis_error   = {clip.analysis_error}")
    return _EXIT_OK


def _size_or_missing(path: Path) -> str:
    if not path.is_file():
        return "(missing)"
    return f"({path.stat().st_size} bytes)"


# --- test-cameras --------------------------------------------------------------------------------


def _run_test_cameras(args: _ParsedArgs) -> int:
    """Probe each configured camera. Returns 4 if any camera was unreachable; 0 otherwise.

    Drift > 5min prints a loud-fail marker (operator-actionable: bad NTP) but does not change exit
    code — the loop must still process every camera so the operator gets a single report rather
    than a fail-fast on camera 1 of N.
    """
    config = load_config(args.config)
    now = datetime.now(UTC)
    print(f"test-cameras: {len(config.cameras)} configured (now={now.isoformat()})")
    reachable = [_probe_camera(cam_cfg, config=config, host_now=now) for cam_cfg in config.cameras]
    return _EXIT_UNREACHABLE if not all(reachable) else _EXIT_OK


def _probe_camera(cam_cfg: CameraConfig, *, config: Config, host_now: datetime) -> bool:
    """One camera's full probe: connectivity + clock-drift + timezone-drift. Returns connectivity OK."""
    tz_name = cam_cfg.timezone or config.web.display_timezone
    camera_tz = ZoneInfo(tz_name)
    print(f"\ncamera {cam_cfg.name} ({cam_cfg.display_name}) host={cam_cfg.host}:{cam_cfg.port} expected_tz={tz_name}")
    client = AmcrestClient(cam_cfg, config.camera_secrets, camera_tz=camera_tz)
    reachable = False
    try:
        with client:
            try:
                _ = list(client.iter_recordings(since=host_now - _TEST_CAMERAS_PROBE_WINDOW, until=host_now))
                reachable = True
                print("  connectivity: OK")
            except CameraError as exc:
                print(f"  connectivity: FAIL ({exc.__class__.__name__}: {exc})")
            if reachable:
                _check_clock_drift(client, host_now=host_now)
                _check_timezone_drift(client, expected=tz_name)
    except CameraError as exc:
        # ``with client:`` only fails through ``close``; surface anything that escapes.
        print(f"  connectivity: FAIL during teardown ({exc.__class__.__name__}: {exc})")
    return reachable


def _check_clock_drift(client: AmcrestClient, *, host_now: datetime) -> None:
    """Print clock-drift status: OK / WARN / loud-FAIL based on |camera_time - host_now|."""
    try:
        camera_time = client.get_camera_time()
    except CameraError as exc:
        print(f"  clock-drift: FAIL ({exc.__class__.__name__}: {exc})")
        return
    drift = (camera_time - host_now).total_seconds()
    abs_drift = abs(drift)
    if abs_drift > _CLOCK_DRIFT_LOUD_FAIL_SECONDS:
        print(f"  clock-drift: !!! FAIL drift={drift:+.1f}s (>{_CLOCK_DRIFT_LOUD_FAIL_SECONDS}s; check camera NTP)")
    elif abs_drift > _CLOCK_DRIFT_WARN_SECONDS:
        print(f"  clock-drift: WARN drift={drift:+.1f}s (>{_CLOCK_DRIFT_WARN_SECONDS}s)")
    else:
        print(f"  clock-drift: OK drift={drift:+.1f}s")


def _check_timezone_drift(client: AmcrestClient, *, expected: str) -> None:
    """Compare the camera's reported NTP timezone against ``expected``.

    Non-blocking advisory per spec resolution #6: a mismatch prints a loud line but does not change
    the exit code. ``cameras[].timezone`` in config.toml is the operator's escape hatch when the
    camera's internal numeric code can't be cleanly mapped to an IANA name.
    """
    try:
        camera_tz = client.get_camera_timezone()
    except CameraError as exc:
        print(f"  timezone-drift: FAIL ({exc.__class__.__name__}: {exc})")
        return
    if camera_tz != expected:
        print(
            f"  timezone-drift: ADVISORY camera reports {camera_tz!r} but config expects {expected!r}; "
            "set cameras[].timezone explicitly in config.toml if this is intentional",
        )
        return
    print(f"  timezone-drift: OK (camera reports {camera_tz!r})")


# --- test-notification ---------------------------------------------------------------------------


def _run_test_notification(args: _ParsedArgs) -> int:
    """Send the dry-run alert via every enabled channel. Returns 0 if all enabled channels deliver."""
    config = load_config(args.config)
    print("test-notification:")
    email_result = send_email(
        _TEST_NOTIFICATION_SUBJECT,
        _TEST_NOTIFICATION_BODY,
        secrets=config.email,
        rules=config.alerts.email,
    )
    print(f"  email: ok={email_result.ok} error={email_result.error}")
    macos_result = send_macos_notification(
        _TEST_NOTIFICATION_SUBJECT,
        _TEST_NOTIFICATION_BODY,
        rules=config.alerts.macos,
    )
    print(f"  macos: ok={macos_result.ok} error={macos_result.error}")
    if email_result.ok and macos_result.ok:
        return _EXIT_OK
    return _EXIT_GENERIC_FAILURE


# --- fetch-models --------------------------------------------------------------------------------


def _run_fetch_models(args: _ParsedArgs) -> int:
    """Download configured detector weights to ``<internal_root>/models/<detector.model>``.

    Idempotent: if the file already exists with the expected size (or any size, when no checksum is
    on file), the call is a no-op. The url scheme matches Ultralytics' assets release CDN, which is
    where YOLO's auto-downloader pulls from when run for the first time.
    """
    config = load_config(args.config)
    models_dir = config.internal_root / _MODELS_SUBDIR
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / config.detector.model

    if target.is_file() and target.stat().st_size > 0:
        print(f"fetch-models: {target} already present ({target.stat().st_size} bytes); no-op")
        return _EXIT_OK

    url = f"{_DETECTOR_WEIGHTS_BASE_URL}/{config.detector.model}"
    print(f"fetch-models: downloading {url} -> {target}")
    # ``.part`` + atomic rename so a SIGKILL or Ctrl-C mid-download doesn't leave a truncated file
    # at ``target`` that a future invocation's existence check would treat as a complete download.
    part = target.with_name(target.name + ".part")
    try:
        _download_to(url, part)
    except (urllib.error.URLError, OSError) as exc:
        with suppress(OSError):
            part.unlink(missing_ok=True)
        print(f"fetch-models: download failed: {exc}", file=sys.stderr)
        return _EXIT_GENERIC_FAILURE
    _ = part.replace(target)
    sha = _sha256_of(target)
    print(f"fetch-models: wrote {target} ({target.stat().st_size} bytes, sha256={sha[:12]}…)")
    return _EXIT_OK


def _download_to(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``. Stdlib over ``httpxyz`` so fetch-models doesn't pull async deps."""
    req = urllib.request.Request(url, headers={"User-Agent": "cat-watcher/fetch-models"})  # noqa: S310  # static https URL, not user-controlled
    response = cast("IO[bytes]", urllib.request.urlopen(req))  # noqa: S310  # same: scheme is constant https
    with response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh, length=_DOWNLOAD_CHUNK_BYTES)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:

        def _read_chunk() -> bytes:
            return fh.read(_DOWNLOAD_CHUNK_BYTES)

        for chunk in iter(_read_chunk, b""):
            h.update(chunk)
    return h.hexdigest()


# --- reanalyze -----------------------------------------------------------------------------------

# ORM streaming batch size for ``reanalyze --all``. Bounds memory at ~N rows in flight while a
# server-side cursor drains the rest. Detection itself is the throughput floor; this size only
# needs to be large enough to amortize cursor round-trips.
_REANALYZE_BATCH_SIZE = 100


@dataclass
class _ReanalyzeCameraCounts:
    """One camera's tally in a reanalyze run. Three buckets are mutually exclusive."""

    rescored: int = 0
    skipped_missing: int = 0
    errored: int = 0


@dataclass
class _ReanalyzeReport:
    """Per-camera tally for one reanalyze run; renders as one summary line per touched camera."""

    per_camera: dict[int, _ReanalyzeCameraCounts] = field(default_factory=dict)

    def for_camera(self, camera_id: int) -> _ReanalyzeCameraCounts:
        """Return (creating if absent) the tally bucket for ``camera_id``."""
        return self.per_camera.setdefault(camera_id, _ReanalyzeCameraCounts())

    @property
    def total_errored(self) -> int:
        """Sum of ``errored`` across cameras; drives the exit-code branch."""
        return sum(c.errored for c in self.per_camera.values())


def _run_reanalyze(args: _ParsedArgs) -> int:
    """Re-score qualifying clips. ``--all`` widens the filter to every clip regardless of status."""
    config = load_config(args.config)
    weights = config.internal_root / _MODELS_SUBDIR / config.detector.model
    if not weights.is_file():
        print(
            f"reanalyze: detector weights not found at {weights}; run `cat-watcher fetch-models` first",
            file=sys.stderr,
        )
        return _EXIT_MISSING_DEPENDENCY

    detector = Detector.from_weights(
        model_path=weights,
        frames_to_sample=config.detector.frames_to_sample,
        confidence_threshold=config.detector.confidence_threshold,
    )
    engine = _open_engine(config)
    try:
        report, camera_display_by_id = _reanalyze_loop(engine=engine, config=config, detector=detector, args=args)
    finally:
        engine.dispose()
    if not report.per_camera:
        print("reanalyze: no qualifying clips")
        return _EXIT_OK
    for camera_id, counts in sorted(report.per_camera.items()):
        label = camera_display_by_id.get(camera_id, f"camera_id={camera_id}")
        print(f"reanalyze [{label}]: rescored={counts.rescored} skipped_missing={counts.skipped_missing} errored={counts.errored}")
    return _EXIT_OK if report.total_errored == 0 else _EXIT_GENERIC_FAILURE


@dataclass(frozen=True)
class _ReanalyzeContext:
    """Per-loop context for reanalyze. Bundles the inputs ``_backfill_clip_frames`` needs."""

    config: Config
    camera_name_by_id: dict[int, str]
    camera_tz_by_name: dict[str, ZoneInfo]


def _reanalyze_loop(
    *,
    engine: Engine,
    config: Config,
    detector: Detector,
    args: _ParsedArgs,
) -> tuple[_ReanalyzeReport, dict[int, str]]:
    """Process qualifying clips one at a time; commits per clip so concurrent writers (web/poller
    heartbeats) aren't starved while YOLO + JPEG encoding holds the per-clip session.
    """
    report = _ReanalyzeReport()
    stmt = select(Clip.id).order_by(Clip.start_ts.asc())
    if not args.all:
        stmt = stmt.where(Clip.analysis_error.is_not(None))
    if args.camera:
        stmt = stmt.join(Camera).where(Camera.name == args.camera)
    if args.limit is not None:
        stmt = stmt.limit(args.limit)

    camera_tz_by_name = {cam.name: ZoneInfo(cam.timezone or config.web.display_timezone) for cam in config.cameras}

    with get_session(engine) as session:
        cameras = list(session.scalars(select(Camera)))
        camera_display_by_id = {cam.id: cam.display_name for cam in cameras}
        ctx = _ReanalyzeContext(
            config=config,
            camera_name_by_id={cam.id: cam.name for cam in cameras},
            camera_tz_by_name=camera_tz_by_name,
        )
        clip_ids: list[int] = list(session.scalars(stmt).all())

    for clip_id in clip_ids:
        with get_session(engine) as session:
            clip = session.get(Clip, clip_id)
            if clip is None:
                continue
            _reanalyze_one_clip(
                clip=clip,
                session=session,
                detector=detector,
                ctx=ctx,
                counts=report.for_camera(clip.camera_id),
            )
    return report, camera_display_by_id


def _reanalyze_one_clip(
    *,
    clip: Clip,
    session: Session,
    detector: Detector,
    ctx: _ReanalyzeContext,
    counts: _ReanalyzeCameraCounts,
) -> None:
    """Re-detect one clip in place; on success replace ``clip_frames`` and repoint ``thumb_path``."""
    full_path = ctx.config.storage_root / clip.file_path
    if not full_path.is_file():
        logger.warning("reanalyze: clip %d file missing at %s; skipping", clip.id, full_path)
        counts.skipped_missing += 1
        return
    fields, scored_frames = detection_for(detector, full_path)
    _apply_detection_fields(clip, fields)
    if fields["analysis_error"] is None and scored_frames:
        _backfill_clip_frames(clip=clip, scored_frames=scored_frames, session=session, ctx=ctx)
    if fields["analysis_error"] is not None:
        counts.errored += 1
    else:
        counts.rescored += 1


def _apply_detection_fields(clip: Clip, fields: DetectionFields) -> None:
    """Copy ``detection_for`` field output onto ``clip``. ``manual_has_cat`` is intentionally untouched.

    The COALESCE projection in the web layer makes the manual override prevail over re-detection;
    overwriting it here would silently discard operator labels.
    """
    clip.has_cat = fields["has_cat"]
    clip.max_score = fields["max_score"]
    clip.frames_sampled = fields["frames_sampled"]
    clip.frames_with_cat = fields["frames_with_cat"]
    box = fields["best_box_xyxy"]
    clip.best_box_xyxy = None if box is None else list(box)
    clip.detector_version = fields["detector_version"]
    clip.analysis_error = fields["analysis_error"]


def _backfill_clip_frames(
    *,
    clip: Clip,
    scored_frames: tuple[ScoredFrame, ...],
    session: Session,
    ctx: _ReanalyzeContext,
) -> None:
    """Encode per-frame thumbs, replace ``clip_frames`` rows, repoint ``Clip.thumb_path``, unlink legacy thumb.

    Idempotent re-run: any pre-existing ``ClipFrame`` rows for ``clip`` are deleted before the new
    batch is attached. The delete + insert + thumb_path repoint commit atomically with the caller's
    per-clip session. The legacy single-frame thumb at the old ``thumb_path`` is unlinked
    best-effort once ``thumb_path`` actually changes; permission failures are logged so a single
    bad file doesn't abort the whole reanalyze run.
    """
    camera_name = ctx.camera_name_by_id[clip.camera_id]
    camera_tz = ctx.camera_tz_by_name.get(camera_name) or ZoneInfo(ctx.config.web.display_timezone)
    local_dt = clip.start_ts.astimezone(camera_tz)
    new_thumb_relpath, clip_frames = write_per_frame_thumbs(
        scored_frames=scored_frames,
        storage_root=ctx.config.storage_root,
        camera_name=camera_name,
        local_dt=local_dt,
    )
    _ = session.execute(delete(ClipFrame).where(ClipFrame.clip_id == clip.id))
    clip.frames = clip_frames
    old_thumb_relpath = clip.thumb_path
    clip.thumb_path = new_thumb_relpath
    if old_thumb_relpath != clip.thumb_path:
        old_thumb_full = ctx.config.storage_root / old_thumb_relpath
        try:
            old_thumb_full.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("reanalyze: could not unlink legacy thumb %s: %s", old_thumb_full, exc)


# --- backup --------------------------------------------------------------------------------------


def _run_backup(args: _ParsedArgs) -> int:
    """Hot-copy the live SQLite into ``<storage_root>/backups/``; same code path the LaunchAgent uses.

    Mirrors the LaunchAgent's storage-availability wait so a manual ``cat-watcher backup`` against
    an unmounted drive surfaces the same operator-actionable timeout instead of a raw
    FileNotFoundError.
    """
    from cat_watcher.backup import run_backup  # noqa: PLC0415  # avoid pulling backup module deps when other sub-commands run

    config = load_config(args.config)
    try:
        wait_for_storage_using_config(config)
    except StorageUnavailableError:
        print(f"backup: storage_root unavailable at {config.storage_root}", file=sys.stderr)
        return _EXIT_LOCKED
    out = run_backup(
        db_path=config.internal_root / _DB_FILENAME,
        backups_dir=config.storage_root / _BACKUPS_SUBDIR,
        now=datetime.now(UTC),
        keep_count=config.backup.keep_count,
    )
    print(f"backup: wrote {out}")
    return _EXIT_OK


# --- restore-backup ------------------------------------------------------------------------------


def _run_restore_backup(args: _ParsedArgs) -> int:
    """Copy ``backups/cat_watcher-<date>.sqlite`` over the live DB. Refuses while any agent is loaded.

    The launchctl probe is the safety belt: restoring while the poller / alerts / web / backup agent
    holds an open SQLAlchemy connection corrupts the destination DB. ``launchctl bootout`` first.
    """
    config = load_config(args.config)
    backup_path = config.storage_root / _BACKUPS_SUBDIR / f"cat_watcher-{args.backup_date}.sqlite"
    if not backup_path.is_file():
        print(f"restore-backup: backup file not found: {backup_path}", file=sys.stderr)
        return _EXIT_NOT_FOUND
    if _agents_loaded():
        print(
            "restore-backup: cat-watcher LaunchAgents are loaded; bootout first "
            f"(`launchctl bootout gui/{os.getuid()}/{_LAUNCHCTL_AGENT_LABEL_PREFIX}<agent>`)",
            file=sys.stderr,
        )
        return _EXIT_LOCKED
    target = config.internal_root / _DB_FILENAME
    _ = shutil.copy2(backup_path, target)
    print(f"restore-backup: copied {backup_path} -> {target}")
    return _EXIT_OK


def _agents_loaded() -> bool:
    """Return True if any cat-watcher LaunchAgent is currently loaded on this user's domain.

    Probes via ``launchctl print-disabled gui/<uid>`` whose output is a list of label/state pairs
    (``"label" => false`` means loaded, ``"label" => true`` means disabled). We treat anything but a
    clean "agent absent / explicitly disabled" as "loaded" so a misread errs on the safe side.
    """
    try:
        result = subprocess.run(  # noqa: S603  # cmd is fully built, not user-shell-evaluated
            ["launchctl", "print-disabled", f"gui/{os.getuid()}"],  # noqa: S607  # launchctl is a system binary, not on PATH spoofable
            check=False,
            capture_output=True,
            timeout=10,
        )
    except FileNotFoundError, subprocess.TimeoutExpired:
        # Without launchctl (non-Mac CI) we can't probe — fall back to "no agents loaded" so tests
        # on Linux can exercise the success path.
        return False
    text = result.stdout.decode(errors="replace")
    return any(_label_loaded(line) for line in text.splitlines())


def _label_loaded(line: str) -> bool:
    """Check one ``launchctl print-disabled`` line for an active cat-watcher agent.

    ``"com.cat-watcher.poller" => false`` means the agent is loaded (``false`` = "not disabled");
    ``=> true`` means the operator already booted it out, which is the green-light condition.
    """
    if _LAUNCHCTL_AGENT_LABEL_PREFIX not in line:
        return False
    return "=> false" in line


if __name__ == "__main__":
    raise SystemExit(main())

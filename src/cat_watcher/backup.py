"""Daily SQLite hot-copy of ``cat_watcher.sqlite`` to ``storage_root/backups/``.

Per spec §4.12. The backup runs as a 4th LaunchAgent (``cat-watcher-backup --once``) at 03:00 local
time. The cross-volume direction is intentional: the live DB lives on internal storage and the
backup lives on the external drive, so a single-drive failure on either side is recoverable.

Three responsibilities:

* :func:`run_backup` — page-by-page hot-copy via SQLite's online backup API
  (:meth:`sqlite3.Connection.backup`), then prune ``backups/cat_watcher-*.sqlite`` to
  ``[backup].keep_count`` newest files by mtime. The online API is safe under WAL without blocking
  writers; it opens its own dedicated connections rather than reusing SQLAlchemy's pool, since the
  copy iterates pages independently of the ORM session.
* :func:`main` + ``--once`` CLI — the LaunchAgent entry point. Performs the §4.13 storage-
  availability wait first (the backup target is on the external drive); only after the drive is
  reachable does the agent insert ``agent_starts(agent_name='backup', ...)`` and run the backup.
  Returns exit 2 on storage timeout (operator-actionable: drive offline / unlock dismissed).

The backup agent does **not** participate in the heartbeat watchdog scheme — its once-daily
cadence would always look stale to a heartbeat-based check. Backup health is monitored by the
``BACKUP_STALE`` rule (:mod:`cat_watcher.alerts`) via filesystem mtime instead.
"""

import argparse
import logging
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cat_watcher.config import load_config
from cat_watcher.db import AgentStart, create_engine, get_session
from cat_watcher.storage import StorageUnavailableError, ensure_storage_layout, wait_for_storage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cat_watcher.config import Config


logger = logging.getLogger(__name__)

_AGENT_NAME = "backup"
_DB_FILENAME = "cat_watcher.sqlite"
_BACKUPS_DIR = "backups"
_BACKUP_FILENAME_TEMPLATE = "cat_watcher-{date}.sqlite"
_BACKUP_GLOB = "cat_watcher-*.sqlite"
_EXIT_STORAGE_UNAVAILABLE = 2


def run_backup(*, db_path: Path, backups_dir: Path, now: datetime, keep_count: int) -> Path:
    """Hot-copy ``db_path`` into ``backups_dir`` and prune older backups to ``keep_count``.

    Returns the path of the just-written backup file. The destination filename is
    ``cat_watcher-YYYY-MM-DD.sqlite`` (UTC date from ``now``), matching the spec §4.12 naming
    convention and the ``BACKUP_STALE`` rule's glob.

    Uses fresh :class:`sqlite3.Connection` objects for both source and destination — the online
    backup API needs dedicated connections so it can iterate pages without contending with the
    SQLAlchemy session pool that may be open against the same source DB.
    """
    backups_dir.mkdir(parents=True, exist_ok=True)
    out = backups_dir / _BACKUP_FILENAME_TEMPLATE.format(date=now.date().isoformat())

    src = sqlite3.connect(db_path)
    dst = sqlite3.connect(out)
    try:
        with dst:
            src.backup(dst)
    finally:
        src.close()
        dst.close()

    _prune(backups_dir, keep_count=keep_count)
    return out


def _prune(backups_dir: Path, *, keep_count: int) -> None:
    """Keep the ``keep_count`` most recent ``cat_watcher-*.sqlite`` files; delete the rest."""
    candidates = sorted(
        backups_dir.glob(_BACKUP_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in candidates[keep_count:]:
        try:
            stale.unlink()
        except OSError as exc:
            logger.warning("backup prune failed for %s: %s", stale, exc)


class _ParsedArgs(argparse.Namespace):
    """Typed view over the parsed ``cat-watcher-backup`` Namespace."""

    once: bool = False
    config: Path | None = None


def _parse_args(argv: Sequence[str] | None) -> _ParsedArgs:
    parser = argparse.ArgumentParser(
        prog="cat-watcher-backup",
        description="Hot-copy the cat-watcher SQLite DB to storage_root/backups/ and prune to keep_count.",
    )
    _ = parser.add_argument(
        "--once",
        action="store_true",
        help="kept for LaunchAgent compat; the backup agent is always one-shot",
    )
    _ = parser.add_argument("--config", type=Path, default=None, help="Override config.toml path")
    return parser.parse_args(argv, namespace=_ParsedArgs())


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    Exit 0 on a clean backup, 2 when the storage-availability wait times out (drive offline or
    unlock dismissed — operator-actionable), 1 on any other unexpected failure.
    """
    args = _parse_args(argv)
    config = load_config(args.config)
    # Backups run once a day under a LaunchAgent; ``%(asctime)s`` makes it possible to read
    # ``backup.stderr.log`` after the fact and tell when a particular tick fired without
    # cross-referencing launchd's calendar interval.
    logging.basicConfig(level=config.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("backup agent starting; storage_root=%s keep_count=%d", config.storage_root, config.backup.keep_count)
    try:
        wait_for_storage(
            config.storage_root,
            interval_seconds=config.storage.wait_interval_seconds,
            timeout_seconds=config.storage.wait_timeout_seconds,
        )
    except StorageUnavailableError:
        logger.exception("backup aborted: storage_root unavailable")
        return _EXIT_STORAGE_UNAVAILABLE

    config.internal_root.mkdir(parents=True, exist_ok=True)
    ensure_storage_layout(internal_root=config.internal_root, storage_root=config.storage_root)

    now = datetime.now(UTC)
    db_path = config.internal_root / _DB_FILENAME
    return _record_start_and_back_up(config=config, db_path=db_path, now=now)


def _record_start_and_back_up(*, config: Config, db_path: Path, now: datetime) -> int:
    """Insert the ``agent_starts`` row, hot-copy the DB, and return the agent's exit code.

    Owns the ``Engine`` lifecycle for the post-wait phase: it's only safe to write ``agent_starts``
    once the storage wait has succeeded (the test contract requires no row when the wait times
    out), so the engine is created here rather than at the top of :func:`main`.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with get_session(engine) as session:
            session.add(AgentStart(agent_name=_AGENT_NAME, started_at=now))
        backup_path = run_backup(
            db_path=db_path,
            backups_dir=config.storage_root / _BACKUPS_DIR,
            now=now,
            keep_count=config.backup.keep_count,
        )
    finally:
        engine.dispose()
    logger.info("backup tick complete: wrote %s", backup_path)
    return 0


if __name__ == "__main__":  # pragma: no cover  # entry-point
    sys.exit(main())

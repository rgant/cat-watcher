"""Tests for cat_watcher.backup."""

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # runtime: pytest fixture annotations are evaluated by collectors
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import pytest

from cat_watcher.backup import _BACKUP_GLOB, main, run_backup
from cat_watcher.db import AgentStart, Base, Camera, create_engine, get_session
from cat_watcher.storage import StorageUnavailableError

if TYPE_CHECKING:
    from collections.abc import Callable

    from cat_watcher.config import Config


_NOW = datetime(2026, 5, 1, 3, 0, 0, tzinfo=UTC)
_DB_FILENAME = "cat_watcher.sqlite"


def _populated_db(internal_root: Path) -> Path:
    """Build a fresh ``cat_watcher.sqlite`` under ``internal_root`` with one Camera row."""
    db_path = internal_root / _DB_FILENAME
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with get_session(engine) as session:
        session.add(Camera(name="pantry", display_name="Pantry", host="cam.example.com"))
    engine.dispose()
    return db_path


def _seed_backup_file(backups_dir: Path, name: str, mtime: datetime) -> Path:
    """Create ``backups_dir/name`` with mtime set to ``mtime``."""
    backups_dir.mkdir(parents=True, exist_ok=True)
    f = backups_dir / name
    _ = f.write_bytes(b"placeholder")
    ts = mtime.timestamp()
    os.utime(f, (ts, ts))
    return f


# --- run_backup -----------------------------------------------------------------------------------


def test_run_backup_writes_valid_sqlite_with_source_rows(tmp_path: Path) -> None:
    """The backup file is a real SQLite DB containing the same Camera row as the source."""
    internal_root = tmp_path / "internal"
    internal_root.mkdir()
    db_path = _populated_db(internal_root)
    backups_dir = tmp_path / "storage" / "backups"

    out = run_backup(db_path=db_path, backups_dir=backups_dir, now=_NOW, keep_count=7)

    assert out.exists()
    assert out.name == "cat_watcher-2026-05-01.sqlite"
    conn = sqlite3.connect(out)
    try:
        rows = list(conn.execute("SELECT name, display_name FROM cameras"))
    finally:
        conn.close()
    assert rows == [("pantry", "Pantry")]


def test_run_backup_creates_backups_dir_when_missing(tmp_path: Path) -> None:
    """``backups_dir`` is mkdir'd on first run (fresh-install case)."""
    internal_root = tmp_path / "internal"
    internal_root.mkdir()
    db_path = _populated_db(internal_root)
    backups_dir = tmp_path / "storage" / "backups"
    assert not backups_dir.exists()

    _ = run_backup(db_path=db_path, backups_dir=backups_dir, now=_NOW, keep_count=7)

    assert backups_dir.is_dir()


@pytest.mark.parametrize("keep_count", [3, 7])
def test_run_backup_prunes_to_keep_count_most_recent_by_mtime(tmp_path: Path, keep_count: int) -> None:
    """Seed ``keep_count + 3`` dummy files with ascending mtimes; only the ``keep_count`` newest survive."""
    internal_root = tmp_path / "internal"
    internal_root.mkdir()
    db_path = _populated_db(internal_root)
    backups_dir = tmp_path / "storage" / "backups"

    seeded = keep_count + 3
    base = _NOW - timedelta(days=seeded + 1)  # all seeded files older than the new backup
    for i in range(seeded):
        _ = _seed_backup_file(
            backups_dir,
            f"cat_watcher-2026-04-{1 + i:02d}.sqlite",
            base + timedelta(days=i),
        )

    new_backup = run_backup(db_path=db_path, backups_dir=backups_dir, now=_NOW, keep_count=keep_count)

    surviving = sorted(p.name for p in backups_dir.glob(_BACKUP_GLOB))
    assert len(surviving) == keep_count
    # The fresh backup just written has the highest mtime, so it always survives.
    assert new_backup.name in surviving


def test_run_backup_overwrites_existing_dated_file(tmp_path: Path) -> None:
    """A second run on the same UTC date replaces the prior backup at the same path."""
    internal_root = tmp_path / "internal"
    internal_root.mkdir()
    db_path = _populated_db(internal_root)
    backups_dir = tmp_path / "storage" / "backups"

    first = run_backup(db_path=db_path, backups_dir=backups_dir, now=_NOW, keep_count=7)
    first_size = first.stat().st_size

    # Add a second row to make the DB strictly larger.
    engine = create_engine(f"sqlite:///{db_path}")
    with get_session(engine) as session:
        session.add(Camera(name="bathroom", display_name="Bathroom", host="cam.example.com"))
    engine.dispose()

    second = run_backup(db_path=db_path, backups_dir=backups_dir, now=_NOW, keep_count=7)
    assert second == first
    assert second.stat().st_size >= first_size

    conn = sqlite3.connect(second)
    try:
        rows = cast("list[tuple[str]]", conn.execute("SELECT name FROM cameras").fetchall())
    finally:
        conn.close()
    assert sorted(row[0] for row in rows) == ["bathroom", "pantry"]


# --- main: storage wait ---------------------------------------------------------------------------


def _build_config(make_config: Callable[..., Config], internal_root: Path, storage_root: Path) -> Config:
    """Standardize the ``main()``-driven test setup: distinct internal/storage roots."""
    internal_root.mkdir(parents=True, exist_ok=True)
    storage_root.mkdir(parents=True, exist_ok=True)
    return make_config(internal_root, storage_root)


def test_main_returns_zero_and_writes_backup_on_clean_run(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """Storage available + DB present → exit 0, backup file written, ``agent_starts`` row inserted."""
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    config = _build_config(make_config, internal_root, storage_root)
    _ = _populated_db(internal_root)

    with (
        patch("cat_watcher.backup.load_config", return_value=config),
        patch("cat_watcher.backup.wait_for_storage_using_config", return_value=None) as wait_mock,
    ):
        rc = main([])

    assert rc == 0
    wait_mock.assert_called_once_with(config)

    backups = list((storage_root / "backups").glob(_BACKUP_GLOB))
    assert len(backups) == 1
    engine = create_engine(f"sqlite:///{internal_root / _DB_FILENAME}")
    try:
        with get_session(engine) as session:
            starts = session.query(AgentStart).filter(AgentStart.agent_name == "backup").all()
    finally:
        engine.dispose()
    assert len(starts) == 1


def test_main_returns_two_when_storage_wait_times_out(tmp_path: Path, make_config: Callable[..., Config]) -> None:
    """``StorageUnavailableError`` from the wait → exit 2, no ``agent_starts`` row written."""
    internal_root = tmp_path / "internal"
    storage_root = tmp_path / "storage"
    config = _build_config(make_config, internal_root, storage_root)
    _ = _populated_db(internal_root)

    with (
        patch("cat_watcher.backup.load_config", return_value=config),
        patch(
            "cat_watcher.backup.wait_for_storage_using_config",
            side_effect=StorageUnavailableError("storage not available within 600s"),
        ),
    ):
        rc = main([])

    assert rc == 2
    assert not list((storage_root / "backups").glob(_BACKUP_GLOB))
    engine = create_engine(f"sqlite:///{internal_root / _DB_FILENAME}")
    try:
        with get_session(engine) as session:
            starts = session.query(AgentStart).filter(AgentStart.agent_name == "backup").all()
    finally:
        engine.dispose()
    assert starts == []


def test_run_backup_prunes_by_mtime_not_filename(tmp_path: Path) -> None:
    """``_prune`` keys on mtime, not filename — a date-old name with a recent mtime survives.

    Pins the spec §4.12 contract: pruning is mtime-based. Required because divergence between
    filename order and mtime order is the only way to distinguish ``key=p.stat().st_mtime`` from
    ``key=p.name``.
    """
    internal_root = tmp_path / "internal"
    internal_root.mkdir()
    db_path = _populated_db(internal_root)
    backups_dir = tmp_path / "storage" / "backups"

    # Old-looking name, freshest mtime. A name-based sort would prune this; an mtime-based sort
    # keeps it.
    fresh_old_name = _seed_backup_file(
        backups_dir,
        "cat_watcher-2026-01-01.sqlite",
        _NOW - timedelta(hours=1),
    )
    # Recent-looking name, oldest mtime. Name-based sort keeps; mtime-based prunes.
    stale_recent_name = _seed_backup_file(
        backups_dir,
        "cat_watcher-2026-12-01.sqlite",
        _NOW - timedelta(days=30),
    )
    # Middle — should be pruned because it's older than fresh_old_name but newer than
    # stale_recent_name.
    middle = _seed_backup_file(
        backups_dir,
        "cat_watcher-2026-06-01.sqlite",
        _NOW - timedelta(days=10),
    )

    new_backup = run_backup(db_path=db_path, backups_dir=backups_dir, now=_NOW, keep_count=2)

    surviving = {p.name for p in backups_dir.glob(_BACKUP_GLOB)}
    assert surviving == {new_backup.name, fresh_old_name.name}
    assert not stale_recent_name.exists()
    assert not middle.exists()


def test_run_backup_prune_ignores_files_outside_glob(tmp_path: Path) -> None:
    """Sibling files in ``backups/`` that don't match ``cat_watcher-*.sqlite`` survive a prune."""
    internal_root = tmp_path / "internal"
    internal_root.mkdir()
    db_path = _populated_db(internal_root)
    backups_dir = tmp_path / "storage" / "backups"

    # Operator-managed sidecars: a notes file and a yearly archive bundle. Neither matches the
    # ``cat_watcher-*.sqlite`` glob; pruning must leave them untouched.
    backups_dir.mkdir(parents=True)
    readme = backups_dir / "RESTORE-NOTES.md"
    _ = readme.write_text("Run cat-watcher restore-backup ...\n")
    archive = backups_dir / "cat_watcher-archive-2025.tar"
    _ = archive.write_bytes(b"tar bytes")

    # Seed enough .sqlite backups to force aggressive pruning (keep_count=1 prunes everything except
    # the just-written file).
    for i in range(3):
        _ = _seed_backup_file(
            backups_dir,
            f"cat_watcher-2026-04-{1 + i:02d}.sqlite",
            _NOW - timedelta(days=10 - i),
        )

    _ = run_backup(db_path=db_path, backups_dir=backups_dir, now=_NOW, keep_count=1)

    assert readme.exists()
    assert archive.exists()
    # Exactly one .sqlite backup survives (the new one).
    assert len(list(backups_dir.glob(_BACKUP_GLOB))) == 1


def test_run_backup_does_not_modify_source_db(tmp_path: Path) -> None:
    """The source DB content is unchanged after the hot-copy runs (read-only-on-source contract)."""
    internal_root = tmp_path / "internal"
    internal_root.mkdir()
    db_path = _populated_db(internal_root)
    backups_dir = tmp_path / "storage" / "backups"

    def read_camera_rows() -> list[tuple[str, str, str]]:
        conn = sqlite3.connect(db_path)
        try:
            return cast(
                "list[tuple[str, str, str]]",
                conn.execute("SELECT name, display_name, host FROM cameras ORDER BY id").fetchall(),
            )
        finally:
            conn.close()

    before = read_camera_rows()
    _ = run_backup(db_path=db_path, backups_dir=backups_dir, now=_NOW, keep_count=7)
    after = read_camera_rows()

    assert before == after
    assert before == [("pantry", "Pantry", "cam.example.com")]

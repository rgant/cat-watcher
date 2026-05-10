"""Periodic retention sweep — prunes aged clip rows + files, agent_starts, alerts_sent.

Two-pass design for clip cleanup (per spec §4.4 step 7):

1. **DB-driven pass** — for each ``Clip`` whose ``start_ts`` is past the retention cutoff, delete
   the row inside its own transaction *first*, then unlink the clip + thumbnail files. Row-first
   ordering means a crash between the two leaves only an orphan file, never a dangling row pointing
   at a missing path.

2. **Orphan filesystem pass** — walks ``<storage_root>/clips`` and ``<storage_root>/thumbs``, and
   for any file whose mtime is past the same cutoff *and* whose relative path is not in the set of
   surviving ``Clip.file_path`` / ``Clip.thumb_path`` values, unlinks it. Cleans up the orphans pass
   1 leaves on crash, plus stragglers from earlier ingestion bugs.

After both passes, walks each ``clips/<slug>/<YYYY-MM-DD>/`` and ``thumbs/<slug>/<YYYY-MM-DD>/``
directory and ``rmdir``s any that are empty (per spec §6.1: "delete entire date directories at
once" — otherwise empty date dirs accumulate forever).

``agent_starts`` and ``alerts_sent`` are pruned by independent cutoffs from
:class:`RetentionConfig`. All filesystem failures (unlink races, non-empty rmdir) are logged at
WARNING and never raised — a sweep that crashed mid-tick would leave the DB in a worse state
than one that logs and continues.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, cast

from sqlalchemy import delete, select

from cat_watcher.db import AgentStart, AlertSent, Clip, ClipFrame, get_session

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import InstrumentedAttribute

    from cat_watcher.config import RetentionConfig


logger = logging.getLogger(__name__)

_CLIPS_DIR = "clips"
_THUMBS_DIR = "thumbs"


@dataclass(frozen=True)
class RetentionReport:
    """Per-sweep counts; useful for logging and for the umbrella CLI ``status`` view."""

    clips_removed_pass1: int
    orphans_removed_pass2: int
    dirs_removed: int
    agent_starts_pruned: int
    alerts_sent_pruned: int


def sweep(
    *,
    engine: Engine,
    storage_root: Path,
    retention: RetentionConfig,
    now: datetime,
) -> RetentionReport:
    """Apply all retention passes in order — aged clip rows, orphan files, empty date dirs, then
    prune ``agent_starts`` and ``alerts_sent``."""
    clip_cutoff = now - timedelta(days=retention.clip_days)
    starts_cutoff = now - timedelta(days=retention.agent_starts_days)
    alerts_cutoff = now - timedelta(days=retention.alerts_sent_days)

    pass1 = _pass1_db_driven(engine, storage_root, clip_cutoff)
    pass2 = _pass2_orphan_files(engine, storage_root, clip_cutoff)
    dirs_removed = _cleanup_empty_date_dirs(storage_root)
    starts_pruned = _prune(engine, AgentStart, AgentStart.started_at, starts_cutoff)
    alerts_pruned = _prune(engine, AlertSent, AlertSent.sent_at, alerts_cutoff)

    return RetentionReport(
        clips_removed_pass1=pass1,
        orphans_removed_pass2=pass2,
        dirs_removed=dirs_removed,
        agent_starts_pruned=starts_pruned,
        alerts_sent_pruned=alerts_pruned,
    )


def _pass1_db_driven(engine: Engine, storage_root: Path, cutoff: datetime) -> int:
    """Delete each aged Clip row inside its own transaction, then unlink its files best-effort."""
    with get_session(engine) as session:
        aged_ids: list[int] = list(session.scalars(select(Clip.id).where(Clip.start_ts < cutoff)).all())

    removed = 0
    for clip_id in aged_ids:
        with get_session(engine) as session:
            clip = session.get(Clip, clip_id)
            if clip is None:
                continue
            file_path = storage_root / clip.file_path
            thumb_path = storage_root / clip.thumb_path
            # Read per-frame state while ``clip`` is still attached. A non-empty ``clip.frames``
            # means the parent of ``thumb_path`` is the per-clip ``<HHMMSS>/`` subdir to rmdir.
            frame_relpaths: list[str] = [f.thumb_path for f in clip.frames]
            per_clip_dir: Path | None = thumb_path.parent if clip.frames else None
            session.delete(clip)
        # Unlink files AFTER the row commit so a crash between the two leaves only an orphan file
        # (pass 2 collects it), never a dangling row pointing at a missing path.
        _ = _unlink_best_effort(file_path)
        _ = _unlink_best_effort(thumb_path)
        for rel in frame_relpaths:
            _ = _unlink_best_effort(storage_root / rel)
        if per_clip_dir is not None:
            try:
                per_clip_dir.rmdir()
            except OSError as exc:
                logger.warning("retention: rmdir failed for %s: %s", per_clip_dir, exc)
        removed += 1
    return removed


def _pass2_orphan_files(engine: Engine, storage_root: Path, cutoff: datetime) -> int:
    """Unlink files under clips/ and thumbs/ that have no surviving Clip row and an old mtime."""
    with get_session(engine) as session:
        survivors: set[str] = set(session.scalars(select(Clip.file_path)).all())
        survivors.update(session.scalars(select(Clip.thumb_path)).all())
        survivors.update(session.scalars(select(ClipFrame.thumb_path)).all())

    cutoff_ts = cutoff.timestamp()
    removed = 0
    for top in (_CLIPS_DIR, _THUMBS_DIR):
        root = storage_root / top
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(storage_root).as_posix()
            if rel in survivors:
                continue
            try:
                if path.stat().st_mtime >= cutoff_ts:
                    continue
            except OSError as exc:
                logger.warning("retention: stat failed for %s: %s", path, exc)
                continue
            if _unlink_best_effort(path):
                removed += 1
    return removed


def _cleanup_empty_date_dirs(storage_root: Path) -> int:
    """``rmdir`` any clips/<slug>/<YYYY-MM-DD>/ or thumbs/<slug>/<YYYY-MM-DD>/ that's empty.

    Under ``thumbs/`` each date dir may contain per-clip ``<HHMMSS>/`` subdirectories alongside flat
    ``<HHMMSS>.jpg`` files. Empty per-clip subdirs are rmdir'd first so they don't block their
    parent date-dir cleanup; ``dirs_removed`` counts only date dirs.
    """
    removed = 0
    for top in (_CLIPS_DIR, _THUMBS_DIR):
        root = storage_root / top
        if not root.is_dir():
            continue
        for slug_dir in root.iterdir():
            if not slug_dir.is_dir():
                continue
            for date_dir in slug_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                if top == _THUMBS_DIR:
                    _rmdir_empty_per_clip_subdirs(date_dir)
                try:
                    date_dir.rmdir()
                except OSError:
                    continue
                removed += 1
    return removed


def _rmdir_empty_per_clip_subdirs(date_dir: Path) -> None:
    """Inside a ``thumbs/<slug>/<YYYY-MM-DD>/`` dir, rmdir any empty per-clip ``<HHMMSS>/`` subdir."""
    for child in date_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            child.rmdir()
        except OSError:
            continue


def _prune(
    engine: Engine,
    model: type[Clip | AgentStart | AlertSent],
    age_column: InstrumentedAttribute[datetime],
    cutoff: datetime,
) -> int:
    """Bulk-delete rows of ``model`` whose ``age_column`` is past ``cutoff``; return row count."""
    with get_session(engine) as session:
        # The runtime CursorResult has ``rowcount``; pyright sees the generic Result superclass.
        result = session.execute(delete(model).where(age_column < cutoff))
        return cast("int", result.rowcount)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]


def _unlink_best_effort(path: Path) -> bool:
    """Unlink ``path``; return True on success, False (with WARNING log) on OSError."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("retention: unlink failed for %s: %s", path, exc)
        return False
    return True

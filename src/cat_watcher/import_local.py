"""Local SD-card import: fold pre-existing camera snapshots into the canonical layout.

The Amcrest cameras spool motion-triggered clips to an internal SD card with this layout:

    <root>/<YYYY-MM-DD>/<NNN>/dav/<HH>/<HH>.<MM>.<SS>-<HH>.<MM>.<SS>[M][0@0][0].mp4
    <root>/<YYYY-MM-DD>/<NNN>/jpg/<HH>/<MM>/<SS>[M][0@0][0].jpg

This module walks such a tree and folds each clip into the canonical storage layout, sharing the
same per-clip primitives the poller uses (``IngestContext``, ``relative_paths_for``,
``extract_thumbnail``, ``detection_fields_for``) so file-before-row and idempotency invariants are
identical. SD-card thumbnails are preferred over ffmpeg-extracted ones because the camera captures
them a few seconds into the motion event (after motion is "confirmed"), which is usually a better
still than the very first frame.
"""

import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from cat_watcher.poller import (
    IngestContext,
    PollerError,
    ensure_db_camera,
    extract_thumbnail,
    materialize_and_persist_clip,
    pid_lock,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.engine import Engine

    from cat_watcher.config import Config
    from cat_watcher.db import Clip
    from cat_watcher.detector import Detector


logger = logging.getLogger(__name__)

# Amcrest event-clip filename: "HH.MM.SS-HH.MM.SS[X][0@0][0].mp4" where X is a single uppercase
# letter for the trigger type. The Amcrest HTTP API PDF (V3.26) shows examples with [M] and [F]
# but doesn't enumerate the letters anywhere we could find; M is presumably "motion" and F is
# probably forced/manual based on context. All trigger types are real clips to import, so we
# accept any letter rather than enumerating.
_DAV_FILENAME_RE = re.compile(
    r"^(?P<sh>\d{2})\.(?P<sm>\d{2})\.(?P<ss>\d{2})"
    r"-(?P<eh>\d{2})\.(?P<em>\d{2})\.(?P<es>\d{2})"
    r"\[[A-Z]\]\[0@0\]\[0\]\.mp4$",
)
_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# SD-card jpg filenames are "<SS>[X][0@0][0].jpg" with the same trigger-type letter as the dav.
_JPG_FILENAME_RE = re.compile(r"^(?P<sec>\d{2})\[[A-Z]\]\[0@0\]\[0\]\.jpg$")


@dataclass(frozen=True)
class _LocalClip:
    """One discovered clip on the source SD-card tree."""

    source_path: Path
    source_filename: str
    start_ts: datetime
    end_ts: datetime
    sd_thumb_dir: Path  # parallel <NNN>/jpg/<HH>/<MM>/ directory; may not exist


@dataclass(frozen=True)
class ImportReport:
    """Outcome of an :func:`import_local` run."""

    inspected: int
    ingested: int
    duplicates: int
    skipped: int
    errors: int


def _find_date_dir(path: Path, source_dir: Path) -> Path | None:
    """Walk parents of ``path`` (up to but not including ``source_dir``) for a YYYY-MM-DD dir."""
    for parent in path.parents:
        if parent == source_dir.parent:
            return None
        if _DATE_DIR_RE.match(parent.name):
            return parent
    return None


def _find_jpg_dir(mp4_path: Path, *, start_hh: int, start_mm: int) -> Path | None:
    """Locate the parallel ``<NNN>/jpg/<HH>/<MM>/`` directory for ``mp4_path``.

    Walks up looking for a ``dav`` ancestor; its parent is the ``<NNN>`` directory under which
    ``jpg/<HH>/<MM>/`` lives. Returns ``None`` if the layout doesn't match (caller will fall back
    to ffmpeg extraction).
    """
    for parent in mp4_path.parents:
        if parent.name == "dav":
            return parent.parent / "jpg" / f"{start_hh:02d}" / f"{start_mm:02d}"
    return None


def _parse_clip_at(path: Path, source_dir: Path, *, camera_tz: ZoneInfo) -> _LocalClip | None:
    """Parse one .mp4 path into a ``_LocalClip``, or ``None`` (with WARNING) if it should skip."""
    match = _DAV_FILENAME_RE.match(path.name)
    if match is None:
        logger.warning("import-local: skipping non-Amcrest filename: %s", path)
        return None
    date_dir = _find_date_dir(path, source_dir)
    if date_dir is None:
        logger.warning("import-local: skipping clip without date-directory ancestor: %s", path)
        return None
    try:
        local_date = date.fromisoformat(date_dir.name)
    except ValueError:
        logger.warning("import-local: skipping invalid date directory %s for %s", date_dir, path)
        return None
    start_t = time(int(match["sh"]), int(match["sm"]), int(match["ss"]))
    end_t = time(int(match["eh"]), int(match["em"]), int(match["es"]))
    local_start = datetime.combine(local_date, start_t, tzinfo=camera_tz)
    # Midnight wrap: if end time is earlier in the day than start, the clip crossed into the next
    # day. The camera's filename layout doesn't encode the end-day, so we infer.
    end_local_date = local_date if end_t >= start_t else date.fromordinal(local_date.toordinal() + 1)
    local_end = datetime.combine(end_local_date, end_t, tzinfo=camera_tz)
    return _LocalClip(
        source_path=path,
        source_filename=path.name,
        start_ts=local_start.astimezone(UTC),
        end_ts=local_end.astimezone(UTC),
        sd_thumb_dir=_find_jpg_dir(path, start_hh=start_t.hour, start_mm=start_t.minute) or path.parent,
    )


def _scan_source(source_dir: Path, *, camera_tz: ZoneInfo) -> tuple[list[_LocalClip], int]:
    """Walk ``source_dir`` for Amcrest motion clips. Returns ``(matched, skipped_count)``.

    Iterates path-sorted so two runs against the same tree process duplicates in the same order
    (matters for log readability, not correctness). For the operator's ~3 GB SD snapshot this is a
    few hundred files — materializing the full list is fine.
    """
    matched: list[_LocalClip] = []
    skipped = 0
    for path in sorted(source_dir.rglob("*.mp4")):
        clip = _parse_clip_at(path, source_dir, camera_tz=camera_tz)
        if clip is None:
            skipped += 1
        else:
            matched.append(clip)
    return matched, skipped


def _locate_sd_thumb(thumb_dir: Path, *, start_sec: int, duration_sec: int) -> Path | None:
    """Pick the SD-card thumbnail nearest the clip's start, within the clip's duration window.

    The camera writes snapshots at the moment motion is "confirmed" (typically 5-10s into the clip),
    so the first jpg in the start-minute directory is usually the best one. If none fall within the
    clip window, returns ``None`` and the caller falls back to ffmpeg extraction.
    """
    if not thumb_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for jpg in thumb_dir.iterdir():
        match = _JPG_FILENAME_RE.match(jpg.name)
        if match is None:
            continue
        sec = int(match["sec"])
        if 0 <= sec - start_sec <= max(duration_sec, 1):
            candidates.append((sec, jpg))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def _atomic_copy_with_fsync(source: Path, dest: Path) -> None:
    """Copy ``source`` -> ``<dest>.part``, fsync, then atomic ``replace``. Mirrors poller download."""
    part = dest.with_suffix(dest.suffix + ".part")
    _ = shutil.copy2(source, part)
    fd = os.open(part, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    _ = part.replace(dest)


def _materialize_thumbnail(clip: _LocalClip, *, local_dt: datetime, clip_full: Path, thumb_full: Path) -> None:
    """Copy the SD-card jpg if available; otherwise fall back to ffmpeg extraction.

    ``local_dt`` is the camera-local start time. The SD-card jpg layout keys on local seconds, so we
    use ``local_dt.second`` rather than ``clip.start_ts.second`` (UTC) to handle non-zero-offset
    timezones correctly.
    """
    duration_sec = max(int((clip.end_ts - clip.start_ts).total_seconds()), 1)
    sd_thumb = _locate_sd_thumb(clip.sd_thumb_dir, start_sec=local_dt.second, duration_sec=duration_sec)
    if sd_thumb is not None:
        _atomic_copy_with_fsync(sd_thumb, thumb_full)
        logger.debug("import-local: used SD-card thumbnail %s for %s", sd_thumb, clip.source_filename)
        return
    extract_thumbnail(clip_full, thumb_full)


def _import_one(clip: _LocalClip, *, ctx: IngestContext) -> Clip | None:
    """Wrap the shared per-clip pipeline with copy-from-disk + sd-or-ffmpeg-thumb materializers."""
    local_dt = clip.start_ts.astimezone(ctx.camera_tz)

    def copy_clip(dest: Path) -> None:
        _atomic_copy_with_fsync(clip.source_path, dest)

    def make_thumb(clip_full: Path, thumb_full: Path) -> None:
        _materialize_thumbnail(clip, local_dt=local_dt, clip_full=clip_full, thumb_full=thumb_full)

    return materialize_and_persist_clip(
        source_filename=clip.source_filename,
        start_ts=clip.start_ts,
        end_ts=clip.end_ts,
        materialize_clip=copy_clip,
        materialize_thumb=make_thumb,
        ctx=ctx,
    )


def import_local(  # noqa: PLR0913  # config + identity + IO knobs all need to be threaded through
    *,
    engine: Engine,
    config: Config,
    camera_name: str,
    source_dir: Path,
    detector: Detector | None,
    limit: int | None,
    now: datetime,
) -> ImportReport:
    """Walk ``source_dir`` and ingest matching clips for ``camera_name``.

    Acquires the poller PID lock to coordinate with the LaunchAgent (default fail-loudly per plan
    Task 17b: a concurrent poller tick raises :class:`cat_watcher.poller.PollerLockedError`, which
    the CLI translates into a non-zero exit + actionable message).
    """
    cam_cfg = next((c for c in config.cameras if c.name == camera_name), None)
    if cam_cfg is None:
        msg = f"camera {camera_name!r} is not in the configured camera list"
        raise ValueError(msg)
    camera_tz = ZoneInfo(cam_cfg.timezone or config.web.display_timezone)
    db_camera = ensure_db_camera(engine, cam_cfg)
    ctx = IngestContext(
        engine=engine,
        storage_root=config.storage_root,
        camera_name=camera_name,
        cam_id=db_camera.id,
        camera_tz=camera_tz,
        detector=detector,
        now=now,
    )

    matched, skipped = _scan_source(source_dir, camera_tz=camera_tz)
    if limit is not None:
        matched = matched[:limit]

    with pid_lock(config.internal_root):
        return _ingest_loop(matched, ctx=ctx, skipped=skipped)


def _ingest_loop(matched: list[_LocalClip], *, ctx: IngestContext, skipped: int) -> ImportReport:
    """Run ``_import_one`` over each clip, counting outcomes; per-clip errors are isolated."""
    inspected = ingested = duplicates = errors = 0
    for local_clip in matched:
        inspected += 1
        try:
            outcome = _import_one(local_clip, ctx=ctx)
        except OSError, PollerError:
            logger.exception("import-local failed for %s", local_clip.source_path)
            errors += 1
            continue
        if outcome is None:
            duplicates += 1
        else:
            ingested += 1
    return ImportReport(
        inspected=inspected,
        ingested=ingested,
        duplicates=duplicates,
        skipped=skipped,
        errors=errors,
    )

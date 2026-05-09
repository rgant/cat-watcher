"""Poller LaunchAgent: discovers + ingests Amcrest motion clips, updates camera state.

Per spec §4.4. One process per host (PID-locked at ``<internal_root>/.poller.pid``); LaunchAgent
schedules a tick every ``[poller].cadence_seconds``. A tick:

1. Acquires an exclusive non-blocking PID lock; exits 0 if already held.
2. Waits for ``storage_root`` to be available (per §4.13 storage wait).
3. Inserts an ``agent_starts`` row.
4. For each configured camera (or one if ``--camera``), discovers + ingests new recordings via
   :class:`cat_watcher.amcrest_client.AmcrestClient`. Strict file-before-row ordering: download +
   thumbnail + fsync land on disk before the ``clips`` row commits, so the web UI never points
   at a partial file.
5. Updates camera state with preservation semantics (each field follows its own rule — see
   :func:`update_camera_state_success`).
6. Calls :func:`cat_watcher.retention.sweep` at end of tick.
7. Upserts the ``poller`` heartbeat.
8. Checks the ``alerts`` heartbeat and dispatches ``ALERTS_STUCK`` if stale (cool-down honored
   via :func:`cat_watcher.alerts.dispatch_alert`).

CLI flags ``--config / --camera / --since / --until / --limit / --no-detect / --list-only`` per
spec §4.10.
"""

import argparse
import contextlib
import fcntl
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from zoneinfo import ZoneInfo

from sqlalchemy import select

from cat_watcher import retention, thumbnails
from cat_watcher.alerts import dispatch_candidate, evaluate_heartbeat_watchdog
from cat_watcher.amcrest_client import AmcrestClient, CameraAPIError, CameraAuthError, CameraUnreachableError
from cat_watcher.config import Config, load_config
from cat_watcher.db import AgentStart, AlertType, Camera, Clip, ClipFrame, Heartbeat, PollStatus, create_engine, get_session
from cat_watcher.detector import Detector, DetectorError
from cat_watcher.logging_setup import setup_logging
from cat_watcher.storage import ensure_storage_layout, wait_for_storage

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator, Sequence

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.amcrest_client import Recording
    from cat_watcher.config import CameraConfig
    from cat_watcher.detector import ScoredFrame


logger = logging.getLogger(__name__)

_PID_FILE_NAME = ".poller.pid"
_FFMPEG_TIMEOUT_SECONDS = 30
_NO_DETECT_MARKER = "skipped: --no-detect"
_DETECTOR_VERSION_NO_DETECT = "skipped"
_AGENT_NAME = "poller"


class PollerError(RuntimeError):
    """Base class for poller failures."""


class PollerLockedError(PollerError):
    """Raised when another poller process already holds the PID lock."""


@contextlib.contextmanager
def pid_lock(internal_root: Path) -> Generator[None]:
    """Hold an exclusive non-blocking ``fcntl.flock`` on ``<internal_root>/.poller.pid``.

    Per spec §4.4 step 1: a manual ``pixi run poll-once`` that overlaps with a LaunchAgent tick
    should see ``PollerLockedError`` (which ``main`` translates into a clean exit 0). The PID file
    is created if absent; on entry we truncate and write the current PID for diagnostic value.
    """
    internal_root.mkdir(parents=True, exist_ok=True)
    pid_path = internal_root / _PID_FILE_NAME
    fp = pid_path.open("a+")
    try:
        try:
            fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            msg = f"poller PID lock held: {pid_path}"
            raise PollerLockedError(msg) from exc
        _ = fp.seek(0)
        _ = fp.truncate()
        _ = fp.write(str(os.getpid()))
        fp.flush()
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
    finally:
        fp.close()


def update_camera_state_success(
    session: Session,
    *,
    camera_id: int,
    ingested_clips: Sequence[Clip],
    now: datetime,
    advance_cursor: bool = True,
) -> None:
    """Apply per-field preservation semantics for a successful tick (per spec §4.4 step 5).

    - ``last_polled_at`` advances to ``now`` only when ``advance_cursor`` is true. Scoped queries
      (``--since`` / ``--until`` / ``--limit``) cannot prove they covered the full
      ``[last_polled_at, now]`` window and so must leave the resume cursor untouched —
      otherwise the next default tick would skip whatever the scoped run did not cover.
    - ``last_clip_at`` advances to ``max(start_ts of new clips)`` only if any clips were ingested.
    - ``last_cat_seen_at`` advances to the latest ``COALESCE(manual_has_cat, has_cat) = true`` clip
      only if at least one new clip is cat-positive (model or operator override).
    - ``poll_status`` -> OK; ``poll_status_since`` -> NULL; ``poll_error`` -> NULL on the OK
      transition.
    """
    cam = session.get(Camera, camera_id)
    if cam is None:
        msg = f"camera {camera_id!r} not found"
        raise ValueError(msg)
    if advance_cursor:
        cam.last_polled_at = now
    if ingested_clips:
        cam.last_clip_at = max(clip.start_ts for clip in ingested_clips)
    cat_positive_starts = [
        clip.start_ts for clip in ingested_clips if (clip.manual_has_cat if clip.manual_has_cat is not None else clip.has_cat)
    ]
    if cat_positive_starts:
        cam.last_cat_seen_at = max(cat_positive_starts)
    cam.poll_status = PollStatus.OK
    cam.poll_status_since = None
    cam.poll_error = None


def update_camera_state_failure(  # noqa: PLR0913  # state-update sibling of update_camera_state_success; all 6 args are independent inputs to a single coherent responsibility (apply per-field failure semantics)
    session: Session,
    *,
    camera_id: int,
    status: PollStatus,
    error: str,
    now: datetime,
    advance_cursor: bool = True,
) -> None:
    """Apply per-field semantics for a failed tick.

    - ``last_polled_at`` advances to ``now`` only when ``advance_cursor`` is true (same window-
      coverage rule as :func:`update_camera_state_success`). Scoped queries leave it in place.
    - ``poll_status`` -> the supplied non-OK status.
    - ``poll_status_since`` is set to ``now`` only on the transition from OK to non-OK; if the
      camera is already non-OK, the original transition timestamp is preserved.
    - ``poll_error`` updates each tick to the latest error message.
    """
    cam = session.get(Camera, camera_id)
    if cam is None:
        msg = f"camera {camera_id!r} not found"
        raise ValueError(msg)
    if advance_cursor:
        cam.last_polled_at = now
    if cam.poll_status == PollStatus.OK:
        cam.poll_status_since = now
    cam.poll_status = status
    cam.poll_error = error


def upsert_heartbeat(session: Session, *, agent_name: str, now: datetime) -> None:
    """Insert or update the ``heartbeats`` row for ``agent_name`` to ``now``."""
    existing = session.scalar(select(Heartbeat).where(Heartbeat.agent_name == agent_name))
    if existing is None:
        session.add(Heartbeat(agent_name=agent_name, last_seen_at=now))
    else:
        existing.last_seen_at = now


# --- thumbnail extraction -------------------------------------------------------------------------


def extract_thumbnail(clip_path: Path, thumb_path: Path) -> None:
    """Extract the first frame of ``clip_path`` to ``thumb_path`` as a JPEG, then ``fsync`` it.

    Per spec §4.4 step 4: thumbnails are fsynced before the ``clips`` row commits so the web UI
    never points at a partial file.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        msg = "ffmpeg not on PATH"
        raise PollerError(msg)
    proc = subprocess.run(  # noqa: S603  # cmd is fully constructed, not user-shell-evaluated
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(clip_path), "-frames:v", "1", "-q:v", "5", str(thumb_path)],
        check=False,
        capture_output=True,
        timeout=_FFMPEG_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        msg = f"ffmpeg thumbnail failed for {clip_path}: {stderr or f'exit {proc.returncode}'}"
        raise PollerError(msg)
    fd = os.open(thumb_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# --- per-clip ingestion ---------------------------------------------------------------------------


def relative_paths_for(camera_name: str, start_ts_local: datetime, suffix_mp4: str = "mp4", suffix_jpg: str = "jpg") -> tuple[str, str]:
    """Compute the canonical ``clips/<slug>/<YYYY-MM-DD>/<HHMMSS>.mp4`` + thumbs sibling.

    Both paths use the camera-local date and time so the on-disk layout matches what the operator
    sees on the camera itself (and what the SD-card filenames encode).
    """
    date_dir = start_ts_local.strftime("%Y-%m-%d")
    hhmmss = start_ts_local.strftime("%H%M%S")
    return (
        f"clips/{camera_name}/{date_dir}/{hhmmss}.{suffix_mp4}",
        f"thumbs/{camera_name}/{date_dir}/{hhmmss}.{suffix_jpg}",
    )


class DetectionFields(TypedDict):
    """Schema returned by :func:`detection_fields_for`. Spreadable into ``Clip(**fields)``."""

    has_cat: bool
    max_score: float
    frames_sampled: int
    frames_with_cat: int
    best_box_xyxy: list[float] | None
    detector_version: str
    analysis_error: str | None


def detection_for(detector: Detector | None, clip_full: Path) -> tuple[DetectionFields, tuple[ScoredFrame, ...]]:
    """Run the detector (or substitute ``--no-detect`` markers) and return ``(DetectionFields, scored_frames)``.

    The returned tuple's second element is empty when ``detector`` is ``None`` or detection
    raised; on success it is the detector's full per-frame buffer (one entry per sampled frame).
    Callers branch on ``analysis_error is None and scored_frames`` to choose between the per-frame
    thumb pipeline and the legacy single-frame fallback.
    """
    if detector is None:
        fields = DetectionFields(
            has_cat=False,
            max_score=0.0,
            frames_sampled=0,
            frames_with_cat=0,
            best_box_xyxy=None,
            detector_version=_DETECTOR_VERSION_NO_DETECT,
            analysis_error=_NO_DETECT_MARKER,
        )
        return fields, ()
    try:
        result = detector.detect(clip_full)
    except (DetectorError, OSError) as exc:
        logger.warning("detector failed for %s: %s", clip_full, exc)
        fields = DetectionFields(
            has_cat=False,
            max_score=0.0,
            frames_sampled=0,
            frames_with_cat=0,
            best_box_xyxy=None,
            detector_version=detector.version,
            analysis_error=f"detect failed: {exc}",
        )
        return fields, ()
    fields = DetectionFields(
        has_cat=result.has_cat,
        max_score=result.max_score,
        frames_sampled=result.frames_sampled,
        frames_with_cat=result.frames_with_cat,
        best_box_xyxy=list(result.best_box_xyxy) if result.best_box_xyxy is not None else None,
        detector_version=result.detector_version,
        analysis_error=None,
    )
    return fields, result.scored_frames


def detection_fields_for(detector: Detector | None, clip_full: Path) -> DetectionFields:
    """Run the detector and return the Clip kwargs; thin shim around :func:`detection_for` for callers
    (reanalyze CLI, unit tests) that don't need the per-frame buffer.
    """
    fields, _ = detection_for(detector, clip_full)
    return fields


@dataclass(frozen=True)
class IngestContext:
    """Bundle of per-camera state shared by the poller and ``import_local`` per-clip pipelines."""

    engine: Engine
    storage_root: Path
    camera_name: str
    cam_id: int
    camera_tz: ZoneInfo
    detector: Detector | None
    now: datetime


def write_per_frame_thumbs(
    *,
    scored_frames: tuple[ScoredFrame, ...],
    storage_root: Path,
    camera_name: str,
    local_dt: datetime,
) -> tuple[str, list[ClipFrame]]:
    """Encode every ``ScoredFrame`` to ``thumbs/<camera>/<date>/<HHMMSS>/`` and build matching ``ClipFrame`` rows.

    Returns ``(best_thumb_relpath, clip_frames)`` — the relpath of the highest-scoring frame
    (suitable for ``Clip.thumb_path``) and the unattached ``ClipFrame`` instances ordered by
    ordinal. Files are fsynced before return; the caller commits the rows in its own session.
    """
    per_clip_dir = thumbnails.per_clip_thumb_dir(camera_name, local_dt)
    (storage_root / per_clip_dir).mkdir(parents=True, exist_ok=True)
    records = thumbnails.write_clip_frames(scored_frames, storage_root=storage_root, per_clip_dir=per_clip_dir)
    clip_frames = [
        ClipFrame(
            ordinal=record.ordinal,
            t_offset_seconds=record.t_offset_seconds,
            score=record.score,
            thumb_path=record.thumb_relpath,
        )
        for record in records
    ]
    return thumbnails.best_frame_relpath(records), clip_frames


def _materialize_thumbs(  # noqa: PLR0913  # explicit args trace the IO surface; bundling adds noise
    *,
    detect_kwargs: DetectionFields,
    scored_frames: tuple[ScoredFrame, ...],
    clip_full: Path,
    local_dt: datetime,
    materialize_thumb: Callable[[Path, Path], None],
    ctx: IngestContext,
) -> tuple[str, list[ClipFrame]]:
    """Pick the per-frame thumb pipeline or the legacy single-thumb fallback; encode the JPEG(s).

    Returns ``(rel_thumb, clip_frames)`` — the relpath that becomes ``Clip.thumb_path`` and the
    pending ``ClipFrame`` rows (empty in the fallback branch). All files are fsynced before return.
    """
    if detect_kwargs["analysis_error"] is None and scored_frames:
        return write_per_frame_thumbs(
            scored_frames=scored_frames,
            storage_root=ctx.storage_root,
            camera_name=ctx.camera_name,
            local_dt=local_dt,
        )
    rel_thumb_legacy = relative_paths_for(ctx.camera_name, local_dt)[1]
    thumb_full = ctx.storage_root / rel_thumb_legacy
    thumb_full.parent.mkdir(parents=True, exist_ok=True)
    materialize_thumb(clip_full, thumb_full)
    return rel_thumb_legacy, []


def _clip_already_ingested(ctx: IngestContext, source_filename: str) -> bool:
    """True iff a Clip row already exists for ``(ctx.cam_id, source_filename)`` (idempotency guard)."""
    with get_session(ctx.engine) as session:
        existing = session.scalar(select(Clip.id).where(Clip.camera_id == ctx.cam_id, Clip.source_filename == source_filename))
    return existing is not None


def materialize_and_persist_clip(  # noqa: PLR0913  # 6 args is the irreducible per-clip contract (identity + IO + ctx)
    *,
    source_filename: str,
    start_ts: datetime,
    end_ts: datetime,
    materialize_clip: Callable[[Path], None],
    materialize_thumb: Callable[[Path, Path], None],
    ctx: IngestContext,
) -> Clip | None:
    """Run the per-clip pipeline shared by the poller and ``import_local``.

    Steps: duplicate-check (idempotent on ``(camera_id, source_filename)``) -> compute canonical
    paths + mkdir -> ``materialize_clip(clip_full)`` -> detect. On the success path the detector's
    per-frame buffer is encoded to one JPEG per frame plus a ``ClipFrame`` row each, and
    ``Clip.thumb_path`` points at the highest-scoring frame; on the no-detect / detection-error
    fallback ``materialize_thumb`` extracts a single legacy thumb at ``<HHMMSS>.jpg``.

    All file IO (clip download, frame encodes, fsync) completes before any DB insert, preserving
    file-before-row ordering per spec §4.4. ``Clip`` and its ``ClipFrame`` rows commit in the same
    transaction so a crash leaves either both or neither.
    """
    if _clip_already_ingested(ctx, source_filename):
        logger.info("skip duplicate: camera=%s source=%s", ctx.camera_name, source_filename)
        return None

    local_dt = start_ts.astimezone(ctx.camera_tz)
    rel_clip = relative_paths_for(ctx.camera_name, local_dt)[0]
    clip_full = ctx.storage_root / rel_clip
    clip_full.parent.mkdir(parents=True, exist_ok=True)

    materialize_clip(clip_full)
    detect_kwargs, scored_frames = detection_for(ctx.detector, clip_full)
    rel_thumb, clip_frames = _materialize_thumbs(
        detect_kwargs=detect_kwargs,
        scored_frames=scored_frames,
        clip_full=clip_full,
        local_dt=local_dt,
        materialize_thumb=materialize_thumb,
        ctx=ctx,
    )

    clip = Clip(
        camera_id=ctx.cam_id,
        source_filename=source_filename,
        start_ts=start_ts,
        end_ts=end_ts,
        duration_seconds=(end_ts - start_ts).total_seconds(),
        file_path=rel_clip,
        thumb_path=rel_thumb,
        file_size_bytes=clip_full.stat().st_size,
        ingested_at=ctx.now,
        frames=clip_frames,
        **detect_kwargs,
    )
    with get_session(ctx.engine) as session:
        session.add(clip)
    return clip


def _ingest_recording(rec: Recording, *, client: AmcrestClient, ctx: IngestContext) -> Clip | None:
    """Poller per-clip wrapper: download via the camera client, then run the shared pipeline."""

    def download(dest: Path) -> None:
        # Amcrest client fsyncs the .part before atomic rename — see Task 14.
        client.download_recording(rec.camera_path, dest=dest)

    return materialize_and_persist_clip(
        source_filename=rec.source_filename,
        start_ts=rec.start_ts,
        end_ts=rec.end_ts,
        materialize_clip=download,
        materialize_thumb=extract_thumbnail,
        ctx=ctx,
    )


# --- per-camera tick ------------------------------------------------------------------------------


@dataclass
class PollerArgs:  # pylint: disable=too-many-instance-attributes  # flat CLI-arg bag; the rule targets behavior-rich classes, not data containers
    """Resolved CLI arguments. Empty strings / None mean "use defaults"."""

    config_path: Path | None = None
    camera: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int | None = None
    no_detect: bool = False
    list_only: bool = False
    verbose: bool = False

    @property
    def truncates_default_window(self) -> bool:
        """True when ``--since`` / ``--until`` / ``--limit`` narrow the search below the full
        ``[last_polled_at, now]`` default. ``run_tick`` reads this to decide whether to advance
        ``cameras.last_polled_at`` after a successful (or failed) tick: a scoped query cannot
        prove it covered the full window, so the resume cursor must stay where it is — otherwise
        the next default tick would skip whatever the scoped run missed. ``--list-only`` is
        handled separately (entire DB-write block is gated upstream).
        """
        return self.since is not None or self.until is not None or self.limit is not None


def _resolve_window(*, db_camera: Camera, args: PollerArgs, retention_days: int, now: datetime) -> tuple[datetime, datetime]:
    """Compute the ``(since, until)`` recording-search window per spec §4.4 step 2."""
    if args.since is not None:
        since = args.since
    elif db_camera.last_polled_at is not None:
        since = db_camera.last_polled_at
    else:
        since = now - timedelta(days=retention_days)
    until = args.until if args.until is not None else now
    return since, until


@dataclass(frozen=True)
class _PollWindow:
    """Camera-local time window the tick searched, for human-facing display only.

    ``local_since`` / ``local_until`` are formatted in the camera's tz (per-camera ``timezone`` if
    set, else ``web.display_timezone``) so the operator reads them in the same clock as the cat.
    ``tz_name`` is the IANA zone for the suffix in the print summary. The DB stores datetimes as
    UTC; these are derived strings for display only.
    """

    local_since: str
    local_until: str
    tz_name: str


@dataclass
class _CameraTickResult:
    """Outcome of one camera's tick — what state to apply afterwards."""

    success: bool
    window: _PollWindow
    status_on_failure: PollStatus = PollStatus.OK  # only consulted when success=False
    error_msg: str = ""
    ingested: list[Clip] = field(default_factory=list)


def _poll_camera(  # noqa: PLR0913  # pylint: disable=too-many-locals  # orchestration helper; dataclass-bundling these would just nest the noise
    *,
    config: Config,
    db_camera: Camera,
    cam_cfg: CameraConfig,
    engine: Engine,
    args: PollerArgs,
    detector: Detector | None,
    now: datetime,
) -> _CameraTickResult:
    """Run one camera's tick. Returns an outcome the caller applies to camera state.

    Catches typed Amcrest errors and maps them to the appropriate ``PollStatus``; any clip-level
    failure (detector error etc.) is recorded inside the clip row and does NOT fail the tick.
    """
    tz_name = cam_cfg.timezone or config.web.display_timezone
    camera_tz = ZoneInfo(tz_name)
    since, until = _resolve_window(db_camera=db_camera, args=args, retention_days=config.retention.clip_days, now=now)
    window = _PollWindow(
        local_since=since.astimezone(camera_tz).strftime("%Y-%m-%d %H:%M:%S"),
        local_until=until.astimezone(camera_tz).strftime("%Y-%m-%d %H:%M:%S"),
        tz_name=tz_name,
    )
    ctx = IngestContext(
        engine=engine,
        storage_root=config.storage_root,
        camera_name=cam_cfg.name,
        cam_id=db_camera.id,
        camera_tz=camera_tz,
        detector=detector,
        now=now,
    )

    client = AmcrestClient(cam_cfg, config.camera_secrets, camera_tz=camera_tz)
    ingested: list[Clip] = []
    try:
        with client:
            for rec in _limited(client.iter_recordings(since=since, until=until), args.limit):
                if args.list_only:
                    logger.info("list-only: camera=%s source=%s start=%s", cam_cfg.name, rec.source_filename, rec.start_ts.isoformat())
                    continue
                clip = _ingest_recording(rec, client=client, ctx=ctx)
                if clip is not None:
                    ingested.append(clip)
    except CameraUnreachableError as exc:
        logger.warning("camera %s unreachable: %s", cam_cfg.name, exc)
        return _CameraTickResult(
            success=False,
            window=window,
            status_on_failure=PollStatus.UNREACHABLE,
            error_msg=str(exc),
            ingested=ingested,
        )
    except (CameraAuthError, CameraAPIError) as exc:
        logger.warning("camera %s API error: %s", cam_cfg.name, exc)
        return _CameraTickResult(
            success=False,
            window=window,
            status_on_failure=PollStatus.ERROR,
            error_msg=str(exc),
            ingested=ingested,
        )
    return _CameraTickResult(success=True, window=window, ingested=ingested)


def _limited[ItemT](items: Iterable[ItemT], limit: int | None) -> Iterator[ItemT]:
    """Yield at most ``limit`` items (or all if limit is None)."""
    if limit is None:
        yield from items
        return
    for idx, item in enumerate(items):
        if idx >= limit:
            return
        yield item


# --- main orchestration ---------------------------------------------------------------------------


class _ParsedArgs(argparse.Namespace):
    """Typed view over the parsed ``cat-watcher-poller`` Namespace."""

    config: Path | None = None
    camera: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int | None = None
    no_detect: bool = False
    list_only: bool = False
    once: bool = False
    verbose: bool = False


def _parse_args(argv: Sequence[str] | None) -> PollerArgs:
    """Parse the ``cat-watcher-poller`` CLI surface (per spec §4.10 + Task 17)."""
    parser = argparse.ArgumentParser(
        prog="cat-watcher-poller",
        description="Poll configured Amcrest cameras for new motion clips and ingest them.",
    )
    _ = parser.add_argument(
        "--once",
        action="store_true",
        help="kept for LaunchAgent compat; the poller is always one-shot",
    )
    _ = parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="path to config.toml; default precedence is --config > $CAT_WATCHER_CONFIG > ./config.toml",
    )
    _ = parser.add_argument(
        "--camera",
        type=str,
        default=None,
        metavar="NAME",
        help="poll only this camera (must match a name in [[cameras]]); default polls all cameras",
    )
    _ = parser.add_argument(
        "--since",
        type=_parse_iso_datetime,
        default=None,
        metavar="ISO8601",
        help=(
            "start of the recording window; ISO 8601 (e.g. 2026-05-04T00:00:00 or 2026-05-04T00:00:00-04:00). "
            "Naive values are interpreted as OS-local time and converted to UTC. "
            "Default: cameras.last_polled_at, falling back to now - retention.clip_days days"
        ),
    )
    _ = parser.add_argument(
        "--until",
        type=_parse_iso_datetime,
        default=None,
        metavar="ISO8601",
        help="end of the recording window; same parsing rules as --since. Default: now (UTC)",
    )
    _ = parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="cap the number of recordings processed per camera per tick (default: no cap)",
    )
    _ = parser.add_argument(
        "--no-detect",
        action="store_true",
        help="skip the YOLO detector; clips ingest with analysis_error='skipped: --no-detect'",
    )
    _ = parser.add_argument(
        "--list-only",
        action="store_true",
        help="strict dry-run: list matching recordings without downloading, persisting, or mutating any state",
    )
    _ = parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="show diagnostic logs at INFO level (httpxyz requests, empty-window notes, retention details). "
        "Default suppresses these; the per-tick summary always prints to stdout.",
    )
    ns = parser.parse_args(argv, namespace=_ParsedArgs())
    return PollerArgs(
        config_path=ns.config,
        camera=ns.camera,
        since=ns.since,
        until=ns.until,
        limit=ns.limit,
        no_detect=ns.no_detect,
        list_only=ns.list_only,
        verbose=ns.verbose,
    )


def _parse_iso_datetime(raw: str) -> datetime:
    """Parse ISO 8601 datetime; naive values are interpreted as OS-local time, then converted to UTC.

    Explicit offsets (``...+00:00``, ``...Z``, etc.) are honored as-is and converted to UTC. Naive
    interpretation as OS-local relies on Python's ``datetime.astimezone()`` documented behavior: a
    naive ``self`` is presumed to be in the system timezone. Operators type input in their own
    clock; the DB stores UTC. Display in the camera's tz happens at print time.
    """
    return datetime.fromisoformat(raw).astimezone(UTC)


def ensure_db_camera(engine: Engine, cam_cfg: CameraConfig) -> Camera:
    """Return the ``Camera`` row for ``cam_cfg.name``, creating it if absent."""
    with get_session(engine) as session:
        existing = session.scalar(select(Camera).where(Camera.name == cam_cfg.name))
        if existing is not None:
            session.expunge(existing)
            return existing
        cam = Camera(name=cam_cfg.name, display_name=cam_cfg.display_name, host=cam_cfg.host)
        session.add(cam)
        session.flush()
        session.expunge(cam)
        return cam


def run_tick(*, config: Config, args: PollerArgs, engine: Engine, detector: Detector | None, now: datetime) -> None:
    """Execute one full poller tick. Caller is responsible for the PID lock + storage wait.

    ``args.list_only`` is a strict dry-run: no ``Clip`` rows, no camera-state updates, no
    ``AgentStart`` row, no retention sweep, no heartbeat. ``ensure_db_camera`` still runs because
    its create-if-missing is benign init (the row is read for ``last_polled_at`` regardless).
    """
    if not args.list_only:
        with get_session(engine) as session:
            session.add(AgentStart(agent_name=_AGENT_NAME, started_at=now))

    cameras_to_poll: list[CameraConfig] = [c for c in config.cameras if c.name == args.camera] if args.camera else list(config.cameras)
    if args.camera and not cameras_to_poll:
        msg = f"--camera {args.camera!r} not found in config"
        raise PollerError(msg)

    for cam_cfg in cameras_to_poll:
        db_camera = ensure_db_camera(engine, cam_cfg)
        try:
            outcome = _poll_camera(
                config=config,
                db_camera=db_camera,
                cam_cfg=cam_cfg,
                engine=engine,
                args=args,
                detector=detector,
                now=now,
            )
        except Exception:
            logger.exception("unexpected failure polling camera %s", cam_cfg.name)
            _emit(f"{cam_cfg.name}: tick failed (error) — unexpected exception (re-run with --verbose for traceback)")
            if not args.list_only:
                with get_session(engine) as session:
                    update_camera_state_failure(
                        session,
                        camera_id=db_camera.id,
                        status=PollStatus.ERROR,
                        error="unexpected exception",
                        now=now,
                        advance_cursor=not args.truncates_default_window,
                    )
            continue
        _print_camera_summary(cam_cfg=cam_cfg, outcome=outcome, list_only=args.list_only)
        if not args.list_only:
            with get_session(engine) as session:
                if outcome.success:
                    update_camera_state_success(
                        session,
                        camera_id=db_camera.id,
                        ingested_clips=outcome.ingested,
                        now=now,
                        advance_cursor=not args.truncates_default_window,
                    )
                else:
                    update_camera_state_failure(
                        session,
                        camera_id=db_camera.id,
                        status=outcome.status_on_failure,
                        error=outcome.error_msg,
                        now=now,
                        advance_cursor=not args.truncates_default_window,
                    )

    if not args.list_only:
        report = retention.sweep(engine=engine, storage_root=config.storage_root, retention=config.retention, now=now)
        _print_retention_summary(report)
        with get_session(engine) as session:
            upsert_heartbeat(session, agent_name=_AGENT_NAME, now=now)
        _check_alerts_stuck(config=config, engine=engine, now=now)


def _check_alerts_stuck(*, config: Config, engine: Engine, now: datetime) -> None:
    """Fire ``ALERTS_STUCK`` if the ``alerts`` heartbeat is older than ``alerts_stuck_minutes``.

    Per Task 17 carve-out: the poller watches the alerts agent (since the alerts agent can't watch
    itself). The poller does *not* own cool-down state; routing flows through
    :func:`cat_watcher.alerts.dispatch_alert`, which honors the same cool-down + suppression rules
    as the alerts agent's own dispatches.
    """
    with get_session(engine) as session:
        cand = evaluate_heartbeat_watchdog(
            session,
            alert_type=AlertType.ALERTS_STUCK,
            agent_name="alerts",
            stale_minutes=config.alerts.alerts_stuck_minutes,
            public_url=config.web.public_url,
            tz_name=config.web.display_timezone,
            now=now,
        )
    if cand is None:
        return
    dispatch_candidate(cand, config=config, engine=engine, now=now)


def _emit(line: str) -> None:
    """Write one line of human-facing summary output to stdout.

    Wraps ``sys.stdout.write`` so the print/no-print rule (ruff T201 forbids ``print``) lives in one
    place. Logging goes to stderr via the configured handler; this stream is the primary user-facing
    output of an interactive ``cat-watcher-poller`` invocation.
    """
    _ = sys.stdout.write(line + "\n")


def _print_camera_summary(*, cam_cfg: CameraConfig, outcome: _CameraTickResult, list_only: bool) -> None:
    """One line per camera per tick to stdout. Always shown regardless of ``--verbose``."""
    window = f" (window {outcome.window.local_since} .. {outcome.window.local_until} {outcome.window.tz_name})"
    if not outcome.success:
        _emit(f"{cam_cfg.name}: tick failed ({outcome.status_on_failure.value}) — {outcome.error_msg}{window}")
        return
    if list_only:
        _emit(f"{cam_cfg.name}: list-only complete{window}")
    elif outcome.ingested:
        _emit(f"{cam_cfg.name}: ingested {len(outcome.ingested)} clip(s){window}")
    else:
        _emit(f"{cam_cfg.name}: no new recordings{window}")


def _print_retention_summary(report: retention.RetentionReport) -> None:
    """One line for the retention sweep result. Always shown unless ``--list-only`` skipped it."""
    total = (
        report.clips_removed_pass1
        + report.orphans_removed_pass2
        + report.dirs_removed
        + report.agent_starts_pruned
        + report.alerts_sent_pruned
    )
    if total == 0:
        _emit("retention: nothing to clean up")
        return
    _emit(
        f"retention: clips={report.clips_removed_pass1} orphans={report.orphans_removed_pass2} "
        f"dirs={report.dirs_removed} agent_starts={report.agent_starts_pruned} "
        f"alerts_sent={report.alerts_sent_pruned}",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    Default log level is ``WARNING`` so a healthy interactive run only emits the per-camera
    summary + retention line via stdout; problems surface on stderr through the warning/error
    loggers. ``--verbose`` raises the level to ``INFO`` to expose httpxyz requests, the empty-window
    note from ``amcrest_client``, and other diagnostic detail. The full structured-logging design
    (Task 26b in the plan) replaces this when it lands.
    """
    args = _parse_args(argv)
    config = load_config(args.config_path)
    setup_logging(
        agent_name="poller",
        internal_root=config.internal_root,
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    ensure_storage_layout(internal_root=config.internal_root, storage_root=config.storage_root)

    try:
        with pid_lock(config.internal_root):
            wait_for_storage(
                config.storage_root,
                interval_seconds=config.storage.wait_interval_seconds,
                timeout_seconds=config.storage.wait_timeout_seconds,
            )
            engine = create_engine(f"sqlite:///{config.internal_root / 'cat_watcher.sqlite'}")
            try:
                detector = (
                    None
                    if args.no_detect or args.list_only
                    else Detector.from_weights(
                        model_path=config.internal_root / "models" / config.detector.model,
                        frames_to_sample=config.detector.frames_to_sample,
                        confidence_threshold=config.detector.confidence_threshold,
                    )
                )
                run_tick(config=config, args=args, engine=engine, detector=detector, now=datetime.now(UTC))
            finally:
                engine.dispose()
    except PollerLockedError:
        logger.info("another poller already holds the PID lock; exiting")
        return 0
    except PollerError:
        logger.exception("poller failed")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover  # entry-point
    sys.exit(main())

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
8. (Task 18) checks the ``alerts`` heartbeat and dispatches ``ALERTS_STUCK`` if stale.

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
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from sqlalchemy import select

from cat_watcher import retention
from cat_watcher.amcrest_client import AmcrestClient, CameraAPIError, CameraAuthError, CameraUnreachableError
from cat_watcher.config import Config, load_config
from cat_watcher.db import AgentStart, Camera, Clip, Heartbeat, PollStatus, create_engine, get_session
from cat_watcher.detector import Detector, DetectorError
from cat_watcher.storage import ensure_storage_layout, wait_for_storage

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator, Sequence

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.amcrest_client import Recording
    from cat_watcher.config import CameraConfig

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
) -> None:
    """Apply per-field preservation semantics for a successful tick (per spec §4.4 step 5).

    - ``last_polled_at`` advances to ``now`` every time.
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


def update_camera_state_failure(
    session: Session,
    *,
    camera_id: int,
    status: PollStatus,
    error: str,
    now: datetime,
) -> None:
    """Apply per-field semantics for a failed tick.

    - ``last_polled_at`` advances to ``now`` (the tick still happened, even if it errored).
    - ``poll_status`` -> the supplied non-OK status.
    - ``poll_status_since`` is set to ``now`` only on the transition from OK to non-OK; if the
      camera is already non-OK, the original transition timestamp is preserved.
    - ``poll_error`` updates each tick to the latest error message.
    """
    cam = session.get(Camera, camera_id)
    if cam is None:
        msg = f"camera {camera_id!r} not found"
        raise ValueError(msg)
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


def _extract_thumbnail(clip_path: Path, thumb_path: Path) -> None:
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


def _relative_paths_for(camera_name: str, start_ts_local: datetime, suffix_mp4: str = "mp4", suffix_jpg: str = "jpg") -> tuple[str, str]:
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


def _detection_fields_for(detector: Detector | None, clip_full: Path) -> dict[str, object]:
    """Run the detector (or substitute the ``--no-detect`` markers) and return the Clip kwargs."""
    if detector is None:
        return {
            "has_cat": False,
            "max_score": 0.0,
            "frames_sampled": 0,
            "frames_with_cat": 0,
            "best_box_xyxy": None,
            "detector_version": _DETECTOR_VERSION_NO_DETECT,
            "analysis_error": _NO_DETECT_MARKER,
        }
    try:
        result = detector.detect(clip_full)
    except (DetectorError, OSError) as exc:
        logger.warning("detector failed for %s: %s", clip_full, exc)
        return {
            "has_cat": False,
            "max_score": 0.0,
            "frames_sampled": 0,
            "frames_with_cat": 0,
            "best_box_xyxy": None,
            "detector_version": detector.version,
            "analysis_error": f"detect failed: {exc}",
        }
    return {
        "has_cat": result.has_cat,
        "max_score": result.max_score,
        "frames_sampled": result.frames_sampled,
        "frames_with_cat": result.frames_with_cat,
        "best_box_xyxy": list(result.best_box_xyxy) if result.best_box_xyxy is not None else None,
        "detector_version": result.detector_version,
        "analysis_error": None,
    }


@dataclass(frozen=True)
class _IngestContext:
    """Bundle of per-camera state passed into ``_ingest_recording`` to keep its signature small."""

    engine: Engine
    storage_root: Path
    camera_name: str
    cam_id: int
    camera_tz: ZoneInfo
    detector: Detector | None
    now: datetime


def _ingest_recording(rec: Recording, *, client: AmcrestClient, ctx: _IngestContext) -> Clip | None:
    """Download + thumbnail + detect + insert. Returns the Clip on success, None if duplicate."""
    with get_session(ctx.engine) as session:
        existing = session.scalar(
            select(Clip.id).where(Clip.camera_id == ctx.cam_id, Clip.source_filename == rec.source_filename),
        )
    if existing is not None:
        logger.info("skip duplicate: camera=%s source=%s", ctx.camera_name, rec.source_filename)
        return None

    local_dt = rec.start_ts.astimezone(ctx.camera_tz)
    rel_clip, rel_thumb = _relative_paths_for(ctx.camera_name, local_dt)
    clip_full = ctx.storage_root / rel_clip
    thumb_full = ctx.storage_root / rel_thumb
    clip_full.parent.mkdir(parents=True, exist_ok=True)
    thumb_full.parent.mkdir(parents=True, exist_ok=True)

    # Download (Amcrest client fsyncs the .part before atomic rename — see Task 14).
    client.download_recording(rec.camera_path, dest=clip_full)
    # Thumbnail + fsync.
    _extract_thumbnail(clip_full, thumb_full)
    # Detector outcome (real result or no-detect / error markers).
    detect_kwargs = _detection_fields_for(ctx.detector, clip_full)

    duration = (rec.end_ts - rec.start_ts).total_seconds()
    clip = Clip(
        camera_id=ctx.cam_id,
        source_filename=rec.source_filename,
        start_ts=rec.start_ts,
        end_ts=rec.end_ts,
        duration_seconds=duration,
        file_path=rel_clip,
        thumb_path=rel_thumb,
        file_size_bytes=clip_full.stat().st_size,
        ingested_at=ctx.now,
        **detect_kwargs,
    )
    with get_session(ctx.engine) as session:
        session.add(clip)
    return clip


# --- per-camera tick ------------------------------------------------------------------------------


@dataclass
class PollerArgs:
    """Resolved CLI arguments. Empty strings / None mean "use defaults"."""

    config_path: Path | None = None
    camera: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int | None = None
    no_detect: bool = False
    list_only: bool = False


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


@dataclass
class _CameraTickResult:
    """Outcome of one camera's tick — what state to apply afterwards."""

    success: bool
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
    camera_tz = ZoneInfo(cam_cfg.timezone or config.web.display_timezone)
    since, until = _resolve_window(db_camera=db_camera, args=args, retention_days=config.retention.clip_days, now=now)
    ctx = _IngestContext(
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
        return _CameraTickResult(success=False, status_on_failure=PollStatus.UNREACHABLE, error_msg=str(exc), ingested=ingested)
    except (CameraAuthError, CameraAPIError) as exc:
        logger.warning("camera %s API error: %s", cam_cfg.name, exc)
        return _CameraTickResult(success=False, status_on_failure=PollStatus.ERROR, error_msg=str(exc), ingested=ingested)
    return _CameraTickResult(success=True, ingested=ingested)


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


def _parse_args(argv: Sequence[str] | None) -> PollerArgs:
    """Parse the ``cat-watcher-poller`` CLI surface (per spec §4.10 + Task 17)."""
    parser = argparse.ArgumentParser(prog="cat-watcher-poller")
    _ = parser.add_argument("--once", action="store_true", help="kept for LaunchAgent compat; the poller is always one-shot")
    _ = parser.add_argument("--config", type=Path, default=None)
    _ = parser.add_argument("--camera", type=str, default=None)
    _ = parser.add_argument("--since", type=_parse_iso_datetime, default=None)
    _ = parser.add_argument("--until", type=_parse_iso_datetime, default=None)
    _ = parser.add_argument("--limit", type=int, default=None)
    _ = parser.add_argument("--no-detect", action="store_true")
    _ = parser.add_argument("--list-only", action="store_true")
    ns = parser.parse_args(argv, namespace=_ParsedArgs())
    return PollerArgs(
        config_path=ns.config,
        camera=ns.camera,
        since=ns.since,
        until=ns.until,
        limit=ns.limit,
        no_detect=ns.no_detect,
        list_only=ns.list_only,
    )


def _parse_iso_datetime(raw: str) -> datetime:
    """Parse ``YYYY-MM-DDTHH:MM:SS`` (treated as UTC if no offset)."""
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _ensure_db_camera(engine: Engine, cam_cfg: CameraConfig) -> Camera:
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
    """Execute one full poller tick. Caller is responsible for the PID lock + storage wait."""
    with get_session(engine) as session:
        session.add(AgentStart(agent_name=_AGENT_NAME, started_at=now))

    cameras_to_poll: list[CameraConfig] = [c for c in config.cameras if c.name == args.camera] if args.camera else list(config.cameras)
    if args.camera and not cameras_to_poll:
        msg = f"--camera {args.camera!r} not found in config"
        raise PollerError(msg)

    for cam_cfg in cameras_to_poll:
        db_camera = _ensure_db_camera(engine, cam_cfg)
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
            with get_session(engine) as session:
                update_camera_state_failure(
                    session,
                    camera_id=db_camera.id,
                    status=PollStatus.ERROR,
                    error="unexpected exception",
                    now=now,
                )
            continue
        with get_session(engine) as session:
            if outcome.success:
                update_camera_state_success(session, camera_id=db_camera.id, ingested_clips=outcome.ingested, now=now)
            else:
                update_camera_state_failure(
                    session,
                    camera_id=db_camera.id,
                    status=outcome.status_on_failure,
                    error=outcome.error_msg,
                    now=now,
                )

    if not args.list_only:
        report = retention.sweep(engine=engine, storage_root=config.storage_root, retention=config.retention, now=now)
        logger.info("retention sweep: %s", report)
        with get_session(engine) as session:
            upsert_heartbeat(session, agent_name=_AGENT_NAME, now=now)
        # The ALERTS_STUCK watchdog (read the ``alerts`` heartbeat, dispatch via
        # ``cat_watcher.alerts.dispatch_alert`` if older than ``config.alerts.alerts_stuck_minutes``)
        # is wired up in Task 18, when the alerts module exists. The poller intentionally does not
        # own cool-down state.


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)
    config = load_config(args.config_path)
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

"""HTTP routes for the cat-watcher web UI.

The full route surface lands across Tasks 20-24; this module today owns:

* ``/health`` (Task 20) — auth-bypassed liveness probe.
* ``/clips`` and ``/clips/{id}`` (Task 21) — clip listing + detail (HTML).
* ``/media/clip/{id}.mp4`` (Task 21) — MP4 streaming with HTTP byte-Range support.
* ``/media/thumb/{id}.jpg`` (Task 21) — thumbnail JPEG.

Routes read state via the SQLAlchemy engine and Jinja2 templates attached to ``app.state`` — they
do **not** instantiate their own engine or templates loader.

**Storage-offline degradation (spec §4.13):** the ``/media/...`` routes distinguish two failure
modes for missing files:

* ``503 Service Unavailable`` — the external drive is offline (probed via
  ``storage_root.is_dir()``). The timeline template's ``onerror`` handler swaps in a placeholder
  SVG and the storage-offline banner shows.
* ``410 Gone`` — the drive is mounted but this specific clip's file isn't on disk anymore (e.g.
  retention sweep removed the file but hasn't yet pruned the row). Distinct response code keeps
  the operator-visible signal in logs separable from the bulk-offline case.
"""

from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Protocol, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy import desc, select

from cat_watcher.db import Camera, Clip, Heartbeat, get_session

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fastapi.templating import Jinja2Templates
    from sqlalchemy.engine import Engine

    from cat_watcher.config import Config


class _AppState(Protocol):
    """Typed view of the attributes :func:`cat_watcher.web.app.build_app` writes onto ``app.state``.

    FastAPI types ``app.state`` as ``Any`` (it's a free-form attribute bag), so every read would
    otherwise need a ``cast`` + ``# pyright: ignore[reportAny]`` pair. Centralizing the cast in
    :func:`_state` and projecting through this protocol gives handlers fully-typed access to the
    three pieces of shared state — engine, config, and templates — without per-callsite ceremony.
    """

    engine: Engine
    config: Config
    templates: Jinja2Templates


def _state(request: Request) -> _AppState:
    """Return a typed view of ``request.app.state``; the only place the cast is needed."""
    return cast("_AppState", request.app.state)  # pyright: ignore[reportAny]  # FastAPI types ``request.app`` as Any


_AGENT_NAME_WEB = "web"
_CLIPS_LIST_LIMIT = 200
_THUMB_MEDIA_TYPE = "image/jpeg"
_VIDEO_MEDIA_TYPE = "video/mp4"


health_router = APIRouter()
clips_router = APIRouter()
media_router = APIRouter()


@health_router.get("/health")
async def health(request: Request) -> dict[str, str | None]:
    """Read-only liveness probe.

    Returns ``{status, heartbeat, now}`` where ``heartbeat`` is the latest persisted
    ``heartbeats('web')`` row's ``last_seen_at`` (ISO 8601, UTC) and ``now`` is the current server
    time. Always 200 — staleness interpretation is the alerts agent's job, not this route's. The
    route does **not** write its own heartbeat; only the periodic background task in the lifespan
    keeps the row fresh.
    """
    state = _state(request)
    now = datetime.now(UTC)
    with get_session(state.engine) as session:
        hb = session.get(Heartbeat, _AGENT_NAME_WEB)
    heartbeat = hb.last_seen_at.isoformat() if hb is not None else None
    return {"status": "ok", "heartbeat": heartbeat, "now": now.isoformat()}


@clips_router.get("/clips")
async def list_clips(
    request: Request,
    *,
    camera: str | None = None,
    has_cat: bool | None = None,
    date_str: str | None = None,
) -> object:
    """Render the clip-listing page filtered by ``camera`` / ``has_cat`` / ``date_str``.

    Ordering is ``start_ts DESC`` and the result set is capped at :data:`_CLIPS_LIST_LIMIT` so a
    runaway query against a populated DB doesn't render a 50k-row page. The ``date_str`` filter
    interprets dates as UTC days (``[start_of_day, start_of_day + 1d)``); the spec defers
    timezone-aware filtering to a later iteration.
    """
    state = _state(request)

    stmt = select(Clip).join(Camera).order_by(desc(Clip.start_ts)).limit(_CLIPS_LIST_LIMIT)
    if camera is not None:
        stmt = stmt.where(Camera.name == camera)
    if has_cat is not None:
        stmt = stmt.where(Clip.has_cat.is_(has_cat))
    if date_str is not None:
        day = date.fromisoformat(date_str)
        day_start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        stmt = stmt.where(Clip.start_ts >= day_start).where(Clip.start_ts < day_start + timedelta(days=1))

    with get_session(state.engine) as session:
        clips = list(session.scalars(stmt))
        cameras = list(session.scalars(select(Camera).order_by(Camera.name)))
        # Build a {camera_id: display_name} mapping eagerly so the template can render the camera
        # display name without triggering lazy-loaded attribute access on a closed session.
        clip_rows = [_clip_summary(clip, cameras) for clip in clips]

    return state.templates.TemplateResponse(
        request,
        "clips.html.jinja",
        {
            "clip_rows": clip_rows,
            "cameras": cameras,
            "filters": {"camera": camera, "has_cat": has_cat, "date": date_str},
            "tz": state.config.web.display_timezone,
        },
    )


@clips_router.get("/clips/{clip_id}")
async def clip_detail(request: Request, clip_id: int) -> object:
    """Render the detail page for ``clip_id``: video player + detection metadata + label form.

    The label-form HTML (POST/DELETE to ``/clips/{id}/label``) is rendered here per Task 21's
    contract; the endpoints themselves land in Task 22.
    """
    state = _state(request)

    with get_session(state.engine) as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        camera = session.get(Camera, clip.camera_id)
        # Detach the rows from the session so the template can read attributes after exit.
        session.expunge(clip)
        if camera is not None:
            session.expunge(camera)

    return state.templates.TemplateResponse(
        request,
        "clip_detail.html.jinja",
        {"clip": clip, "camera": camera, "tz": state.config.web.display_timezone},
    )


def _clip_video_relpath(clip: Clip) -> str:
    """Accessor for ``Clip.file_path``; named so callsites read intent rather than dotted access."""
    return clip.file_path


def _clip_thumb_relpath(clip: Clip) -> str:
    """Accessor for ``Clip.thumb_path``; pairs with :func:`_clip_video_relpath`."""
    return clip.thumb_path


@media_router.get("/media/clip/{clip_id}.mp4")
async def media_clip(request: Request, clip_id: int) -> FileResponse:
    """Serve the MP4 file for ``clip_id``. ``FileResponse`` handles HTTP byte-Range itself
    (``<video>`` seeking, RFC 7233 § 4): ``200`` for no Range header, ``206`` with a correct
    ``Content-Range`` for valid ranges, ``400`` for malformed Range syntax, ``416`` for ranges
    that fall past EOF. Plus our 404/503/410 from :func:`_resolve_media_path`.
    """
    state = _state(request)
    file_path = _resolve_media_path(engine=state.engine, clip_id=clip_id, get_relpath=_clip_video_relpath, config=state.config)
    return FileResponse(file_path, media_type=_VIDEO_MEDIA_TYPE)


@media_router.get("/media/thumb/{clip_id}.jpg")
async def media_thumb(request: Request, clip_id: int) -> FileResponse:
    """Serve the JPEG thumbnail for ``clip_id``. Same 404/503/410 semantics as :func:`media_clip`.

    No Range support — thumbnails are small (a few KB) so a single ``FileResponse`` is fine.
    """
    state = _state(request)
    file_path = _resolve_media_path(engine=state.engine, clip_id=clip_id, get_relpath=_clip_thumb_relpath, config=state.config)
    return FileResponse(file_path, media_type=_THUMB_MEDIA_TYPE)


def _clip_summary(clip: Clip, cameras: list[Camera]) -> dict[str, object]:
    """Project a Clip + its Camera into a flat dict the template can render without lazy loads."""
    by_id = {cam.id: cam for cam in cameras}
    cam = by_id.get(clip.camera_id)
    return {
        "id": clip.id,
        "camera_display_name": cam.display_name if cam is not None else "",
        "camera_name": cam.name if cam is not None else "",
        "source_filename": clip.source_filename,
        "start_ts": clip.start_ts,
        "duration_seconds": clip.duration_seconds,
        "has_cat": clip.has_cat,
        "manual_has_cat": clip.manual_has_cat,
        "max_score": clip.max_score,
    }


def _resolve_media_path(
    *,
    engine: Engine,
    clip_id: int,
    get_relpath: Callable[[Clip], str],
    config: Config,
) -> Path:
    """Look up ``clip_id`` and return the on-disk path for the relpath returned by ``get_relpath``.

    Raises ``HTTPException(404)`` if the row is missing, ``HTTPException(503)`` if ``storage_root``
    is offline (spec §4.13), and ``HTTPException(410)`` if the row exists but the specific file is
    gone (data-integrity drift, distinct from the bulk-offline case).
    """
    with get_session(engine) as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        relative = get_relpath(clip)
    if not _storage_root_available(config.storage_root):
        raise HTTPException(status_code=503, detail="external storage offline")
    full = config.storage_root / relative
    if not full.is_file():
        raise HTTPException(status_code=410, detail="media file unavailable")
    return full


def _storage_root_available(storage_root: Path) -> bool:
    """Probe whether the external drive is currently accessible per spec §4.13.

    The poller / backup agents run a write-probe on startup; the web agent only needs a read probe
    (it never writes to ``storage_root``), so checking ``is_dir()`` suffices. A stricter write
    probe would conflict with read-only mounts that operators sometimes use during data recovery.
    """
    return storage_root.is_dir()

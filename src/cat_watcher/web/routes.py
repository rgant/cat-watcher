"""HTTP routes for the cat-watcher web UI.

The full route surface lands across Tasks 20-24; this module today owns:

* ``/health`` (Task 20) — auth-bypassed liveness probe.
* ``/clips`` and ``/clips/{id}`` (Task 21) — clip listing + detail (HTML).
* ``/media/clip/{id}.mp4`` (Task 21) — MP4 streaming with HTTP byte-Range support.
* ``/media/thumb/{id}.jpg`` (Task 21) — thumbnail JPEG.
* ``POST /clips/{id}/label`` and ``DELETE /clips/{id}/label`` (Task 22) — manual label
  set/clear; the POST redirects 303 back to the detail page so the form survives a refresh.
* ``/`` and ``/timeline`` (Task 23) — per-camera SVG activity timeline; switches between per-clip
  markers and per-hour heatmap buckets at the 24h threshold (spec §4.7.1).
* ``/cameras``, ``/stats``, ``/alerts`` (Task 24) — per-camera health, 30-day daily activity
  aggregation (with manual-label override applied via ``COALESCE(manual_has_cat, has_cat)``),
  and alert dispatch history.

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

import operator
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Annotated, Protocol, cast
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import Integer, desc, func, select

from cat_watcher.db import AlertSent, Camera, Clip, Heartbeat, get_session

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from typing import TypedDict

    from fastapi.templating import Jinja2Templates
    from sqlalchemy.engine import Engine

    from cat_watcher.config import Config

    class _CameraRow(TypedDict):
        """Flat projection of a :class:`Camera` ORM row consumed by the timeline view-model."""

        id: int
        name: str
        display_name: str


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
# Spec §4.7: ``/cameras`` shows the most recent N alerts per camera; 5 is the same N the design
# spec calls out, kept here as a constant so the template doesn't have to know it.
_CAMERA_RECENT_ALERTS_LIMIT = 5
# Spec §4.7: ``/stats`` and ``/alerts`` cap their windows at 30 days so a long-running deployment
# doesn't grow into a multi-thousand-row scroll.
_HISTORY_DAYS = 30
_NO_CAMERA_PLACEHOLDER = "—"


health_router = APIRouter()
clips_router = APIRouter()
label_router = APIRouter()
media_router = APIRouter()
timeline_router = APIRouter()
cameras_router = APIRouter()
stats_router = APIRouter()
alerts_router = APIRouter()


_TIMELINE_RANGES: dict[str, timedelta] = {
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
}
_TIMELINE_DEFAULT_RANGE = "24h"
# Spec §4.7.1: bucketing kicks in for windows wider than 24h to keep marker count below the
# pixel-per-clip resolution. Hardcoded — operators don't choose, the rendering does.
_TIMELINE_BUCKET_THRESHOLD = timedelta(hours=24)
_BUCKET_SECONDS = 3600  # one bin per hour


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
    if camera:
        stmt = stmt.where(Camera.name == camera)
    if has_cat is not None:
        stmt = stmt.where(Clip.has_cat.is_(has_cat))
    if date_str:
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


@label_router.post("/clips/{clip_id}/label")
async def set_label(
    request: Request,
    clip_id: int,
    *,
    has_cat: Annotated[bool, Form()],
    notes: Annotated[str, Form()] = "",
) -> Response:
    """Persist a manual label override for ``clip_id``.

    Empty ``notes`` collapses to ``NULL`` so the column distinguishes "no notes provided" from a
    deliberate empty-string label. Redirects 303 back to the detail page; that's the
    POST-Redirect-GET pattern the form relies on so a browser refresh after submit doesn't resubmit
    the form.
    """
    state = _state(request)
    with get_session(state.engine) as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        clip.manual_has_cat = has_cat
        clip.manual_label_notes = notes or None
        clip.manual_label_at = datetime.now(UTC)
    return RedirectResponse(request.url_for("clip_detail", clip_id=clip_id), status_code=303)


@label_router.delete("/clips/{clip_id}/label")
async def clear_label(request: Request, clip_id: int) -> Response:
    """Clear all three manual-label columns for ``clip_id`` back to ``NULL``.

    Returns 204 (No Content) so the htmx caller can ``window.location.reload()`` without parsing a
    body. 404 if the clip row is gone.
    """
    state = _state(request)
    with get_session(state.engine) as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        clip.manual_has_cat = None
        clip.manual_label_notes = None
        clip.manual_label_at = None
    return Response(status_code=204)


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


@timeline_router.get("/")
async def root(request: Request, range: str = _TIMELINE_DEFAULT_RANGE) -> object:  # noqa: A002  # ``range`` is the public query-param name
    """Render the activity timeline at the default 24h window (or whatever ``?range=`` overrides to).

    Same handler as :func:`timeline`; the distinct route name is what the nav's ``url_for('root')``
    resolves to so the "Timeline" link in the header has a stable name.
    """
    return _render_timeline(request, range_key=range)


@timeline_router.get("/timeline")
async def timeline(request: Request, range: str = _TIMELINE_DEFAULT_RANGE) -> object:  # noqa: A002
    """Render the activity timeline scoped to ``?range=`` (one of 6h / 24h / 7d / 30d).

    Lives at a separate path so HTMX can ``hx-get`` the partial without rewriting the page URL.
    """
    return _render_timeline(request, range_key=range)


def _render_timeline(request: Request, *, range_key: str) -> object:
    """Build the timeline view-model and dispatch to ``timeline.html.jinja``.

    Density bucketing kicks in when the requested window is wider than
    :data:`_TIMELINE_BUCKET_THRESHOLD` (spec §4.7.1). Below the threshold we hand the template a
    flat list of per-clip markers; above it we collapse to per-hour bins keyed by lane.
    """
    state = _state(request)
    delta = _TIMELINE_RANGES.get(range_key, _TIMELINE_RANGES[_TIMELINE_DEFAULT_RANGE])
    if range_key not in _TIMELINE_RANGES:
        # Snap to the default rather than 400-ing — operators following an old bookmark should
        # still get a usable page, just at the standard window.
        range_key = _TIMELINE_DEFAULT_RANGE
    start_window = datetime.now(UTC) - delta
    display_tz = ZoneInfo(state.config.web.display_timezone)

    camera_rows, clip_markers_by_camera, alert_markers = _load_timeline_data(
        engine=state.engine,
        start_window=start_window,
        total_seconds=delta.total_seconds(),
        display_tz=display_tz,
    )

    use_buckets = delta > _TIMELINE_BUCKET_THRESHOLD
    lanes, lanes_have_clips, thumb_cards = _build_lanes_view(
        camera_rows=camera_rows,
        clip_markers_by_camera=clip_markers_by_camera,
        total_seconds=delta.total_seconds(),
        use_buckets=use_buckets,
    )

    return state.templates.TemplateResponse(
        request,
        "timeline.html.jinja",
        {
            "cameras": camera_rows,
            "lanes": lanes,
            "lanes_have_clips": lanes_have_clips,
            "next_longer_range_key": _next_longer_range(range_key),
            "alerts": alert_markers,
            "thumb_cards": thumb_cards,
            "time_axis_marks": _time_axis_marks(
                range_key=range_key,
                start_window=start_window,
                total_seconds=delta.total_seconds(),
                display_tz=display_tz,
            ),
            "range_key": range_key,
            "ranges": list(_TIMELINE_RANGES),
            "use_buckets": use_buckets,
            "storage_online": _storage_root_available(state.config.storage_root),
            "tz": state.config.web.display_timezone,
        },
    )


def _load_timeline_data(
    *,
    engine: Engine,
    start_window: datetime,
    total_seconds: float,
    display_tz: ZoneInfo,
) -> tuple[list[_CameraRow], dict[int, list[dict[str, object]]], list[dict[str, object]]]:
    """Pull the cameras / clips / alerts in the window and project them into template view-models.

    All session-bound work happens inside one ``get_session`` so the rows are projected to plain
    dicts before the session closes — the caller never touches a detached ORM instance.
    """
    with get_session(engine) as session:
        cameras = list(session.scalars(select(Camera).order_by(Camera.name)))
        clips = list(session.scalars(select(Clip).where(Clip.start_ts >= start_window).order_by(Clip.start_ts)))
        alerts = list(session.scalars(select(AlertSent).where(AlertSent.sent_at >= start_window).order_by(AlertSent.sent_at)))
        camera_rows = [_camera_lane(cam) for cam in cameras]
        clip_markers_by_camera: dict[int, list[dict[str, object]]] = defaultdict(list)
        for clip in clips:
            clip_markers_by_camera[clip.camera_id].append(
                _clip_marker(clip, start_window=start_window, total_seconds=total_seconds, display_tz=display_tz),
            )
        alert_markers = [_alert_marker(alert, start_window=start_window, total_seconds=total_seconds) for alert in alerts]
    return camera_rows, clip_markers_by_camera, alert_markers


def _build_lanes_view(
    *,
    camera_rows: list[_CameraRow],
    clip_markers_by_camera: dict[int, list[dict[str, object]]],
    total_seconds: float,
    use_buckets: bool,
) -> tuple[dict[int, list[dict[str, object]]], bool, list[dict[str, object]]]:
    """Project per-camera clip markers into the SVG lanes view-model + flat newest-first thumb_cards.

    The thumb-card list is sorted by ``start_ts`` DESC and interleaved across cameras so the strip
    reads top-left to most-recent regardless of which camera produced it. Each card carries the
    camera's display_name precomputed so the template iterates a single sequence.
    """
    if use_buckets:
        lanes: dict[int, list[dict[str, object]]] = {
            cam_row["id"]: _bucket_markers(
                clip_markers_by_camera.get(cam_row["id"], []),
                total_seconds=total_seconds,
            )
            for cam_row in camera_rows
        }
    else:
        lanes = {cam_row["id"]: clip_markers_by_camera.get(cam_row["id"], []) for cam_row in camera_rows}
    lanes_have_clips = any(lanes.get(cam_row["id"]) for cam_row in camera_rows)
    camera_display_by_id = {cam_row["id"]: cam_row["display_name"] for cam_row in camera_rows}
    thumb_cards = sorted(
        [
            {**marker, "camera_display_name": camera_display_by_id[cam_id]}
            for cam_id, markers in clip_markers_by_camera.items()
            for marker in markers
        ],
        key=operator.itemgetter("start_ts"),
        reverse=True,
    )
    return lanes, lanes_have_clips, thumb_cards


def _camera_lane(cam: Camera) -> _CameraRow:
    """Project a Camera row into a flat dict the template can render without lazy loads."""
    return {"id": cam.id, "name": cam.name, "display_name": cam.display_name}


def _clip_marker(
    clip: Clip,
    *,
    start_window: datetime,
    total_seconds: float,
    display_tz: ZoneInfo,
) -> dict[str, object]:
    """Project a Clip into the SVG-and-card view-model with a precomputed local-time stamp.

    ``css_classes`` and ``display_stamp`` are precomputed (rather than templated as conditionals or
    filter chains) so djlint's HTML reformatter can't insert newlines into the class attribute, and
    so Jinja stays free of timezone arithmetic.
    """
    offset_seconds = (clip.start_ts - start_window).total_seconds()
    effective = clip.has_cat if clip.manual_has_cat is None else clip.manual_has_cat
    classes = ["clip", "clip-cat" if effective else "clip-no-cat"]
    if clip.manual_has_cat is not None:
        classes.append("clip-manual")
    if clip.analysis_error:
        classes.append("clip-error")
    return {
        "id": clip.id,
        "start_ts": clip.start_ts,
        "duration_seconds": clip.duration_seconds,
        "max_score": clip.max_score,
        "has_cat": effective,
        "manual_label": clip.manual_has_cat is not None,
        "analysis_error": bool(clip.analysis_error),
        "css_classes": " ".join(classes),
        "display_stamp": clip.start_ts.astimezone(display_tz).strftime("%H:%M:%S"),
        "display_start": clip.start_ts.astimezone(display_tz).strftime("%a %H:%M:%S"),
        # Fractional positions in [0, 1]; the template multiplies by the lane's pixel width.
        "x_frac": max(0.0, min(1.0, offset_seconds / total_seconds)),
        "w_frac": max(0.0, min(1.0, clip.duration_seconds / total_seconds)),
    }


def _alert_marker(alert: AlertSent, *, start_window: datetime, total_seconds: float) -> dict[str, object]:
    """Project an AlertSent into the template view-model: position fraction + label + type."""
    offset_seconds = (alert.sent_at - start_window).total_seconds()
    return {
        "sent_at": alert.sent_at,
        "alert_type": alert.alert_type.value,
        "x_frac": max(0.0, min(1.0, offset_seconds / total_seconds)),
    }


def _bucket_markers(markers: list[dict[str, object]], *, total_seconds: float) -> list[dict[str, object]]:
    """Collapse per-clip markers into per-hour bins with precomputed opacity and fill-class.

    Each output dict carries: ``bin_index``, ``x_frac``, ``w_frac``, ``count``, ``cat_count``,
    ``opacity`` (0.20-0.95 scaled to this *lane's* max count so a quiet camera doesn't get washed
    out by a busy one), and ``fill_class`` (``bucket-cat`` or ``bucket-no-cat``).
    """
    buckets: dict[int, dict[str, int]] = defaultdict(lambda: {"count": 0, "cat_count": 0})
    bucket_w_frac = _BUCKET_SECONDS / total_seconds
    for marker in markers:
        x_frac = cast("float", marker["x_frac"])
        bin_index = int(x_frac * total_seconds // _BUCKET_SECONDS)
        bucket = buckets[bin_index]
        bucket["count"] += 1
        if marker["has_cat"]:
            bucket["cat_count"] += 1
    if not buckets:
        return []
    lane_max = max(b["count"] for b in buckets.values())
    return [
        {
            "bin_index": bin_index,
            "x_frac": (bin_index * _BUCKET_SECONDS) / total_seconds,
            "w_frac": bucket_w_frac,
            "count": stats["count"],
            "cat_count": stats["cat_count"],
            "opacity": round(0.20 + 0.75 * (stats["count"] / lane_max), 3),
            "fill_class": "bucket-cat" if stats["cat_count"] > 0 else "bucket-no-cat",
        }
        for bin_index, stats in sorted(buckets.items())
    ]


def _format_tick_label_hour_minute(dt_local: datetime) -> str:
    """Used at 6h and 24h: ``HH:MM`` clock label."""
    return dt_local.strftime("%H:%M")


def _format_tick_label_weekday_hour(dt_local: datetime) -> str:
    """Used at 7d: ``Mon 14:00`` so a label survives a date crossing without a separate marker."""
    return dt_local.strftime("%a %H:%M")


def _format_tick_label_day_month(dt_local: datetime) -> str:
    """Used at 30d: ``5 May`` — no clock component needed when ticks are 24h apart."""
    return dt_local.strftime("%-d %b")


@dataclass(frozen=True, slots=True)
class _TickConfig:
    """Per-range tick cadence + labelling rule for the time-axis row."""

    seconds: int  # spacing between adjacent ticks
    label_every: int  # n-th tick gets a text label
    formatter: Callable[[datetime], str]


_TICK_CONFIG: dict[str, _TickConfig] = {
    "6h": _TickConfig(seconds=30 * 60, label_every=2, formatter=_format_tick_label_hour_minute),
    "24h": _TickConfig(seconds=60 * 60, label_every=1, formatter=_format_tick_label_hour_minute),
    "7d": _TickConfig(seconds=6 * 60 * 60, label_every=2, formatter=_format_tick_label_weekday_hour),
    "30d": _TickConfig(seconds=24 * 60 * 60, label_every=1, formatter=_format_tick_label_day_month),
}


def _format_day_label(dt_local: datetime, *, range_key: str, end_local: datetime) -> str | None:
    """Choose the label that sits next to a midnight day-boundary marker, by range.

    ``None`` means render the boundary line with no label (6h windows are too short to need a date
    prompt). ``today``/``yesterday`` is used at 24h so the operator can read the boundary without
    parsing dates. 7d and 30d get a full ``5 May`` style date.
    """
    if range_key == "6h":
        return None
    if range_key == "24h":
        return "today" if dt_local.date() == end_local.date() else "yesterday"
    return dt_local.strftime("%-d %b")


def _tick_marks(
    *,
    range_key: str,
    start_window: datetime,
    total_seconds: float,
    display_tz: ZoneInfo,
) -> list[dict[str, object]]:
    """Build the per-range tick row: every nth tick gets a label, the rest are unlabeled."""
    tick_config = _TICK_CONFIG[range_key]
    n_ticks = int(total_seconds // tick_config.seconds)
    marks: list[dict[str, object]] = []
    for i in range(1, n_ticks + 1):
        offset = i * tick_config.seconds
        tick_dt_local = (start_window + timedelta(seconds=offset)).astimezone(display_tz)
        label = tick_config.formatter(tick_dt_local) if i % tick_config.label_every == 0 else None
        marks.append({"x_frac": offset / total_seconds, "label": label, "kind": "tick"})
    return marks


def _day_boundary_marks(
    *,
    range_key: str,
    start_window: datetime,
    total_seconds: float,
    display_tz: ZoneInfo,
) -> list[dict[str, object]]:
    """Build the per-midnight day-boundary marks in ``display_tz`` for the window.

    Iteration is in *calendar-day* space: ``date + timedelta(days=1)`` always advances exactly one
    calendar day, and ``datetime.combine`` re-resolves the UTC offset for each midnight. Adding
    ``timedelta(days=1)`` to a tz-aware ``datetime`` instead would carry the start-of-window's
    offset across DST transitions and place the boundary an hour off (or on the wrong date).
    """
    start_local = start_window.astimezone(display_tz)
    end_local = (start_window + timedelta(seconds=total_seconds)).astimezone(display_tz)
    marks: list[dict[str, object]] = []
    day = start_local.date() + timedelta(days=1)
    while day <= end_local.date():
        cursor = datetime.combine(day, datetime.min.time(), tzinfo=display_tz)
        offset = (cursor - start_local).total_seconds()
        if 0 < offset < total_seconds:
            marks.append(
                {
                    "x_frac": offset / total_seconds,
                    "label": _format_day_label(cursor, range_key=range_key, end_local=end_local),
                    "kind": "day",
                },
            )
        day += timedelta(days=1)
    return marks


def _time_axis_marks(
    *,
    range_key: str,
    start_window: datetime,
    total_seconds: float,
    display_tz: ZoneInfo,
) -> list[dict[str, object]]:
    """Build the SVG time-axis view-model: tick rows, day boundaries, and a 'now' marker.

    Each output dict carries ``x_frac`` ([0, 1] left fraction), an optional ``label`` (string or
    ``None``), and a ``kind`` discriminator: ``"tick"``, ``"day"``, or ``"now"``. The template
    consumes the list in order and picks the SVG element type per kind.
    """
    return [
        *_tick_marks(range_key=range_key, start_window=start_window, total_seconds=total_seconds, display_tz=display_tz),
        *_day_boundary_marks(range_key=range_key, start_window=start_window, total_seconds=total_seconds, display_tz=display_tz),
        {"x_frac": 1.0, "label": None, "kind": "now"},
    ]


def _next_longer_range(range_key: str) -> str | None:
    """Return the next preset wider than ``range_key`` in :data:`_TIMELINE_RANGES`, or ``None``.

    Used by the empty-state CTA: at 6h -> 24h, at 24h -> 7d, at 7d -> 30d, at 30d -> ``None``.
    """
    keys = list(_TIMELINE_RANGES)
    if range_key not in keys:
        return None
    idx = keys.index(range_key)
    return keys[idx + 1] if idx + 1 < len(keys) else None


@cameras_router.get("/cameras")
async def cameras_page(request: Request) -> object:
    """Render the per-camera health table (spec §4.7).

    Each row surfaces the camera's polling state — display name, ``poll_status``, the timestamp
    poll-status went non-OK (``poll_status_since``), the last poll attempt, the last clip ingested,
    the last cat detection, and a truncated poll error — plus the camera's most recent alerts so an
    operator can correlate "this camera went unreachable at HH:MM" with "an INACTIVITY alert fired
    N hours later". A separate non-camera-scoped section covers ``camera_id IS NULL`` alerts on
    ``/alerts``; this page is camera-scoped only.
    """
    state = _state(request)
    with get_session(state.engine) as session:
        cameras = list(session.scalars(select(Camera).order_by(Camera.name)))
        recent_by_camera: dict[int, list[dict[str, object]]] = {}
        for cam in cameras:
            recent_alerts = list(
                session.scalars(
                    select(AlertSent)
                    .where(AlertSent.camera_id == cam.id)
                    .order_by(desc(AlertSent.sent_at))
                    .limit(_CAMERA_RECENT_ALERTS_LIMIT),
                ),
            )
            recent_by_camera[cam.id] = [_alert_summary(a, camera_display_name=cam.display_name) for a in recent_alerts]
        camera_rows = [_camera_row(cam, recent_alerts=recent_by_camera[cam.id]) for cam in cameras]

    return state.templates.TemplateResponse(
        request,
        "cameras.html.jinja",
        {"cameras": camera_rows, "tz": state.config.web.display_timezone},
    )


@stats_router.get("/stats")
async def stats_page(request: Request) -> object:
    """Render the 30-day daily clip aggregation (spec §4.7).

    Groups by ``(camera_id, date(start_ts))`` and computes total clips + cat-positive clips per
    bucket. Cat-positive uses ``COALESCE(manual_has_cat, has_cat)`` so corrected manual labels flow
    into the stat — a clip the detector got wrong but a human re-labeled is now counted on the
    human's call. ``CAST … AS INTEGER`` is required because SQLite doesn't sum booleans directly;
    with the cast each truthy bit becomes 1 and the SUM gives a per-day integer count.
    """
    state = _state(request)
    cutoff = datetime.now(UTC) - timedelta(days=_HISTORY_DAYS)
    cat_expr = func.coalesce(Clip.manual_has_cat, Clip.has_cat).cast(Integer)

    with get_session(state.engine) as session:
        cameras = list(session.scalars(select(Camera).order_by(Camera.name)))
        camera_display_by_id = {cam.id: cam.display_name for cam in cameras}
        date_label = func.date(Clip.start_ts).label("d")
        # ``func.count`` is callable at runtime via SQLAlchemy's GenericFunction proxy; pylint can't
        # see through the proxy and flags ``not-callable``, so we disable it on this one line.
        rows = session.execute(
            select(
                Clip.camera_id,
                date_label,
                func.count().label("total"),  # pylint: disable=not-callable
                func.sum(cat_expr).label("cat_total"),
            )
            .where(Clip.start_ts >= cutoff)
            .group_by(Clip.camera_id, date_label)
            .order_by(date_label.desc(), Clip.camera_id),
        ).all()

    stat_rows = [_stat_row(row, camera_display_by_id=camera_display_by_id) for row in rows]
    return state.templates.TemplateResponse(
        request,
        "stats.html.jinja",
        {"rows": stat_rows, "tz": state.config.web.display_timezone},
    )


@alerts_router.get("/alerts")
async def alerts_page(request: Request) -> object:
    """Render the last 30 days of dispatched alerts (spec §4.7).

    Sorted newest-first. Camera-scoped alerts (``camera_id`` set) render the camera's display name;
    non-camera alerts (``WEB_DOWN``, ``DISK_LOW``, etc.) render :data:`_NO_CAMERA_PLACEHOLDER` so
    operators can scan the column for "which subsystem fired this" without losing the row to a
    blank cell.
    """
    state = _state(request)
    cutoff = datetime.now(UTC) - timedelta(days=_HISTORY_DAYS)

    with get_session(state.engine) as session:
        cameras = list(session.scalars(select(Camera)))
        camera_display_by_id = {cam.id: cam.display_name for cam in cameras}
        alerts = list(
            session.scalars(
                select(AlertSent).where(AlertSent.sent_at >= cutoff).order_by(desc(AlertSent.sent_at)),
            ),
        )
        alert_rows = [
            _alert_summary(
                alert,
                camera_display_name=camera_display_by_id.get(alert.camera_id) if alert.camera_id is not None else None,
            )
            for alert in alerts
        ]

    return state.templates.TemplateResponse(
        request,
        "alerts.html.jinja",
        {"alerts": alert_rows, "tz": state.config.web.display_timezone},
    )


def _stat_row(row: object, *, camera_display_by_id: dict[int, str]) -> dict[str, object]:
    """Project a stats query Row into a flat dict the template can render.

    The query selects ``(camera_id, date_label, total, cat_total)`` so we destructure positionally
    via ``cast`` + tuple-unpack rather than attribute access; SQLAlchemy types Row column accessors
    as ``Any``, which would otherwise blossom into per-attribute ``reportAny`` warnings here. The
    label expressions are typed positionally — column 0 is ``camera_id``, column 3 is the
    ``func.sum(cat_expr)`` result which is ``None`` for an all-NULL bucket — so we coerce the
    nullable last value to ``int`` for template arithmetic.
    """
    camera_id, date_value, total, cat_total = cast("tuple[int, object, int, int | None]", tuple(cast("Sequence[object]", row)))
    return {
        "camera_id": camera_id,
        "camera_display_name": camera_display_by_id.get(camera_id, ""),
        "date": date_value,
        "total": total,
        "cat_total": int(cat_total or 0),
    }


def _camera_row(cam: Camera, *, recent_alerts: list[dict[str, object]]) -> dict[str, object]:
    """Project a Camera row into a flat dict the template can render without lazy loads.

    Includes the precomputed ``recent_alerts`` list so the template's outer ``{% for cam %}`` loop
    can render the camera's recent-alert sub-table without re-running a query per row.
    ``poll_error`` is truncated here so the template doesn't have to know the cap (and the cap can
    move without sweeping the templates).
    """
    return {
        "id": cam.id,
        "name": cam.name,
        "display_name": cam.display_name,
        "host": cam.host,
        "poll_status": cam.poll_status.value,
        "poll_status_since": cam.poll_status_since,
        "last_polled_at": cam.last_polled_at,
        "last_clip_at": cam.last_clip_at,
        "last_cat_seen_at": cam.last_cat_seen_at,
        "poll_error": _truncate(cam.poll_error, 200),
        "recent_alerts": recent_alerts,
    }


def _alert_summary(alert: AlertSent, *, camera_display_name: str | None) -> dict[str, object]:
    """Project an AlertSent row into a flat dict the alerts/cameras templates can render.

    ``camera_display`` collapses the ``camera_id is None`` and ``camera_id is set but its row was
    deleted from config`` cases into the same em-dash placeholder — both correspond to "no live
    camera attached", which is what the operator cares about.
    """
    return {
        "sent_at": alert.sent_at,
        "alert_type": alert.alert_type.value,
        "camera_display": camera_display_name or _NO_CAMERA_PLACEHOLDER,
        "subject": alert.subject,
        "email_ok": alert.email_ok,
        "macos_ok": alert.macos_ok,
    }


def _truncate(value: str | None, limit: int) -> str | None:
    """Cap ``value`` at ``limit`` characters with a single trailing ellipsis when truncated."""
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"

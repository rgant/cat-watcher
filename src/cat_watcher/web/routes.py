"""HTTP routes for the cat-watcher web UI.

Routes read state via the SQLAlchemy engine and Jinja2 templates attached to ``app.state``.

**Storage-offline degradation (spec §4.13):** ``/media/...`` returns ``503`` when ``storage_root``
is offline, ``410`` when mounted but the file is gone.
"""

import logging
import operator
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from sqlalchemy import Integer, desc, func, select
from sqlalchemy import cast as sql_cast

from cat_watcher.db import (
    AlertSent,
    Camera,
    Clip,
    ClipFrame,
    ClipFrameSubject,
    ClipLabelSummary,
    Heartbeat,
    Subject,
    get_session,
)
from cat_watcher.web._app_state import get_state
from cat_watcher.web.clips_routes import clips_router

__all__ = [
    "alerts_router",
    "cameras_router",
    "clips_router",
    "health_router",
    "media_router",
    "membership_router",
    "review_router",
    "stats_router",
    "timeline_router",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from typing import TypedDict

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config

    class _CameraRow(TypedDict):
        id: int
        name: str
        display_name: str

    class _ClipLabelInfo(TypedDict):
        effective_has_cat: bool
        show_manual_badge: bool


logger = logging.getLogger(__name__)


_AGENT_NAME_WEB = "web"
_THUMB_MEDIA_TYPE = "image/jpeg"
_VIDEO_MEDIA_TYPE = "video/mp4"
# Spec §4.7: ``/cameras`` shows the most recent N alerts per camera; 5 is the same N the design spec
# calls out, kept here as a constant so the template doesn't have to know it.
_CAMERA_RECENT_ALERTS_LIMIT = 5
# Spec §4.7: ``/stats`` and ``/alerts`` cap their windows at 30 days so a long-running deployment
# doesn't grow into a multi-thousand-row scroll.
_HISTORY_DAYS = 30
_NO_CAMERA_PLACEHOLDER = "—"


health_router = APIRouter()
review_router = APIRouter()
membership_router = APIRouter()
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
    state = get_state(request)
    now = datetime.now(UTC)
    with get_session(state.engine) as session:
        hb = session.get(Heartbeat, _AGENT_NAME_WEB)
    heartbeat = hb.last_seen_at.isoformat() if hb is not None else None
    return {"status": "ok", "heartbeat": heartbeat, "now": now.isoformat()}


@review_router.post("/clips/{clip_id}/reviewed", status_code=204)
async def mark_clip_reviewed(request: Request, clip_id: int) -> Response:
    """Set ``clips.reviewed_at`` to now, marking the operator review complete.

    Idempotent: a clip already reviewed is left unchanged (its original timestamp is preserved).
    Returns 204 on success (both the state-change and idempotent no-op cases) and 404 if the clip
    row does not exist.
    """
    state = get_state(request)
    with get_session(state.engine) as session:
        clip = session.execute(select(Clip).where(Clip.id == clip_id)).scalar_one_or_none()
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        if clip.reviewed_at is None:
            clip.reviewed_at = datetime.now(UTC)
            logger.info("clip_reviewed", extra={"clip_id": clip.id})
        session.commit()
    return Response(status_code=204)


@review_router.delete("/clips/{clip_id}/reviewed", status_code=204)
async def reopen_clip_review(request: Request, clip_id: int) -> Response:
    """Clear ``clips.reviewed_at``, re-opening the clip for operator review.

    Idempotent: clearing an already-unreviewed clip is a no-op. Memberships
    (``clip_frame_subjects``) are NOT touched — the re-open workflow keeps existing frame tags
    intact. Returns 204 on success and 404 if the clip row does not exist.
    """
    state = get_state(request)
    with get_session(state.engine) as session:
        clip = session.execute(select(Clip).where(Clip.id == clip_id)).scalar_one_or_none()
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        if clip.reviewed_at is not None:
            clip.reviewed_at = None
            logger.info("clip_review_reopened", extra={"clip_id": clip.id})
        session.commit()
    return Response(status_code=204)


def _validate_frame_membership(session: Session, *, clip_id: int, frame_id: int, subject_id: int) -> Subject:
    if session.get(Clip, clip_id) is None:
        raise HTTPException(status_code=404, detail="clip not found")
    frame = session.get(ClipFrame, frame_id)
    if frame is None or frame.clip_id != clip_id:
        raise HTTPException(status_code=404, detail="frame not found")
    subj = session.get(Subject, subject_id)
    if subj is None:
        raise HTTPException(status_code=404, detail="subject not found")
    return subj


@membership_router.put("/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}", status_code=204)
async def add_frame_subject(request: Request, clip_id: int, frame_id: int, subject_id: int) -> Response:
    """Add ``subject_id`` to ``frame_id``'s subject set (insert-or-ignore). 409 if subject is archived."""
    state = get_state(request)
    with get_session(state.engine) as session:
        subj = _validate_frame_membership(session, clip_id=clip_id, frame_id=frame_id, subject_id=subject_id)
        if subj.archived_at is not None:
            raise HTTPException(status_code=409, detail="subject is archived")
        if session.get(ClipFrameSubject, (frame_id, subject_id)) is None:
            session.add(ClipFrameSubject(clip_frame_id=frame_id, subject_id=subject_id))
            session.commit()
            logger.info("clip_frame_subject_added", extra={"clip_id": clip_id, "frame_id": frame_id, "subject_slug": subj.slug})
    return Response(status_code=204)


@membership_router.delete("/clips/{clip_id}/frames/{frame_id}/subjects/{subject_id}", status_code=204)
async def remove_frame_subject(request: Request, clip_id: int, frame_id: int, subject_id: int) -> Response:
    """Remove ``subject_id`` from ``frame_id``'s subject set (no-op if absent). Archived subjects allowed."""
    state = get_state(request)
    with get_session(state.engine) as session:
        subj = _validate_frame_membership(session, clip_id=clip_id, frame_id=frame_id, subject_id=subject_id)
        existing = session.get(ClipFrameSubject, (frame_id, subject_id))
        if existing is not None:
            session.delete(existing)
            session.commit()
            logger.info("clip_frame_subject_removed", extra={"clip_id": clip_id, "frame_id": frame_id, "subject_slug": subj.slug})
    return Response(status_code=204)


def _clip_video_relpath(clip: Clip) -> str:
    return clip.file_path


def _clip_thumb_relpath(clip: Clip) -> str:
    return clip.thumb_path


@media_router.get("/media/clip/{clip_id}.mp4")
async def media_clip(request: Request, clip_id: int) -> FileResponse:
    """Serve the MP4 file for ``clip_id``.

    ``FileResponse`` handles HTTP byte-Range itself (``<video>`` seeking, RFC 7233 § 4): ``200``
    for no Range header, ``206`` with a correct ``Content-Range`` for valid ranges, ``400`` for
    malformed Range syntax, ``416`` for ranges that fall past EOF. Plus our 404/503/410 from
    :func:`_resolve_media_path`.
    """
    state = get_state(request)
    file_path = _resolve_media_path(engine=state.engine, clip_id=clip_id, get_relpath=_clip_video_relpath, config=state.config)
    return FileResponse(file_path, media_type=_VIDEO_MEDIA_TYPE)


@media_router.get("/media/thumb/{clip_id}.jpg")
async def media_thumb(request: Request, clip_id: int) -> FileResponse:
    """Serve the JPEG thumbnail for ``clip_id``. Same 404/503/410 semantics as :func:`media_clip`.

    No Range support — thumbnails are small (a few KB) so a single ``FileResponse`` is fine.
    """
    state = get_state(request)
    file_path = _resolve_media_path(engine=state.engine, clip_id=clip_id, get_relpath=_clip_thumb_relpath, config=state.config)
    return FileResponse(file_path, media_type=_THUMB_MEDIA_TYPE)


@media_router.get("/media/frame/{frame_id}.jpg", name="media_frame")
async def media_frame(request: Request, frame_id: int) -> FileResponse:
    """Serve the per-frame JPEG keyed by ``ClipFrame.id``; same 404/503/410 semantics as :func:`media_thumb`."""
    state = get_state(request)
    file_path = _resolve_frame_media_path(engine=state.engine, frame_id=frame_id, config=state.config)
    return FileResponse(file_path, media_type=_THUMB_MEDIA_TYPE)


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


def _resolve_frame_media_path(*, engine: Engine, frame_id: int, config: Config) -> Path:
    """Look up ``ClipFrame.id`` and return the on-disk path for its ``thumb_path``.

    Mirrors :func:`_resolve_media_path` but keyed on a per-frame row instead of a clip.
    """
    with get_session(engine) as session:
        frame = session.get(ClipFrame, frame_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="frame not found")
        relative = frame.thumb_path
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

    Distinct route name from :func:`timeline` so the nav's ``url_for('root')`` has a stable target.
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
    state = get_state(request)
    delta = _TIMELINE_RANGES.get(range_key, _TIMELINE_RANGES[_TIMELINE_DEFAULT_RANGE])
    if range_key not in _TIMELINE_RANGES:
        # Snap to the default rather than 400-ing — operators following an old bookmark should still
        # get a usable page, just at the standard window.
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
        summary_by_clip = {
            summary.clip_id: summary
            for summary in session.scalars(
                select(ClipLabelSummary).where(ClipLabelSummary.clip_id.in_([clip.id for clip in clips])),
            )
        }
        alerts = list(session.scalars(select(AlertSent).where(AlertSent.sent_at >= start_window).order_by(AlertSent.sent_at)))
        camera_rows = [_camera_lane(cam) for cam in cameras]
        clip_markers_by_camera: dict[int, list[dict[str, object]]] = defaultdict(list)
        for clip in clips:
            summary = summary_by_clip[clip.id]
            clip_markers_by_camera[clip.camera_id].append(
                _clip_marker(
                    clip,
                    label={
                        "effective_has_cat": summary.effective_has_cat,
                        "show_manual_badge": summary.has_manual_cat and clip.reviewed_at is not None,
                    },
                    start_window=start_window,
                    total_seconds=total_seconds,
                    display_tz=display_tz,
                ),
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
    return {"id": cam.id, "name": cam.name, "display_name": cam.display_name}


def _clip_marker(
    clip: Clip,
    *,
    label: _ClipLabelInfo,
    start_window: datetime,
    total_seconds: float,
    display_tz: ZoneInfo,
) -> dict[str, object]:
    """Project a Clip into the SVG-and-card view-model with a precomputed local-time stamp.

    ``css_classes`` and ``display_stamp`` are precomputed (rather than templated as conditionals or
    filter chains) so djlint's HTML reformatter can't insert newlines into the class attribute, and
    so Jinja stays free of timezone arithmetic.

    ``label`` carries ``effective_has_cat`` from the ``clip_label_summary`` view and
    ``show_manual_badge`` (``has_manual_cat AND reviewed_at IS NOT NULL``) — the badge fires only
    when the clip is reviewed AND has cat frame tags, not for partially-tagged unreviewed clips.
    """
    offset_seconds = (clip.start_ts - start_window).total_seconds()
    effective_has_cat = label["effective_has_cat"]
    show_manual_badge = label["show_manual_badge"]
    classes = ["clip", "clip-cat" if effective_has_cat else "clip-no-cat"]
    if show_manual_badge:
        classes.append("clip-manual")
    if clip.analysis_error:
        classes.append("clip-error")
    return {
        "id": clip.id,
        "start_ts": clip.start_ts,
        "duration_seconds": clip.duration_seconds,
        "max_score": clip.max_score,
        "has_cat": effective_has_cat,
        "manual_label": show_manual_badge,
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
    return dt_local.strftime("%H:%M")


def _format_tick_label_weekday_hour(dt_local: datetime) -> str:
    """At 7d: ``Mon 14:00`` so a label survives a date crossing without a separate marker."""
    return dt_local.strftime("%a %H:%M")


def _format_tick_label_day_month(dt_local: datetime) -> str:
    """At 30d: ``5 May`` — no clock component needed when ticks are 24h apart."""
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
    state = get_state(request)
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
    bucket. Cat-positive reads ``effective_has_cat`` from the ``clip_label_summary`` view.
    ``CAST … AS INTEGER`` is required because SQLite doesn't sum booleans directly; with the cast
    each truthy bit becomes 1 and the SUM gives a per-day integer count.
    """
    state = get_state(request)
    cutoff = datetime.now(UTC) - timedelta(days=_HISTORY_DAYS)
    cat_expr = sql_cast(ClipLabelSummary.effective_has_cat, Integer)

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
                func.count().label("total"),  # pylint: disable=not-callable  # sqlalchemy func.count() is a generative construct, not the builtin; pylint false positive
                func.sum(cat_expr).label("cat_total"),
            )
            .join(ClipLabelSummary, ClipLabelSummary.clip_id == Clip.id)
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
    operators can scan the column for "which subsystem fired this" without losing the row to a blank
    cell.
    """
    state = get_state(request)
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

    Destructured positionally via ``cast`` + tuple-unpack because SQLAlchemy types Row column
    accessors as ``Any`` and per-attribute access would blossom into ``reportAny`` warnings. The
    nullable ``func.sum(cat_expr)`` is coerced to ``int`` for template arithmetic.
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

    Embeds the precomputed ``recent_alerts`` list so the template's ``{% for cam %}`` loop renders
    the recent-alert sub-table without re-running a query per row. ``poll_error`` is truncated here
    so the cap can move without sweeping templates.
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
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"

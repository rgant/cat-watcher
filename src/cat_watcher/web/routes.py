"""HTTP routes for the cat-watcher web UI.

Routes read state via the SQLAlchemy engine and Jinja2 templates attached to ``app.state``.

**Storage-offline degradation (spec §4.13):** ``/media/...`` returns ``503`` when ``storage_root``
is offline, ``410`` when mounted but the file is gone.
"""

import logging
import math
import operator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from sqlalchemy import desc, select
from sqlalchemy.orm import selectinload

from cat_watcher.config import active_unsure_threshold
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
from cat_watcher.identity import query_cat_subjects
from cat_watcher.labels import is_manual_override
from cat_watcher.timefmt import local_date, local_stamp
from cat_watcher.visits import query_cat_clips
from cat_watcher.web._app_state import get_state
from cat_watcher.web.clips_routes import clips_router, identity_chip
from cat_watcher.web.pagination import PageInfo, page_info, parse_page
from cat_watcher.web.stats_routes import HISTORY_DAYS
from cat_watcher.web.time_axis import time_axis_marks

__all__ = [
    "alerts_router",
    "cameras_router",
    "clips_router",
    "health_router",
    "media_router",
    "membership_router",
    "review_router",
    "timeline_router",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import tzinfo
    from pathlib import Path
    from typing import TypedDict

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config
    from cat_watcher.identity import ClipIdentity

    class _CameraRow(TypedDict):
        id: int
        display_name: str

    class _Lane(TypedDict):
        """One SVG lane. ``key`` is unique in the view. ``label`` is the text at the left of the lane."""

        key: str
        label: str
        markers: list[dict[str, object]]

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
_NO_CAMERA_PLACEHOLDER = "—"


health_router = APIRouter()
review_router = APIRouter()
membership_router = APIRouter()
media_router = APIRouter()
timeline_router = APIRouter()
cameras_router = APIRouter()
alerts_router = APIRouter()


_TIMELINE_RANGES: dict[str, timedelta] = {
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "3d": timedelta(days=3),
}
_TIMELINE_DEFAULT_RANGE = "24h"
#: The ``lanes`` values and the text of their toggle links.
_LANE_VIEW_LABELS: dict[str, str] = {"cats": "Cats", "boxes": "Boxes"}
_DEFAULT_LANE_VIEW = "cats"
#: The ``clips`` values and the text of their toggle links.
_STRIP_FILTER_LABELS: dict[str, str] = {"cat": "Cat clips", "all": "All clips"}
_DEFAULT_STRIP_FILTER = "cat"
_STRIP_PAGE_SIZE = 48
# A cat lane key holds a subject id, so no cat lane key can be this key.
_UNKNOWN_CAT_LANE_KEY = "unknown-cat"
_UNKNOWN_CAT_LABEL = "Unknown cat"
# SVG viewBox units. ``.lane-label`` in ``style.css`` is a 10-unit monospace font with .06em letter
# spacing, so one uppercase character takes about 6.6 units. A change to that rule must change this.
_LANE_LABEL_CHAR_ADVANCE = 6.6
_LANE_LABEL_PADDING = 8
_LANE_LABEL_MIN_WIDTH = 80


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


def _reviewed_at_payload(reviewed_at: datetime | None, *, tz: tzinfo) -> dict[str, str | None]:
    """Render ``reviewed_at`` the way the detail page would, for the client to insert verbatim.

    The browser cannot derive these itself: ``new Date()`` near local midnight puts the badge on
    tomorrow's date, and the display zone is a server-side config value the page never sees.
    """
    if reviewed_at is None:
        return {"reviewed_at_iso": None, "reviewed_at_stamp": None, "reviewed_at_date": None}
    return {
        "reviewed_at_iso": reviewed_at.isoformat(),
        "reviewed_at_stamp": local_stamp(reviewed_at, tz=tz),
        "reviewed_at_date": local_date(reviewed_at, tz=tz),
    }


@review_router.post("/clips/{clip_id}/reviewed")
async def mark_clip_reviewed(request: Request, clip_id: int) -> dict[str, str | None]:
    """Set ``clips.reviewed_at`` to now, marking the operator review complete.

    Idempotent: a clip already reviewed is left unchanged (its original timestamp is preserved), and
    the response repeats the stored value. Returns the rendered timestamp so the page can update
    without a reload; 404 if the clip row does not exist.
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
        reviewed_at = clip.reviewed_at
    return _reviewed_at_payload(reviewed_at, tz=ZoneInfo(state.config.web.display_timezone))


@review_router.delete("/clips/{clip_id}/reviewed")
async def reopen_clip_review(request: Request, clip_id: int) -> dict[str, str | None]:
    """Clear ``clips.reviewed_at``, re-opening the clip for operator review.

    Idempotent: clearing an already-unreviewed clip is a no-op. Memberships
    (``clip_frame_subjects``) are NOT touched — the re-open workflow keeps existing frame tags
    intact. Returns null timestamp fields; 404 if the clip row does not exist.
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
    return _reviewed_at_payload(None, tz=ZoneInfo(state.config.web.display_timezone))


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
async def root(
    request: Request,
    range: str = _TIMELINE_DEFAULT_RANGE,  # noqa: A002  # ``range`` is the public query-param name
    lanes: str = _DEFAULT_LANE_VIEW,
    clips: str = _DEFAULT_STRIP_FILTER,
    page: str | None = None,
) -> object:
    """Render the activity timeline at the default 24h window (or whatever ``?range=`` overrides to).

    Distinct route name from :func:`timeline` so the nav's ``url_for('root')`` has a stable target.
    """
    return _render_timeline(request, view=_timeline_view(range_key=range, lanes=lanes, clips=clips), page=page)


@timeline_router.get("/timeline")
async def timeline(
    request: Request,
    range: str = _TIMELINE_DEFAULT_RANGE,  # noqa: A002
    lanes: str = _DEFAULT_LANE_VIEW,
    clips: str = _DEFAULT_STRIP_FILTER,
    page: str | None = None,
) -> object:
    """Render the activity timeline scoped to ``?range=`` (one of 6h / 24h / 3d).

    Lives at a separate path so HTMX can ``hx-get`` the partial without rewriting the page URL.
    """
    return _render_timeline(request, view=_timeline_view(range_key=range, lanes=lanes, clips=clips), page=page)


@dataclass(frozen=True, slots=True)
class _TimelineView:
    """The view state that every timeline link carries. Each field holds a known value."""

    range_key: str
    lanes: str
    clips: str


def _timeline_view(*, range_key: str, lanes: str, clips: str) -> _TimelineView:
    """Give the view state of a request. An unknown value snaps to its default, so an old bookmark still works."""
    return _TimelineView(
        range_key=range_key if range_key in _TIMELINE_RANGES else _TIMELINE_DEFAULT_RANGE,
        lanes=lanes if lanes in _LANE_VIEW_LABELS else _DEFAULT_LANE_VIEW,
        clips=clips if clips in _STRIP_FILTER_LABELS else _DEFAULT_STRIP_FILTER,
    )


def _timeline_url(view: _TimelineView, *, page: int = 1) -> str:
    """Build the URL of a timeline link.

    ``range`` is always in the URL. A ``lanes``, ``clips``, or ``page`` value at its default stays
    out of the URL. A range, view, or filter link has no ``page``, so it opens page 1.
    """
    params = {"range": view.range_key}
    if view.lanes != _DEFAULT_LANE_VIEW:
        params["lanes"] = view.lanes
    if view.clips != _DEFAULT_STRIP_FILTER:
        params["clips"] = view.clips
    if page > 1:
        params["page"] = str(page)
    return f"/timeline?{urlencode(params)}"


def _view_links(view: _TimelineView) -> dict[str, list[dict[str, object]]]:
    """Build the range presets, the view toggle, and the strip filter toggle."""
    return {
        "range_links": [
            {"href": _timeline_url(replace(view, range_key=key)), "label": key, "current": key == view.range_key}
            for key in _TIMELINE_RANGES
        ],
        "lane_links": [
            {"href": _timeline_url(replace(view, lanes=key)), "label": label, "current": key == view.lanes}
            for key, label in _LANE_VIEW_LABELS.items()
        ],
        "strip_links": [
            {"href": _timeline_url(replace(view, clips=key)), "label": label, "current": key == view.clips}
            for key, label in _STRIP_FILTER_LABELS.items()
        ],
    }


def _render_timeline(request: Request, *, view: _TimelineView, page: str | None) -> object:
    """Build the timeline view-model and dispatch to ``timeline.html.jinja``.

    Every range shows one marker per clip.
    """
    state = get_state(request)
    delta = _TIMELINE_RANGES[view.range_key]
    now = datetime.now(UTC)
    window = _Window(start=now - delta, end=now, total_seconds=delta.total_seconds())
    display_tz = ZoneInfo(state.config.web.display_timezone)

    data = _load_timeline_data(
        engine=state.engine,
        window=window,
        display_tz=display_tz,
        unsure_threshold=active_unsure_threshold(state.config),
    )
    lanes, strip = _build_lanes_view(view=view, data=data, requested_page=parse_page(page))
    next_longer_range_key = _next_longer_range(view.range_key)

    return state.templates.TemplateResponse(
        request,
        "timeline.html.jinja",
        {
            "lanes": lanes,
            "label_width": _lane_label_width(lanes),
            "window_has_clips": bool(data.clip_markers),
            "no_cat_visited": view.lanes == "cats" and not data.identity_by_clip,
            "box_view_url": _timeline_url(replace(view, lanes="boxes")),
            "all_clips_url": _timeline_url(replace(view, clips="all")) if view.clips == "cat" else "",
            "next_longer_range_key": next_longer_range_key,
            "next_longer_range_url": _timeline_url(replace(view, range_key=next_longer_range_key)) if next_longer_range_key else "",
            "alerts": data.alert_markers,
            "thumb_cards": strip.cards,
            "strip_page": strip.page,
            "previous_page_url": _timeline_url(view, page=strip.page.number - 1) if strip.page.has_previous else "",
            "next_page_url": _timeline_url(view, page=strip.page.number + 1) if strip.page.has_next else "",
            "time_axis_marks": time_axis_marks(
                range_key=view.range_key,
                start_window=window.start,
                total_seconds=window.total_seconds,
                display_tz=display_tz,
            ),
            "range_key": view.range_key,
            **_view_links(view),
            "storage_online": _storage_root_available(state.config.storage_root),
        },
    )


@dataclass(frozen=True, slots=True)
class _Window:
    """The time span of the timeline, from ``start`` to ``end``, with ``end`` exclusive.

    The markers and the cat clips read the same span. A clip with a start after ``end`` comes from
    a camera clock that is fast. It shows in no view.
    """

    start: datetime
    end: datetime
    total_seconds: float


@dataclass(frozen=True, slots=True)
class _TimelineData:
    """The rows of one timeline window, projected before the session closes.

    ``identity_by_clip`` holds the cat clips of the window only. A clip id outside it is not a cat clip.
    """

    camera_rows: list[_CameraRow]
    cats: list[Subject]
    clip_markers: list[dict[str, object]]
    identity_by_clip: dict[int, ClipIdentity]
    alert_markers: list[dict[str, object]]


def _load_timeline_data(
    *,
    engine: Engine,
    window: _Window,
    display_tz: ZoneInfo,
    unsure_threshold: float,
) -> _TimelineData:
    """Pull the cameras, cats, clips, and alerts of the window in one session.

    The cameras, clips, and alerts become plain dicts before the session closes. The cat subjects
    stay ORM objects. The session does not expire them, so their columns stay readable.
    """
    with get_session(engine) as session:
        cameras = list(session.scalars(select(Camera).order_by(Camera.name)))
        cats = query_cat_subjects(session)
        clip_markers = _query_clip_markers(session, cameras=cameras, window=window, display_tz=display_tz)
        cat_clips = query_cat_clips(session, since=window.start, until=window.end, unsure_threshold=unsure_threshold, cats=cats)
        alerts = session.scalars(select(AlertSent).where(AlertSent.sent_at >= window.start).order_by(AlertSent.sent_at))
        return _TimelineData(
            camera_rows=[_camera_lane(cam) for cam in cameras],
            cats=cats,
            clip_markers=clip_markers,
            identity_by_clip={cat_clip.clip_id: cat_clip.identity for cat_clip in cat_clips},
            alert_markers=[_alert_marker(alert, start_window=window.start, total_seconds=window.total_seconds) for alert in alerts],
        )


def _query_clip_markers(
    session: Session,
    *,
    cameras: Sequence[Camera],
    window: _Window,
    display_tz: ZoneInfo,
) -> list[dict[str, object]]:
    """Give one marker per clip of the window, in ``start_ts`` order.

    Each marker names its camera, so a lane of any kind can show it.
    """
    clips = list(
        session.scalars(select(Clip).where(Clip.start_ts >= window.start, Clip.start_ts < window.end).order_by(Clip.start_ts)),
    )
    summary_by_clip = {
        summary.clip_id: summary
        for summary in session.scalars(select(ClipLabelSummary).where(ClipLabelSummary.clip_id.in_([clip.id for clip in clips])))
    }
    camera_display_by_id = {cam.id: cam.display_name for cam in cameras}
    return [
        {
            **_clip_marker(
                clip,
                label={
                    "effective_has_cat": summary_by_clip[clip.id].effective_has_cat,
                    "show_manual_badge": is_manual_override(
                        has_cat=clip.has_cat,
                        has_manual_cat=summary_by_clip[clip.id].has_manual_cat,
                        reviewed=clip.reviewed_at is not None,
                    ),
                },
                start_window=window.start,
                total_seconds=window.total_seconds,
                display_tz=display_tz,
            ),
            "camera_id": clip.camera_id,
            "camera_display_name": camera_display_by_id[clip.camera_id],
        }
        for clip in clips
    ]


@dataclass(frozen=True, slots=True)
class _StripPage:
    """The thumb cards of one strip page, newest first, and the page they are."""

    cards: list[dict[str, object]]
    page: PageInfo


def _build_lanes_view(*, view: _TimelineView, data: _TimelineData, requested_page: int | None) -> tuple[list[_Lane], _StripPage]:
    """Project the clip markers into the SVG lanes of ``view`` and one page of thumb cards.

    The strip does not depend on ``view.lanes``. The thumb cards interleave the cameras, so the
    strip reads from the newest clip whatever camera produced it.
    """
    if view.lanes == "cats":
        lanes = _cat_lanes(cats=data.cats, clip_markers=data.clip_markers, identity_by_clip=data.identity_by_clip)
    else:
        lanes = _box_lanes(camera_rows=data.camera_rows, clip_markers=data.clip_markers)
    newest_first = sorted(data.clip_markers, key=operator.itemgetter("start_ts"), reverse=True)
    if view.clips == "cat":
        newest_first = [marker for marker in newest_first if marker["id"] in data.identity_by_clip]
    info = page_info(requested=requested_page, total=len(newest_first), page_size=_STRIP_PAGE_SIZE)
    cats_by_slug = {cat.slug: cat for cat in data.cats}
    cards = [
        _thumb_card(marker, identity=data.identity_by_clip.get(_marker_id(marker)), cats_by_slug=cats_by_slug)
        for marker in newest_first[info.offset : info.offset + info.limit]
    ]
    return lanes, _StripPage(cards=cards, page=info)


def _thumb_card(marker: dict[str, object], *, identity: ClipIdentity | None, cats_by_slug: Mapping[str, Subject]) -> dict[str, object]:
    """Copy ``marker`` into a thumb card that names the cats of its clip.

    ``identity`` is ``None`` for a clip that is not a cat clip. Such a card names no cat.
    """
    if identity is None:
        return {**marker, "identity_chips": [], "identity_fallback_label": None}
    if not identity.cats:
        return {**marker, "identity_chips": [], "identity_fallback_label": _UNKNOWN_CAT_LABEL}
    return {**marker, "identity_chips": [identity_chip(named, cats_by_slug) for named in identity.cats], "identity_fallback_label": None}


def _marker_id(marker: dict[str, object]) -> int:
    return cast("int", marker["id"])


def _cat_lanes(
    *,
    cats: Sequence[Subject],
    clip_markers: list[dict[str, object]],
    identity_by_clip: Mapping[int, ClipIdentity],
) -> list[_Lane]:
    """Give one lane per cat, then an "Unknown cat" lane.

    Each active cat gets a lane, in ``cats`` order. An archived cat gets a lane only if it has a clip
    in the window. The "Unknown cat" lane shows only if a cat clip in the window names no cat. A
    clip shows in the lane of each cat that it names. A clip that is not a cat clip shows in no lane.
    """
    slugs_by_clip = {clip_id: {named.slug for named in identity.cats} for clip_id, identity in identity_by_clip.items()}
    cat_markers = [marker for marker in clip_markers if marker["id"] in slugs_by_clip]
    lanes: list[_Lane] = []
    for cat in cats:
        markers = [_cat_lane_marker(marker, color=cat.color) for marker in cat_markers if cat.slug in slugs_by_clip[_marker_id(marker)]]
        if markers or cat.archived_at is None:
            lanes.append({"key": f"cat-{cat.id}", "label": cat.display_name, "markers": markers})
    unknown = [_cat_lane_marker(marker, color=None) for marker in cat_markers if not slugs_by_clip[_marker_id(marker)]]
    if unknown:
        lanes.append({"key": _UNKNOWN_CAT_LANE_KEY, "label": _UNKNOWN_CAT_LABEL, "markers": unknown})
    return lanes


def _cat_lane_marker(marker: dict[str, object], *, color: str | None) -> dict[str, object]:
    """Copy ``marker`` for one cat lane, because one clip can show in more than one lane.

    The copy fills with ``color``. An empty ``lane_color`` lets the CSS fill with the text color.
    """
    return {**marker, "css_classes": f"{marker['css_classes']} clip-cat-lane", "lane_color": color or ""}


def _box_lanes(*, camera_rows: list[_CameraRow], clip_markers: list[dict[str, object]]) -> list[_Lane]:
    """Give one lane per camera, in ``camera_rows`` order. A camera with no clip gets an empty lane."""
    return [
        {
            "key": f"camera-{cam_row['id']}",
            "label": cam_row["display_name"],
            "markers": [marker for marker in clip_markers if marker["camera_id"] == cam_row["id"]],
        }
        for cam_row in camera_rows
    ]


def _lane_label_width(lanes: list[_Lane]) -> int:
    """Give the width of the lane-label column, so the longest label ends before the track starts."""
    longest = max((len(lane["label"]) for lane in lanes), default=0)
    return max(_LANE_LABEL_MIN_WIDTH, math.ceil(longest * _LANE_LABEL_CHAR_ADVANCE) + _LANE_LABEL_PADDING)


def _camera_lane(cam: Camera) -> _CameraRow:
    return {"id": cam.id, "display_name": cam.display_name}


def _clip_marker(
    clip: Clip,
    *,
    label: _ClipLabelInfo,
    start_window: datetime,
    total_seconds: float,
    display_tz: ZoneInfo,
) -> dict[str, object]:
    """Project a Clip into the SVG-and-card view-model with a precomputed local-time stamp.

    The route builds ``css_classes`` and ``display_start``, not the template. Then djlint cannot put
    a line break in the class attribute. Jinja also does no time zone arithmetic.

    ``label`` carries ``effective_has_cat`` from the ``clip_label_summary`` view and
    ``show_manual_badge`` (see :func:`cat_watcher.labels.is_manual_override`) — the badge fires only
    when the reviewed verdict disagrees with the detector, not when the operator merely agrees.
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


def _next_longer_range(range_key: str) -> str | None:
    """Return the next preset wider than ``range_key`` in :data:`_TIMELINE_RANGES`, or ``None``.

    Used by the empty-state CTA: at 6h -> 24h, at 24h -> 3d, at 3d -> ``None``.
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
            recent_by_camera[cam.id] = [_alert_summary(a, source_display=cam.display_name) for a in recent_alerts]
        camera_rows = [_camera_row(cam, recent_alerts=recent_by_camera[cam.id]) for cam in cameras]

    return state.templates.TemplateResponse(
        request,
        "cameras.html.jinja",
        {"cameras": camera_rows},
    )


@alerts_router.get("/alerts")
async def alerts_page(request: Request) -> object:
    """Render the last 30 days of dispatched alerts (spec §4.7).

    Sorted newest-first. The ``Source`` column shows the cat of a per-cat alert, the camera of a
    camera alert, or :data:`_NO_CAMERA_PLACEHOLDER` for any other alert (``WEB_DOWN``, ``DISK_LOW``).
    The placeholder keeps the column scannable. One query loads the cats of every row.
    """
    state = get_state(request)
    cutoff = datetime.now(UTC) - timedelta(days=HISTORY_DAYS)

    with get_session(state.engine) as session:
        cameras = list(session.scalars(select(Camera)))
        camera_display_by_id = {cam.id: cam.display_name for cam in cameras}
        alerts = list(
            session.scalars(
                select(AlertSent).options(selectinload(AlertSent.cat)).where(AlertSent.sent_at >= cutoff).order_by(desc(AlertSent.sent_at)),
            ),
        )
        alert_rows = [_alert_summary(alert, source_display=_alert_source(alert, camera_display_by_id)) for alert in alerts]

    return state.templates.TemplateResponse(
        request,
        "alerts.html.jinja",
        {"alerts": alert_rows},
    )


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


def _alert_source(alert: AlertSent, camera_display_by_id: Mapping[int, str]) -> str | None:
    """Return the cat display name of a per-cat alert, else the camera display name, else ``None``."""
    if alert.cat is not None:
        return alert.cat.display_name
    if alert.camera_id is not None:
        return camera_display_by_id.get(alert.camera_id)
    return None


def _alert_summary(alert: AlertSent, *, source_display: str | None) -> dict[str, object]:
    """Project an AlertSent row into a flat dict the alerts/cameras templates can render.

    ``source`` collapses "no source" and "a camera row that is gone" into the same em-dash
    placeholder. Both mean "no live camera or cat attached".
    """
    return {
        "sent_at": alert.sent_at,
        "alert_type": alert.alert_type.value,
        "source": source_display or _NO_CAMERA_PLACEHOLDER,
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

"""HTTP routes for the clips listing and detail pages.

Routes read state via the SQLAlchemy engine and Jinja2 templates attached to ``app.state``.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import desc, func, select

from cat_watcher.db import Camera, Clip, ClipFrameSubject, ClipLabelSummary, Subject, get_session
from cat_watcher.labels import build_tag_summary, query_cat_frame_counts
from cat_watcher.web._app_state import get_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import Select


_CLIPS_LIST_LIMIT = 200

clips_router = APIRouter()


@dataclass(frozen=True, slots=True)
class _ClipsFilter:
    """``/clips`` filter params bundled so query helpers stay under pylint's local-variable ceiling."""

    reviewed: Literal["any", "no", "yes"]
    camera: str | None
    has_cat: bool | None
    date_str: str | None


def _build_filter_qs(f: _ClipsFilter) -> str:
    """Serialize the filter set as a querystring for row-link carry-through.

    ``reviewed`` is always included so the detail page can construct the back-link.
    """
    params: list[tuple[str, str]] = [("reviewed", f.reviewed)]
    if f.camera:
        params.append(("camera", f.camera))
    if f.has_cat is not None:
        params.append(("has_cat", str(f.has_cat).lower()))
    if f.date_str:
        params.append(("date_str", f.date_str))
    return urlencode(params)


def _clip_query(f: _ClipsFilter) -> Select[tuple[int]]:
    """Return a base SELECT for ``Clip.id`` scoped to filter ``f``, without order/limit.

    Callers add ``WHERE`` for relative position and ``ORDER BY`` / ``LIMIT 1`` to find prev/next.
    """
    stmt: Select[tuple[int]] = select(Clip.id).join(Camera)
    if f.camera:
        stmt = stmt.where(Camera.name == f.camera)
    if f.has_cat is not None:
        stmt = stmt.where(Clip.has_cat.is_(f.has_cat))
    if f.date_str:
        _d = date.fromisoformat(f.date_str)
        day_start = datetime(_d.year, _d.month, _d.day, tzinfo=UTC)
        stmt = stmt.where(Clip.start_ts >= day_start).where(Clip.start_ts < day_start + timedelta(days=1))
    if f.reviewed == "no":
        stmt = stmt.where(Clip.reviewed_at.is_(None))
    elif f.reviewed == "yes":
        stmt = stmt.where(Clip.reviewed_at.is_not(None))
    return stmt


def _build_filtered_nav_urls(engine: Engine, clip: Clip, f: _ClipsFilter, filter_qs: str) -> tuple[str, str]:
    """Return ``(prev_url, next_url)`` scoped to filter ``f``.

    ``prev_url`` points to the clip with ``start_ts > clip.start_ts`` (← Newer) within the filtered
    set. ``next_url`` points to the clip with ``start_ts < clip.start_ts`` (Older →). For
    ``reviewed=yes``, ordering follows ``reviewed_at`` instead of ``start_ts``.

    When no neighboring clip satisfies the filter, the URL falls back to ``/clips?{filter_qs}`` so
    the keyboard handler's click returns the operator to the queue index.
    """
    base = _clip_query(f)
    fallback = f"/clips?{filter_qs}"

    if f.reviewed == "yes" and clip.reviewed_at is not None:
        prev_stmt = base.where(Clip.reviewed_at > clip.reviewed_at).order_by(Clip.reviewed_at.asc(), Clip.id.asc()).limit(1)
        next_stmt = base.where(Clip.reviewed_at < clip.reviewed_at).order_by(desc(Clip.reviewed_at), Clip.id.asc()).limit(1)
    else:
        prev_stmt = base.where(Clip.start_ts > clip.start_ts).order_by(Clip.start_ts.asc(), Clip.id.asc()).limit(1)
        next_stmt = base.where(Clip.start_ts < clip.start_ts).order_by(desc(Clip.start_ts), Clip.id.asc()).limit(1)

    with get_session(engine) as session:
        prev_id = session.scalar(prev_stmt)
        next_id = session.scalar(next_stmt)

    prev_url = f"/clips/{prev_id}?{filter_qs}" if prev_id is not None else fallback
    next_url = f"/clips/{next_id}?{filter_qs}" if next_id is not None else fallback
    return prev_url, next_url


def _parse_detail_filter(request: Request) -> _ClipsFilter | None:
    """Extract filter params from the detail-page querystring.

    Returns ``None`` when no recognized filter key is present, triggering legacy nav behavior.
    ``reviewed`` defaults to ``"no"`` (matching the ``/clips`` default) when absent but another
    filter key is present — matching how ``/clips`` behaves.
    """
    params = request.query_params
    known_keys = {"reviewed", "camera", "has_cat", "date_str"}
    if not known_keys.intersection(params.keys()):
        return None
    reviewed_raw = params.get("reviewed", "no")
    valid_reviewed: tuple[Literal["any", "no", "yes"], ...] = ("any", "no", "yes")
    reviewed: Literal["any", "no", "yes"] = reviewed_raw if reviewed_raw in valid_reviewed else "no"
    has_cat_raw = params.get("has_cat")
    has_cat: bool | None = None
    if has_cat_raw == "true":
        has_cat = True
    elif has_cat_raw == "false":
        has_cat = False
    return _ClipsFilter(
        reviewed=reviewed,
        camera=params.get("camera"),
        has_cat=has_cat,
        date_str=params.get("date_str"),
    )


def _resolve_nav_urls(engine: Engine, clip: Clip, request: Request) -> tuple[str, str]:
    """Return ``(prev_url, next_url)`` for the detail-page nav, scoped to the request filter if present.

    Falls back to empty strings when no filter querystring is present — the template then renders
    the legacy ``url_for``-based links using ``prev_clip_id`` / ``next_clip_id`` from the data
    object.
    """
    clip_filter = _parse_detail_filter(request)
    if clip_filter is None:
        return "", ""
    filter_qs = _build_filter_qs(clip_filter)
    return _build_filtered_nav_urls(engine, clip, clip_filter, filter_qs)


def _button_title(subj: Subject) -> str:
    """Return the tooltip string: display_name, with description in parens if set."""
    if subj.description:
        return f"{subj.display_name} ({subj.description})"
    return subj.display_name


def _build_subject_button(
    subj: Subject,
    frame_id: int,
    clip_id: int,
    tagged: set[int],
) -> dict[str, object]:
    """Build the precomputed button dict for one subject on one frame."""
    is_pressed = subj.id in tagged
    membership_url = f"/clips/{clip_id}/frames/{frame_id}/subjects/{subj.id}"
    return {
        "glyph": subj.display_name[0].upper(),
        "title": _button_title(subj),
        "slug": subj.slug,
        "subject_id": subj.id,
        "frame_id": frame_id,
        "is_pressed": is_pressed,
        "button_class": "tag-btn tag-btn-on" if is_pressed else "tag-btn tag-btn-off",
        "hx_put": None if is_pressed else membership_url,
        "hx_delete": membership_url if is_pressed else None,
        "color": subj.color,
    }


def _build_frame_tag_rows(
    clip_id: int,
    frames: Sequence[Mapping[str, object]],
    subjects_by_kind: dict[str, list[Subject]],
    frame_memberships: dict[int, set[int]],
) -> list[dict[str, object]]:
    """Precompute the per-frame tag-button structures so the template stays free of conditionals.

    Returns one entry per frame. Each entry has ``frame`` (the frame dict) and ``button_groups``: a
    list of ``{"kind": str, "buttons": [...]}`` for each non-empty subject kind. Button dicts carry
    precomputed CSS class strings and HTMX method/URL — djlint must not reflow class conditionals,
    so the template receives final strings, not logic.
    """
    rows: list[dict[str, object]] = []
    for frame in frames:
        frame_id = int(str(frame["id"]))
        tagged = frame_memberships.get(frame_id, set())
        button_groups: list[dict[str, object]] = [
            {"kind": kind, "buttons": [_build_subject_button(s, frame_id, clip_id, tagged) for s in subjects]}
            for kind, subjects in subjects_by_kind.items()
            if subjects
        ]
        rows.append({"frame": frame, "button_groups": button_groups})
    return rows


def _extract_tagged_slugs(label_row: ClipLabelSummary | None) -> set[str]:
    """Parse ``tagged_subject_slugs`` from a ``clip_label_summary`` row into a slug set."""
    if label_row is None:
        return set()
    raw_slugs = label_row.tagged_subject_slugs
    return set(raw_slugs.split(",")) if raw_slugs else set()


@clips_router.get("/clips")
async def list_clips(
    request: Request,
    *,
    reviewed: Literal["any", "no", "yes"] = "no",
    camera: str | None = None,
    has_cat: bool | None = None,
    date_str: str | None = None,
) -> object:
    """Render the clip-listing page.

    ``reviewed=no`` (default) shows unreviewed clips oldest-first — the operator review queue.
    ``reviewed=yes`` shows reviewed clips newest-reviewed-first. ``reviewed=any`` preserves
    ``start_ts DESC``. Capped at :data:`_CLIPS_LIST_LIMIT`; ``date_str`` is interpreted as a UTC day.
    """
    state = get_state(request)
    display_tz = ZoneInfo(state.config.web.display_timezone)
    clip_filter = _ClipsFilter(reviewed=reviewed, camera=camera, has_cat=has_cat, date_str=date_str)
    clip_rows, cameras, total_count, reviewed_count = _query_clips_list(state.engine, clip_filter, display_tz=display_tz)
    return state.templates.TemplateResponse(
        request,
        "clips.html.jinja",
        {
            "clip_rows": clip_rows,
            "cameras": cameras,
            "filters": {"camera": camera, "has_cat": has_cat, "date": date_str, "reviewed": reviewed},
            "filter_qs": _build_filter_qs(clip_filter),
            "progress": {"reviewed": reviewed_count, "total": total_count},
            "tz": state.config.web.display_timezone,
        },
    )


def _project_one_clip_row(
    clip: Clip,
    summary: ClipLabelSummary,
    cameras: list[Camera],
    *,
    display_tz: ZoneInfo,
) -> dict[str, object]:
    """Project a Clip + its label summary into a flat dict the template can render."""
    return {
        **_clip_summary(
            clip,
            cameras,
            display_tz=display_tz,
            effective_has_cat=summary.effective_has_cat,
            show_manual_badge=summary.has_manual_cat and clip.reviewed_at is not None,
        ),
        **_reviewed_at_fields(clip.reviewed_at),
    }


def _query_clips_list(
    engine: Engine,
    f: _ClipsFilter,
    *,
    display_tz: ZoneInfo,
) -> tuple[list[dict[str, object]], list[Camera], int, int]:
    """Run the clips-list + progress COUNT queries and return ``(clip_rows, cameras, total, reviewed)``."""
    base_stmt = select(Clip).join(Camera)
    count_stmt = select(func.count()).select_from(Clip).join(Camera)  # pylint: disable=not-callable  # sqlalchemy func.count() is a generative construct, not the builtin; pylint false positive
    if f.camera:
        base_stmt = base_stmt.where(Camera.name == f.camera)
        count_stmt = count_stmt.where(Camera.name == f.camera)
    if f.has_cat is not None:
        base_stmt = base_stmt.where(Clip.has_cat.is_(f.has_cat))
        count_stmt = count_stmt.where(Clip.has_cat.is_(f.has_cat))
    if f.date_str:
        _d = date.fromisoformat(f.date_str)
        day_start = datetime(_d.year, _d.month, _d.day, tzinfo=UTC)
        base_stmt = base_stmt.where(Clip.start_ts >= day_start).where(Clip.start_ts < day_start + timedelta(days=1))
        count_stmt = count_stmt.where(Clip.start_ts >= day_start).where(Clip.start_ts < day_start + timedelta(days=1))
    if f.reviewed == "no":
        stmt = base_stmt.where(Clip.reviewed_at.is_(None)).order_by(Clip.start_ts.asc(), Clip.id.asc()).limit(_CLIPS_LIST_LIMIT)
    elif f.reviewed == "yes":
        stmt = base_stmt.where(Clip.reviewed_at.is_not(None)).order_by(desc(Clip.reviewed_at), Clip.id.asc()).limit(_CLIPS_LIST_LIMIT)
    else:
        stmt = base_stmt.order_by(desc(Clip.start_ts)).limit(_CLIPS_LIST_LIMIT)
    with get_session(engine) as session:
        clips = list(session.scalars(stmt))
        summary_by_clip = {
            summary.clip_id: summary
            for summary in session.scalars(
                select(ClipLabelSummary).where(ClipLabelSummary.clip_id.in_([clip.id for clip in clips])),
            )
        }
        cameras = list(session.scalars(select(Camera).order_by(Camera.name)))
        total_count = session.scalar(count_stmt) or 0
        reviewed_count = session.scalar(count_stmt.where(Clip.reviewed_at.is_not(None))) or 0
        clip_rows = [_project_one_clip_row(clip, summary_by_clip[clip.id], cameras, display_tz=display_tz) for clip in clips]
    return clip_rows, cameras, total_count, reviewed_count


@dataclass(frozen=True, slots=True)
class _ClipDetailData:
    """All ORM data needed for the clip detail page, detached from the session."""

    clip: Clip
    camera: Camera | None
    frames: list[dict[str, object]]
    subjects_by_kind: dict[str, list[Subject]]
    frame_memberships: dict[int, set[int]]
    prev_clip_id: int | None
    next_clip_id: int | None


def _query_clip_detail(engine: Engine, clip_id: int, confidence_threshold: float) -> _ClipDetailData:
    """Load and detach all ORM objects needed for the clip detail page.

    Projecting ``clip.frames`` into dicts inside the session avoids lazy-load calls after expunge.
    The membership bulk-load uses ``IN (frame_ids)`` to cap it at one round-trip per page view.
    """
    with get_session(engine) as session:
        clip = session.get(Clip, clip_id)
        if clip is None:
            raise HTTPException(status_code=404, detail="clip not found")
        camera = session.get(Camera, clip.camera_id)
        frames: list[dict[str, object]] = [
            {
                "id": f.id,
                "ordinal": f.ordinal,
                "t_offset_seconds": f.t_offset_seconds,
                "display_offset": f"{int(f.t_offset_seconds // 60):d}:{int(f.t_offset_seconds % 60):02d}",
                "score": f.score,
                "display_score": f"{f.score:.2f}",
                "below_threshold": f.score < confidence_threshold,
            }
            for f in clip.frames
        ]
        # Active subjects only (``archived_at IS NULL``), ordered for display.
        subjects_by_kind: dict[str, list[Subject]] = {"cat": [], "event": []}
        for subj in session.scalars(select(Subject).where(Subject.archived_at.is_(None)).order_by(Subject.kind, Subject.display_order)):
            if subj.kind in subjects_by_kind:
                subjects_by_kind[subj.kind].append(subj)
        # Single bulk query: frame_id → set of subject_ids. Includes archived-subject memberships
        # (no filter on subject here) so review UI can show historically tagged subjects.
        frame_memberships: dict[int, set[int]] = {}
        if frames:
            for row in session.execute(
                select(ClipFrameSubject.clip_frame_id, ClipFrameSubject.subject_id).where(
                    ClipFrameSubject.clip_frame_id.in_([cast("int", fr["id"]) for fr in frames]),
                ),
            ).all():
                fid, sid = cast("tuple[int, int]", tuple(row))
                frame_memberships.setdefault(fid, set()).add(sid)
        # Global next/prev neighbors by ``start_ts`` — listing-order independent.
        prev_id = session.scalar(select(Clip.id).where(Clip.start_ts > clip.start_ts).order_by(Clip.start_ts.asc()).limit(1))
        next_id = session.scalar(select(Clip.id).where(Clip.start_ts < clip.start_ts).order_by(desc(Clip.start_ts)).limit(1))
        # Detach the rows from the session so the template can read attributes after exit.
        session.expunge(clip)
        if camera is not None:
            session.expunge(camera)
        for subj in [s for kind in subjects_by_kind.values() for s in kind]:
            session.expunge(subj)
    return _ClipDetailData(
        clip=clip,
        camera=camera,
        frames=frames,
        subjects_by_kind=subjects_by_kind,
        frame_memberships=frame_memberships,
        prev_clip_id=prev_id,
        next_clip_id=next_id,
    )


@clips_router.get("/clips/{clip_id}")
async def clip_detail(request: Request, clip_id: int) -> object:
    """Render the detail page for ``clip_id``: video player + detection metadata + label form."""
    state = get_state(request)
    display_tz = ZoneInfo(state.config.web.display_timezone)
    data = _query_clip_detail(state.engine, clip_id, state.config.detector.confidence_threshold)

    prev_url, next_url = _resolve_nav_urls(state.engine, data.clip, request)

    with get_session(state.engine) as label_session:
        label_row = label_session.scalars(
            select(ClipLabelSummary).where(ClipLabelSummary.clip_id == clip_id),
        ).one_or_none()
        if label_row is not None:
            label_session.expunge(label_row)

    cat_frame_counts = query_cat_frame_counts(state.engine, clip_id)
    tagged_slugs = _extract_tagged_slugs(label_row)

    # The Amcrest video has the camera-local clock burned into the OSD; rendering the heading in
    # display_timezone keeps the page label aligned with what the user sees in the player.
    display_start = data.clip.start_ts.astimezone(display_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    frame_tag_rows = _build_frame_tag_rows(clip_id, data.frames, data.subjects_by_kind, data.frame_memberships)
    tag_summary = build_tag_summary(cat_frame_counts, tagged_slugs, data.subjects_by_kind["event"])
    has_manual_cat = label_row.has_manual_cat if label_row is not None else False

    return state.templates.TemplateResponse(
        request,
        "clip_detail.html.jinja",
        {
            "clip": data.clip,
            "camera": data.camera,
            "frames": data.frames,
            "display_start": display_start,
            "prev_clip_id": data.prev_clip_id,
            "next_clip_id": data.next_clip_id,
            "prev_url": prev_url,
            "next_url": next_url,
            "tz": state.config.web.display_timezone,
            "subjects_by_kind": data.subjects_by_kind,
            "frame_memberships": data.frame_memberships,
            "label_summary": label_row,
            "frame_tag_rows": frame_tag_rows,
            "tag_summary": tag_summary,
            "has_manual_cat": has_manual_cat,
            **_build_review_context(clip_id, data.clip.reviewed_at),
        },
    )


def _clip_summary(
    clip: Clip,
    cameras: list[Camera],
    *,
    display_tz: ZoneInfo,
    effective_has_cat: bool,
    show_manual_badge: bool,
) -> dict[str, object]:
    """Project a Clip + its Camera into a flat dict the template can render without lazy loads.

    ``display_start`` is precomputed in ``display_tz`` so the visible cell aligns with the
    camera-OSD time burned into the video; the raw ``start_ts`` (UTC) is also passed for the HTML5
    ``<time datetime="…">`` attribute. ``badge_class`` and ``manual_badge`` are precomputed here
    (not in the template) so djlint reflow cannot break test substring assertions on the class
    attribute.
    """
    by_id = {cam.id: cam for cam in cameras}
    cam = by_id.get(clip.camera_id)
    badge_class = "badge-cat" if effective_has_cat else "badge-no-cat"
    if show_manual_badge:
        badge_class = f"{badge_class} badge-manual"
    return {
        "id": clip.id,
        "camera_display_name": cam.display_name if cam is not None else "",
        "camera_name": cam.name if cam is not None else "",
        "source_filename": clip.source_filename,
        "start_ts": clip.start_ts,
        "display_start": clip.start_ts.astimezone(display_tz).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "duration_seconds": clip.duration_seconds,
        "max_score": clip.max_score,
        "effective_has_cat": effective_has_cat,
        "badge_class": badge_class,
        "manual_badge": show_manual_badge,
    }


def _build_review_context(clip_id: int, reviewed_at: datetime | None) -> dict[str, str]:
    """Precompute the review-button and reviewed_at display fields for the clip detail template.

    Returns ``reviewed_url``, ``review_label``, ``review_hx_method``, ``reviewed_at_iso``,
    and ``reviewed_at_short`` as a flat dict. Empty strings for the timestamp fields keep Jinja2
    ``{% if %}`` checks at the display level rather than inside attribute strings (djlint reflows
    multi-line attribute expressions and breaks test substring assertions).
    """
    url = f"/clips/{clip_id}/reviewed"
    if reviewed_at is not None:
        return {
            "reviewed_url": url,
            "review_label": "Re-open for review",
            "review_hx_method": "delete",
            "reviewed_at_iso": reviewed_at.isoformat(),
            "reviewed_at_short": reviewed_at.strftime("%Y-%m-%d"),
        }
    return {
        "reviewed_url": url,
        "review_label": "Mark reviewed",
        "review_hx_method": "post",
        "reviewed_at_iso": "",
        "reviewed_at_short": "",
    }


def _reviewed_at_fields(reviewed_at: datetime | None) -> dict[str, str]:
    """Precompute ``reviewed_at_short`` / ``reviewed_at_iso`` for the clips-list template.

    Empty strings keep the template ``{% if %}`` at the display level (not inside attribute strings,
    which djlint reflows).
    """
    if reviewed_at is not None:
        return {"reviewed_at_short": reviewed_at.strftime("%Y-%m-%d"), "reviewed_at_iso": reviewed_at.isoformat()}
    return {"reviewed_at_short": "", "reviewed_at_iso": ""}

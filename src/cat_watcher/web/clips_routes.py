"""HTTP routes for the clips listing and detail pages.

Routes read state via the SQLAlchemy engine and Jinja2 templates attached to ``app.state``.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import desc, func, select

from cat_watcher.db import Camera, Clip, ClipFrameSubject, ClipLabelSummary, Subject, get_session
from cat_watcher.labels import build_tag_summary, is_manual_override, query_cat_frame_counts
from cat_watcher.web._app_state import get_state
from cat_watcher.web.clip_filters import (
    ClipsFilter,
    ParsedClipsFilter,
    apply_clip_filters,
    build_filter_qs,
    build_ignored_notice,
    parse_clips_filter,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime, tzinfo

    from sqlalchemy.engine import Engine
    from sqlalchemy.sql import Select


_CLIPS_LIST_LIMIT = 200

clips_router = APIRouter()


def _camera_names(engine: Engine) -> set[str]:
    """Return the vocabulary a ``camera`` filter value is validated against.

    The ``cameras`` table, not ``config.cameras``: the filter form's Camera select is built from
    these rows, so validating against config would let the page render an option that reports
    itself ignored when chosen.
    """
    with get_session(engine) as session:
        return set(session.scalars(select(Camera.name)))


def _build_filtered_nav_urls(engine: Engine, clip: Clip, f: ClipsFilter, filter_qs: str, *, display_tz: tzinfo) -> tuple[str, str]:
    """Return ``(prev_url, next_url)`` scoped to filter ``f``.

    ``prev_url`` points to the clip with ``start_ts > clip.start_ts`` (← Newer) within the filtered
    set. ``next_url`` points to the clip with ``start_ts < clip.start_ts`` (Older →). For
    ``reviewed=yes``, ordering follows ``reviewed_at`` instead of ``start_ts``.

    When no neighboring clip satisfies the filter, the URL falls back to ``/clips?{filter_qs}`` so
    the keyboard handler's click returns the operator to the queue index.
    """
    base: Select[tuple[int]] = apply_clip_filters(select(Clip.id), f, display_tz=display_tz)
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


def _resolve_nav_urls(engine: Engine, clip: Clip, parsed: ParsedClipsFilter, *, display_tz: tzinfo) -> tuple[str, str]:
    """Return ``(prev_url, next_url)`` for the detail-page nav, scoped to the request filter if present.

    Falls back to empty strings when the querystring carries no recognized filter key — the template
    then renders the legacy ``url_for``-based links using ``prev_clip_id`` / ``next_clip_id`` from
    the data object.
    """
    if not parsed.any_key_present:
        return "", ""
    filter_qs = build_filter_qs(parsed.clips_filter)
    return _build_filtered_nav_urls(engine, clip, parsed.clips_filter, filter_qs, display_tz=display_tz)


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
        "label": subj.display_name,
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


def _compute_label_summary(engine: Engine, clip_id: int) -> tuple[str, bool]:
    """Compute ``(tag_summary, has_manual_cat)`` for a clip from its current frame memberships.

    Shared by the detail-page render and the ``/clips/{id}/label-summary`` endpoint so the value
    shown live after a frame-tag toggle matches a full page reload exactly.
    """
    cat_frame_counts = query_cat_frame_counts(engine, clip_id)
    with get_session(engine) as session:
        label_row = session.scalars(select(ClipLabelSummary).where(ClipLabelSummary.clip_id == clip_id)).one_or_none()
        tagged_slugs = _extract_tagged_slugs(label_row)
        has_manual_cat = label_row.has_manual_cat if label_row is not None else False
        event_subjects = list(
            session.scalars(
                select(Subject)  # fmt: wrap
                .where(Subject.kind == "event", Subject.archived_at.is_(None))
                .order_by(Subject.display_order),
            ),
        )
        tag_summary = build_tag_summary(cat_frame_counts, tagged_slugs, event_subjects)
    return tag_summary, has_manual_cat


@clips_router.get("/clips")
async def list_clips(request: Request) -> object:
    """Render the clip-listing page.

    ``reviewed=no`` (default) shows unreviewed clips oldest-first — the operator review queue.
    ``reviewed=yes`` shows reviewed clips newest-reviewed-first. ``reviewed=any`` preserves
    ``start_ts DESC``. Capped at :data:`_CLIPS_LIST_LIMIT`; ``date_str`` selects a calendar day in
    ``display_timezone``.

    Filters are read from ``request.query_params`` rather than declared as typed route parameters:
    a ``Literal`` parameter cannot accept the empty string the filter form submits for "unset", and
    a parsed one cannot report a bad value instead of raising.
    """
    state = get_state(request)
    display_tz = ZoneInfo(state.config.web.display_timezone)
    parsed = parse_clips_filter(request.query_params, camera_names=_camera_names(state.engine))
    clip_rows, cameras, total_count, reviewed_count = _query_clips_list(state.engine, parsed.clips_filter, display_tz=display_tz)
    return state.templates.TemplateResponse(
        request,
        "clips.html.jinja",
        {
            "clip_rows": clip_rows,
            "cameras": cameras,
            "filters": parsed.clips_filter,
            "filter_qs": build_filter_qs(parsed.clips_filter),
            "filter_notice": build_ignored_notice(parsed.ignored),
            "progress": {"reviewed": reviewed_count, "total": total_count},
        },
    )


def _project_one_clip_row(clip: Clip, summary: ClipLabelSummary, cameras: list[Camera]) -> dict[str, object]:
    """Project a Clip + its label summary into a flat dict the template can render."""
    return _clip_summary(
        clip,
        cameras,
        effective_has_cat=summary.effective_has_cat,
        show_manual_badge=is_manual_override(
            has_cat=clip.has_cat,
            has_manual_cat=summary.has_manual_cat,
            reviewed=clip.reviewed_at is not None,
        ),
    )


def _query_clips_list(
    engine: Engine,
    f: ClipsFilter,
    *,
    display_tz: tzinfo,
) -> tuple[list[dict[str, object]], list[Camera], int, int]:
    """Run the clips-list + progress COUNT queries and return ``(clip_rows, cameras, total, reviewed)``.

    The progress indicator deliberately counts across every review state — it answers "how far
    through this camera/day/cat slice am I", which a ``reviewed``-scoped count cannot — so the COUNT
    is built from a filter with ``reviewed`` neutralized and derives its reviewed half from the same
    statement.
    """
    base_stmt = apply_clip_filters(select(Clip), f, display_tz=display_tz)
    count_stmt = apply_clip_filters(
        select(func.count()).select_from(Clip),  # pylint: disable=not-callable  # sqlalchemy func.count() is a generative construct, not the builtin; pylint false positive
        replace(f, reviewed="any"),
        display_tz=display_tz,
    )
    if f.reviewed == "no":
        stmt = base_stmt.order_by(Clip.start_ts.asc(), Clip.id.asc()).limit(_CLIPS_LIST_LIMIT)
    elif f.reviewed == "yes":
        stmt = base_stmt.order_by(desc(Clip.reviewed_at), Clip.id.asc()).limit(_CLIPS_LIST_LIMIT)
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
        clip_rows = [_project_one_clip_row(clip, summary_by_clip[clip.id], cameras) for clip in clips]
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

    parsed = parse_clips_filter(request.query_params, camera_names=_camera_names(state.engine))
    prev_url, next_url = _resolve_nav_urls(state.engine, data.clip, parsed, display_tz=display_tz)
    # Carrying the filter back keeps the operator's queue alive across a round trip through a clip;
    # without it "All clips" silently resets to the default queue.
    filter_qs = build_filter_qs(parsed.clips_filter) if parsed.any_key_present else ""
    all_clips_url = f"/clips?{filter_qs}" if filter_qs else "/clips"

    tag_summary, has_manual_cat = _compute_label_summary(state.engine, clip_id)

    frame_tag_rows = _build_frame_tag_rows(clip_id, data.frames, data.subjects_by_kind, data.frame_memberships)

    return state.templates.TemplateResponse(
        request,
        "clip_detail.html.jinja",
        {
            "clip": data.clip,
            "camera": data.camera,
            "frames": data.frames,
            "prev_clip_id": data.prev_clip_id,
            "next_clip_id": data.next_clip_id,
            "prev_url": prev_url,
            "next_url": next_url,
            "all_clips_url": all_clips_url,
            "filter_notice": build_ignored_notice(parsed.ignored),
            "subjects_by_kind": data.subjects_by_kind,
            "frame_memberships": data.frame_memberships,
            "frame_tag_rows": frame_tag_rows,
            "tag_summary": tag_summary,
            "has_manual_cat": has_manual_cat,
            **_build_review_context(clip_id, data.clip.reviewed_at),
        },
    )


@clips_router.get("/clips/{clip_id}/label-summary")
async def clip_label_summary(request: Request, clip_id: int) -> dict[str, object]:
    """Return the recomputed ``tag_summary`` / ``has_manual_cat`` as JSON for live update after a tag toggle."""
    state = get_state(request)
    tag_summary, has_manual_cat = _compute_label_summary(state.engine, clip_id)
    return {"tag_summary": tag_summary, "has_manual_cat": has_manual_cat}


def _clip_summary(
    clip: Clip,
    cameras: list[Camera],
    *,
    effective_has_cat: bool,
    show_manual_badge: bool,
) -> dict[str, object]:
    """Project a Clip + its Camera into a flat dict the template can render without lazy loads.

    Datetimes are passed raw; the template renders them through the ``localstamp`` / ``localdate``
    filters and keeps the UTC ISO value for the ``<time datetime="…">`` attribute. ``badge_class``
    and ``manual_badge`` are precomputed here (not in the template) so djlint reflow cannot break
    test substring assertions on the class attribute.
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
        "reviewed_at": clip.reviewed_at,
        "duration_seconds": clip.duration_seconds,
        "max_score": clip.max_score,
        "effective_has_cat": effective_has_cat,
        "badge_class": badge_class,
        "manual_badge": show_manual_badge,
    }


def _build_review_context(clip_id: int, reviewed_at: datetime | None) -> dict[str, str]:
    """Precompute the review-button fields for the clip detail template.

    Returns ``reviewed_url``, ``review_label``, and ``review_hx_method``. The timestamp itself is
    read off ``clip.reviewed_at`` in the template, so this stays a flat string mapping.
    """
    url = f"/clips/{clip_id}/reviewed"
    if reviewed_at is not None:
        return {"reviewed_url": url, "review_label": "Re-open for review", "review_hx_method": "delete"}
    return {"reviewed_url": url, "review_label": "Mark reviewed", "review_hx_method": "post"}

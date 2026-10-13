"""HTTP routes for the clips listing and detail pages.

Routes read state via the SQLAlchemy engine and Jinja2 templates attached to ``app.state``.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple, cast
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import func, select, tuple_

from cat_watcher.config import active_unsure_threshold
from cat_watcher.db import Camera, Clip, ClipFrameSubject, ClipLabelSummary, Subject, get_session
from cat_watcher.identity import ClipIdentity, FrameIdentity, NamedCat, query_cat_subjects, query_clip_detail, query_clip_identities
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
from cat_watcher.web.pagination import PageInfo, page_info

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import datetime, tzinfo

    from sqlalchemy import ColumnElement, UnaryExpression
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session
    from sqlalchemy.sql import Select

    from cat_watcher.web.clip_filters import ClipsSort


_CLIPS_PAGE_SIZE = 100

#: The detail-page nav labels for each sort, as ``(prev, next)``. ``prev`` is the row above in the list.
_NAV_LABELS: dict[ClipsSort, tuple[str, str]] = {"newest": ("← Newer", "Older →"), "oldest": ("← Older", "Newer →")}

clips_router = APIRouter()


def _camera_names(engine: Engine) -> set[str]:
    """Return the vocabulary a ``camera`` filter value is validated against.

    The ``cameras`` table, not ``config.cameras``: the filter form's Camera select is built from
    these rows, so validating against config would let the page render an option that reports
    itself ignored when chosen.
    """
    with get_session(engine) as session:
        return set(session.scalars(select(Camera.name)))


def _list_order(sort: ClipsSort) -> tuple[UnaryExpression[datetime], UnaryExpression[int]]:
    """Return the ``ORDER BY`` of the list for ``sort``. ``id`` breaks a tie on ``start_ts``."""
    if sort == "oldest":
        return Clip.start_ts.asc(), Clip.id.asc()
    return Clip.start_ts.desc(), Clip.id.desc()


def _after(clip: Clip) -> ColumnElement[bool]:
    """Match the clips after ``clip`` in the oldest-first order. A clip with the same ``start_ts`` and a higher ``id`` matches."""
    return tuple_(Clip.start_ts, Clip.id) > tuple_(clip.start_ts, clip.id)


def _before(clip: Clip) -> ColumnElement[bool]:
    """Match the clips before ``clip`` in the oldest-first order. A clip with the same ``start_ts`` and a lower ``id`` matches."""
    return tuple_(Clip.start_ts, Clip.id) < tuple_(clip.start_ts, clip.id)


class _NavUrls(NamedTuple):
    """The prev, next, and back links of the detail page. An empty string selects the legacy links."""

    prev_url: str
    next_url: str
    back_url: str


def _build_filtered_nav_urls(engine: Engine, clip: Clip, f: ClipsFilter, *, display_tz: tzinfo) -> _NavUrls:
    """Return the nav links scoped to filter ``f``, in the list order of ``f.sort``.

    ``prev_url`` points to the row above ``clip`` in the list and ``next_url`` to the row below. So
    under ``newest`` prev is the newer neighbor, and under ``oldest`` it is the older neighbor. The
    neighbor links carry no ``page``, because a neighbor can be on a different page.

    ``back_url`` opens the list page that holds ``clip``, from :func:`_back_url`. When no neighbor
    satisfies the filter, the prev or next URL is ``back_url``, so the keyboard handler's click
    returns the operator to the queue index.
    """
    base: Select[int] = apply_clip_filters(select(Clip.id), f, display_tz=display_tz)
    nav_qs = build_filter_qs(replace(f, page=None))

    newer_stmt = base.where(_after(clip)).order_by(*_list_order("oldest")).limit(1)
    older_stmt = base.where(_before(clip)).order_by(*_list_order("newest")).limit(1)
    prev_stmt, next_stmt = (newer_stmt, older_stmt) if f.sort == "newest" else (older_stmt, newer_stmt)

    with get_session(engine) as session:
        prev_id = session.scalar(prev_stmt)
        next_id = session.scalar(next_stmt)
        back_url = _back_url(session, clip, f, display_tz=display_tz)

    return _NavUrls(
        prev_url=f"/clips/{prev_id}?{nav_qs}" if prev_id is not None else back_url,
        next_url=f"/clips/{next_id}?{nav_qs}" if next_id is not None else back_url,
        back_url=back_url,
    )


def _back_url(session: Session, clip: Clip, f: ClipsFilter, *, display_tz: tzinfo) -> str:
    """Return the URL of the list page that holds ``clip`` under filter ``f``.

    One COUNT of the filtered rows above ``clip`` in the list order gives the page. If ``clip`` does
    not match ``f``, the count still gives its place in the list order. That page holds its
    neighbors. The list route clamps a page past the end, so this needs no total.
    """
    above_clip = _after(clip) if f.sort == "newest" else _before(clip)
    rows_above = session.scalar(_count_stmt(f, display_tz=display_tz).where(above_clip)) or 0
    return _page_url(f, rows_above // _CLIPS_PAGE_SIZE + 1)


def _resolve_nav_urls(engine: Engine, clip: Clip, parsed: ParsedClipsFilter, *, display_tz: tzinfo) -> _NavUrls:
    """Return the detail-page nav links of the request filter.

    If the querystring has no recognized filter key, the prev and next links are empty and the back
    link is ``/clips``. The template then renders the legacy ``url_for``-based links from
    ``prev_clip_id`` / ``next_clip_id`` of the data object.
    """
    if not parsed.any_key_present:
        return _NavUrls(prev_url="", next_url="", back_url="/clips")
    return _build_filtered_nav_urls(engine, clip, parsed.clips_filter, display_tz=display_tz)


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


_NOT_CLASSIFIED = FrameIdentity(kind="not_classified", slugs=(), confidence=None)


@dataclass(frozen=True)
class _DetailIdentity:
    """The identity data of the detail page: model result per frame, cats by slug, clip fields."""

    model_results: Mapping[int, FrameIdentity]
    cats_by_slug: Mapping[str, Subject]
    fields: Mapping[str, object]


def _model_label(result: FrameIdentity, cats_by_slug: Mapping[str, Subject]) -> dict[str, str | None]:
    """Build the label, CSS class, and color of the stored model result of one frame.

    The label is plain text beside the tag buttons. It never uses a ``tag-btn`` class, because the
    keyboard handler and the tag toggle script act on those classes.
    """
    if result.kind != "model_cat":
        label = _FALLBACK_LABELS.get(result.kind, _FALLBACK_LABELS["not_classified"])
        if result.kind == "unsure" and result.confidence is not None:
            label = f"{label} ({result.confidence:.2f})"
        return {
            "model_label": label,
            "model_css_class": "prediction prediction-fallback",
            "model_color": None,
        }
    slug = result.slugs[0]
    subject = cats_by_slug.get(slug)
    name = subject.display_name if subject is not None else slug
    confidence = f" {result.confidence:.2f}" if result.confidence is not None else ""
    return {
        "model_label": f"{name}{confidence}",
        "model_css_class": "prediction prediction-model",
        "model_color": subject.color if subject is not None else None,
    }


def _build_frame_tag_rows(
    clip_id: int,
    frames: Sequence[Mapping[str, object]],
    subjects_by_kind: dict[str, list[Subject]],
    frame_memberships: dict[int, set[int]],
    identity: _DetailIdentity,
) -> list[dict[str, object]]:
    """Precompute the per-frame tag-button structures so the template stays free of conditionals.

    Returns one entry per frame. Each entry has ``frame`` (the frame dict), ``button_groups``: a
    list of ``{"kind": str, "buttons": [...]}`` for each non-empty subject kind, and the
    ``model_label`` fields of the stored model result. Button dicts carry
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
        model = _model_label(identity.model_results.get(frame_id, _NOT_CLASSIFIED), identity.cats_by_slug)
        rows.append({"frame": frame, "button_groups": button_groups, **model})
    return rows


def _extract_tagged_slugs(label_row: ClipLabelSummary | None) -> set[str]:
    """Parse ``tagged_subject_slugs`` from a ``clip_label_summary`` row into a slug set."""
    if label_row is None:
        return set()
    raw_slugs = label_row.tagged_subject_slugs
    return set(raw_slugs.split(",")) if raw_slugs else set()


class _LabelSummary(NamedTuple):
    """The tag summary text of a clip and the two cat verdicts of its ``clip_label_summary`` row."""

    tag_summary: str
    has_manual_cat: bool
    effective_has_cat: bool


def _compute_label_summary(engine: Engine, clip_id: int) -> _LabelSummary:
    """Compute the label summary of a clip from its current frame memberships.

    Shared by the detail-page render and the ``/clips/{id}/label-summary`` endpoint so the value
    shown live after a frame-tag toggle matches a full page reload exactly.
    """
    cat_frame_counts = query_cat_frame_counts(engine, clip_id)
    with get_session(engine) as session:
        label_row = session.scalars(select(ClipLabelSummary).where(ClipLabelSummary.clip_id == clip_id)).one_or_none()
        tagged_slugs = _extract_tagged_slugs(label_row)
        has_manual_cat = label_row.has_manual_cat if label_row is not None else False
        effective_has_cat = label_row.effective_has_cat if label_row is not None else False
        event_subjects = list(
            session.scalars(
                select(Subject)  # fmt: wrap
                .where(Subject.kind == "event", Subject.archived_at.is_(None))
                .order_by(Subject.display_order),
            ),
        )
        tag_summary = build_tag_summary(cat_frame_counts, tagged_slugs, event_subjects)
    return _LabelSummary(tag_summary, has_manual_cat, effective_has_cat)


@clips_router.get("/clips")
async def list_clips(request: Request) -> object:
    """Render one page of the clip list.

    ``reviewed=no`` (default) shows unreviewed clips, the operator review queue. ``sort`` orders
    every ``reviewed`` value by ``start_ts``, newest first by default. A page holds
    :data:`_CLIPS_PAGE_SIZE` clips. ``date_str`` selects a calendar day in ``display_timezone``.

    Filters are read from ``request.query_params`` rather than declared as typed route parameters:
    a ``Literal`` parameter cannot accept the empty string the filter form submits for "unset", and
    a parsed one cannot report a bad value instead of raising.
    """
    state = get_state(request)
    display_tz = ZoneInfo(state.config.web.display_timezone)
    parsed = parse_clips_filter(request.query_params, camera_names=_camera_names(state.engine))
    unsure_threshold = active_unsure_threshold(state.config)
    clips_page = _query_clips_list(
        state.engine,
        parsed.clips_filter,
        display_tz=display_tz,
        unsure_threshold=unsure_threshold,
        confidence_threshold=state.config.detector.confidence_threshold,
    )
    info = clips_page.page
    return state.templates.TemplateResponse(
        request,
        "clips.html.jinja",
        {
            "clip_rows": clips_page.clip_rows,
            "cameras": clips_page.cameras,
            "filters": parsed.clips_filter,
            # The page shown, not the page asked for, so a row link returns to the rows it came from.
            "filter_qs": build_filter_qs(_on_page(parsed.clips_filter, info.number)),
            "filter_notice": build_ignored_notice(parsed.ignored),
            "progress": {"reviewed": clips_page.progress_reviewed, "total": clips_page.progress_total},
            "page": info,
            "previous_page_url": _page_url(parsed.clips_filter, info.number - 1) if info.has_previous else "",
            "next_page_url": _page_url(parsed.clips_filter, info.number + 1) if info.has_next else "",
        },
    )


def _on_page(f: ClipsFilter, number: int) -> ClipsFilter:
    """Return ``f`` on page ``number``, with the first page as ``None``."""
    return replace(f, page=number if number > 1 else None)


def _page_url(f: ClipsFilter, number: int) -> str:
    """Return the list URL of page ``number`` of filter ``f``."""
    return f"/clips?{build_filter_qs(_on_page(f, number))}"


_FALLBACK_LABELS = {"unsure": "unsure", "not_classified": "not classified", "no_box": "no cat box found"}
_NO_CAT_NAMED_LABEL = "no cat named"
_NOT_CAT_POSITIVE_LABEL = "no cat"


def identity_chip(named: NamedCat, cats_by_slug: Mapping[str, Subject]) -> dict[str, str]:
    """Build the chip dict for one named cat. A model chip and an operator chip differ only in class."""
    subject = cats_by_slug.get(named.slug)
    name = subject.display_name if subject is not None else named.slug
    return {
        "label": name,
        "css_class": f"identity-chip identity-chip-{named.source}",
        "color": subject.color or "" if subject is not None else "",
        "title": f"{name} ({named.source})",
    }


def _identity_fields(
    identity: ClipIdentity | None,
    cats_by_slug: Mapping[str, Subject],
    *,
    effective_has_cat: bool,
    label_non_cat: bool,
) -> dict[str, object]:
    """Build the identity chips and the fallback label of a clip.

    A clip with no result row, or one that names nothing and has no fallback, gets a label too,
    so the cell is never empty. A clip that is not cat-positive gets ``no cat`` when ``label_non_cat``
    is set. The clips list sets it to ``False``, because its badge already says ``no cat``.

    Rule 1 comes before the cat-positive check. A tag does not mark a clip reviewed. So an operator
    cat tag on an unreviewed clip names its cat, also when the detector calls the clip ``no cat``.
    A model cat needs a cat-positive clip.
    """
    if not effective_has_cat:
        operator_cats = [] if identity is None else [named for named in identity.cats if named.source == "operator"]
        if operator_cats:
            return {"identity_chips": [identity_chip(named, cats_by_slug) for named in operator_cats], "identity_fallback_label": None}
        return {"identity_chips": [], "identity_fallback_label": _NOT_CAT_POSITIVE_LABEL if label_non_cat else None}
    if identity is None:
        return {"identity_chips": [], "identity_fallback_label": _FALLBACK_LABELS["not_classified"]}
    chips = [identity_chip(named, cats_by_slug) for named in identity.cats]
    if chips:
        return {"identity_chips": chips, "identity_fallback_label": None}
    label = _FALLBACK_LABELS[identity.fallback] if identity.fallback is not None else _NO_CAT_NAMED_LABEL
    return {"identity_chips": [], "identity_fallback_label": label}


def _identity_fields_by_clip(
    session: Session,
    clips: Sequence[Clip],
    summary_by_clip: Mapping[int, ClipLabelSummary],
    *,
    unsure_threshold: float,
    confidence_threshold: float,
) -> dict[int, dict[str, object]]:
    """Build the identity fields of every clip on the page with a fixed number of queries."""
    cats = query_cat_subjects(session)
    identity_by_clip = query_clip_identities(
        session,
        [clip.id for clip in clips],
        unsure_threshold=unsure_threshold,
        confidence_threshold=confidence_threshold,
        cats=cats,
    )
    cats_by_slug = {subj.slug: subj for subj in cats}
    return {
        clip.id: _identity_fields(
            identity_by_clip.get(clip.id),
            cats_by_slug,
            effective_has_cat=summary_by_clip[clip.id].effective_has_cat,
            label_non_cat=False,
        )
        for clip in clips
    }


def _project_one_clip_row(
    clip: Clip,
    summary: ClipLabelSummary,
    cameras: list[Camera],
    *,
    identity_fields: Mapping[str, object],
) -> dict[str, object]:
    """Project a Clip + its label summary + its identity into a flat dict the template can render."""
    summary_fields = _clip_summary(
        clip,
        cameras,
        effective_has_cat=summary.effective_has_cat,
        show_manual_badge=is_manual_override(
            has_cat=clip.has_cat,
            has_manual_cat=summary.has_manual_cat,
            reviewed=clip.reviewed_at is not None,
        ),
    )
    return {**summary_fields, **identity_fields}


def _count_stmt(f: ClipsFilter, *, display_tz: tzinfo) -> Select[int]:
    """Return the COUNT of the clips that ``f`` selects."""
    return apply_clip_filters(select(func.count()).select_from(Clip), f, display_tz=display_tz)


class _Progress(NamedTuple):
    """The two progress counts of filter ``f`` with ``reviewed="any"``."""

    total: int
    reviewed: int

    def list_total(self, f: ClipsFilter) -> int:
        """Return the row count of the list of ``f``. ``reviewed`` is the only filter the counts leave out."""
        return {"any": self.total, "yes": self.reviewed, "no": self.total - self.reviewed}[f.reviewed]


def _progress_counts(session: Session, f: ClipsFilter, *, display_tz: tzinfo) -> _Progress:
    """Count the clips of ``f`` with ``reviewed="any"``, and the reviewed ones among them."""
    count_stmt = _count_stmt(replace(f, reviewed="any"), display_tz=display_tz)
    total = session.scalar(count_stmt) or 0
    reviewed = session.scalar(count_stmt.where(Clip.reviewed_at.is_not(None))) or 0
    return _Progress(total, reviewed)


class _ClipsListPage(NamedTuple):
    """One page of list rows, the page it is, and the two halves of the progress count."""

    clip_rows: list[dict[str, object]]
    cameras: list[Camera]
    page: PageInfo
    progress_total: int
    progress_reviewed: int


def _query_clips_list(
    engine: Engine,
    f: ClipsFilter,
    *,
    display_tz: tzinfo,
    unsure_threshold: float,
    confidence_threshold: float,
) -> _ClipsListPage:
    """Run the progress COUNT queries and the list query of one page.

    The progress COUNT uses the filter with ``reviewed="any"``. It tells how far the operator is
    through this camera, day, or cat slice. A count scoped to ``reviewed`` cannot tell that. The
    reviewed half of the progress comes from the same statement. The two halves also size the
    list, because ``reviewed`` is the only filter that the progress count leaves out.
    """
    base_stmt = apply_clip_filters(select(Clip), f, display_tz=display_tz)
    with get_session(engine) as session:
        progress = _progress_counts(session, f, display_tz=display_tz)
        info = page_info(requested=f.page, total=progress.list_total(f), page_size=_CLIPS_PAGE_SIZE)
        clips = list(session.scalars(base_stmt.order_by(*_list_order(f.sort)).offset(info.offset).limit(info.limit)))
        summary_by_clip = {
            summary.clip_id: summary
            for summary in session.scalars(
                select(ClipLabelSummary).where(ClipLabelSummary.clip_id.in_([clip.id for clip in clips])),
            )
        }
        cameras = list(session.scalars(select(Camera).order_by(Camera.name)))
        identity_fields = _identity_fields_by_clip(
            session,
            clips,
            summary_by_clip,
            unsure_threshold=unsure_threshold,
            confidence_threshold=confidence_threshold,
        )
        clip_rows = [
            _project_one_clip_row(clip, summary_by_clip[clip.id], cameras, identity_fields=identity_fields[clip.id]) for clip in clips
        ]
    return _ClipsListPage(clip_rows, cameras, info, progress.total, progress.reviewed)


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
        # Global next/prev neighbors in the newest-first order. No filter applies.
        prev_id = session.scalar(select(Clip.id).where(_after(clip)).order_by(*_list_order("oldest")).limit(1))
        next_id = session.scalar(select(Clip.id).where(_before(clip)).order_by(*_list_order("newest")).limit(1))
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


def _query_detail_identity(
    engine: Engine,
    clip_id: int,
    *,
    unsure_threshold: float,
    confidence_threshold: float,
    effective_has_cat: bool,
) -> _DetailIdentity:
    """Load the model results, the cats, and the clip identity fields in one session.

    ``effective_has_cat`` comes from the label summary that the caller already read.
    """
    with get_session(engine) as session:
        cats = query_cat_subjects(session)
        identity, model_results = query_clip_detail(
            session,
            clip_id,
            unsure_threshold=unsure_threshold,
            confidence_threshold=confidence_threshold,
            cats=cats,
        )
        cats_by_slug = {subj.slug: subj for subj in cats}
        fields = _identity_fields(identity, cats_by_slug, effective_has_cat=effective_has_cat, label_non_cat=True)
        session.expunge_all()
    return _DetailIdentity(model_results, cats_by_slug, fields)


@clips_router.get("/clips/{clip_id}")
async def clip_detail(request: Request, clip_id: int) -> object:
    """Render the detail page for ``clip_id``: video player + detection metadata + label form."""
    state = get_state(request)
    display_tz = ZoneInfo(state.config.web.display_timezone)
    data = _query_clip_detail(state.engine, clip_id, state.config.detector.confidence_threshold)

    parsed = parse_clips_filter(request.query_params, camera_names=_camera_names(state.engine))
    nav = _resolve_nav_urls(state.engine, data.clip, parsed, display_tz=display_tz)
    prev_label, next_label = _NAV_LABELS[parsed.clips_filter.sort]

    label_summary = _compute_label_summary(state.engine, clip_id)

    identity = _query_detail_identity(
        state.engine,
        clip_id,
        unsure_threshold=active_unsure_threshold(state.config),
        confidence_threshold=state.config.detector.confidence_threshold,
        effective_has_cat=label_summary.effective_has_cat,
    )
    frame_tag_rows = _build_frame_tag_rows(clip_id, data.frames, data.subjects_by_kind, data.frame_memberships, identity)

    return state.templates.TemplateResponse(
        request,
        "clip_detail.html.jinja",
        {
            "clip": data.clip,
            "camera": data.camera,
            "frames": data.frames,
            "prev_clip_id": data.prev_clip_id,
            "next_clip_id": data.next_clip_id,
            "prev_url": nav.prev_url,
            "next_url": nav.next_url,
            "prev_label": prev_label,
            "next_label": next_label,
            "all_clips_url": nav.back_url,
            "filter_notice": build_ignored_notice(parsed.ignored),
            "subjects_by_kind": data.subjects_by_kind,
            "frame_memberships": data.frame_memberships,
            "frame_tag_rows": frame_tag_rows,
            "tag_summary": label_summary.tag_summary,
            "has_manual_cat": label_summary.has_manual_cat,
            **identity.fields,
            **_build_review_context(clip_id, data.clip.reviewed_at),
        },
    )


@clips_router.get("/clips/{clip_id}/label-summary")
async def clip_label_summary(request: Request, clip_id: int) -> dict[str, object]:
    """Return ``tag_summary``, ``has_manual_cat``, and ``identity_html`` as JSON for live update.

    The page calls this after a tag toggle and after a review toggle. ``identity_html`` comes from
    the same partial as the page, so the live clip result matches a reload.
    """
    state = get_state(request)
    label_summary = _compute_label_summary(state.engine, clip_id)
    identity = _query_detail_identity(
        state.engine,
        clip_id,
        unsure_threshold=active_unsure_threshold(state.config),
        confidence_threshold=state.config.detector.confidence_threshold,
        effective_has_cat=label_summary.effective_has_cat,
    )
    identity_html = state.templates.get_template("_clip_identity.html.jinja").render(
        chips=identity.fields["identity_chips"],
        fallback_label=identity.fields["identity_fallback_label"],
    )
    return {"tag_summary": label_summary.tag_summary, "has_manual_cat": label_summary.has_manual_cat, "identity_html": identity_html}


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

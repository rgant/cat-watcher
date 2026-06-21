"""Label-summary helpers shared between the CLI and web routes.

Both ``cat_watcher.__main__`` (``inspect`` sub-command) and ``cat_watcher.web.clips_routes``
(detail page) need the same ``_query_cat_frame_counts`` / ``_build_tag_summary`` logic; this module
is the single home so neither caller duplicates it.
"""

from typing import TYPE_CHECKING, cast

from sqlalchemy import and_, func, select

from cat_watcher.db import ClipFrame, ClipFrameSubject, Subject, get_session

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.engine import Engine


def query_cat_frame_counts(engine: Engine, clip_id: int) -> list[tuple[str, str, datetime | None, int]]:
    """Return ``(slug, display_name, archived_at, frame_count)`` for every cat subject.

    Active subjects come first (``archived_at IS NULL`` ordered first), then archived. Both groups
    are ordered by ``display_order``. The LEFT JOIN on ``clip_frames`` restricts the count to
    frames belonging to ``clip_id``; subjects with no frames for this clip get 0.
    """
    stmt = (
        select(
            Subject.slug,
            Subject.display_name,
            Subject.archived_at,
            func.count(ClipFrameSubject.clip_frame_id).label("frame_count"),  # pylint: disable=not-callable  # sqlalchemy func.count() is a generative construct, not the builtin; pylint false positive
        )
        .select_from(Subject)
        .outerjoin(ClipFrameSubject, ClipFrameSubject.subject_id == Subject.id)
        .outerjoin(
            ClipFrame,
            and_(ClipFrame.id == ClipFrameSubject.clip_frame_id, ClipFrame.clip_id == clip_id),
        )
        .where(Subject.kind == "cat")
        .group_by(Subject.id, Subject.slug, Subject.display_name, Subject.archived_at)
        .order_by(Subject.archived_at.is_(None).desc(), Subject.display_order)
    )
    with get_session(engine) as session:
        rows = session.execute(stmt).all()
    return [(cast("str", r[0]), cast("str", r[1]), cast("datetime | None", r[2]), cast("int", r[3])) for r in rows]


def build_tag_summary(
    cat_frame_counts: list[tuple[str, str, datetime | None, int]],
    tagged_slugs: set[str],
    active_event_subjects: list[Subject],
) -> str:
    """Compose the tag_summary string per spec rules.

    Cats: all cat subjects listed with frame counts (zeros explicit). Archived cats with any
    tagged frames are appended after active cats, using display_name + "(archived)".
    Events: only active event subjects with ≥1 tagged frame on this clip, bare slug.
    Groups joined by ', '; separated by '; '. Returns '—' when no cats configured and no events.
    """
    cat_parts: list[str] = []
    for slug, display_name, archived_at, frame_count in cat_frame_counts:
        if archived_at is None:
            cat_parts.append(f"{slug}: {frame_count}")
        elif frame_count > 0:
            cat_parts.append(f"{display_name} (archived): {frame_count}")

    event_parts: list[str] = [s.slug for s in active_event_subjects if s.slug in tagged_slugs]

    if not cat_parts and not event_parts:
        return "—"
    if not event_parts:
        return ", ".join(cat_parts)
    if not cat_parts:
        return ", ".join(event_parts)
    return f"{', '.join(cat_parts)}; {', '.join(event_parts)}"

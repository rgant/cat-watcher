"""DB reads that feed the classifier dataset export and the batch spot-check. Neither writes.

Every query here reads operator tags recorded through the review UI (``ClipFrameSubject``).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from sqlalchemy import and_, func, select

from cat_watcher.db import Camera, Clip, ClipFrame, ClipFrameSubject, Subject, get_session

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class CatFrameRow:
    """One frame tagged with exactly one cat, ready for crop export."""

    clip_id: int
    frame_id: int
    ordinal: int
    t_offset_seconds: float
    cat_slug: str
    clip_file_path: str
    frame_thumb_path: str


def resolve_cat_classes(engine: Engine) -> tuple[str, ...]:
    """Return active cat-kind subject slugs, ordered by ``display_order``.

    This tuple is the canonical class order. Train and benchmark record it, so it must not
    reorder once a model trains against it. An archived cat drops out, an event subject never
    enters.
    """
    stmt = select(Subject.slug).where(Subject.kind == "cat", Subject.archived_at.is_(None)).order_by(Subject.display_order)
    with get_session(engine) as session:
        slugs = session.execute(stmt).scalars().all()
    return tuple(slugs)


def query_single_cat_frames(engine: Engine) -> list[CatFrameRow]:
    """Return every clip frame tagged with exactly one cat subject, and that cat is active.

    Whether or not a subject is archived, every ``kind='cat'`` tag on a frame counts toward the
    total. A frame with zero or two-or-more such tags is excluded. An event tag does not add to
    the cat-tag count, so it stays included. An archived cat is absent from
    ``resolve_cat_classes``. Even a frame whose only cat tag names an archived cat is excluded.
    Rows sort by ``clip_id`` then ``ordinal``, so a downstream split stays deterministic.
    """
    single_cat_frame_ids = (
        select(ClipFrameSubject.clip_frame_id)
        .join(Subject, Subject.id == ClipFrameSubject.subject_id)
        .where(Subject.kind == "cat")
        .group_by(ClipFrameSubject.clip_frame_id)
        .having(func.count(ClipFrameSubject.subject_id) == 1)  # pylint: disable=not-callable  # sqlalchemy func.count() is a generative construct, not the builtin; pylint false positive
        .subquery()
    )
    stmt = (
        select(
            ClipFrame.clip_id,
            ClipFrame.id.label("frame_id"),
            ClipFrame.ordinal,
            ClipFrame.t_offset_seconds,
            Subject.slug.label("cat_slug"),
            Clip.file_path.label("clip_file_path"),
            ClipFrame.thumb_path.label("frame_thumb_path"),
        )
        .select_from(single_cat_frame_ids)
        .join(ClipFrame, ClipFrame.id == single_cat_frame_ids.c.clip_frame_id)
        .join(ClipFrameSubject, ClipFrameSubject.clip_frame_id == ClipFrame.id)
        .join(Subject, and_(Subject.id == ClipFrameSubject.subject_id, Subject.kind == "cat"))
        .join(Clip, Clip.id == ClipFrame.clip_id)
        .where(Subject.archived_at.is_(None))
        .order_by(ClipFrame.clip_id, ClipFrame.ordinal)
    )
    with get_session(engine) as session:
        rows = session.execute(stmt).all()
    return [
        CatFrameRow(
            clip_id=cast("int", r.clip_id),
            frame_id=cast("int", r.frame_id),
            ordinal=cast("int", r.ordinal),
            t_offset_seconds=cast("float", r.t_offset_seconds),
            cat_slug=cast("str", r.cat_slug),
            clip_file_path=cast("str", r.clip_file_path),
            frame_thumb_path=cast("str", r.frame_thumb_path),
        )
        for r in rows
    ]


@dataclass(frozen=True)
class ClipCandidate:  # pylint: disable=too-many-instance-attributes  # flat candidate row; the rule targets behavior-rich classes, not data containers
    """One ``has_cat`` clip with no operator tag yet. It carries its best frame."""

    clip_id: int
    camera_name: str
    start_ts: datetime
    clip_file_path: str
    frame_id: int
    ordinal: int
    t_offset_seconds: float
    frame_thumb_path: str


def query_untagged_cat_clips(
    engine: Engine,
    *,
    camera: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 25,
) -> list[ClipCandidate]:
    """Return ``has_cat`` clips with no ``kind='cat'`` tag on any frame, newest ``start_ts`` first.

    Each row carries the clip's highest-score ``ClipFrame``, a tie broken by the lower
    ``ordinal``. These clips carry no operator tag yet, so a spot check can still judge them. A
    ``has_cat`` clip with zero ``ClipFrame`` rows never appears, since it has no frame to show.
    """
    tagged_clip_ids = (
        select(ClipFrame.clip_id)
        .join(ClipFrameSubject, ClipFrameSubject.clip_frame_id == ClipFrame.id)
        .join(Subject, Subject.id == ClipFrameSubject.subject_id)
        .where(Subject.kind == "cat")
    )
    ranked_frames = select(
        ClipFrame.id.label("frame_id"),
        ClipFrame.clip_id,
        ClipFrame.ordinal,
        ClipFrame.t_offset_seconds,
        ClipFrame.thumb_path.label("frame_thumb_path"),
        func.row_number().over(partition_by=ClipFrame.clip_id, order_by=[ClipFrame.score.desc(), ClipFrame.ordinal.asc()]).label("rank"),
    ).subquery()
    stmt = (
        select(
            Clip.id.label("clip_id"),
            Camera.name.label("camera_name"),
            Clip.start_ts,
            Clip.file_path.label("clip_file_path"),
            ranked_frames.c.frame_id,
            ranked_frames.c.ordinal,
            ranked_frames.c.t_offset_seconds,
            ranked_frames.c.frame_thumb_path,
        )
        .select_from(Clip)
        .join(Camera, Camera.id == Clip.camera_id)
        .join(ranked_frames, and_(ranked_frames.c.clip_id == Clip.id, ranked_frames.c.rank == 1))
        .where(Clip.has_cat.is_(True), Clip.id.notin_(tagged_clip_ids))
    )
    if camera is not None:
        stmt = stmt.where(Camera.name == camera)
    if since is not None:
        stmt = stmt.where(Clip.start_ts >= since)
    if until is not None:
        stmt = stmt.where(Clip.start_ts <= until)
    stmt = stmt.order_by(Clip.start_ts.desc()).limit(limit)

    with get_session(engine) as session:
        rows = session.execute(stmt).all()
    return [
        ClipCandidate(
            clip_id=cast("int", r.clip_id),
            camera_name=cast("str", r.camera_name),
            start_ts=cast("datetime", r.start_ts),
            clip_file_path=cast("str", r.clip_file_path),
            frame_id=cast("int", r.frame_id),
            ordinal=cast("int", r.ordinal),
            t_offset_seconds=cast("float", r.t_offset_seconds),
            frame_thumb_path=cast("str", r.frame_thumb_path),
        )
        for r in rows
    ]

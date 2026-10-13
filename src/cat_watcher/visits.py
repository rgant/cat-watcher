"""Group cat clips into visits.

A visit is a run of clips of one named cat on one camera. A clip joins the visit of the previous
clip if its start is less than ``VISIT_GAP`` after the end of that clip. The rule lives here and
nowhere else. The stats and the alerts call these functions.

A visit belongs to a window only if its first clip starts in the window. A clip in the window can
continue a visit that started before the window. Such a visit is not in the window.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select

from cat_watcher.config import active_unsure_threshold
from cat_watcher.db import Clip, ClipLabelSummary
from cat_watcher.identity import query_clip_identities

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from sqlalchemy import ColumnElement
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config
    from cat_watcher.db import Subject
    from cat_watcher.identity import ClipIdentity

__all__ = (
    "VISIT_GAP",
    "CatClip",
    "Visit",
    "group_visits",
    "last_visits_for",
    "query_cat_clips",
    "query_last_visits",
    "query_visits",
)

VISIT_GAP = timedelta(minutes=2)

# A window end is exclusive. This step makes ``now`` itself fall inside the window.
_ONE_TICK = timedelta(microseconds=1)


@dataclass(frozen=True)
class Visit:
    """One cat on one camera, from the start of its first clip to the end of its last clip."""

    slug: str
    camera_id: int
    start_ts: datetime
    end_ts: datetime
    clip_ids: tuple[int, ...]


@dataclass(frozen=True)
class CatClip:
    """A cat clip with the cats that its identity names."""

    clip_id: int
    camera_id: int
    start_ts: datetime
    end_ts: datetime
    identity: ClipIdentity


def query_cat_clips(
    session: Session,
    *,
    since: datetime,
    until: datetime,
    unsure_threshold: float,
    cats: Sequence[Subject],
) -> list[CatClip]:
    """Return every cat clip with ``since <= start_ts < until``, oldest first, with its identity.

    ``cats`` comes from :func:`cat_watcher.identity.query_cat_subjects`.
    The number of queries is fixed.
    """
    return _read_cat_clips(session, (Clip.start_ts >= since, Clip.start_ts < until), unsure_threshold=unsure_threshold, cats=cats)


def query_visits(
    session: Session,
    *,
    since: datetime,
    until: datetime,
    unsure_threshold: float,
    cats: Sequence[Subject],
) -> dict[str, list[Visit]]:
    """Map each named cat to its visits that start in ``since <= start_ts < until``, oldest first.

    The read includes a clip that ends less than ``VISIT_GAP`` before ``since``. Thus a clip in the
    window that continues an earlier visit joins that visit, and the visit drops out.
    The number of queries is fixed.
    """
    clips = _read_cat_clips(
        session,
        (Clip.end_ts >= since - VISIT_GAP, Clip.start_ts < until),
        unsure_threshold=unsure_threshold,
        cats=cats,
    )
    in_window: dict[str, list[Visit]] = {}
    for slug, visits in group_visits(clips).items():
        kept = [visit for visit in visits if visit.start_ts >= since]
        if kept:
            in_window[slug] = kept
    return in_window


def _read_cat_clips(
    session: Session,
    window: tuple[ColumnElement[bool], ...],
    *,
    unsure_threshold: float,
    cats: Sequence[Subject],
) -> list[CatClip]:
    """Return the cat clips that match every condition in ``window``, oldest first, with identities."""
    rows = session.execute(
        select(Clip.id, Clip.camera_id, Clip.start_ts, Clip.end_ts)
        .join(ClipLabelSummary, ClipLabelSummary.clip_id == Clip.id)
        .where(ClipLabelSummary.effective_has_cat, *window)
        .order_by(Clip.start_ts, Clip.id),
    ).all()
    identities = query_clip_identities(session, [row[0] for row in rows], unsure_threshold=unsure_threshold, cats=cats)
    return [
        CatClip(clip_id=clip_id, camera_id=camera_id, start_ts=start_ts, end_ts=end_ts, identity=identities[clip_id])
        for clip_id, camera_id, start_ts, end_ts in rows
    ]


def group_visits(clips: Sequence[CatClip]) -> dict[str, list[Visit]]:
    """Map each named cat to its visits, oldest first.

    A clip that names two cats joins a visit of each. A clip that names no cat joins none.
    """
    by_key: dict[tuple[str, int], list[CatClip]] = defaultdict(list)
    for clip in sorted(clips, key=lambda clip: (clip.start_ts, clip.clip_id)):
        for cat in clip.identity.cats:
            by_key[cat.slug, clip.camera_id].append(clip)
    visits: dict[str, list[Visit]] = defaultdict(list)
    for (slug, camera_id), run in by_key.items():
        visits[slug].extend(_split_run(slug, camera_id, run))
    return {slug: sorted(found, key=lambda visit: visit.start_ts) for slug, found in visits.items()}


def _split_run(slug: str, camera_id: int, run: Sequence[CatClip]) -> list[Visit]:
    """Split the oldest-first clips of one cat on one camera at each gap of ``VISIT_GAP`` or more."""
    groups: list[list[CatClip]] = []
    for clip in run:
        if groups and clip.start_ts - groups[-1][-1].end_ts < VISIT_GAP:
            groups[-1].append(clip)
        else:
            groups.append([clip])
    return [
        Visit(
            slug=slug,
            camera_id=camera_id,
            start_ts=group[0].start_ts,
            end_ts=group[-1].end_ts,
            clip_ids=tuple(clip.clip_id for clip in group),
        )
        for group in groups
    ]


def query_last_visits(
    session: Session,
    *,
    now: datetime,
    retention_days: int,
    unsure_threshold: float,
    cats: Sequence[Subject],
) -> dict[str, Visit]:
    """Map each cat with a visit in the retention window to its newest visit.

    The number of queries is fixed.
    """
    visits = query_visits(
        session,
        since=now - timedelta(days=retention_days),
        until=now + _ONE_TICK,
        unsure_threshold=unsure_threshold,
        cats=cats,
    )
    return {slug: found[-1] for slug, found in visits.items()}


def last_visits_for(session: Session, config: Config, *, now: datetime, cats: Sequence[Subject]) -> dict[str, Visit]:
    """Map each cat to its last visit, with the retention window and the active threshold of ``config``.

    The stats and the INACTIVITY rule call this, so both use the same last visit.
    """
    return query_last_visits(
        session,
        now=now,
        retention_days=config.retention.clip_days,
        unsure_threshold=active_unsure_threshold(config),
        cats=cats,
    )

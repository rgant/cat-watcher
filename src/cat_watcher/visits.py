"""Group cat clips into visits.

A visit is a run of clips of one named cat on one camera. A clip joins the open visit if its start
is less than ``VISIT_GAP`` after the latest end of a clip in that visit. The rule lives here and
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
    from collections.abc import Mapping, Sequence
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
    "last_visits_in_retention",
    "query_visits",
    "visits_for",
    "visits_since",
)

VISIT_GAP = timedelta(minutes=2)

# A window end is exclusive. This step makes ``now`` itself fall inside the window.
_ONE_TICK = timedelta(microseconds=1)


@dataclass(frozen=True)
class Visit:
    """One cat on one camera, from the start of its first clip to the latest end of its clips."""

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


def query_visits(session: Session, config: Config, *, since: datetime, until: datetime, cats: Sequence[Subject]) -> dict[str, list[Visit]]:
    """Map each named cat to its visits that start in ``since <= start_ts < until``, oldest first.

    The read includes a clip that ends less than ``VISIT_GAP`` before ``since``. Thus a clip in the
    window that continues an earlier visit joins that visit, and the visit drops out. The identity
    thresholds come from ``config``. The number of queries is fixed.
    """
    clips = _read_cat_clips(session, config, (Clip.end_ts >= since - VISIT_GAP, Clip.start_ts < until), cats=cats)
    return visits_since(group_visits(clips), since)


def visits_since(visits_by_slug: Mapping[str, Sequence[Visit]], since: datetime) -> dict[str, list[Visit]]:
    """Keep the visits that start at or after ``since``, and drop a cat that has none left.

    This is the window rule of every visit window. If ``visits_by_slug`` comes from a read that
    starts at or before ``since``, the result equals a read that starts at ``since``.
    """
    in_window: dict[str, list[Visit]] = {}
    for slug, visits in visits_by_slug.items():
        kept = [visit for visit in visits if visit.start_ts >= since]
        if kept:
            in_window[slug] = kept
    return in_window


def _read_cat_clips(
    session: Session,
    config: Config,
    window: tuple[ColumnElement[bool], ...],
    *,
    cats: Sequence[Subject],
) -> list[CatClip]:
    """Return the cat clips that match every condition in ``window``, oldest first, with identities."""
    rows = session.execute(
        select(Clip.id, Clip.camera_id, Clip.start_ts, Clip.end_ts)
        .join(ClipLabelSummary, ClipLabelSummary.clip_id == Clip.id)
        .where(ClipLabelSummary.effective_has_cat, *window)
        .order_by(Clip.start_ts, Clip.id),
    ).all()
    identities = query_clip_identities(
        session,
        [row[0] for row in rows],
        unsure_threshold=active_unsure_threshold(config),
        confidence_threshold=config.detector.confidence_threshold,
        cats=cats,
    )
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
    """Split the oldest-first clips of one cat on one camera at each gap of ``VISIT_GAP`` or more.

    The gap starts at the latest end in the visit so far, not at the end of the previous clip. A
    short clip inside a long clip thus cannot split the visit. This also makes a split depend only
    on the clips that end less than ``VISIT_GAP`` before it, which :func:`query_visits` reads.
    """
    groups: list[tuple[list[CatClip], datetime]] = []
    for clip in run:
        if groups and clip.start_ts - groups[-1][1] < VISIT_GAP:
            members, end = groups[-1]
            members.append(clip)
            groups[-1] = (members, max(end, clip.end_ts))
        else:
            groups.append(([clip], clip.end_ts))
    return [
        Visit(
            slug=slug,
            camera_id=camera_id,
            start_ts=members[0].start_ts,
            end_ts=end,
            clip_ids=tuple(clip.clip_id for clip in members),
        )
        for members, end in groups
    ]


def visits_for(session: Session, config: Config, *, now: datetime, since: datetime, cats: Sequence[Subject]) -> dict[str, list[Visit]]:
    """Map each cat to its visits from the earlier of ``since`` and the retention start, through ``now``.

    One read serves the last visit and a window that starts at ``since``. Pass the result to
    :func:`last_visits_in_retention` and to :func:`visits_since`. ``now`` is in the read, so a clip
    that starts at ``now`` counts.
    """
    return query_visits(session, config, since=min(since, _retention_start(config, now)), until=now + _ONE_TICK, cats=cats)


def last_visits_in_retention(visits_by_slug: Mapping[str, Sequence[Visit]], config: Config, *, now: datetime) -> dict[str, Visit]:
    """Map each cat with a visit in the retention window to its newest visit.

    ``visits_by_slug`` comes from :func:`visits_for`. The stats and the INACTIVITY rule call this,
    so both use the same last visit.
    """
    return {slug: found[-1] for slug, found in visits_since(visits_by_slug, _retention_start(config, now)).items()}


def _retention_start(config: Config, now: datetime) -> datetime:
    return now - timedelta(days=config.retention.clip_days)

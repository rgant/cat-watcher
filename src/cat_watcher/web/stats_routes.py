"""HTTP route for the stats page: visits per cat and clips per camera.

Routes read state via the SQLAlchemy engine and Jinja2 templates attached to ``app.state``.
"""

from collections import Counter
from datetime import UTC, datetime, time, timedelta
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from sqlalchemy import select

from cat_watcher.db import Camera, Clip, ClipLabelSummary, get_session
from cat_watcher.identity import query_cat_subjects
from cat_watcher.timefmt import ago
from cat_watcher.visits import last_visits_in_retention, visits_for, visits_since
from cat_watcher.web._app_state import get_state

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from datetime import date, tzinfo

    from sqlalchemy.orm import Session

    from cat_watcher.config import Config
    from cat_watcher.db import Subject
    from cat_watcher.visits import Visit

__all__ = ["HISTORY_DAYS", "bucket_clips_by_local_day", "stats_router"]

# Spec §4.7: the per-camera ``/stats`` table and ``/alerts`` show 30 days at most. This keeps the
# tables short on a deployment that runs for a long time.
HISTORY_DAYS = 30
#: The per-cat table covers this many local days. Today is the last of them.
_CAT_STAT_DAYS = 7
_NO_LAST_VISIT = "none"

stats_router = APIRouter()


def bucket_clips_by_local_day(
    rows: Iterable[tuple[int, datetime, bool]],
    *,
    tz: tzinfo,
) -> dict[tuple[int, date], tuple[int, int]]:
    """Aggregate ``(camera_id, start_ts, effective_has_cat)`` into per-``(camera, local day)`` counts.

    Returns ``{(camera_id, local_date): (total, cat_total)}``.

    This runs in Python, not in SQL. SQLite has no timezone database. ``func.date(start_ts)``
    gives UTC days only. A fixed hour offset is wrong on each side of a DST change. The function
    is pure, so a test can reach both changes. Through the route, each change falls in a
    30-day window only in autumn or spring.
    """
    buckets: dict[tuple[int, date], tuple[int, int]] = {}
    for camera_id, start_ts, effective_has_cat in rows:
        key = (camera_id, start_ts.astimezone(tz).date())
        total, cat_total = buckets.get(key, (0, 0))
        buckets[key] = (total + 1, cat_total + (1 if effective_has_cat else 0))
    return buckets


@stats_router.get("/stats")
async def stats_page(request: Request) -> object:
    """Render the per-cat visits of 7 local days and the per-camera clips of 30 days.

    The per-camera table counts the clips and the cat-positive clips of each camera on each local
    day. Its Date column names the same day as the ``/clips`` Start column. Cat-positive reads
    ``effective_has_cat`` from the ``clip_label_summary`` view.
    """
    state = get_state(request)
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=HISTORY_DAYS)
    display_tz = ZoneInfo(state.config.web.display_timezone)

    with get_session(state.engine) as session:
        cameras = list(session.scalars(select(Camera).order_by(Camera.name)))
        camera_display_by_id = {cam.id: cam.display_name for cam in cameras}
        rows = session.execute(
            select(Clip.camera_id, Clip.start_ts, ClipLabelSummary.effective_has_cat)
            .join(ClipLabelSummary, ClipLabelSummary.clip_id == Clip.id)
            .where(Clip.start_ts >= cutoff),
        ).all()
        cat_rows = _cat_stat_rows(session, config=state.config, now=now, camera_display_by_id=camera_display_by_id)

    buckets = bucket_clips_by_local_day(_stat_inputs(rows), tz=display_tz)
    stat_rows = [
        _stat_row(camera_id, day, total, cat_total, camera_display_by_id=camera_display_by_id)
        for (camera_id, day), (total, cat_total) in sorted(buckets.items(), key=_stat_sort_key)
    ]
    return state.templates.TemplateResponse(
        request,
        "stats.html.jinja",
        {"rows": stat_rows, "cat_rows": cat_rows, "day_headers": [_day_label(day) for day in _stat_days(now, display_tz)]},
    )


def _stat_days(now: datetime, tz: tzinfo) -> list[date]:
    """Return the local days of the per-cat table, oldest first. Today is the last day."""
    today = now.astimezone(tz).date()
    return [today - timedelta(days=offset) for offset in range(_CAT_STAT_DAYS - 1, -1, -1)]


def _day_label(day: date) -> str:
    return day.strftime("%a %-d")


def _cat_stat_rows(
    session: Session,
    *,
    config: Config,
    now: datetime,
    camera_display_by_id: Mapping[int, str],
) -> list[dict[str, object]]:
    """Give one row of visit counts per cat, in :func:`query_cat_subjects` order.

    Each active cat gets a row. An archived cat gets a row only if it has a visit in the 7 days.
    The window starts at local midnight of the first day, so a daylight-saving change in the week
    does not move it. The number of queries is fixed.
    """
    tz = ZoneInfo(config.web.display_timezone)
    days = _stat_days(now, tz)
    cats = query_cat_subjects(session)
    window_start = datetime.combine(days[0], time.min, tzinfo=tz)
    read = visits_for(session, config, now=now, since=window_start, cats=cats)
    visits_by_slug = visits_since(read, window_start)
    last_visits = last_visits_in_retention(read, config, now=now)
    rows: list[dict[str, object]] = []
    for cat in cats:
        visits = visits_by_slug.get(cat.slug, [])
        if cat.archived_at is not None and not visits:
            continue
        last_visit_text = _last_visit_text(last_visits.get(cat.slug), now=now, tz=tz, camera_display_by_id=camera_display_by_id)
        rows.append(_cat_stat_row(cat, visits, days=days, tz=tz, last_visit_text=last_visit_text))
    return rows


def _cat_stat_row(cat: Subject, visits: Sequence[Visit], *, days: Sequence[date], tz: tzinfo, last_visit_text: str) -> dict[str, object]:
    """Count the visits of one cat on each of ``days``. A visit counts on the local day of its start."""
    per_day = Counter(visit.start_ts.astimezone(tz).date() for visit in visits)
    counts = [per_day[day] for day in days]
    complete_days = len(days) - 1
    return {
        "chip": {
            "label": cat.display_name,
            "css_class": "identity-chip identity-chip-operator",
            "color": cat.color or "",
            "title": cat.display_name,
        },
        "last_visit": last_visit_text,
        "day_counts": [{"day": day.isoformat(), "label": _day_label(day), "count": count} for day, count in zip(days, counts, strict=True)],
        "total": sum(counts),
        "mean": round(sum(counts[:complete_days]) / complete_days, 1),
    }


def _last_visit_text(visit: Visit | None, *, now: datetime, tz: tzinfo, camera_display_by_id: Mapping[int, str]) -> str:
    """Render the start of ``visit`` as local time, how long ago, and the camera. No visit gives "none"."""
    if visit is None:
        return _NO_LAST_VISIT
    local = visit.start_ts.astimezone(tz)
    return f"{local:%a %Y-%m-%d %H:%M} · {ago(visit.start_ts, now)} · {camera_display_by_id.get(visit.camera_id, '')}"


def _stat_inputs(rows: Iterable[object]) -> list[tuple[int, datetime, bool]]:
    """Convert stats query Rows into plain tuples for :func:`bucket_clips_by_local_day`.

    The code reads the columns by position with ``cast`` and a tuple unpack. SQLAlchemy types
    each Row column accessor as ``Any``, and a per-element access causes ``reportAny`` warnings.
    """
    return [cast("tuple[int, datetime, bool]", tuple(cast("Sequence[object]", row))) for row in rows]


def _stat_sort_key(item: tuple[tuple[int, date], tuple[int, int]]) -> tuple[int, int]:
    """Order stats rows newest day first, then camera_id ascending."""
    (camera_id, day), _counts = item
    return (-day.toordinal(), camera_id)


def _stat_row(camera_id: int, day: date, total: int, cat_total: int, *, camera_display_by_id: dict[int, str]) -> dict[str, object]:
    """Project one aggregated bucket into a flat dict the template can render."""
    return {
        "camera_id": camera_id,
        "camera_display_name": camera_display_by_id.get(camera_id, ""),
        "date": day,
        "total": total,
        "cat_total": cat_total,
    }

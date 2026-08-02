"""Parsing, reporting, serialization, and SQL application of the ``/clips`` filter set.

Both ``list_clips`` and the clip detail page's prev/next navigation go through this module, so the
two cannot disagree about what a filter value means. Every value is parsed leniently: an
unrecognized value snaps to the control's default and is reported through
:func:`build_ignored_notice` rather than raising. Declaring these as typed route parameters is what
made a malformed ``date_str`` a 500 and an empty ``reviewed`` a 422.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlencode

from cat_watcher.db import Camera, Clip, ClipLabelSummary

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from datetime import tzinfo

    from sqlalchemy.sql import Select

#: The querystring keys the page understands, in the order the notice lists them. Public because
#: the tests parametrize over it, so a control added later is covered the day it lands.
RECOGNIZED_KEYS: tuple[str, ...] = ("reviewed", "camera", "has_cat", "date_str")

_Reviewed = Literal["any", "no", "yes"]
_REVIEWED_VALUES: tuple[_Reviewed, ...] = ("any", "no", "yes")
_HAS_CAT_VALUES: dict[str, bool] = {"true": True, "false": False}


@dataclass(frozen=True, slots=True)
class ClipsFilter:
    """The resolved filter set. Every field holds a usable value, never a raw querystring string."""

    reviewed: _Reviewed = "no"
    camera: str | None = None
    has_cat: bool | None = None
    day: date | None = None


@dataclass(frozen=True, slots=True)
class IgnoredFilter:
    """One rejected value, carried to the notice so the operator learns what was dropped."""

    param: str
    value: str


@dataclass(frozen=True, slots=True)
class ParsedClipsFilter:
    """A resolved filter plus what the parser rejected getting there.

    ``any_key_present`` is true when any recognized key appeared at all, whatever its value. The
    detail page needs it: a URL with no filter keys keeps the legacy global prev/next navigation.
    """

    clips_filter: ClipsFilter
    ignored: tuple[IgnoredFilter, ...]
    any_key_present: bool


def parse_clips_filter(params: Mapping[str, str], *, camera_names: Collection[str]) -> ParsedClipsFilter:
    """Resolve ``params`` into a filter, snapping every unusable value to its control's default.

    An empty string is the filter form's encoding for "unset", so it selects the default and is
    never reported. A non-empty unrecognized value also selects the default but *is* reported. Keys
    outside :data:`RECOGNIZED_KEYS` are ignored silently. A repeated key takes the last occurrence,
    which is what ``QueryParams.get`` does.

    ``camera_names`` is the vocabulary the ``camera`` value is checked against; passing it in keeps
    this function pure while letting both routes share one answer.
    """
    reviewed, reviewed_bad = _parse_reviewed(params.get("reviewed"))
    camera, camera_bad = _parse_camera(params.get("camera"), camera_names)
    has_cat, has_cat_bad = _parse_has_cat(params.get("has_cat"))
    day, day_bad = _parse_day(params.get("date_str"))
    # Ordered by RECOGNIZED_KEYS, not by querystring order, so the notice text is stable.
    ignored = tuple(bad for bad in (reviewed_bad, camera_bad, has_cat_bad, day_bad) if bad is not None)
    return ParsedClipsFilter(
        clips_filter=ClipsFilter(reviewed=reviewed, camera=camera, has_cat=has_cat, day=day),
        ignored=ignored,
        any_key_present=any(key in params for key in RECOGNIZED_KEYS),
    )


def _parse_reviewed(raw: str | None) -> tuple[_Reviewed, IgnoredFilter | None]:
    if not raw:
        return "no", None
    if raw in _REVIEWED_VALUES:
        return raw, None
    return "no", IgnoredFilter(param="reviewed", value=raw)


def _parse_camera(raw: str | None, camera_names: Collection[str]) -> tuple[str | None, IgnoredFilter | None]:
    if not raw:
        return None, None
    if raw in camera_names:
        return raw, None
    return None, IgnoredFilter(param="camera", value=raw)


def _parse_has_cat(raw: str | None) -> tuple[bool | None, IgnoredFilter | None]:
    if not raw:
        return None, None
    if raw in _HAS_CAT_VALUES:
        return _HAS_CAT_VALUES[raw], None
    return None, IgnoredFilter(param="has_cat", value=raw)


def _parse_day(raw: str | None) -> tuple[date | None, IgnoredFilter | None]:
    if not raw:
        return None, None
    try:
        return date.fromisoformat(raw), None
    except ValueError:
        return None, IgnoredFilter(param="date_str", value=raw)


def build_ignored_notice(ignored: Sequence[IgnoredFilter]) -> str:
    """Render the operator-facing notice, or ``""`` when nothing was rejected.

    The returned string is unescaped. Rendered into a page it picks up Jinja's autoescaping, so the
    markup reads ``date_str=&#34;abc&#34;`` — these values are operator-supplied and must never be
    marked safe.
    """
    if not ignored:
        return ""
    entries = ", ".join(f'{item.param}="{item.value}"' for item in ignored)
    return f"Ignored invalid filter values: {entries}."


def build_filter_qs(f: ClipsFilter) -> str:
    """Serialize ``f`` for row-link and prev/next carry-through.

    ``reviewed`` is always emitted so the detail page can rebuild a complete back-link. Rejected
    values cannot appear here: the filter holds what the parser snapped to, not what arrived.
    """
    params: list[tuple[str, str]] = [("reviewed", f.reviewed)]
    if f.camera:
        params.append(("camera", f.camera))
    if f.has_cat is not None:
        params.append(("has_cat", str(f.has_cat).lower()))
    if f.day is not None:
        params.append(("date_str", f.day.isoformat()))
    return urlencode(params)


def apply_clip_filters[TP: tuple[object, ...]](stmt: Select[TP], f: ClipsFilter, *, display_tz: tzinfo) -> Select[TP]:
    """Apply ``f`` to a statement selecting from ``Clip``, preserving its row type.

    ``TP`` is bound rather than fixed so one implementation serves ``select(Clip.id)``,
    ``select(Clip)``, and ``select(func.count()).select_from(Clip)`` without the caller losing its
    row type.

    Contract, relied on by every caller:

    * ``stmt`` must already select from ``Clip``; the conditional joins infer their left side from
      the statement's FROM.
    * The caller must not have joined ``Camera`` or ``ClipLabelSummary`` itself.
    * Row cardinality is preserved — the ``Camera`` join is on a ``NOT NULL`` FK and the view has
      exactly one row per clip, so neither can drop or duplicate rows.
    * ``ORDER BY``, ``LIMIT``, and any further ``WHERE`` on ``Clip`` stay with the caller.

    Callers that deliberately count across review states — the progress indicator — must pass
    ``replace(f, reviewed="any")`` rather than relying on this function to skip the clause.
    """
    if f.camera:
        stmt = stmt.join(Camera, Camera.id == Clip.camera_id).where(Camera.name == f.camera)
    if f.has_cat is not None:
        # ``effective_has_cat``, not ``Clip.has_cat``: this is the column the Cat? badge renders, and
        # they diverge once an operator's frame tags override the detector.
        stmt = stmt.join(ClipLabelSummary, ClipLabelSummary.clip_id == Clip.id).where(
            ClipLabelSummary.effective_has_cat.is_(f.has_cat),
        )
    if f.day is not None:
        # Wall-clock arithmetic inside the zone, so the window is 23h on a spring-forward day and 25h
        # on a fall-back one. UtcDateTime converts the aware bounds on bind.
        day_start = datetime(f.day.year, f.day.month, f.day.day, tzinfo=display_tz)
        stmt = stmt.where(Clip.start_ts >= day_start).where(Clip.start_ts < day_start + timedelta(days=1))
    if f.reviewed == "no":
        stmt = stmt.where(Clip.reviewed_at.is_(None))
    elif f.reviewed == "yes":
        stmt = stmt.where(Clip.reviewed_at.is_not(None))
    return stmt

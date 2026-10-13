"""The time axis of the timeline SVG: ticks, day boundaries, and the "now" marker.

Each function takes the window and the display zone and gives plain dicts. The template reads them
in order.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from zoneinfo import ZoneInfo

__all__ = ["time_axis_marks"]


def _format_tick_label_hour_minute(dt_local: datetime) -> str:
    return dt_local.strftime("%H:%M")


def _format_tick_label_weekday_hour(dt_local: datetime) -> str:
    """At 3d: ``Mon 14:00`` so a label survives a date crossing without a separate marker."""
    return dt_local.strftime("%a %H:%M")


@dataclass(frozen=True, slots=True)
class _TickConfig:
    """The tick spacing and the label rule of one range."""

    seconds: int  # spacing between adjacent ticks
    label_every: int  # n-th tick gets a text label
    formatter: Callable[[datetime], str]


_TICK_CONFIG: dict[str, _TickConfig] = {
    "6h": _TickConfig(seconds=30 * 60, label_every=2, formatter=_format_tick_label_hour_minute),
    "24h": _TickConfig(seconds=60 * 60, label_every=1, formatter=_format_tick_label_hour_minute),
    "3d": _TickConfig(seconds=6 * 60 * 60, label_every=2, formatter=_format_tick_label_weekday_hour),
}


def _format_day_label(dt_local: datetime, *, range_key: str, end_local: datetime) -> str | None:
    """Choose the label that sits next to a midnight day-boundary marker, by range.

    ``None`` means render the boundary line with no label (6h windows are too short to need a date
    prompt). ``today``/``yesterday`` is used at 24h so the operator can read the boundary without
    parsing dates. 3d gets a full ``5 May`` style date.
    """
    if range_key == "6h":
        return None
    if range_key == "24h":
        return "today" if dt_local.date() == end_local.date() else "yesterday"
    return dt_local.strftime("%-d %b")


def _tick_marks(
    *,
    range_key: str,
    start_window: datetime,
    total_seconds: float,
    display_tz: ZoneInfo,
) -> list[dict[str, object]]:
    """Build the per-range tick row: every nth tick gets a label, the rest are unlabeled."""
    tick_config = _TICK_CONFIG[range_key]
    n_ticks = int(total_seconds // tick_config.seconds)
    marks: list[dict[str, object]] = []
    for i in range(1, n_ticks + 1):
        offset = i * tick_config.seconds
        tick_dt_local = (start_window + timedelta(seconds=offset)).astimezone(display_tz)
        label = tick_config.formatter(tick_dt_local) if i % tick_config.label_every == 0 else None
        marks.append({"x_frac": offset / total_seconds, "label": label, "kind": "tick"})
    return marks


def _day_boundary_marks(
    *,
    range_key: str,
    start_window: datetime,
    total_seconds: float,
    display_tz: ZoneInfo,
) -> list[dict[str, object]]:
    """Build the per-midnight day-boundary marks in ``display_tz`` for the window.

    The loop steps in calendar days. ``date + timedelta(days=1)`` always gives the next calendar day.
    ``datetime.combine`` then finds the UTC offset of each midnight again. A step on a tz-aware
    ``datetime`` keeps the offset of the window start. Across a DST change, that puts the boundary
    one hour off, or on the wrong date.
    """
    start_local = start_window.astimezone(display_tz)
    end_local = (start_window + timedelta(seconds=total_seconds)).astimezone(display_tz)
    marks: list[dict[str, object]] = []
    day = start_local.date() + timedelta(days=1)
    while day <= end_local.date():
        cursor = datetime.combine(day, datetime.min.time(), tzinfo=display_tz)
        offset = (cursor - start_local).total_seconds()
        if 0 < offset < total_seconds:
            marks.append(
                {
                    "x_frac": offset / total_seconds,
                    "label": _format_day_label(cursor, range_key=range_key, end_local=end_local),
                    "kind": "day",
                },
            )
        day += timedelta(days=1)
    return marks


def time_axis_marks(
    *,
    range_key: str,
    start_window: datetime,
    total_seconds: float,
    display_tz: ZoneInfo,
) -> list[dict[str, object]]:
    """Build the SVG time-axis view-model: tick rows, day boundaries, and a 'now' marker.

    Each output dict carries ``x_frac`` ([0, 1] left fraction), an optional ``label`` (string or
    ``None``), and a ``kind`` discriminator: ``"tick"``, ``"day"``, or ``"now"``. The template
    consumes the list in order and picks the SVG element type per kind.
    """
    return [
        *_tick_marks(range_key=range_key, start_window=start_window, total_seconds=total_seconds, display_tz=display_tz),
        *_day_boundary_marks(range_key=range_key, start_window=start_window, total_seconds=total_seconds, display_tz=display_tz),
        {"x_frac": 1.0, "label": None, "kind": "now"},
    ]

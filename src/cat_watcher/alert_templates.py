"""Pure renderers for alert subject + body + macOS summary, one function per alert type.

Per spec §4.14. Each ``render_*`` function takes plain values (camera display name, datetimes, paths,
counts, etc.) and returns an :class:`AlertContent` carrying:

* ``subject`` — short email subject line (also used as the macOS notification title).
* ``body`` — multi-line plain-text email body.
* ``macos_summary`` — single-line message body for the ``osascript`` notification banner; capped at
  120 characters so it never overflows the macOS UI surface.

These functions are I/O-free and config-free: callers (the alerts agent and the poller's
``ALERTS_STUCK`` watchdog) read whatever state they need from the DB / filesystem and pass it in.
That keeps the templates trivially testable as snapshots and avoids coupling the alerts agent's rule
evaluators to the on-disk layout the renderers happen to display.

All datetimes are formatted in the operator's display timezone (``cfg.web.display_timezone``) with
the IANA abbreviation + UTC offset trailing the local time (``2026-05-01 09:42:11 EDT (-04:00)``).
Relative ages (``14h 18m ago``) come from the ``now`` argument so tests can pin the rendered output.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from cat_watcher.db import AlertType, PollStatus


_MACOS_SUMMARY_MAX = 120
_BYTES_PER_GB = 1_000_000_000
_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60


# Inactivity-rule branch labels. The `poll_status_unreachable` branch is parameterized at render
# time because it embeds the unreachable-since timestamp; the other two are static strings.
INACTIVITY_BRANCH_NO_CATS = "no cats seen"
INACTIVITY_BRANCH_NO_CLIPS = "no clips received"


@dataclass(frozen=True)
class AlertContent:
    """Rendered alert payload shared between email and macOS dispatch paths."""

    subject: str
    body: str
    macos_summary: str


def _fmt(dt: datetime, tz_name: str) -> str:
    """Render ``dt`` in ``tz_name`` as ``YYYY-MM-DD HH:MM:SS ZZZ (+HH:MM)`` per spec §4.14."""
    local = dt.astimezone(ZoneInfo(tz_name))
    abbr = local.strftime("%Z")
    raw_offset = local.strftime("%z")
    offset_fmt = f"{raw_offset[:3]}:{raw_offset[3:]}" if raw_offset else ""
    return f"{local.strftime('%Y-%m-%d %H:%M:%S')} {abbr} ({offset_fmt})"


def _ago(dt: datetime, now: datetime) -> str:
    """Render ``now - dt`` as ``45s ago`` / ``12m ago`` / ``14h 18m ago`` per spec §4.14 examples."""
    delta = now - dt
    seconds = max(0, int(delta.total_seconds()))
    if seconds < _SECONDS_PER_MINUTE:
        return f"{seconds}s ago"
    minutes_total = seconds // _SECONDS_PER_MINUTE
    if minutes_total < _MINUTES_PER_HOUR:
        return f"{minutes_total}m ago"
    hours, minutes = divmod(minutes_total, _MINUTES_PER_HOUR)
    return f"{hours}h {minutes}m ago"


def _fmt_with_relative(dt: datetime, tz_name: str, now: datetime) -> str:
    """Combine :func:`_fmt` with :func:`_ago` as ``<absolute> — <ago>`` (em-dash separator per §4.14)."""
    return f"{_fmt(dt, tz_name)} — {_ago(dt, now)}"


def _fmt_with_relative_or_never(dt: datetime | None, tz_name: str, now: datetime) -> str:
    """``_fmt_with_relative`` when set, ``never`` otherwise."""
    if dt is None:
        return "never"
    return _fmt_with_relative(dt, tz_name, now)


def _capped_summary(text: str) -> str:
    """Truncate ``text`` to :data:`_MACOS_SUMMARY_MAX` chars; appends ``…`` when it overflows."""
    if len(text) <= _MACOS_SUMMARY_MAX:
        return text
    return text[: _MACOS_SUMMARY_MAX - 1] + "…"


def inactivity_branch_unreachable(poll_status_since: datetime, tz_name: str) -> str:
    """Render the ``camera unreachable since {timestamp}`` branch label (spec §4.14)."""
    return f"camera unreachable since {_fmt(poll_status_since, tz_name)}"


def render_inactivity(  # noqa: PLR0913  # spec §4.14 INACTIVITY body has 9 distinct fields
    *,
    camera_display_name: str,
    branch: str,
    last_polled_at: datetime | None,
    last_clip_at: datetime | None,
    last_cat_seen_at: datetime | None,
    poll_status: PollStatus,
    public_url: str,
    tz_name: str,
    now: datetime,
) -> AlertContent:
    """Render the ``INACTIVITY`` alert per spec §4.14.

    ``branch`` is one of :data:`INACTIVITY_BRANCH_NO_CATS`, :data:`INACTIVITY_BRANCH_NO_CLIPS`, or a
    string from :func:`inactivity_branch_unreachable` (caller-built so the unreachable-since
    timestamp survives without re-passing ``poll_status_since`` here).
    """
    subject = f"[cat-watcher] INACTIVITY: {camera_display_name}"
    body = (
        f"Branch:         {branch}\n"
        f"Last poll:      {_fmt_with_relative_or_never(last_polled_at, tz_name, now)}\n"
        f"Last clip:      {_fmt_with_relative_or_never(last_clip_at, tz_name, now)}\n"
        f"Last cat seen:  {_fmt_with_relative_or_never(last_cat_seen_at, tz_name, now)}\n"
        f"Poll status:    {poll_status.value}\n"
        f"Web UI:         {public_url}\n"
    )
    summary = _capped_summary(f"{camera_display_name}: {branch}")
    return AlertContent(subject=subject, body=body, macos_summary=summary)


def render_inactivity_no_cats_global(  # noqa: PLR0913  # spec §4.14 INACTIVITY (no-cats global) body has 4 distinct fields plus tz + clock
    *,
    last_cat_seen_at: datetime,
    last_cat_seen_camera_name: str,
    inactivity_hours: int,
    public_url: str,
    tz_name: str,
    now: datetime,
) -> AlertContent:
    """Render the fleet-wide ``no cats seen on any camera`` INACTIVITY alert per spec §4.14.

    Distinct from :func:`render_inactivity` because the per-camera fields (``Last poll``,
    ``Last clip``, ``Poll status``) don't apply to a fleet-wide check. The ``last_cat_seen_at`` line
    identifies which camera was the most recent to see a cat, so the operator knows where to start
    looking when activity resumes.
    """
    subject = "[cat-watcher] INACTIVITY: no cats seen on any camera"
    body = (
        "Branch:         no cats seen on any camera\n"
        f"Last cat seen:  {_fmt_with_relative(last_cat_seen_at, tz_name, now)} ({last_cat_seen_camera_name})\n"
        f"Threshold:      {inactivity_hours}h\n"
        f"Web UI:         {public_url}\n"
    )
    summary = _capped_summary(f"no cats seen on any camera in {_ago(last_cat_seen_at, now).removesuffix(' ago')}")
    return AlertContent(subject=subject, body=body, macos_summary=summary)


def render_frequency(  # noqa: PLR0913  # spec §4.14 FREQUENCY body has 8 distinct fields
    *,
    camera_display_name: str,
    count: int,
    window_hours: int,
    threshold: int,
    first_in_window: datetime,
    latest_in_window: datetime,
    public_url: str,
    tz_name: str,
) -> AlertContent:
    """Render the ``FREQUENCY`` alert per spec §4.14."""
    subject = f"[cat-watcher] FREQUENCY: {camera_display_name} ({count} in {window_hours}h)"
    body = (
        f"Camera:                          {camera_display_name}\n"
        f"Cat-positive clips in last {window_hours}h:   {count}\n"
        f"Threshold:                       {threshold}\n"
        f"First in window:                 {_fmt(first_in_window, tz_name)}\n"
        f"Latest:                          {_fmt(latest_in_window, tz_name)}\n"
        f"Web UI:                          {public_url}\n"
    )
    summary = _capped_summary(
        f"{camera_display_name}: {count} cat-positive clips in {window_hours}h (threshold {threshold})",
    )
    return AlertContent(subject=subject, body=body, macos_summary=summary)


def render_heartbeat_watchdog(  # noqa: PLR0913  # spec §4.14 POLLER_STUCK / WEB_DOWN / ALERTS_STUCK template has 6 distinct fields
    *,
    alert_type: AlertType,
    agent_name: str,
    last_heartbeat: datetime,
    public_url: str,
    tz_name: str,
    now: datetime,
) -> AlertContent:
    """Render ``POLLER_STUCK`` / ``WEB_DOWN`` / ``ALERTS_STUCK`` per spec §4.14 (shared template)."""
    subject = f"[cat-watcher] {alert_type.value}"
    body = (
        f"Agent:           {agent_name}\n"
        f"Last heartbeat:  {_fmt_with_relative(last_heartbeat, tz_name, now)}\n"
        f"Web UI:          {public_url}\n"
    )
    summary = _capped_summary(f"{agent_name} heartbeat {_ago(last_heartbeat, now)}")
    return AlertContent(subject=subject, body=body, macos_summary=summary)


def render_poller_empty_after_quiet(  # noqa: PLR0913  # body has camera + 3 timestamps + 1 derived duration + tz + clock
    *,
    camera_display_name: str,
    last_clip_at: datetime,
    queried_since: datetime,
    queried_until: datetime,
    public_url: str,
    tz_name: str,
    now: datetime,
) -> AlertContent:
    """Render ``POLLER_EMPTY_AFTER_QUIET``: safety-net widened query returned zero rows.

    Fired by the poller (not the alerts agent) when a camera has been quiet for
    ``safety_net_hours``, the poller widens its query to ``[last_clip_at - overlap_minutes, now]``,
    and Amcrest's ``findFile`` returns zero rows. Indicates the camera is alive enough to answer
    the query but is not surfacing recordings: firmware/index issue, SD card filled or failed,
    camera rebooted without preserving recordings, or the camera is not recording.
    """
    subject = f"[cat-watcher] POLLER_EMPTY_AFTER_QUIET: {camera_display_name}"
    body = (
        f"Camera:         {camera_display_name}\n"
        f"Last clip:      {_fmt_with_relative(last_clip_at, tz_name, now)}\n"
        f"Quiet for:      {_ago(last_clip_at, now).removesuffix(' ago')}\n"
        f"Queried since:  {_fmt(queried_since, tz_name)}\n"
        f"Queried until:  {_fmt(queried_until, tz_name)}\n"
        f"Result:         findFile returned 0 rows\n"
        f"What to check:  camera power, SD card health, camera's own recording UI\n"
        f"Web UI:         {public_url}\n"
    )
    summary = _capped_summary(f"{camera_display_name}: no clips after {_ago(last_clip_at, now).removesuffix(' ago')} quiet")
    return AlertContent(subject=subject, body=body, macos_summary=summary)


def render_web_flapping(  # noqa: PLR0913  # spec §4.14 WEB_FLAPPING body has 7 distinct fields
    *,
    restart_count: int,
    window_minutes: int,
    first_restart: datetime,
    last_restart: datetime,
    log_tail: str,
    public_url: str,
    tz_name: str,
) -> AlertContent:
    """Render ``WEB_FLAPPING`` per spec §4.14, including the trailing log tail block."""
    subject = f"[cat-watcher] WEB_FLAPPING ({restart_count} restarts in last {window_minutes}m)"
    body = (
        f"Agent:                  web\n"
        f"Restarts in last {window_minutes}m:   {restart_count}\n"
        f"First in window:        {_fmt(first_restart, tz_name)}\n"
        f"Most recent:            {_fmt(last_restart, tz_name)}\n"
        f"Web UI:                 {public_url}\n"
        f"\n--- Last 50 lines of web.stderr.log ---\n{log_tail}"
    )
    summary = _capped_summary(f"web: {restart_count} restarts in last {window_minutes}m")
    return AlertContent(subject=subject, body=body, macos_summary=summary)


def render_disk_low(
    *,
    storage_root: Path,
    free_bytes: int,
    total_bytes: int,
    threshold_fraction: float,
    public_url: str,
) -> AlertContent:
    """Render ``DISK_LOW`` per spec §4.14. ``total_bytes==0`` is reported as ``0.0%``."""
    fraction = (free_bytes / total_bytes) if total_bytes > 0 else 0.0
    free_gb = free_bytes // _BYTES_PER_GB
    total_gb = total_bytes // _BYTES_PER_GB
    threshold_pct = threshold_fraction * 100
    fraction_pct = fraction * 100
    subject = f"[cat-watcher] DISK_LOW ({fraction_pct:.1f}% free on {storage_root})"
    body = (
        f"Mount:           {storage_root}\n"
        f"Free:            {free_gb} GB / {total_gb} GB ({fraction_pct:.1f}%)\n"
        f"Threshold:       {threshold_pct:.0f}%\n"
        f"Web UI:          {public_url}\n"
    )
    summary = _capped_summary(
        f"{storage_root}: {fraction_pct:.1f}% free (threshold {threshold_pct:.0f}%)",
    )
    return AlertContent(subject=subject, body=body, macos_summary=summary)


def render_storage_unavailable(
    *,
    storage_root: Path,
    first_detected: datetime,
    tz_name: str,
    now: datetime,
) -> AlertContent:
    """Render ``STORAGE_UNAVAILABLE`` per spec §4.14.

    ``first_detected`` is the tick's ``now`` from the alerts agent — there is no separate
    "first-detected" persistence, so the field reflects the tick that observed the failure.
    """
    subject = "[cat-watcher] STORAGE_UNAVAILABLE (external drive not mounted)"
    body = (
        f"storage_root:  {storage_root}\n"
        f"First detected: {_fmt_with_relative(first_detected, tz_name, now)}\n"
        f"Probable cause: external drive not unlocked yet, disconnected, or unlock dismissed\n"
        f"Affected:      poller (cannot ingest new clips), backup (cannot write nightly backup)\n"
        f"Unaffected:    alerts (this email), web UI (DB on internal storage)\n"
    )
    summary = _capped_summary(f"storage_root unavailable: {storage_root}")
    return AlertContent(subject=subject, body=body, macos_summary=summary)


def render_backup_stale(  # noqa: PLR0913  # spec §4.14 BACKUP_STALE body has 6 distinct fields
    *,
    newest_backup_name: str,
    mtime: datetime,
    threshold_hours: int,
    storage_root: Path,
    tz_name: str,
    now: datetime,
) -> AlertContent:
    """Render ``BACKUP_STALE`` per spec §4.14. ``hours_since`` rounds down to whole hours."""
    delta_seconds = max(0, int((now - mtime).total_seconds()))
    hours_since = delta_seconds // 3600
    subject = f"[cat-watcher] BACKUP_STALE (no backup in {hours_since}h)"
    body = (
        f"Newest backup:  {newest_backup_name}\n"
        f"Mtime:          {_fmt_with_relative(mtime, tz_name, now)}\n"
        f"Threshold:      {threshold_hours}h\n"
        f"Storage root:   {storage_root} (mounted: yes)\n"
    )
    summary = _capped_summary(f"No backup in {hours_since}h (threshold {threshold_hours}h)")
    return AlertContent(subject=subject, body=body, macos_summary=summary)

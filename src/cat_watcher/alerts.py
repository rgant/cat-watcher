"""Alerts agent: rule evaluators + cool-down-aware dispatch helper + LaunchAgent entry point.

Per spec §4.5 + Task 18 plan. Three responsibilities, one module:

1. **Rule evaluators** — pure functions that read DB / filesystem state and return an
   :class:`AlertCandidate` (or None, or a list for storage rules) describing what to fire. Per-camera
   rules: ``INACTIVITY``, ``FREQUENCY``. Cross-agent watchdogs: ``POLLER_STUCK``, ``WEB_DOWN``,
   ``WEB_FLAPPING``. Storage rules: ``STORAGE_UNAVAILABLE``, ``DISK_LOW``, ``BACKUP_STALE``.
   Evaluators don't render templates or send anything — they invoke the renderers in
   :mod:`cat_watcher.alert_templates` and hand the result back to the caller.

2. **Shared dispatcher** — :func:`dispatch_alert` does cool-down + send + record. This is the only
   place that talks to :mod:`cat_watcher.notifier` and writes to ``alerts_sent``. The poller imports
   this for its ``ALERTS_STUCK`` watchdog (Task 17 carve-out): the poller does *not* own cool-down
   state, so all routing — including from inside the poller — must funnel through this helper.
   Suppressed events log at INFO and write **no** ``alerts_sent`` row (per Task 18: an active
   24h-cool-down alert must not generate ~96 rows/day per type).

3. **Tick orchestrator** (:func:`run_alerts_tick`) + ``main`` — for the ``cat-watcher-alerts --once``
   LaunchAgent. Skips the §4.13 storage wait (the alerts agent's state lives on internal storage, so
   it must run even when the external drive is offline so it can fire ``STORAGE_UNAVAILABLE``).
"""

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from cat_watcher.alert_templates import (
    INACTIVITY_BRANCH_NO_CATS,
    INACTIVITY_BRANCH_NO_CLIPS,
    AlertContent,
    inactivity_branch_unreachable,
    render_backup_stale,
    render_disk_low,
    render_frequency,
    render_heartbeat_watchdog,
    render_inactivity,
    render_storage_unavailable,
    render_web_flapping,
)
from cat_watcher.config import load_config
from cat_watcher.db import (
    AgentStart,
    AlertSent,
    AlertType,
    Camera,
    Clip,
    Heartbeat,
    PollStatus,
    create_engine,
    get_session,
)
from cat_watcher.notifier import EmailResult, NotifResult, send_email, send_macos_notification
from cat_watcher.storage import ensure_storage_layout, storage_available

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import AlertConfig, Config, EmailSecrets


logger = logging.getLogger(__name__)

_AGENT_NAME = "alerts"
_DB_FILENAME = "cat_watcher.sqlite"
_DELIVERY_ERROR_MAX_LEN = 500
_LOG_TAIL_LINES = 50


# Per spec §4.5: these alert types use hard-coded cool-down overrides instead of the config default
# (`[alerts].cooldown_hours`). The values follow from the rule semantics — a flap detector with a 6h
# cool-down would miss the next flap; a once-per-24h backup's staleness check would re-fire inside a
# single missed run if the cool-down were hours-scale. One-line edit if an operator truly needs to
# change them; no config schema churn.
_HARDCODED_COOLDOWN_OVERRIDES_HOURS: dict[AlertType, int] = {
    AlertType.WEB_FLAPPING: 1,
    AlertType.DISK_LOW: 24,
    AlertType.BACKUP_STALE: 24,
}


@dataclass(frozen=True)
class AlertCandidate:
    """Rule evaluator output: a fully-rendered alert ready for :func:`dispatch_alert`.

    Carrying ``alert_type`` here (rather than implying it from the evaluator that produced it)
    means the orchestrator can collect candidates from multiple evaluators into a single list and
    dispatch them uniformly without a parallel ``alert_type`` array.
    """

    alert_type: AlertType
    camera_id: int | None
    content: AlertContent


@dataclass(frozen=True, kw_only=True, slots=True)
class DispatchEnv:
    """External dependencies for :func:`dispatch_alert` — config + infrastructure, separate from
    the per-call alert identity (``alert_type`` + ``camera_id`` + ``content``).

    ``now`` is optional so production callers omit it and get :func:`datetime.now(UTC)`; tests pin
    it to keep cool-down assertions deterministic.
    """

    secrets: EmailSecrets
    rules: AlertConfig
    session: Session
    now: datetime | None = None


# --- shared cool-down + dispatch ------------------------------------------------------------------


def cooldown_for(alert_type: AlertType, rules: AlertConfig) -> int:
    """Return the cool-down window in hours for ``alert_type``.

    Hard-coded overrides win over ``rules.cooldown_hours``; everything else falls back to the config
    default.
    """
    return _HARDCODED_COOLDOWN_OVERRIDES_HOURS.get(alert_type, rules.cooldown_hours)


def dispatch_alert(
    alert_type: AlertType,
    *,
    camera_id: int | None,
    content: AlertContent,
    env: DispatchEnv,
) -> None:
    """Send ``content`` for ``alert_type`` if the cool-down window is clear.

    Cool-down key is ``(camera_id, alert_type)`` for camera alerts and ``(NULL, alert_type)`` for
    the seven non-camera types (per plan §13 resolution 5: one global cool-down per non-camera
    alert type). Suppression logs at INFO and writes **no** ``alerts_sent`` row; only dispatched
    events persist a row. The helper never raises — both senders return typed results that are
    recorded on the row.
    """
    effective_now = env.now if env.now is not None else datetime.now(UTC)
    cooldown_h = cooldown_for(alert_type, env.rules)
    cutoff = effective_now - timedelta(hours=cooldown_h)

    last_sent_at = _most_recent_sent_at(
        env.session,
        alert_type=alert_type,
        camera_id=camera_id,
        cutoff=cutoff,
    )
    if last_sent_at is not None:
        remaining = (last_sent_at + timedelta(hours=cooldown_h)) - effective_now
        logger.info(
            "alert suppressed: type=%s camera_id=%s remaining_cooldown_seconds=%d",
            alert_type.value,
            camera_id,
            max(0, int(remaining.total_seconds())),
        )
        return

    email_result = send_email(content.subject, content.body, secrets=env.secrets, rules=env.rules.email)
    notif_result = send_macos_notification(content.subject, content.macos_summary, rules=env.rules.macos)

    delivery_error = _format_delivery_error(email_result, notif_result)
    env.session.add(
        AlertSent(
            alert_type=alert_type,
            camera_id=camera_id,
            sent_at=effective_now,
            subject=content.subject,
            body=content.body,
            email_ok=email_result.ok,
            macos_ok=notif_result.ok,
            delivery_error=delivery_error,
        ),
    )
    if not email_result.ok and not notif_result.ok:
        logger.critical(
            "alert dispatch failed on both channels: type=%s camera_id=%s email=%s macos=%s",
            alert_type.value,
            camera_id,
            email_result.error,
            notif_result.error,
        )


def _most_recent_sent_at(
    session: Session,
    *,
    alert_type: AlertType,
    camera_id: int | None,
    cutoff: datetime,
) -> datetime | None:
    """Return the latest ``alerts_sent.sent_at`` matching ``(alert_type, camera_id)`` since ``cutoff``."""
    camera_predicate = AlertSent.camera_id.is_(None) if camera_id is None else AlertSent.camera_id == camera_id
    stmt = (
        select(AlertSent.sent_at)
        .where(AlertSent.alert_type == alert_type)
        .where(camera_predicate)
        .where(AlertSent.sent_at >= cutoff)
        .order_by(AlertSent.sent_at.desc())
        .limit(1)
    )
    return session.scalar(stmt)


def _format_delivery_error(email_result: EmailResult, notif_result: NotifResult) -> str | None:
    """Combine non-OK errors into a single ``email: ...; macos: ...`` string capped at 500 chars."""
    parts: list[str] = []
    if not email_result.ok and email_result.error:
        parts.append(f"email: {email_result.error}")
    if not notif_result.ok and notif_result.error:
        parts.append(f"macos: {notif_result.error}")
    if not parts:
        return None
    joined = "; ".join(parts)
    return joined[:_DELIVERY_ERROR_MAX_LEN]


# --- rule evaluators ------------------------------------------------------------------------------


def evaluate_inactivity(
    cam: Camera,
    *,
    inactivity_hours: int,
    public_url: str,
    tz_name: str,
    now: datetime,
) -> AlertCandidate | None:
    """Evaluate the ``INACTIVITY`` rule per spec §4.5.

    Branch order (first match wins):

    1. ``poll_status != ok`` — fires immediately, no time threshold; cool-down still applies.
    2. ``last_clip_at`` set and older than ``inactivity_hours`` — camera producing nothing.
    3. ``last_cat_seen_at`` set and older than ``inactivity_hours`` — camera works, no cats.

    Returns ``None`` when no branch fires. If both ``last_cat_seen_at`` and ``last_clip_at`` are
    NULL the rule does not fire (no baseline yet — covered by the first poll's 30-day backfill).
    The ``poll_status_since`` value embeds in the unreachable branch label so the body distinguishes
    *which* branch fired (operator looks at cat / camera / network accordingly).
    """
    threshold = timedelta(hours=inactivity_hours)
    branch: str | None = None
    if cam.poll_status != PollStatus.OK:
        since = cam.poll_status_since or now
        branch = inactivity_branch_unreachable(since, tz_name)
    elif cam.last_clip_at is not None and (now - cam.last_clip_at) > threshold:
        branch = INACTIVITY_BRANCH_NO_CLIPS
    elif cam.last_cat_seen_at is not None and (now - cam.last_cat_seen_at) > threshold:
        branch = INACTIVITY_BRANCH_NO_CATS

    if branch is None:
        return None

    content = render_inactivity(
        camera_display_name=cam.display_name,
        branch=branch,
        last_polled_at=cam.last_polled_at,
        last_clip_at=cam.last_clip_at,
        last_cat_seen_at=cam.last_cat_seen_at,
        poll_status=cam.poll_status,
        public_url=public_url,
        tz_name=tz_name,
        now=now,
    )
    return AlertCandidate(alert_type=AlertType.INACTIVITY, camera_id=cam.id, content=content)


def evaluate_frequency(  # noqa: PLR0913  # rule reads camera clips against 4 thresholds, formats output with tz + clock
    session: Session,
    cam: Camera,
    *,
    window_hours: int,
    threshold: int,
    public_url: str,
    tz_name: str,
    now: datetime,
) -> AlertCandidate | None:
    """Evaluate the ``FREQUENCY`` rule per spec §4.5: ``count(cat-positive) >= threshold`` in window.

    ``COALESCE(manual_has_cat, has_cat)`` is the cat-positive projection — manual labels override
    the model output, so a clip you marked ``manual_has_cat=False`` does not count toward the
    threshold even if ``has_cat=True``. Conversely a model-false clip you marked ``True`` does
    count.
    """
    cutoff = now - timedelta(hours=window_hours)
    rows = list(
        session.scalars(
            select(Clip)
            .where(Clip.camera_id == cam.id)
            .where(func.coalesce(Clip.manual_has_cat, Clip.has_cat).is_(True))
            .where(Clip.start_ts >= cutoff)
            .order_by(Clip.start_ts),
        ).all(),
    )
    count = len(rows)
    if count < threshold:
        return None
    content = render_frequency(
        camera_display_name=cam.display_name,
        count=count,
        window_hours=window_hours,
        threshold=threshold,
        first_in_window=rows[0].start_ts,
        latest_in_window=rows[-1].start_ts,
        public_url=public_url,
        tz_name=tz_name,
    )
    return AlertCandidate(alert_type=AlertType.FREQUENCY, camera_id=cam.id, content=content)


def evaluate_heartbeat_watchdog(  # noqa: PLR0913  # watchdog needs the alert identity, the staleness threshold, and tz + clock for the rendered body
    session: Session,
    *,
    alert_type: AlertType,
    agent_name: str,
    stale_minutes: int,
    public_url: str,
    tz_name: str,
    now: datetime,
) -> AlertCandidate | None:
    """Evaluate ``POLLER_STUCK`` / ``WEB_DOWN`` per spec §4.5 (heartbeat older than threshold).

    ``ALERTS_STUCK`` is fired by the **poller** (it watches the alerts agent's heartbeat); this
    helper is shared by the poller's wiring in :mod:`cat_watcher.poller` so the rendering + cooldown
    bookkeeping flows through the same code path.
    """
    hb = session.get(Heartbeat, agent_name)
    if hb is None or now - hb.last_seen_at <= timedelta(minutes=stale_minutes):
        return None
    content = render_heartbeat_watchdog(
        alert_type=alert_type,
        agent_name=agent_name,
        last_heartbeat=hb.last_seen_at,
        public_url=public_url,
        tz_name=tz_name,
        now=now,
    )
    return AlertCandidate(alert_type=alert_type, camera_id=None, content=content)


def _read_log_tail(log_path: Path, line_count: int = _LOG_TAIL_LINES) -> str:
    """Best-effort read of the last ``line_count`` lines of ``log_path`` (empty string on miss/IO error)."""
    if not log_path.is_file():
        return ""
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        logger.warning("failed to read web log tail at %s: %s", log_path, exc)
        return ""
    return "".join(lines[-line_count:])


def evaluate_web_flapping(  # noqa: PLR0913  # rule reads agent_starts against window/threshold and renders with log tail + tz + clock
    session: Session,
    *,
    window_minutes: int,
    threshold: int,
    log_path: Path,
    public_url: str,
    tz_name: str,
    now: datetime,
) -> AlertCandidate | None:
    """Evaluate ``WEB_FLAPPING`` per spec §4.5: ≥``threshold`` web restarts within ``window_minutes``."""
    cutoff = now - timedelta(minutes=window_minutes)
    rows = list(
        session.scalars(
            select(AgentStart)  # dprint-ignore
            .where(AgentStart.agent_name == "web")
            .where(AgentStart.started_at >= cutoff)
            .order_by(AgentStart.started_at),
        ).all(),
    )
    if len(rows) < threshold:
        return None
    content = render_web_flapping(
        restart_count=len(rows),
        window_minutes=window_minutes,
        first_restart=rows[0].started_at,
        last_restart=rows[-1].started_at,
        log_tail=_read_log_tail(log_path),
        public_url=public_url,
        tz_name=tz_name,
    )
    return AlertCandidate(alert_type=AlertType.WEB_FLAPPING, camera_id=None, content=content)


def evaluate_storage(
    *,
    storage_root: Path,
    threshold_fraction: float,
    public_url: str,
    tz_name: str,
    now: datetime,
) -> list[AlertCandidate]:
    """Evaluate ``STORAGE_UNAVAILABLE`` + ``DISK_LOW`` per spec §4.5.

    Returns ``[]``, ``[STORAGE_UNAVAILABLE]``, or ``[DISK_LOW]`` (never both: ``DISK_LOW`` is
    skipped when the drive is unmounted, since ``STORAGE_UNAVAILABLE`` covers that case).
    """
    candidates: list[AlertCandidate] = []
    if not storage_available(storage_root):
        content = render_storage_unavailable(
            storage_root=storage_root,
            first_detected=now,
            tz_name=tz_name,
            now=now,
        )
        candidates.append(AlertCandidate(alert_type=AlertType.STORAGE_UNAVAILABLE, camera_id=None, content=content))
        return candidates
    usage = shutil.disk_usage(storage_root)
    fraction = (usage.free / usage.total) if usage.total > 0 else 0.0
    if fraction < threshold_fraction:
        content = render_disk_low(
            storage_root=storage_root,
            free_bytes=usage.free,
            total_bytes=usage.total,
            threshold_fraction=threshold_fraction,
            public_url=public_url,
        )
        candidates.append(AlertCandidate(alert_type=AlertType.DISK_LOW, camera_id=None, content=content))
    return candidates


def evaluate_backup_stale(  # noqa: PLR0913  # rule reads filesystem mtime + alerts_sent against two cool-down windows and renders with tz + clock
    session: Session,
    *,
    storage_root: Path,
    threshold_hours: int,
    storage_unavailable_cooldown_hours: int,
    tz_name: str,
    now: datetime,
) -> AlertCandidate | None:
    """Evaluate ``BACKUP_STALE`` per spec §4.5.

    Suppression: returns ``None`` when the most recent ``alerts_sent`` row for
    ``STORAGE_UNAVAILABLE`` is newer than ``now - storage_unavailable_cooldown_hours`` (its cool-
    down window is still open). This lives in the rule rather than the dispatcher because both
    alerts share the same root cause; suppressing in the dispatcher would still write a
    ``BACKUP_STALE`` cool-down row, defeating the purpose.
    """
    backups_dir = storage_root / "backups"
    if not backups_dir.is_dir():
        return None

    suppression_cutoff = now - timedelta(hours=storage_unavailable_cooldown_hours)
    storage_unavail_recent = session.scalar(
        select(AlertSent.id)
        .where(AlertSent.alert_type == AlertType.STORAGE_UNAVAILABLE)
        .where(AlertSent.camera_id.is_(None))
        .where(AlertSent.sent_at >= suppression_cutoff)
        .limit(1),
    )
    if storage_unavail_recent is not None:
        return None

    backups = sorted(
        backups_dir.glob("cat_watcher-*.sqlite"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not backups:
        return None
    newest = backups[0]
    mtime = datetime.fromtimestamp(newest.stat().st_mtime, tz=UTC)
    if (now - mtime) <= timedelta(hours=threshold_hours):
        return None
    content = render_backup_stale(
        newest_backup_name=newest.name,
        mtime=mtime,
        threshold_hours=threshold_hours,
        storage_root=storage_root,
        tz_name=tz_name,
        now=now,
    )
    return AlertCandidate(alert_type=AlertType.BACKUP_STALE, camera_id=None, content=content)


# --- tick orchestrator + entry point --------------------------------------------------------------


def _upsert_alerts_heartbeat(session: Session, *, now: datetime) -> None:
    """Insert or update the ``alerts`` heartbeat row."""
    existing = session.get(Heartbeat, _AGENT_NAME)
    if existing is None:
        session.add(Heartbeat(agent_name=_AGENT_NAME, last_seen_at=now))
    else:
        existing.last_seen_at = now


def dispatch_candidate(
    cand: AlertCandidate,
    *,
    config: Config,
    engine: Engine,
    now: datetime,
) -> None:
    """Open a fresh session and dispatch one :class:`AlertCandidate` via :func:`dispatch_alert`.

    Public so the poller can reuse it for ``ALERTS_STUCK`` (Task 17 carve-out): the alert routing
    layer must funnel through this single helper to keep cool-down state honored uniformly.
    """
    with get_session(engine) as session:
        dispatch_alert(
            cand.alert_type,
            camera_id=cand.camera_id,
            content=cand.content,
            env=DispatchEnv(secrets=config.email, rules=config.alerts, session=session, now=now),
        )


def _dispatch_each(
    candidates: Sequence[AlertCandidate],
    *,
    config: Config,
    engine: Engine,
    now: datetime,
) -> None:
    """Dispatch each candidate in its own session so a SMTP error on one doesn't unravel the tick."""
    for cand in candidates:
        dispatch_candidate(cand, config=config, engine=engine, now=now)


def _camera_candidates(
    session: Session,
    cam: Camera,
    *,
    config: Config,
    now: datetime,
) -> Iterator[AlertCandidate]:
    """Yield the per-camera ``INACTIVITY`` + ``FREQUENCY`` candidates for ``cam`` (each may be None)."""
    tz_name = config.web.display_timezone
    public_url = config.web.public_url
    ic = evaluate_inactivity(
        cam,
        inactivity_hours=config.alerts.inactivity_hours,
        public_url=public_url,
        tz_name=tz_name,
        now=now,
    )
    if ic is not None:
        yield ic
    fc = evaluate_frequency(
        session,
        cam,
        window_hours=config.alerts.frequency_window_hours,
        threshold=config.alerts.frequency_threshold_count,
        public_url=public_url,
        tz_name=tz_name,
        now=now,
    )
    if fc is not None:
        yield fc


def _watchdog_candidates(
    session: Session,
    *,
    config: Config,
    now: datetime,
) -> Iterator[AlertCandidate]:
    """Yield ``POLLER_STUCK`` / ``WEB_DOWN`` / ``WEB_FLAPPING`` candidates for the tick."""
    tz_name = config.web.display_timezone
    public_url = config.web.public_url
    for agent_name, alert_type, stale_minutes in (
        ("poller", AlertType.POLLER_STUCK, config.alerts.poller_stuck_minutes),
        ("web", AlertType.WEB_DOWN, config.alerts.web_down_minutes),
    ):
        cand = evaluate_heartbeat_watchdog(
            session,
            alert_type=alert_type,
            agent_name=agent_name,
            stale_minutes=stale_minutes,
            public_url=public_url,
            tz_name=tz_name,
            now=now,
        )
        if cand is not None:
            yield cand
    wf = evaluate_web_flapping(
        session,
        window_minutes=config.alerts.web_flapping_window_minutes,
        threshold=config.alerts.web_flapping_threshold_count,
        log_path=config.internal_root / "logs" / "web.stderr.log",
        public_url=public_url,
        tz_name=tz_name,
        now=now,
    )
    if wf is not None:
        yield wf


def _evaluate_storage_block(
    session: Session,
    *,
    config: Config,
    now: datetime,
) -> AlertCandidate | None:
    """Evaluate ``BACKUP_STALE`` against the current ``alerts_sent`` table.

    Run *after* the storage candidates have already dispatched in their own sessions, so a freshly
    inserted ``STORAGE_UNAVAILABLE`` row triggers the BACKUP_STALE suppression for this same tick.
    """
    return evaluate_backup_stale(
        session,
        storage_root=config.storage_root,
        threshold_hours=config.alerts.backup_stale_hours,
        storage_unavailable_cooldown_hours=cooldown_for(AlertType.STORAGE_UNAVAILABLE, config.alerts),
        tz_name=config.web.display_timezone,
        now=now,
    )


def run_alerts_tick(*, config: Config, engine: Engine, now: datetime) -> None:
    """Run one alerts agent tick: rule evaluation, dispatch, heartbeat.

    Order matters for the storage block: ``STORAGE_UNAVAILABLE`` (when firing) must dispatch — and
    therefore land in ``alerts_sent`` — *before* :func:`evaluate_backup_stale` runs, so the
    BACKUP_STALE suppression check sees the row.
    """
    with get_session(engine) as session:
        session.add(AgentStart(agent_name=_AGENT_NAME, started_at=now))

    pre_storage: list[AlertCandidate] = []
    with get_session(engine) as session:
        for cam in session.scalars(select(Camera)).all():
            pre_storage.extend(_camera_candidates(session, cam, config=config, now=now))
        pre_storage.extend(_watchdog_candidates(session, config=config, now=now))
    _dispatch_each(pre_storage, config=config, engine=engine, now=now)

    storage_cands = evaluate_storage(
        storage_root=config.storage_root,
        threshold_fraction=config.alerts.disk_low_threshold_fraction,
        public_url=config.web.public_url,
        tz_name=config.web.display_timezone,
        now=now,
    )
    _dispatch_each(storage_cands, config=config, engine=engine, now=now)

    with get_session(engine) as session:
        bs = _evaluate_storage_block(session, config=config, now=now)
    if bs is not None:
        dispatch_candidate(bs, config=config, engine=engine, now=now)

    with get_session(engine) as session:
        _upsert_alerts_heartbeat(session, now=now)


class _ParsedArgs(argparse.Namespace):
    """Typed view over the parsed ``cat-watcher-alerts`` Namespace."""

    once: bool = False
    config: Path | None = None


def _parse_args(argv: Sequence[str] | None) -> _ParsedArgs:
    parser = argparse.ArgumentParser(prog="cat-watcher-alerts", description="Evaluate alert rules and dispatch one tick.")
    _ = parser.add_argument("--once", action="store_true", help="kept for LaunchAgent compat; the alerts agent is always one-shot")
    _ = parser.add_argument("--config", type=Path, default=None, help="Override config.toml path")
    return parser.parse_args(argv, namespace=_ParsedArgs())


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    The alerts agent intentionally skips the §4.13 storage wait — its DB lives on internal storage
    and the ``STORAGE_UNAVAILABLE`` rule depends on the agent running while the external drive is
    offline. :func:`evaluate_storage` does its own per-tick write probe.
    """
    args = _parse_args(argv)
    config = load_config(args.config)
    logging.basicConfig(level=config.log_level, format="%(levelname)s %(name)s: %(message)s")
    # The internal root must exist (DB + logs); the storage_root may not (drive offline). We
    # don't call ensure_storage_layout here because that requires storage_root to be a directory;
    # the alerts agent specifically tolerates an unmounted drive.
    config.internal_root.mkdir(parents=True, exist_ok=True)
    if storage_available(config.storage_root):
        ensure_storage_layout(internal_root=config.internal_root, storage_root=config.storage_root)

    engine = create_engine(f"sqlite:///{config.internal_root / _DB_FILENAME}")
    try:
        run_alerts_tick(config=config, engine=engine, now=datetime.now(UTC))
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":  # pragma: no cover  # entry-point
    sys.exit(main())

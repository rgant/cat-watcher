"""Tests for cat_watcher.alert_templates.

The renderers are pure functions: every test passes plain values and asserts the full subject + body
text. No mocks, no DB, no time-of-day flakiness — ``now`` is always pinned. Snapshot-style equality
catches accidental whitespace shifts; the ``_macos_summary_under_cap`` parameterized test enforces
the spec §4.14 ≤120-character ceiling on ``macos_summary``.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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
from cat_watcher.db import AlertType, PollStatus

_NOW = datetime(2026, 5, 1, 13, 42, 11, tzinfo=UTC)  # 09:42:11 EDT (-04:00)
_TZ_NY = "America/New_York"
_PUBLIC_URL = "http://litterbox.local:8000"


def test_inactivity_no_clips_branch_renders_full_body() -> None:
    """Branch ``no clips received`` produces the spec §4.14 example body verbatim."""
    last_poll = datetime(2026, 5, 1, 13, 30, 0, tzinfo=UTC)  # 09:30:00 EDT — 12m ago
    last_clip = datetime(2026, 5, 1, 1, 24, 0, tzinfo=UTC)  # 21:24:00 (prev day) EDT — 12h 18m ago
    last_cat = datetime(2026, 5, 1, 1, 24, 0, tzinfo=UTC)

    content = render_inactivity(
        camera_display_name="Pantry Litter Box Camera",
        branch=INACTIVITY_BRANCH_NO_CLIPS,
        last_polled_at=last_poll,
        last_clip_at=last_clip,
        last_cat_seen_at=last_cat,
        poll_status=PollStatus.OK,
        public_url=_PUBLIC_URL,
        tz_name=_TZ_NY,
        now=_NOW,
    )

    assert content.subject == "[cat-watcher] INACTIVITY: Pantry Litter Box Camera"
    assert content.body == (
        "Branch:         no clips received\n"
        "Last poll:      2026-05-01 09:30:00 EDT (-04:00) — 12m ago\n"
        "Last clip:      2026-04-30 21:24:00 EDT (-04:00) — 12h 18m ago\n"
        "Last cat seen:  2026-04-30 21:24:00 EDT (-04:00) — 12h 18m ago\n"
        "Poll status:    ok\n"
        f"Web UI:         {_PUBLIC_URL}\n"
    )
    assert content.macos_summary == "Pantry Litter Box Camera: no clips received"


def test_inactivity_no_cats_branch_uses_correct_label() -> None:
    """Branch label is the static ``no cats seen`` constant (not synthesized)."""
    content = render_inactivity(
        camera_display_name="Pantry",
        branch=INACTIVITY_BRANCH_NO_CATS,
        last_polled_at=_NOW - timedelta(minutes=1),
        last_clip_at=_NOW - timedelta(minutes=30),
        last_cat_seen_at=_NOW - timedelta(hours=13),
        poll_status=PollStatus.OK,
        public_url=_PUBLIC_URL,
        tz_name=_TZ_NY,
        now=_NOW,
    )

    assert "Branch:         no cats seen\n" in content.body
    assert content.macos_summary == "Pantry: no cats seen"


def test_inactivity_unreachable_branch_includes_since_timestamp() -> None:
    """``inactivity_branch_unreachable`` embeds the unreachable-since absolute timestamp."""
    since = datetime(2026, 5, 1, 13, 30, 0, tzinfo=UTC)  # 09:30:00 EDT
    branch = inactivity_branch_unreachable(since, _TZ_NY)

    content = render_inactivity(
        camera_display_name="Pantry",
        branch=branch,
        last_polled_at=since,
        last_clip_at=None,
        last_cat_seen_at=None,
        poll_status=PollStatus.UNREACHABLE,
        public_url=_PUBLIC_URL,
        tz_name=_TZ_NY,
        now=_NOW,
    )

    assert "Branch:         camera unreachable since 2026-05-01 09:30:00 EDT (-04:00)\n" in content.body
    assert "Last clip:      never\n" in content.body
    assert "Last cat seen:  never\n" in content.body
    assert "Poll status:    unreachable\n" in content.body


def test_frequency_renders_count_and_window_in_subject() -> None:
    """Subject embeds ``({count} in {window_hours}h)`` per spec §4.14."""
    first = datetime(2026, 5, 1, 8, 45, 0, tzinfo=UTC)  # 04:45 EDT
    latest = datetime(2026, 5, 1, 13, 58, 11, tzinfo=UTC)  # 09:58:11 EDT

    content = render_frequency(
        camera_display_name="Pantry Litter Box Camera",
        count=12,
        window_hours=6,
        threshold=8,
        first_in_window=first,
        latest_in_window=latest,
        public_url=_PUBLIC_URL,
        tz_name=_TZ_NY,
    )

    assert content.subject == "[cat-watcher] FREQUENCY: Pantry Litter Box Camera (12 in 6h)"
    assert "Cat-positive clips in last 6h:   12\n" in content.body
    assert "Threshold:                       8\n" in content.body
    assert "First in window:                 2026-05-01 04:45:00 EDT (-04:00)\n" in content.body
    assert "Latest:                          2026-05-01 09:58:11 EDT (-04:00)\n" in content.body
    assert content.macos_summary.startswith("Pantry Litter Box Camera: 12 cat-positive clips in 6h")


@pytest.mark.parametrize(
    ("alert_type", "agent_name"),
    [
        (AlertType.POLLER_STUCK, "poller"),
        (AlertType.WEB_DOWN, "web"),
    ],
)
def test_heartbeat_watchdog_uses_shared_template(alert_type: AlertType, agent_name: str) -> None:
    """Watchdog types share the spec §4.14 body shape; the subject + agent name vary by type.

    Two parametrizations is enough to prove the alert_type / agent_name fields are interpolated
    rather than hard-coded — a third (``ALERTS_STUCK``) would re-test the same code path.
    """
    last_hb = _NOW - timedelta(minutes=14)

    content = render_heartbeat_watchdog(
        alert_type=alert_type,
        agent_name=agent_name,
        last_heartbeat=last_hb,
        public_url=_PUBLIC_URL,
        tz_name=_TZ_NY,
        now=_NOW,
    )

    assert content.subject == f"[cat-watcher] {alert_type.value}"
    assert f"Agent:           {agent_name}\n" in content.body
    assert "— 14m ago" in content.body
    assert agent_name in content.macos_summary
    assert "14m ago" in content.macos_summary


def test_web_flapping_includes_log_tail_section() -> None:
    """Body trails with ``--- Last 50 lines of web.stderr.log ---`` then the literal tail."""
    first = _NOW - timedelta(minutes=27)  # 09:15:11 EDT
    latest = _NOW - timedelta(minutes=1)
    log_tail = "Traceback (most recent call last):\n  File 'x.py' line 1\nValueError: boom\n"

    content = render_web_flapping(
        restart_count=6,
        window_minutes=30,
        first_restart=first,
        last_restart=latest,
        log_tail=log_tail,
        public_url=_PUBLIC_URL,
        tz_name=_TZ_NY,
    )

    assert content.subject == "[cat-watcher] WEB_FLAPPING (6 restarts in last 30m)"
    assert "Restarts in last 30m:   6\n" in content.body
    assert "Agent:                  web\n" in content.body
    assert "--- Last 50 lines of web.stderr.log ---" in content.body
    assert log_tail in content.body
    assert content.macos_summary == "web: 6 restarts in last 30m"


def test_disk_low_renders_fraction_and_threshold() -> None:
    """``Free`` line prints ``GB / GB (fraction%)``; subject embeds ``%`` and ``storage_root``."""
    free = 164 * 1_000_000_000
    total = 2_000 * 1_000_000_000
    storage_root = Path("/Volumes/Data")

    content = render_disk_low(
        storage_root=storage_root,
        free_bytes=free,
        total_bytes=total,
        threshold_fraction=0.10,
        public_url=_PUBLIC_URL,
    )

    assert content.subject == "[cat-watcher] DISK_LOW (8.2% free on /Volumes/Data)"
    assert "Mount:           /Volumes/Data\n" in content.body
    assert "Free:            164 GB / 2000 GB (8.2%)\n" in content.body
    assert "Threshold:       10%\n" in content.body
    assert content.macos_summary == "/Volumes/Data: 8.2% free (threshold 10%)"


def test_disk_low_handles_zero_total_gracefully() -> None:
    """``total_bytes==0`` reports 0.0% and does not divide-by-zero."""
    content = render_disk_low(
        storage_root=Path("/dev/null"),
        free_bytes=0,
        total_bytes=0,
        threshold_fraction=0.10,
        public_url=_PUBLIC_URL,
    )
    assert "0.0%" in content.subject
    assert "0 GB / 0 GB (0.0%)" in content.body


def test_storage_unavailable_includes_first_detected_with_relative() -> None:
    """``First detected`` line carries an absolute timestamp and ``— 0s ago`` when ``now == first_detected``."""
    content = render_storage_unavailable(
        storage_root=Path("/Volumes/Data/cat-watcher"),
        first_detected=_NOW,
        tz_name=_TZ_NY,
        now=_NOW,
    )

    assert content.subject == "[cat-watcher] STORAGE_UNAVAILABLE (external drive not mounted)"
    assert "storage_root:  /Volumes/Data/cat-watcher\n" in content.body
    assert "First detected: 2026-05-01 09:42:11 EDT (-04:00) — 0s ago\n" in content.body
    assert "Probable cause:" in content.body
    assert content.macos_summary == "storage_root unavailable: /Volumes/Data/cat-watcher"


def test_backup_stale_reports_hours_since_mtime() -> None:
    """``hours_since`` is derived from ``now - mtime`` and rounds down to whole hours."""
    mtime = _NOW - timedelta(hours=38, minutes=42)

    content = render_backup_stale(
        newest_backup_name="cat_watcher-2026-04-29.sqlite",
        mtime=mtime,
        threshold_hours=36,
        storage_root=Path("/Volumes/Data/cat-watcher"),
        tz_name=_TZ_NY,
        now=_NOW,
    )

    assert content.subject == "[cat-watcher] BACKUP_STALE (no backup in 38h)"
    assert "Newest backup:  cat_watcher-2026-04-29.sqlite\n" in content.body
    assert "— 38h 42m ago\n" in content.body
    assert "Threshold:      36h\n" in content.body
    assert "Storage root:   /Volumes/Data/cat-watcher (mounted: yes)\n" in content.body
    assert content.macos_summary == "No backup in 38h (threshold 36h)"


@pytest.mark.parametrize(
    "content",
    [
        render_inactivity(
            camera_display_name="A" * 200,
            branch=INACTIVITY_BRANCH_NO_CATS,
            last_polled_at=_NOW,
            last_clip_at=_NOW,
            last_cat_seen_at=_NOW,
            poll_status=PollStatus.OK,
            public_url=_PUBLIC_URL,
            tz_name=_TZ_NY,
            now=_NOW,
        ),
        render_frequency(
            camera_display_name="B" * 200,
            count=99,
            window_hours=6,
            threshold=8,
            first_in_window=_NOW,
            latest_in_window=_NOW,
            public_url=_PUBLIC_URL,
            tz_name=_TZ_NY,
        ),
        render_heartbeat_watchdog(
            alert_type=AlertType.POLLER_STUCK,
            agent_name="poller",
            last_heartbeat=_NOW - timedelta(minutes=20),
            public_url=_PUBLIC_URL,
            tz_name=_TZ_NY,
            now=_NOW,
        ),
        render_web_flapping(
            restart_count=5,
            window_minutes=30,
            first_restart=_NOW - timedelta(minutes=29),
            last_restart=_NOW,
            log_tail="x" * 5_000,
            public_url=_PUBLIC_URL,
            tz_name=_TZ_NY,
        ),
        render_disk_low(
            storage_root=Path("/" + "very-long-mount-name/" * 10),
            free_bytes=1,
            total_bytes=10,
            threshold_fraction=0.10,
            public_url=_PUBLIC_URL,
        ),
        render_storage_unavailable(
            storage_root=Path("/" + "long-path/" * 15),
            first_detected=_NOW,
            tz_name=_TZ_NY,
            now=_NOW,
        ),
        render_backup_stale(
            newest_backup_name="cat_watcher-2026-04-29.sqlite",
            mtime=_NOW - timedelta(hours=40),
            threshold_hours=36,
            storage_root=Path("/" + "long-path/" * 15),
            tz_name=_TZ_NY,
            now=_NOW,
        ),
    ],
    ids=[
        "inactivity",
        "frequency",
        "heartbeat",
        "web_flapping",
        "disk_low",
        "storage_unavailable",
        "backup_stale",
    ],
)
def test_macos_summary_never_exceeds_cap(content: AlertContent) -> None:
    """Spec §4.14 caps macOS notification summary at 120 chars; renderers truncate or stay under."""
    assert len(content.macos_summary) <= 120


def test_relative_ago_buckets_seconds_minutes_hours() -> None:
    """``_ago`` rolls into ``s ago`` / ``m ago`` / ``h Xm ago`` at 60s and 60m boundaries."""
    cases: list[tuple[timedelta, str]] = [
        (timedelta(seconds=0), "0s ago"),
        (timedelta(seconds=45), "45s ago"),
        (timedelta(seconds=59), "59s ago"),
        (timedelta(minutes=1), "1m ago"),
        (timedelta(minutes=12), "12m ago"),
        (timedelta(minutes=59), "59m ago"),
        (timedelta(hours=1), "1h 0m ago"),
        (timedelta(hours=14, minutes=18), "14h 18m ago"),
        (timedelta(hours=38, minutes=42), "38h 42m ago"),
    ]
    for delta, expected in cases:
        prior = _NOW - delta
        content = render_heartbeat_watchdog(
            alert_type=AlertType.POLLER_STUCK,
            agent_name="poller",
            last_heartbeat=prior,
            public_url=_PUBLIC_URL,
            tz_name=_TZ_NY,
            now=_NOW,
        )
        assert f"— {expected}" in content.body, f"delta={delta!r}, body={content.body!r}"


def test_macos_summary_truncation_appends_visible_ellipsis() -> None:
    """An over-cap summary lands at exactly 120 chars and ends with ``…`` so the cut is visible.

    The companion ``test_macos_summary_never_exceeds_cap`` only verifies length ≤ 120; this pins
    the *visible*-truncation contract that ``_capped_summary`` documents (silent truncation would
    still satisfy the length check but hide content from the operator).
    """
    long_name = "P" * 200  # well over the 120-char cap once "Pantry: no cats seen" prefix is implied
    content = render_inactivity(
        camera_display_name=long_name,
        branch=INACTIVITY_BRANCH_NO_CATS,
        last_polled_at=_NOW,
        last_clip_at=_NOW,
        last_cat_seen_at=_NOW,
        poll_status=PollStatus.OK,
        public_url=_PUBLIC_URL,
        tz_name=_TZ_NY,
        now=_NOW,
    )
    assert len(content.macos_summary) == 120
    assert content.macos_summary.endswith("…")

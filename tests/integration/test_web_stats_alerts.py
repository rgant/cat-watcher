"""Integration tests for the cat-watcher cameras / stats / alerts pages (Task 24).

Covers ``GET /cameras`` (per-camera health table + recent alerts), ``GET /stats`` (30-day daily
aggregation across all cameras with manual-label override applied via
``COALESCE(manual_has_cat, has_cat)``), and ``GET /alerts`` (last 30 days of dispatched alerts with
email/macOS delivery flags). Auth is exercised exhaustively in ``test_web_health.py``; this module
attaches a constant ``Authorization`` header.
"""

import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING

from cat_watcher.db import AlertSent, AlertType, Camera, Clip, PollStatus

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from fastapi.testclient import TestClient
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


_AUTH_HEADER = {"Authorization": f"Basic {base64.b64encode(b'admin:pw').decode()}"}


def _persist_cameras(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    internal_root: Path,
    cameras: list[Camera],
) -> list[int]:
    with db_session_factory(internal_root) as session:
        for cam in cameras:
            session.add(cam)
        session.flush()
        return [cam.id for cam in cameras]


def _persist(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    internal_root: Path,
    rows: list[Clip] | list[AlertSent],
) -> None:
    with db_session_factory(internal_root) as session:
        for row in rows:
            session.add(row)


def _make_clip(
    *,
    camera_id: int,
    start_ts: datetime,
    has_cat: bool,
    manual_has_cat: bool | None = None,
) -> Clip:
    """Build a Clip row; ``source_filename`` derives from the full ``start_ts`` (date + time + µs).

    Each test can mint many clips per camera by varying ``start_ts`` alone — the
    ``(camera_id, source_filename)`` uniqueness constraint is satisfied without per-test
    bookkeeping, even when two seeded clips share the same time-of-day across different dates.
    """
    fname = f"{start_ts.strftime('%Y%m%d-%H%M%S%f')}.mp4"
    return Clip(
        camera_id=camera_id,
        source_filename=fname,
        start_ts=start_ts,
        end_ts=start_ts + timedelta(seconds=10),
        duration_seconds=10.0,
        file_path=f"clips/{fname}",
        thumb_path=f"thumbs/{fname}.jpg",
        file_size_bytes=1024,
        has_cat=has_cat,
        manual_has_cat=manual_has_cat,
        detector_version="yolov11n@deadbeef",
        ingested_at=start_ts,
    )


def test_cameras_page_lists_all_cameras_with_status(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``GET /cameras`` renders both seeded cameras' display names and ``poll_status`` values."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _persist_cameras(
        db_session_factory,
        internal_root,
        [
            Camera(name="pantry", display_name="Pantry Litter Box", host="cam1.example.com", poll_status=PollStatus.OK),
            Camera(name="bath", display_name="Bath Litter Box", host="cam2.example.com", poll_status=PollStatus.UNREACHABLE),
        ],
    )

    with web_test_client(config) as client:
        response = client.get("/cameras", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "Pantry Litter Box" in response.text
    assert "Bath Litter Box" in response.text
    assert "ok" in response.text
    assert "unreachable" in response.text


def test_cameras_page_renders_poll_status_since_elapsed(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A non-OK camera with ``poll_status_since`` set surfaces an elapsed-time string in the page.

    The exact format is presentation; the route's contract is that operators can see "how long has
    this been broken?" without doing arithmetic. We pin the ISO timestamp itself so the test isn't
    tied to a particular humanizer format, plus the textual ``unreachable`` status badge so the row
    is recognizable.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    since = datetime.now(UTC) - timedelta(hours=3, minutes=15)
    _ = _persist_cameras(
        db_session_factory,
        internal_root,
        [
            Camera(
                name="pantry",
                display_name="Pantry",
                host="cam.example.com",
                poll_status=PollStatus.UNREACHABLE,
                poll_status_since=since,
                poll_error="connect timeout",
            ),
        ],
    )

    with web_test_client(config) as client:
        response = client.get("/cameras", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "unreachable" in response.text
    assert since.isoformat() in response.text
    assert "connect timeout" in response.text


def test_cameras_page_includes_recent_alerts_for_each_camera(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Each camera's most recent alerts surface alongside its row on ``/cameras``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    [cam_id] = _persist_cameras(
        db_session_factory,
        internal_root,
        [Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)],
    )
    now = datetime.now(UTC)
    _persist(
        db_session_factory,
        internal_root,
        [
            AlertSent(
                alert_type=AlertType.INACTIVITY,
                camera_id=cam_id,
                sent_at=now - timedelta(hours=2),
                subject="No cat seen for 24h",
                body="alert body",
            ),
        ],
    )

    with web_test_client(config) as client:
        response = client.get("/cameras", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "INACTIVITY" in response.text
    assert "No cat seen for 24h" in response.text


def test_stats_aggregates_clip_counts_per_camera_per_day(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``/stats`` shows total + cat-positive counts per camera per day for the last 30 days.

    Two cameras seeded:

    * pantry: 3 clips today, 2 of which are cat-positive via ``has_cat=True`` (no manual override).
    * bath: 1 clip yesterday with ``has_cat=False`` but ``manual_has_cat=True`` — the override must
      flip the cat-positive count to 1.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    pantry_id, bath_id = _persist_cameras(
        db_session_factory,
        internal_root,
        [
            Camera(name="pantry", display_name="Pantry", host="cam1.example.com", poll_status=PollStatus.OK),
            Camera(name="bath", display_name="Bath", host="cam2.example.com", poll_status=PollStatus.OK),
        ],
    )
    today_anchor = datetime.now(UTC).replace(microsecond=0)
    yesterday_anchor = today_anchor - timedelta(days=1)
    _persist(
        db_session_factory,
        internal_root,
        [
            *(
                _make_clip(camera_id=pantry_id, start_ts=today_anchor - timedelta(minutes=i * 10), has_cat=has_cat)
                for i, has_cat in enumerate([True, True, False])
            ),
            _make_clip(camera_id=bath_id, start_ts=yesterday_anchor, has_cat=False, manual_has_cat=True),
        ],
    )

    with web_test_client(config) as client:
        response = client.get("/stats", headers=_AUTH_HEADER)

    assert response.status_code == 200
    pantry_row = _row_for(response.text, "Pantry", today_anchor.date().isoformat())
    bath_row = _row_for(response.text, "Bath", yesterday_anchor.date().isoformat())
    assert pantry_row is not None, "Expected a Pantry row for today in /stats"
    assert bath_row is not None, "Expected a Bath row for yesterday in /stats"
    assert "3" in pantry_row, f"Pantry total clips=3 missing in row: {pantry_row}"
    assert "2" in pantry_row, f"Pantry cat-positive=2 missing in row: {pantry_row}"
    # 1 / 1 — total and cat both equal 1 thanks to the manual override.
    assert "1" in bath_row


def _row_for(body: str, camera_display_name: str, date_iso: str) -> str | None:
    """Find the single ``<tr>`` block on the stats page containing both ``camera_display_name`` and ``date_iso``.

    Returns the substring spanning the row, or ``None`` if no matching row is found. Used to scope
    substring assertions to the right row instead of the whole page (e.g. so a ``2`` from a
    different row doesn't satisfy the cat-count check).
    """
    cursor = 0
    while True:
        start = body.find("<tr", cursor)
        if start == -1:
            return None
        end = body.find("</tr>", start)
        if end == -1:
            return None
        row = body[start : end + len("</tr>")]
        if camera_display_name in row and date_iso in row:
            return row
        cursor = end + len("</tr>")


def test_alerts_page_lists_recent_alerts(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``/alerts`` lists alerts dispatched in the last 30 days with type, subject, camera, and delivery flags.

    The ``email_ok=True, macos_ok=False`` seed crosses the only branch in each delivery cell — one ✓
    and one ✗ on the same row pin both states with one assertion. Without this, the route could drop
    the delivery columns entirely and the rest of the suite would stay green.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    [cam_id] = _persist_cameras(
        db_session_factory,
        internal_root,
        [Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)],
    )
    now = datetime.now(UTC)
    _persist(
        db_session_factory,
        internal_root,
        [
            AlertSent(
                alert_type=AlertType.INACTIVITY,
                camera_id=cam_id,
                sent_at=now - timedelta(days=1),
                subject="Pantry inactivity 24h",
                body="alert body",
                email_ok=True,
                macos_ok=False,
            ),
        ],
    )

    with web_test_client(config) as client:
        response = client.get("/alerts", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "INACTIVITY" in response.text
    assert "Pantry inactivity 24h" in response.text
    assert "Pantry" in response.text
    # Asserting on the full ``<td data-label="Email">✓</td>`` substring scopes the check to the
    # right column so a stray ``✓`` elsewhere in the page can't satisfy the test.
    assert '<td data-label="Email">✓</td>' in response.text
    assert '<td data-label="macOS">✗</td>' in response.text


def test_alerts_page_renders_em_dash_for_non_camera_alerts(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Alerts with ``camera_id IS NULL`` (e.g. WEB_DOWN, DISK_LOW) show "—" in the camera column."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    now = datetime.now(UTC)
    _persist(
        db_session_factory,
        internal_root,
        [
            AlertSent(
                alert_type=AlertType.DISK_LOW,
                camera_id=None,
                sent_at=now - timedelta(hours=4),
                subject="Internal disk under 10%",
                body="alert body",
            ),
        ],
    )

    with web_test_client(config) as client:
        response = client.get("/alerts", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "DISK_LOW" in response.text
    assert "Internal disk under 10%" in response.text
    # The em-dash placeholder is the contract; tested verbatim so the route can't accidentally fall
    # back to ``None`` / empty / ``-`` (which would silently break operator scanability).
    assert "—" in response.text


def test_alerts_page_omits_alerts_older_than_30_days(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """An alert dispatched > 30 days ago does not appear; one inside the window does.

    Pins the 30-day cutoff documented in spec §4.7 so the route doesn't quietly widen the window
    (which would let stale alerts crowd out current ones in a long-running deployment).
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    now = datetime.now(UTC)
    _persist(
        db_session_factory,
        internal_root,
        [
            AlertSent(
                alert_type=AlertType.INACTIVITY,
                sent_at=now - timedelta(days=45),
                subject="Stale alert from 45 days ago",
                body="alert body",
            ),
            AlertSent(
                alert_type=AlertType.FREQUENCY,
                sent_at=now - timedelta(days=1),
                subject="Recent alert from yesterday",
                body="alert body",
            ),
        ],
    )

    with web_test_client(config) as client:
        response = client.get("/alerts", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "Stale alert from 45 days ago" not in response.text
    assert "Recent alert from yesterday" in response.text


def test_alerts_page_orders_newest_first(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Alerts render in ``sent_at DESC`` order so the newest dispatch is at the top of the page.

    Pins the contract in case a refactor flips the ``order_by(desc(AlertSent.sent_at))`` clause.
    Operators scan this page from the top down looking for "what just fired"; reverse order would
    bury the latest event under historical noise.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    now = datetime.now(UTC)
    _persist(
        db_session_factory,
        internal_root,
        [
            AlertSent(
                alert_type=AlertType.INACTIVITY,
                sent_at=now - timedelta(days=2),
                subject="Older alert subject",
                body="alert body",
            ),
            AlertSent(
                alert_type=AlertType.FREQUENCY,
                sent_at=now - timedelta(hours=1),
                subject="Newer alert subject",
                body="alert body",
            ),
        ],
    )

    with web_test_client(config) as client:
        response = client.get("/alerts", headers=_AUTH_HEADER)

    assert response.status_code == 200
    newer_pos = response.text.find("Newer alert subject")
    older_pos = response.text.find("Older alert subject")
    assert newer_pos != -1, "Newer alert subject missing from /alerts"
    assert older_pos != -1, "Older alert subject missing from /alerts"
    assert newer_pos < older_pos, "Expected newer alert to render before older alert in /alerts"


def test_stats_omits_clips_older_than_30_days(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A clip with ``start_ts`` 45 days back doesn't appear in ``/stats``; one yesterday does.

    Symmetric to ``test_alerts_page_omits_alerts_older_than_30_days``. The 30-day cutoff lives in
    two routes (stats + alerts) sharing a single ``_HISTORY_DAYS`` constant; without this test, a
    regression that flips ``>=`` to ``>`` in the stats query — or widens the window only there —
    would slip through the suite.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    [cam_id] = _persist_cameras(
        db_session_factory,
        internal_root,
        [Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)],
    )
    now = datetime.now(UTC)
    stale_ts = now - timedelta(days=45)
    fresh_ts = now - timedelta(days=1)
    _persist(
        db_session_factory,
        internal_root,
        [
            _make_clip(camera_id=cam_id, start_ts=stale_ts, has_cat=True),
            _make_clip(camera_id=cam_id, start_ts=fresh_ts, has_cat=True),
        ],
    )

    with web_test_client(config) as client:
        response = client.get("/stats", headers=_AUTH_HEADER)

    assert response.status_code == 200
    # The stale clip's ISO date should be absent — its row was filtered out by the cutoff. The fresh
    # clip's date must be present so the test fails if the route accidentally drops *all* clips
    # (e.g. inverted condition).
    assert stale_ts.date().isoformat() not in response.text
    assert fresh_ts.date().isoformat() in response.text


def test_cameras_page_caps_recent_alerts_at_five(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Only the 5 most-recent alerts per camera render on ``/cameras`` (``_CAMERA_RECENT_ALERTS_LIMIT``).

    Seeds 7 alerts spaced one hour apart so each has a distinct ``sent_at`` ordering. The newest 5
    (subjects ``alert-0`` through ``alert-4`` — index 0 is the smallest age, hence newest) must
    render; the two oldest (``alert-5``, ``alert-6``) must not. Without this guard the limit could
    silently widen — or vanish — and the page would just keep growing until the camera-card section
    became unscrollable on mobile.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    [cam_id] = _persist_cameras(
        db_session_factory,
        internal_root,
        [Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)],
    )
    now = datetime.now(UTC)
    seeded_alerts = [
        AlertSent(
            alert_type=AlertType.FREQUENCY,
            camera_id=cam_id,
            sent_at=now - timedelta(hours=i + 1),
            subject=f"alert-{i}",
            body="alert body",
        )
        for i in range(7)
    ]
    _persist(db_session_factory, internal_root, seeded_alerts)

    with web_test_client(config) as client:
        response = client.get("/cameras", headers=_AUTH_HEADER)

    assert response.status_code == 200
    # Newest 5 (smallest hour-offsets) must appear.
    for i in range(5):
        assert f"alert-{i}" in response.text, f"Expected alert-{i} (newest 5) on /cameras"
    # Two oldest must be cut off by the limit.
    assert "alert-5" not in response.text
    assert "alert-6" not in response.text

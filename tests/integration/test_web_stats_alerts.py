"""Integration tests for the cat-watcher cameras / stats / alerts pages.

Covers ``GET /cameras`` (per-camera health table + recent alerts), ``GET /stats`` (per-cat visits of
7 local days, and the 30-day daily clip counts per camera from the ``clip_label_summary`` view), and
``GET /alerts`` (last 30 days of dispatched alerts with email/macOS delivery flags). Auth is
exercised exhaustively in ``test_web_health.py``; this module attaches a constant ``Authorization``
header.
"""

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path  # noqa: TC003  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

from db_helpers import AUTH_HEADER, add_clip, apply_clip_label_summary_view, recorded_statements, seed_cat_subject, tag_clip_frame

from cat_watcher.db import AlertSent, AlertType, Base, Camera, Clip, PollStatus, Subject, create_engine, get_session
from cat_watcher.web.stats_routes import _cat_stat_rows, _stat_days

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from fastapi.testclient import TestClient
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


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
        detector_version="yolov11n@deadbeef",
        ingested_at=start_ts,
    )


def test_cameras_page_lists_all_cameras_with_status(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with alembic_web_test_client(config) as client:
        response = client.get("/cameras", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "Pantry Litter Box" in response.text
    assert "Bath Litter Box" in response.text
    assert "ok" in response.text
    assert "unreachable" in response.text


def test_cameras_page_renders_poll_status_since_elapsed(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with alembic_web_test_client(config) as client:
        response = client.get("/cameras", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "unreachable" in response.text
    assert since.isoformat() in response.text
    assert "connect timeout" in response.text


def test_cameras_page_includes_recent_alerts_for_each_camera(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with alembic_web_test_client(config) as client:
        response = client.get("/cameras", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "INACTIVITY" in response.text
    assert "No cat seen for 24h" in response.text


def test_stats_aggregates_clip_counts_per_camera_per_day(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``/stats`` shows total + cat-positive counts per camera per day for the last 30 days.

    Two cameras seeded:

    * pantry: 3 clips today, 2 of which are cat-positive via ``has_cat=True`` (no operator review).
    * bath: 1 clip yesterday with ``has_cat=False`` but reviewed and tagged with a cat subject — the
      operator override via the review queue must flip the cat-positive count to 1.
    """
    internal_root, storage_root = storage_dirs
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
    # Bath clip: detector says no-cat; operator review will flip it to cat-positive via tagging.
    bath_clip = _make_clip(camera_id=bath_id, start_ts=yesterday_anchor, has_cat=False)
    _persist(
        db_session_factory,
        internal_root,
        [
            *(
                _make_clip(camera_id=pantry_id, start_ts=today_anchor - timedelta(minutes=i * 10), has_cat=has_cat)
                for i, has_cat in enumerate([True, True, False])
            ),
            bath_clip,
        ],
    )
    # Tag the bath clip with a cat subject and mark it reviewed so effective_has_cat becomes True.
    _tag_clips_as_cat(internal_root, [(bath_clip.id, yesterday_anchor)])

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        response = client.get("/stats", headers=AUTH_HEADER)

    assert response.status_code == 200
    # The Date column is a display_timezone day, so a UTC-date lookup fails on any run after
    # 20:00 Eastern — a failure on a date unrelated to the change.
    pantry_row = _row_for(response.text, "Pantry", _local_day(today_anchor))
    bath_row = _row_for(response.text, "Bath", _local_day(yesterday_anchor))
    assert pantry_row is not None, "Expected a Pantry row for today in /stats"
    assert bath_row is not None, "Expected a Bath row for yesterday in /stats"
    assert "3" in pantry_row, f"Pantry total clips=3 missing in row: {pantry_row}"
    assert "2" in pantry_row, f"Pantry cat-positive=2 missing in row: {pantry_row}"
    # 1 / 1 — total and cat both equal 1 thanks to the operator review-queue override.
    assert "1" in bath_row


_DISPLAY_TZ = ZoneInfo("America/New_York")


def _local_day(value: datetime) -> str:
    """Return ``value``'s calendar day in the configured display zone, which is what /stats buckets by."""
    return value.astimezone(_DISPLAY_TZ).date().isoformat()


def _row_positions(body: str, rows: list[tuple[str, str]]) -> list[int]:
    """Return each ``(camera_display_name, date_iso)`` row's offset in ``body``, in the order given."""
    offsets: list[int] = []
    for name, day_iso in rows:
        row = _row_for(body, name, day_iso)
        assert row is not None, f"missing row for {name} on {day_iso}"
        offsets.append(body.index(row))
    return offsets


def _table(body: str, table_id: str) -> str:
    """Return the ``<table>`` element with ``id="table_id"`` in ``body``."""
    start = body.index(f'<table id="{table_id}"')
    return body[start : body.index("</table>", start)]


def _row_for(body: str, camera_display_name: str, date_iso: str) -> str | None:
    """Find the single ``<tr>`` block on the stats page containing both ``camera_display_name`` and ``date_iso``.

    Returns the substring spanning the row, or ``None`` if no matching row is found. Used to scope
    substring assertions to the right row instead of the whole page (e.g. so a ``2`` from a
    different row does not satisfy the cat-count check). The search covers the per-camera table only,
    because a per-cat last-visit cell can also hold a camera name and a date.
    """
    body = _table(body, "camera-stats")
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
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with alembic_web_test_client(config) as client:
        response = client.get("/alerts", headers=AUTH_HEADER)

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
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Alerts with ``camera_id IS NULL``, such as WEB_DOWN and DISK_LOW, show a dash in the Source column."""
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

    with alembic_web_test_client(config) as client:
        response = client.get("/alerts", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "DISK_LOW" in response.text
    assert "Internal disk under 10%" in response.text
    # The em-dash placeholder is the contract; tested verbatim so the route can't accidentally fall
    # back to ``None`` / empty / ``-`` (which would silently break operator scanability).
    assert "—" in response.text


def test_alerts_page_source_column_names_cat_camera_or_placeholder(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A per-cat alert shows the cat name, a camera alert the camera name, and any other alert a dash."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    now = datetime.now(UTC)
    with db_session_factory(internal_root) as session:
        cam = Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)
        cat = Subject(slug="marcel", display_name="Marcel", kind="cat", display_order=1)
        session.add_all([cam, cat])
        session.flush()
        session.add_all(
            [
                AlertSent(
                    alert_type=AlertType.INACTIVITY,
                    subject_id=cat.id,
                    sent_at=now - timedelta(hours=1),
                    subject="cat alert",
                    body="b",
                ),
                AlertSent(
                    alert_type=AlertType.INACTIVITY,
                    camera_id=cam.id,
                    sent_at=now - timedelta(hours=2),
                    subject="camera alert",
                    body="b",
                ),
                AlertSent(alert_type=AlertType.DISK_LOW, sent_at=now - timedelta(hours=3), subject="disk alert", body="b"),
            ],
        )

    with alembic_web_test_client(config) as client:
        response = client.get("/alerts", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert '<th scope="col">Source</th>' in response.text
    cells = re.findall(r'<td data-label="Source">([^<]*)</td>', response.text)
    assert cells == ["Marcel", "Pantry", "—"]


def test_alerts_page_omits_alerts_older_than_30_days(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with alembic_web_test_client(config) as client:
        response = client.get("/alerts", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "Stale alert from 45 days ago" not in response.text
    assert "Recent alert from yesterday" in response.text


def test_alerts_page_orders_newest_first(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with alembic_web_test_client(config) as client:
        response = client.get("/alerts", headers=AUTH_HEADER)

    assert response.status_code == 200
    newer_pos = response.text.find("Newer alert subject")
    older_pos = response.text.find("Older alert subject")
    assert newer_pos != -1, "Newer alert subject missing from /alerts"
    assert older_pos != -1, "Older alert subject missing from /alerts"
    assert newer_pos < older_pos, "Expected newer alert to render before older alert in /alerts"


def test_stats_omits_clips_older_than_30_days(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A clip with ``start_ts`` 45 days back does not appear in ``/stats``. A clip from yesterday does.

    This test pairs with ``test_alerts_page_omits_alerts_older_than_30_days``. The stats route and
    the alerts route share one ``HISTORY_DAYS`` constant. Without this test, two changes can pass
    the suite. One flips ``>=`` to ``>`` in the stats query. The other widens the window there only.
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

    with alembic_web_test_client(config) as client:
        response = client.get("/stats", headers=AUTH_HEADER)

    assert response.status_code == 200
    # The stale clip's ISO date should be absent — its row was filtered out by the cutoff. The fresh
    # clip's date must be present so the test fails if the route accidentally drops *all* clips
    # (e.g. inverted condition).
    assert _local_day(stale_ts) not in response.text
    assert _local_day(fresh_ts) in response.text


def test_cameras_page_caps_recent_alerts_at_five(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with alembic_web_test_client(config) as client:
        response = client.get("/cameras", headers=AUTH_HEADER)

    assert response.status_code == 200
    # Newest 5 (smallest hour-offsets) must appear.
    for i in range(5):
        assert f"alert-{i}" in response.text, f"Expected alert-{i} (newest 5) on /cameras"
    # Two oldest must be cut off by the limit.
    assert "alert-5" not in response.text
    assert "alert-6" not in response.text


def _seed_cat_subject_stats(engine: Engine) -> int:
    """Insert one cat Subject and return its id."""
    with get_session(engine) as session:
        subj = Subject(slug="stats-cat", display_name="Stats Cat", kind="cat", display_order=1)
        session.add(subj)
        session.flush()
        return subj.id


def _tag_clips_as_cat(internal_root: Path, tags: list[tuple[int, datetime | None]]) -> None:
    """Seed one cat Subject and tag each ``(clip_id, reviewed_at)`` clip with it via the review queue.

    ``reviewed_at=None`` tags the frame without marking the clip reviewed (so ``effective_has_cat``
    stays the detector verdict); a non-None value marks it reviewed (the override flips cat-positive).
    """
    engine = create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")
    try:
        subj_id = _seed_cat_subject_stats(engine)
        for clip_id, reviewed_at in tags:
            tag_clip_frame(engine, clip_id=clip_id, subject_id=subj_id, reviewed_at=reviewed_at)
    finally:
        engine.dispose()


def _make_stats_parity_clip(cam_id: int, label: str, today: datetime, *, has_cat: bool, reviewed_at: datetime | None = None) -> Clip:
    """Build a Clip row for the stats parity test; source_filename derives from ``label``."""
    fname = f"stats-parity-{label}.mp4"
    return Clip(
        camera_id=cam_id,
        source_filename=fname,
        start_ts=today,
        end_ts=today + timedelta(seconds=10),
        duration_seconds=10.0,
        file_path=f"clips/{fname}",
        thumb_path=f"thumbs/{fname}.jpg",
        file_size_bytes=1024,
        has_cat=has_cat,
        detector_version="yolov11n@deadbeef",
        ingested_at=today,
        reviewed_at=reviewed_at,
    )


def _seed_parity_clips(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    internal_root: Path,
    cam_id: int,
    today: datetime,
) -> tuple[int, int]:
    """Seed four clips for the stats parity test and return ``(clip_c_id, clip_d_id)``.

    A and B need no frame subjects (we only need their IDs to verify the total count).
    C and D will be tagged with a cat frame subject by the caller after this returns.
    """
    clip_c = _make_stats_parity_clip(cam_id, "C", today, has_cat=False)
    clip_d = _make_stats_parity_clip(cam_id, "D", today, has_cat=False, reviewed_at=today)
    with db_session_factory(internal_root) as session:
        for clip in (
            _make_stats_parity_clip(cam_id, "A", today, has_cat=True),
            _make_stats_parity_clip(cam_id, "B", today, has_cat=True, reviewed_at=today),
            clip_c,
            clip_d,
        ):
            session.add(clip)
        session.flush()
        return clip_c.id, clip_d.id


def test_stats_effective_has_cat_four_combinations(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``/stats`` cat-positive count reflects ``effective_has_cat`` from ``clip_label_summary``.

    Four clips, all on today's date for one camera:

    * A: ``has_cat=True``, unreviewed, no frames → effective=True (detector).
    * B: ``has_cat=True``, reviewed, no frames → effective=False (reviewed with has_manual_cat=FALSE).
    * C: ``has_cat=False``, unreviewed, cat frame → effective=False (detector, unreviewed).
    * D: ``has_cat=False``, reviewed, cat frame → effective=True (has_manual_cat=TRUE, reviewed).

    Expected cat_total = 2 (clips A and D).
    """
    internal_root, storage_root = storage_dirs
    [cam_id] = _persist_cameras(
        db_session_factory,
        internal_root,
        [Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)],
    )
    today = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0)
    clip_c_id, clip_d_id = _seed_parity_clips(db_session_factory, internal_root, cam_id, today)

    # C: tagged but unreviewed → effective stays False. D: tagged and reviewed → effective becomes True.
    _tag_clips_as_cat(internal_root, [(clip_c_id, None), (clip_d_id, today)])

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/stats", headers=AUTH_HEADER).text

    row = _row_for(body, "Pantry", today.date().isoformat())
    assert row is not None, "Expected a Pantry row for today in /stats"
    # Total = 4 clips; cat_total = 2 (A=detector True + D=reviewed manual True).
    assert "4" in row, f"Expected total=4 in Pantry row: {row}"
    assert "2" in row, f"Expected cat_total=2 in Pantry row: {row}"


# --- local-time rendering ------------------------------------------------------------------------


def test_cameras_page_renders_every_timestamp_in_local_time(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Each of the four camera timestamps renders in ``display_timezone``, with a distinct seed.

    All four cells render through the identical template construct, so one shared seed would let a
    template that missed a field still satisfy every assertion.
    """
    internal_root, storage_root = storage_dirs
    # All four are just after UTC midnight, so the local day differs from the UTC one.
    since = datetime(2026, 7, 2, 2, 0, 0, tzinfo=UTC)
    polled = datetime(2026, 7, 2, 2, 1, 0, tzinfo=UTC)
    last_clip = datetime(2026, 7, 2, 2, 2, 0, tzinfo=UTC)
    last_cat = datetime(2026, 7, 2, 2, 3, 0, tzinfo=UTC)
    _ = _persist_cameras(
        db_session_factory,
        internal_root,
        [
            Camera(
                name="pantry",
                display_name="Pantry",
                host="cam1.example.com",
                poll_status=PollStatus.OK,
                poll_status_since=since,
                last_polled_at=polled,
                last_clip_at=last_clip,
                last_cat_seen_at=last_cat,
            ),
        ],
    )

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        response = client.get("/cameras", headers=AUTH_HEADER)

    assert response.status_code == 200
    for value in (since, polled, last_clip, last_cat):
        expected = value.astimezone(_DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
        assert f">{expected}</time>" in response.text
        # The machine-readable attribute keeps UTC ISO, so the negative must be element-anchored.
        assert f">{value.isoformat()}</time>" not in response.text
        assert f'datetime="{value.isoformat()}"' in response.text


def test_alerts_page_renders_sent_in_local_time(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The Sent column renders in ``display_timezone``.

    Seeded relative to now because ``alerts_page`` cuts at a 30-day window, and the expected string
    is derived from the zone rather than hardcoded because a now-relative seed crosses EDT/EST.
    """
    internal_root, storage_root = storage_dirs
    [cam_id] = _persist_cameras(
        db_session_factory,
        internal_root,
        [Camera(name="pantry", display_name="Pantry", host="cam1.example.com", poll_status=PollStatus.OK)],
    )
    # Just after a UTC midnight inside the window, so the local calendar day differs.
    sent_at = (datetime.now(UTC) - timedelta(days=1)).replace(hour=2, minute=0, second=0, microsecond=0)
    _persist(
        db_session_factory,
        internal_root,
        [
            AlertSent(
                camera_id=cam_id,
                alert_type=AlertType.INACTIVITY,
                subject="alert-local",
                body="alert body",
                sent_at=sent_at,
                email_ok=True,
                macos_ok=True,
            ),
        ],
    )

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        response = client.get("/alerts", headers=AUTH_HEADER)

    assert response.status_code == 200
    expected = sent_at.astimezone(_DISPLAY_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    assert f">{expected}</time>" in response.text
    assert f">{sent_at.isoformat()}</time>" not in response.text


def test_stats_buckets_a_cross_midnight_clip_into_its_local_day(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A clip just after UTC midnight belongs to the previous local day, as the Start column shows."""
    internal_root, storage_root = storage_dirs
    [cam_id] = _persist_cameras(
        db_session_factory,
        internal_root,
        [Camera(name="pantry", display_name="Pantry", host="cam1.example.com", poll_status=PollStatus.OK)],
    )
    start_ts = (datetime.now(UTC) - timedelta(days=1)).replace(hour=2, minute=0, second=0, microsecond=0)
    # The premise the assertion depends on, checked rather than assumed.
    assert _local_day(start_ts) != start_ts.date().isoformat()
    _persist(db_session_factory, internal_root, [_make_clip(camera_id=cam_id, start_ts=start_ts, has_cat=True)])

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        response = client.get("/stats", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert _row_for(response.text, "Pantry", _local_day(start_ts)) is not None
    assert _row_for(response.text, "Pantry", start_ts.date().isoformat()) is None


def test_stats_rows_are_ordered_newest_day_then_camera(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """Rows read newest day first, then camera_id ascending.

    The ordering lives in Python rather than an ``ORDER BY`` clause, because bucketing by a
    display_timezone day cannot be expressed in SQLite. Nothing else pins it.
    """
    internal_root, storage_root = storage_dirs
    pantry_id, bath_id = _persist_cameras(
        db_session_factory,
        internal_root,
        [
            Camera(name="pantry", display_name="Pantry", host="cam1.example.com", poll_status=PollStatus.OK),
            Camera(name="bath", display_name="Bath", host="cam2.example.com", poll_status=PollStatus.OK),
        ],
    )
    assert pantry_id < bath_id
    newer = datetime.now(UTC).replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=1)
    older = newer - timedelta(days=1)
    _persist(
        db_session_factory,
        internal_root,
        [_make_clip(camera_id=cam, start_ts=day, has_cat=False) for day in (newer, older) for cam in (pantry_id, bath_id)],
    )

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        response = client.get("/stats", headers=AUTH_HEADER)

    assert response.status_code == 200
    expected = [
        ("Pantry", _local_day(newer)),
        ("Bath", _local_day(newer)),
        ("Pantry", _local_day(older)),
        ("Bath", _local_day(older)),
    ]
    positions = _row_positions(response.text, expected)
    assert positions == sorted(positions), f"rows out of order: {expected} landed at {positions}"


# --- per-cat stats ------------------------------------------------------------------------------


def _stats_engine(internal_root: Path) -> Engine:
    """Return an engine on the test DB with the tables and the ``clip_label_summary`` view."""
    engine = create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")
    Base.metadata.create_all(engine)
    apply_clip_label_summary_view(engine)
    return engine


def _seed_pantry(engine: Engine) -> int:
    """Insert the camera ``Pantry`` and return its id."""
    with get_session(engine) as session:
        cam = Camera(name="pantry", display_name="Pantry", host="cam.example.com", poll_status=PollStatus.OK)
        session.add(cam)
        session.flush()
        return cam.id


def _add_cat_clip(engine: Engine, cam_id: int, subject_id: int, start_ts: datetime) -> int:
    """Insert a 2-second clip that an operator tag names as ``subject_id``, and return its id."""
    clip_id = add_clip(engine, cam_id, start_ts=start_ts, name=f"{start_ts:%Y%m%d-%H%M%S%f}.mp4")
    tag_clip_frame(engine, clip_id=clip_id, subject_id=subject_id, reviewed_at=start_ts)
    return clip_id


def _archive(engine: Engine, subject_id: int) -> None:
    with get_session(engine) as session:
        subject = session.get(Subject, subject_id)
        assert subject is not None
        subject.archived_at = datetime.now(UTC)


def _cat_row(body: str, display_name: str) -> str | None:
    """Return the per-cat table row that names ``display_name``, or ``None``."""
    for match in re.finditer(r"<tr>.*?</tr>", _table(body, "cat-stats"), flags=re.DOTALL):
        if f">{display_name}<" in match.group(0):
            return match.group(0)
    return None


def _day_counts(row: str) -> dict[str, int]:
    """Map the ISO day of each day cell in a per-cat ``row`` to its visit count."""
    return {match.group(1): int(match.group(2)) for match in re.finditer(r'data-day="([0-9-]+)"[^>]*>\s*(\d+)\s*<', row)}


def _local_noon(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 12, tzinfo=_DISPLAY_TZ)


def _only_stat_row(engine: Engine, config: Config, now: datetime) -> dict[str, object]:
    """Return the per-cat row of the one cat in the test DB."""
    with get_session(engine) as session:
        rows = _cat_stat_rows(session, config=config, now=now, camera_display_by_id={})
    assert len(rows) == 1
    return rows[0]


def _day_cells(row: dict[str, object]) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", row["day_counts"])


def _counts(row: dict[str, object]) -> list[object]:
    return [day["count"] for day in _day_cells(row)]


def test_stats_heading_names_no_span_and_each_table_names_its_own(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """The page heading names no span. The per-cat caption names 7 days and the per-camera caption 30 days."""
    internal_root, storage_root = storage_dirs
    engine = _stats_engine(internal_root)
    try:
        cam_id = _seed_pantry(engine)
        _ = _add_cat_clip(engine, cam_id, seed_cat_subject(engine), datetime.now(UTC) - timedelta(hours=1))
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/stats", headers=AUTH_HEADER).text

    assert "<h1>Activity</h1>" in body
    assert "last 7 days" in _table(body, "cat-stats")
    assert "last 30 days" in _table(body, "camera-stats")
    assert body.index('id="cat-stats"') < body.index('id="camera-stats"')


def test_stats_counts_a_2330_visit_on_its_local_day(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A visit at 23:30 local time counts on that local day, not on the next UTC day."""
    internal_root, storage_root = storage_dirs
    today = datetime.now(_DISPLAY_TZ).date()
    yesterday = today - timedelta(days=1)
    start = datetime(yesterday.year, yesterday.month, yesterday.day, 23, 30, tzinfo=_DISPLAY_TZ)
    # The premise the assertion depends on, checked rather than assumed.
    assert start.astimezone(UTC).date() == today
    engine = _stats_engine(internal_root)
    try:
        cam_id = _seed_pantry(engine)
        subj_id = seed_cat_subject(engine)
        _ = _add_cat_clip(engine, cam_id, subj_id, start)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/stats", headers=AUTH_HEADER).text

    row = _cat_row(body, "Marcel")
    assert row is not None
    counts = _day_counts(row)
    assert list(counts) == [(today - timedelta(days=offset)).isoformat() for offset in range(6, -1, -1)]
    assert counts[yesterday.isoformat()] == 1
    assert counts[today.isoformat()] == 0
    # 1 visit over the 6 complete days, shown with one decimal.
    assert ">0.2<" in row
    assert "Pantry" in row
    assert " ago" in row
    assert f"{start:%a %Y-%m-%d %H:%M}" in row


def test_stats_week_across_a_dst_change_has_seven_local_days(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
) -> None:
    """The window starts at local midnight 6 days before today, and each visit counts on its local day.

    America/New_York leaves daylight time on 2026-11-01. A window start derived from the EST offset
    of today is 05:00 UTC, so it drops the 00:30 EDT visit on the first day.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    assert config.web.display_timezone == "America/New_York"
    now = datetime(2026, 11, 4, 17, 0, tzinfo=UTC)
    engine = _stats_engine(internal_root)
    try:
        cam_id = _seed_pantry(engine)
        subj_id = seed_cat_subject(engine)
        for local in (
            datetime(2026, 10, 28, 23, 30, tzinfo=_DISPLAY_TZ),  # before the window
            datetime(2026, 10, 29, 0, 30, tzinfo=_DISPLAY_TZ),  # EDT
            datetime(2026, 10, 31, 23, 30, tzinfo=_DISPLAY_TZ),  # EDT
            datetime(2026, 11, 1, 23, 30, tzinfo=_DISPLAY_TZ),  # EST
        ):
            _ = _add_cat_clip(engine, cam_id, subj_id, local)
        row = _only_stat_row(engine, config, now)
    finally:
        engine.dispose()

    assert _stat_days(now, _DISPLAY_TZ) == [date(2026, 10, 29) + timedelta(days=offset) for offset in range(7)]
    assert [day["day"] for day in _day_cells(row)] == [(date(2026, 10, 29) + timedelta(days=offset)).isoformat() for offset in range(7)]
    assert _counts(row) == [1, 0, 1, 1, 0, 0, 0]
    assert row["total"] == 3


def test_stats_mean_ignores_today(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
) -> None:
    """The mean divides the visits of the 6 complete days by 6. Today's visits count in the total only."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    now = datetime(2026, 10, 12, 20, 0, tzinfo=UTC)
    today = now.astimezone(_DISPLAY_TZ).date()
    engine = _stats_engine(internal_root)
    try:
        cam_id = _seed_pantry(engine)
        subj_id = seed_cat_subject(engine)
        # 4 visits on the complete days, 5 today. Visits are 10 minutes apart so none merge.
        for days_back, offset in ((6, 0), (6, 1), (3, 0), (1, 0)):
            _ = _add_cat_clip(engine, cam_id, subj_id, _local_noon(today - timedelta(days=days_back)) + timedelta(minutes=10 * offset))
        for offset in range(5):
            _ = _add_cat_clip(engine, cam_id, subj_id, _local_noon(today) + timedelta(minutes=10 * offset))
        row = _only_stat_row(engine, config, now)
    finally:
        engine.dispose()

    assert _counts(row) == [2, 0, 0, 1, 0, 1, 5]
    assert row["total"] == 9
    assert row["mean"] == 0.7


def test_stats_two_clips_less_than_2_minutes_apart_count_as_one_visit(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
) -> None:
    """A gap of less than 2 minutes joins a visit. A gap of exactly 2 minutes starts a new visit."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    now = datetime(2026, 10, 12, 20, 0, tzinfo=UTC)
    first = _local_noon(now.astimezone(_DISPLAY_TZ).date())
    engine = _stats_engine(internal_root)
    try:
        cam_id = _seed_pantry(engine)
        subj_id = seed_cat_subject(engine)
        _ = _add_cat_clip(engine, cam_id, subj_id, first)
        # A 2-second clip, then a gap of 58 seconds: one visit.
        _ = _add_cat_clip(engine, cam_id, subj_id, first + timedelta(minutes=1))
        # A gap of exactly 2 minutes from the end of the last clip: a new visit.
        _ = _add_cat_clip(engine, cam_id, subj_id, first + timedelta(minutes=3, seconds=2))
        row = _only_stat_row(engine, config, now)
    finally:
        engine.dispose()

    assert _counts(row)[-1] == 2


def test_stats_visit_that_starts_before_day_1_does_not_count_on_day_1(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
) -> None:
    """A visit counts on the day of its first clip, so a visit from the day before day 1 does not count."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    now = datetime(2026, 10, 12, 20, 0, tzinfo=UTC)
    day_1 = _stat_days(now, _DISPLAY_TZ)[0]
    midnight = datetime(day_1.year, day_1.month, day_1.day, tzinfo=_DISPLAY_TZ)
    engine = _stats_engine(internal_root)
    try:
        cam_id = _seed_pantry(engine)
        subj_id = seed_cat_subject(engine)
        # The 2-second clip at 23:59:00 ends 88 seconds before the next clip starts: one visit.
        _ = _add_cat_clip(engine, cam_id, subj_id, midnight - timedelta(minutes=1))
        _ = _add_cat_clip(engine, cam_id, subj_id, midnight + timedelta(seconds=30))
        row = _only_stat_row(engine, config, now)
    finally:
        engine.dispose()

    assert _counts(row) == [0, 0, 0, 0, 0, 0, 0]


def test_stats_last_visit_says_none_for_a_cat_with_no_visit(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """An active cat with no visit gets a row with "none" and a zero on each day."""
    internal_root, storage_root = storage_dirs
    engine = _stats_engine(internal_root)
    try:
        _ = seed_cat_subject(engine)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/stats", headers=AUTH_HEADER).text

    row = _cat_row(body, "Marcel")
    assert row is not None
    assert re.search(r'data-label="Last visit">\s*none\s*<', row) is not None
    assert set(_day_counts(row).values()) == {0}


def test_stats_archived_cat_gets_a_row_only_with_a_visit_in_the_7_days(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A visit 10 days ago is a last visit, but it is not in the 7 days, so it gives no row."""
    internal_root, storage_root = storage_dirs
    now = datetime.now(UTC)
    engine = _stats_engine(internal_root)
    try:
        cam_id = _seed_pantry(engine)
        _ = seed_cat_subject(engine, slug="active", display_name="Active", display_order=1)
        recent_id = seed_cat_subject(engine, slug="recent", display_name="Recent", display_order=2)
        old_id = seed_cat_subject(engine, slug="old", display_name="Old", display_order=3)
        idle_id = seed_cat_subject(engine, slug="idle", display_name="Idle", display_order=4)
        _ = _add_cat_clip(engine, cam_id, recent_id, now - timedelta(days=3))
        _ = _add_cat_clip(engine, cam_id, old_id, now - timedelta(days=10))
        for subject_id in (recent_id, old_id, idle_id):
            _archive(engine, subject_id)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/stats", headers=AUTH_HEADER).text

    assert _cat_row(body, "Active") is not None
    assert _cat_row(body, "Recent") is not None
    assert _cat_row(body, "Old") is None
    assert _cat_row(body, "Idle") is None


def test_stats_cat_name_carries_its_color_only_when_set(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A cat with a color sets ``--chip-color``. A cat with no color renders with the CSS fallback."""
    internal_root, storage_root = storage_dirs
    engine = _stats_engine(internal_root)
    try:
        _ = seed_cat_subject(engine, slug="marcel", display_name="Marcel", display_order=1, color="#d97706")
        _ = seed_cat_subject(engine, slug="rufus", display_name="Rufus", display_order=2)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        response = client.get("/stats", headers=AUTH_HEADER)

    assert response.status_code == 200
    marcel = _cat_row(response.text, "Marcel")
    rufus = _cat_row(response.text, "Rufus")
    assert marcel is not None
    assert rufus is not None
    assert "--chip-color: #d97706;" in marcel
    assert "style=" not in rufus
    assert response.text.index(">Marcel<") < response.text.index(">Rufus<")


def _seed_daily_visits(engine: Engine, cam_id: int, *, slug: str, order: int, last_noon: datetime) -> None:
    """Seed one cat with a visit on each of the 7 days. ``order`` sets the minute, so no file name repeats."""
    subject_id = seed_cat_subject(engine, slug=slug, display_name=slug, display_order=order)
    for offset in range(7):
        _ = _add_cat_clip(engine, cam_id, subject_id, last_noon - timedelta(days=offset) + timedelta(minutes=order))


def test_stats_per_cat_rows_read_a_fixed_number_of_queries(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
) -> None:
    """More cats and visits on more days do not add a query."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    now = datetime(2026, 10, 12, 20, 0, tzinfo=UTC)
    today = now.astimezone(_DISPLAY_TZ).date()
    engine = _stats_engine(internal_root)
    try:
        cam_id = _seed_pantry(engine)
        first_id = seed_cat_subject(engine, slug="first", display_name="First", display_order=1)
        _ = _add_cat_clip(engine, cam_id, first_id, _local_noon(today))
        with get_session(engine) as session, recorded_statements(engine) as small:
            _ = _cat_stat_rows(session, config=config, now=now, camera_display_by_id={})
        for index in range(3):
            _seed_daily_visits(engine, cam_id, slug=f"more-{index}", order=2 + index, last_noon=_local_noon(today))
        with get_session(engine) as session, recorded_statements(engine) as large:
            rows = _cat_stat_rows(session, config=config, now=now, camera_display_by_id={})
    finally:
        engine.dispose()

    assert len(rows) == 4
    assert len(large) == len(small)

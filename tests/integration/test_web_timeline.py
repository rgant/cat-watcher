"""Integration tests for the cat-watcher timeline routes.

Covers ``GET /`` and ``GET /timeline`` rendering: SVG lanes per camera, range-preset switching via
the ``range`` query parameter, and the storage-offline degradation path (banner + ``onerror``
placeholder).

The tests deliberately avoid asserting on layout numerics (lane width, x-coordinates) because those
are presentation details that the route's contract is silent on; they pin observable HTML contract —
element classes, marker counts, ``aria-label`` text — instead.
"""

import html
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo

import pytest
from db_helpers import AUTH_HEADER, add_clip, add_clip_frame, seed_cat_subject, tag_clip_frame, tag_frame
from sqlalchemy import event as sa_event
from sqlalchemy.engine import Engine as SaEngine

from cat_watcher.db import AlertSent, AlertType, Camera, Clip, Subject, create_engine, get_session

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from fastapi.testclient import TestClient
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


def _seed_camera_row(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    internal_root: Path,
    *,
    name: str = "pantry",
    display_name: str = "Pantry",
) -> int:
    with db_session_factory(internal_root) as session:
        cam = Camera(name=name, display_name=display_name, host="cam.example.com")
        session.add(cam)
        session.flush()
        return cam.id


def _seed_clip_rows(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    internal_root: Path,
    *,
    camera_id: int,
    start_offsets: list[timedelta],
    reference_now: datetime,
) -> None:
    with db_session_factory(internal_root) as session:
        for i, offset in enumerate(start_offsets):
            start_ts = reference_now - offset
            session.add(
                Clip(
                    camera_id=camera_id,
                    source_filename=f"clip-{i:04d}.mp4",
                    start_ts=start_ts,
                    end_ts=start_ts + timedelta(seconds=30),
                    duration_seconds=30.0,
                    file_path=f"clips/pantry/{i:04d}.mp4",
                    thumb_path=f"thumbs/pantry/{i:04d}.jpg",
                    file_size_bytes=1024,
                    has_cat=(i % 2 == 0),
                    detector_version="yolov11n@deadbeef",
                    ingested_at=reference_now,
                ),
            )


def _state_clip_kwargs(
    camera_id: int,
    reference_now: datetime,
    label: str,
    *,
    has_cat: bool = False,
    analysis_error: str | None = None,
) -> dict[str, object]:
    """Build a Clip-row kwargs dict, overriding ``has_cat`` / ``analysis_error`` as needed.

    ``_seed_clip_rows`` doesn't surface those fields, so tests that need them construct rows
    directly via this helper to keep field boilerplate out of test bodies.
    """
    start_ts = reference_now - timedelta(hours=2)
    return {
        "camera_id": camera_id,
        "source_filename": f"state-{label}.mp4",
        "start_ts": start_ts,
        "end_ts": start_ts + timedelta(seconds=30),
        "duration_seconds": 30.0,
        "file_path": f"clips/pantry/state-{label}.mp4",
        "thumb_path": f"thumbs/pantry/state-{label}.jpg",
        "file_size_bytes": 1024,
        "has_cat": has_cat,
        "analysis_error": analysis_error,
        "detector_version": "yolov11n@deadbeef",
        "ingested_at": reference_now,
    }


def _seed_alert_row(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    internal_root: Path,
    *,
    camera_id: int,
    sent_at: datetime,
    alert_type: AlertType = AlertType.FREQUENCY,
) -> None:
    with db_session_factory(internal_root) as session:
        session.add(
            AlertSent(
                alert_type=alert_type,
                camera_id=camera_id,
                sent_at=sent_at,
                subject="alert",
                body="alert body",
            ),
        )


def test_root_renders_svg_with_camera_display_name(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``GET /`` returns 200 with an inline ``<svg>`` and the camera's display name in a lane label."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root, display_name="Pantry Litter Box")
    now_ish = datetime.now(UTC)
    _seed_clip_rows(db_session_factory, internal_root, camera_id=cam_id, start_offsets=[timedelta(hours=2)], reference_now=now_ish)

    with alembic_web_test_client(config) as client:
        response = client.get("/", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "<svg" in response.text
    assert "Pantry Litter Box" in response.text


def test_timeline_route_accepts_range_param(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``GET /timeline?range=6h`` renders a 200 with an SVG lane container."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    fixed_now = datetime.now(UTC)
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[timedelta(hours=2)],
        reference_now=fixed_now,
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/timeline?range=6h", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert 'class="timeline-svg"' in response.text


def test_timeline_renders_per_clip_markers_at_6h(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A 6h window with 10 seeded clips renders 10 distinct per-clip ``<rect class="clip">`` elements."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now_ish = datetime.now(UTC)
    # 10 clips spread over 4.5 hours, well inside the 6h window.
    offsets = [timedelta(hours=5) - timedelta(minutes=i * 30) for i in range(10)]
    _seed_clip_rows(db_session_factory, internal_root, camera_id=cam_id, start_offsets=offsets, reference_now=now_ish)

    with alembic_web_test_client(config) as client:
        response = client.get("/timeline?range=6h&lanes=boxes", headers=AUTH_HEADER)

    assert response.status_code == 200
    # Per-clip SVG markers carry ``<rect class="clip ...">`` (the leading ``clip`` plus modifier
    # classes for cat/no-cat/manual/error styling). Scoping the substring to the ``<rect`` element
    # keeps the count robust against modifier-class additions and against the same ``css_classes``
    # being reused on the thumb-strip ``<li>`` cards.
    assert response.text.count('<rect class="clip ') == 10


def test_timeline_renders_per_clip_markers_at_3d(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A 3d window gives one marker per clip, and the thumb strip shows."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now_ish = datetime.now(UTC)
    # The clips share one hour, two days back. A per-hour aggregate merges them into one cell.
    offsets = [timedelta(days=2) + timedelta(minutes=i) for i in range(5)]
    _seed_clip_rows(db_session_factory, internal_root, camera_id=cam_id, start_offsets=offsets, reference_now=now_ish)

    with alembic_web_test_client(config) as client:
        response = client.get("/timeline?range=3d&lanes=boxes", headers=AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    assert body.count('<rect class="clip ') == 5
    assert "bucket" not in body
    assert 'class="thumb-strip"' in body


@pytest.mark.parametrize("range_key", ["7d", "30d", "bogus"])
def test_timeline_unknown_range_renders_24h_view(
    range_key: str,
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A range outside ``6h``, ``24h``, and ``3d`` renders the ``24h`` view."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[timedelta(hours=1)],
        reference_now=datetime.now(UTC),
    )

    with alembic_web_test_client(config) as client:
        response = client.get(f"/timeline?range={range_key}", headers=AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    assert 'aria-label="Activity over the last 24h"' in body
    assert re.search(r'\?range=24h"[^>]*aria-current="page"', body) is not None
    assert body.count('class="axis-tick"') == 24
    assert "range=7d" not in body
    assert "range=30d" not in body


def test_timeline_3d_ticks_show_weekday_and_day_boundaries_show_date(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """At 3d, every second tick has a weekday label, and a midnight line has a day-and-month label."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[timedelta(hours=1)],
        reference_now=datetime.now(UTC),
    )

    with alembic_web_test_client(config) as client:
        body = client.get("/timeline?range=3d", headers=AUTH_HEADER).text

    tick_labels: list[str] = re.findall(r'class="axis-tick-label"[^>]*>([^<]*)</text>', body)
    assert len(tick_labels) == 6
    for label in tick_labels:
        assert re.fullmatch(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d\d:\d\d", label), label
    day_labels: list[str] = re.findall(r'class="axis-day-label"[^>]*>([^<]*)</text>', body)
    assert day_labels
    for label in day_labels:
        assert re.fullmatch(r"\d{1,2} [A-Z][a-z]{2}", label), label


def test_box_lane_label_is_camera_display_name(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Each lane label is the camera display name. A camera with no clip still gets a lane."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root, name="pantry", display_name="Pantry Box")
    _ = _seed_camera_row(db_session_factory, internal_root, name="hall", display_name="Hall Box")
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[timedelta(hours=1)],
        reference_now=datetime.now(UTC),
    )

    with alembic_web_test_client(config) as client:
        body = client.get("/timeline?range=24h&lanes=boxes", headers=AUTH_HEADER).text

    assert re.findall(r'class="lane-label">([^<]*)</text>', body) == ["Hall Box", "Pantry Box"]


@pytest.mark.parametrize(
    "case",
    [
        ("Pantry", 80.0),
        # 24 characters at 6.6 units each, rounded up, plus 8 units of padding.
        ("Pantry Litter Box Camera", 167.0),
    ],
)
def test_lane_label_column_fits_the_longest_lane_label(
    case: tuple[str, float],
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The lane track starts after the longest lane label. A short label keeps the 80-unit minimum."""
    display_name, expected_track_start = case
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root, display_name=display_name)
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[timedelta(hours=1)],
        reference_now=datetime.now(UTC),
    )

    with alembic_web_test_client(config) as client:
        body = client.get("/timeline?range=24h&lanes=boxes", headers=AUTH_HEADER).text

    track_starts: list[str] = re.findall(r'<line x1="([^"]+)"[^>]*class="lane-axis"', body)
    assert {float(x1) for x1 in track_starts} == {expected_track_start}


def test_marker_carries_camera_display_name_and_weekday_start(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A marker names its camera and the weekday and time of its start, for the tooltip and the link label."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root, display_name="Pantry Box")
    reference_now = datetime.now(UTC).replace(microsecond=0)
    _seed_clip_rows(db_session_factory, internal_root, camera_id=cam_id, start_offsets=[timedelta(hours=3)], reference_now=reference_now)

    with alembic_web_test_client(config) as client:
        body = client.get("/timeline?range=24h", headers=AUTH_HEADER).text

    expected_start = (reference_now - timedelta(hours=3)).astimezone(ZoneInfo(config.web.display_timezone)).strftime("%a %H:%M:%S")
    assert f'data-camera="Pantry Box" data-start="{expected_start}"' in body
    assert f'aria-label="Pantry Box · {expected_start} · score' in body


def test_timeline_renders_offline_banner_when_storage_root_unmounted(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """When ``storage_root`` is missing, the response includes the offline banner from spec §4.13."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_row(db_session_factory, internal_root)
    # Drop the storage_root after config validation but before the request — mimics the drive being
    # ejected at runtime. ``_storage_root_available`` calls ``is_dir()``, so a non-existent path
    # returns False without raising.
    storage_root.rmdir()

    with alembic_web_test_client(config) as client:
        response = client.get("/", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "External storage offline" in response.text


def test_timeline_header_carries_htmx_attrs_and_banner_aria_live(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Range presets carry hx-indicator + hx-push-url; storage banner carries aria-live."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_row(db_session_factory, internal_root)
    # Drop the storage_root after config validation so the probe (`is_dir()`) returns False — same
    # trick the existing test_timeline_renders_offline_banner_when_storage_root_unmounted uses.
    storage_root.rmdir()

    with alembic_web_test_client(config) as client:
        response = client.get("/timeline?range=24h", headers=AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    assert 'class="timeline-header"' in body
    # The range links, the view links, and the strip filter links carry the HTMX attributes.
    assert body.count('hx-indicator="#timeline-region"') == 7
    assert body.count('hx-push-url="true"') == 7
    assert 'aria-live="polite"' in body
    assert 'class="banner banner-offline"' in body


def test_timeline_omits_offline_banner_when_storage_root_mounted(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A mounted storage root produces no offline banner."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_row(db_session_factory, internal_root)

    with alembic_web_test_client(config) as client:
        response = client.get("/", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "External storage offline" not in response.text


def test_timeline_thumb_imgs_reference_placeholder_in_onerror(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Each rendered thumbnail ``<img>`` carries an ``onerror`` that points at the bundled placeholder.

    Confirms the spec §4.13 fallback path: when the media route returns 503, the inline placeholder
    is shown instead of a broken-image glyph. Per-clip markers render an `<img>` inside the lane
    group, so we seed at least one clip for the assertion.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now_ish = datetime.now(UTC)
    _seed_clip_rows(db_session_factory, internal_root, camera_id=cam_id, start_offsets=[timedelta(hours=1)], reference_now=now_ish)

    with alembic_web_test_client(config) as client:
        response = client.get("/timeline?range=6h", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert re.search(r"/static/clip-placeholder\.[0-9a-f]{12}\.svg", response.text)
    assert "onerror" in response.text


def test_root_lane_includes_alert_marker(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """An ``alerts_sent`` row inside the window renders a vertical marker tagged with the alert type."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now_ish = datetime.now(UTC)
    # Seed a clip so the SVG renders (per spec §5, the empty-state replaces the SVG when no clips
    # exist in the window). Alert-marker rendering is what this test exercises.
    _seed_clip_rows(db_session_factory, internal_root, camera_id=cam_id, start_offsets=[timedelta(hours=1)], reference_now=now_ish)
    _seed_alert_row(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        sent_at=now_ish - timedelta(hours=2),
        alert_type=AlertType.FREQUENCY,
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/", headers=AUTH_HEADER)

    assert response.status_code == 200
    # Alert markers carry the alert type as a label so operators can read them at a glance.
    assert "FREQUENCY" in response.text


def test_timeline_renders_time_axis_with_per_range_tick_count(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The SVG includes a <g class="time-axis"> group whose tick count matches the per-range cadence."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # Seed a clip so the SVG renders (the no-clips path renders the empty-state block instead).
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=_seed_camera_row(db_session_factory, internal_root),
        start_offsets=[timedelta(hours=1)],
        reference_now=datetime.now(UTC),
    )

    cases = [
        ("6h", 12, "tick interval 30 min, 6h window -> 12 ticks"),
        ("24h", 24, "tick interval 1h, 24h window -> 24 ticks"),
        ("3d", 12, "tick interval 6h, 3d window -> 12 ticks"),
    ]

    with alembic_web_test_client(config) as client:
        for range_key, expected_ticks, msg in cases:
            response = client.get(f"/timeline?range={range_key}", headers=AUTH_HEADER)
            assert response.status_code == 200
            body = response.text
            assert '<g class="time-axis">' in body, f"{range_key}: missing time-axis group"
            tick_count = body.count('class="axis-tick"')
            assert tick_count == expected_ticks, f"{range_key}: {msg}; got {tick_count}"
            assert 'class="axis-now"' in body, f"{range_key}: missing now indicator"


def test_timeline_renders_day_boundary_markers_when_window_crosses_midnight(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """At 24h, the time-axis renders a day-boundary marker per midnight in display_timezone."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    # Seed a clip so the SVG renders (the no-clips path renders the empty-state block instead).
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[timedelta(hours=1)],
        reference_now=datetime.now(UTC),
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/timeline?range=24h", headers=AUTH_HEADER)

    # A 24h window in display_timezone always crosses exactly one midnight boundary.
    assert response.status_code == 200
    assert response.text.count('class="axis-day-boundary"') == 1


def test_thumb_card_renders_weekday_and_time_in_configured_timezone(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A thumb card shows the weekday and time of the clip start in display_timezone."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    # Anchor on the actual current time so the seeded clip stays inside the route's 24h window
    # regardless of when the test runs. ``microsecond=0`` keeps the formatted ``HH:MM:SS``
    # round-trip exact across the seed/render boundary.
    reference_now = datetime.now(UTC).replace(microsecond=0)
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[timedelta(hours=4)],
        reference_now=reference_now,
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/?range=24h", headers=AUTH_HEADER)

    assert response.status_code == 200
    expected_local = (reference_now - timedelta(hours=4)).astimezone(ZoneInfo(config.web.display_timezone))
    expected_stamp = expected_local.strftime("%a %H:%M:%S")
    assert f'<span class="thumb-time">{expected_stamp}</span>' in response.text


def test_timeline_empty_state_links_to_next_longer_range(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """With no clips at 24h, the empty state offers 3d and a /cameras link."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_row(db_session_factory, internal_root)  # camera but no clips

    with alembic_web_test_client(config) as client:
        response = client.get("/timeline?range=24h", headers=AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    assert 'class="empty-state' in body, "expected empty-state block"
    assert "No activity in this range" in body
    assert re.search(r'class="empty-cta-next-longer-range"\s+href="[^"]*\?range=3d"', body), "expected a 3d CTA"
    assert 'class="empty-cta-cameras"' in body, "expected /cameras CTA"


def test_timeline_empty_state_omits_next_longer_link_at_3d(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """At the longest preset (3d), the empty state hides the next-longer-range CTA."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_row(db_session_factory, internal_root)

    with alembic_web_test_client(config) as client:
        response = client.get("/timeline?range=3d", headers=AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    assert 'class="empty-state' in body
    assert "next-longer-range" not in body
    assert "/cameras" in body


def test_thumb_card_li_carries_clip_css_classes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Each <li> in the thumb strip carries the clip's css_classes so the state border applies."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    fixed_now = datetime.now(UTC)
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[timedelta(hours=2)],
        reference_now=fixed_now,
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/?range=24h", headers=AUTH_HEADER)

    assert response.status_code == 200
    # The seeded clip has has_cat=True (i % 2 == 0 for i=0). Expect <li class="clip clip-cat">.
    assert '<li class="clip clip-cat"' in response.text


def test_thumb_card_renders_clip_manual_and_clip_error_state_classes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A reviewed clip with cat frame tagging emits ``clip-manual``; a detector-error clip emits ``clip-error``."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    reference_now = datetime.now(UTC).replace(microsecond=0)
    with db_session_factory(internal_root) as session:
        manual_clip = Clip(**_state_clip_kwargs(cam_id, reference_now, "manual"))
        session.add(manual_clip)
        session.add(Clip(**_state_clip_kwargs(cam_id, reference_now, "error", analysis_error="ffmpeg returned 1: bad header")))
        session.flush()
        manual_clip_id = manual_clip.id

    # Tag the manual clip with a cat frame subject and mark it reviewed.
    engine = _engine_for(internal_root)
    try:
        subj_id = seed_cat_subject(engine)
        tag_clip_frame(engine, clip_id=manual_clip_id, subject_id=subj_id, reviewed_at=reference_now)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/?range=24h&clips=all", headers=AUTH_HEADER).text

    # Reviewed + cat-tagged frames: both clip-cat and clip-manual modifier classes present.
    assert _has_class(body, "clip-cat"), "expected cat-tagged reviewed clip to carry clip-cat class"
    assert "clip-manual" in body, "expected cat-tagged reviewed clip to carry clip-manual class"
    # Detector-error clip: clip-error modifier present on a thumb card.
    assert "clip-error" in body, "expected detector-error clip to carry clip-error class"


def _has_class(body: str, name: str) -> bool:
    """Return whether an element in ``body`` has the class ``name`` as a whole token.

    A substring test is not enough, because ``clip-cat`` is also the start of ``clip-cat-lane``.
    """
    return re.search(rf'class="(?:[^"]* )?{re.escape(name)}(?: [^"]*)?"', body) is not None


def _engine_for(internal_root: Path) -> Engine:
    """Return a short-lived engine for the test DB so ``tag_clip_frame`` can write frames."""
    return create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")


def test_clip_manual_badge_absent_for_unreviewed_clip_with_frame_membership(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A clip with cat-tagged frames but ``reviewed_at IS NULL`` must NOT carry ``clip-manual``.

    Partially-tagged but unreviewed clips should not show the manual badge so the operator
    knows the review is incomplete.
    """
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    reference_now = datetime.now(UTC).replace(microsecond=0)
    with db_session_factory(internal_root) as session:
        clip = Clip(**_state_clip_kwargs(cam_id, reference_now, "unreviewed-tagged"))
        session.add(clip)
        session.flush()
        clip_id = clip.id

    engine = _engine_for(internal_root)
    try:
        subj_id = seed_cat_subject(engine)
        tag_clip_frame(engine, clip_id=clip_id, subject_id=subj_id)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        response = client.get("/?range=24h&lanes=boxes&clips=all", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "clip-manual" not in response.text, "unreviewed clip must not carry clip-manual badge"


def test_clip_manual_badge_present_for_reviewed_clip_with_frame_membership(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A clip with cat-tagged frames AND ``reviewed_at IS NOT NULL`` carries ``clip-manual``."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    reference_now = datetime.now(UTC).replace(microsecond=0)
    with db_session_factory(internal_root) as session:
        clip = Clip(**_state_clip_kwargs(cam_id, reference_now, "reviewed-tagged"))
        session.add(clip)
        session.flush()
        clip_id = clip.id

    engine = _engine_for(internal_root)
    try:
        subj_id = seed_cat_subject(engine)
        tag_clip_frame(engine, clip_id=clip_id, subject_id=subj_id, reviewed_at=reference_now)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        response = client.get("/?range=24h", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "clip-manual" in response.text, "reviewed + tagged clip must carry clip-manual badge"


def test_clip_manual_badge_absent_when_operator_agrees_with_detector(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``has_cat=True`` + reviewed + cat-tagged → ``clip-cat`` but NO ``clip-manual``.

    The operator's label agrees with the detector, so the manual marker is suppressed.
    """
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    reference_now = datetime.now(UTC).replace(microsecond=0)
    with db_session_factory(internal_root) as session:
        clip = Clip(**_state_clip_kwargs(cam_id, reference_now, "reviewed-agree", has_cat=True))
        session.add(clip)
        session.flush()
        clip_id = clip.id

    engine = _engine_for(internal_root)
    try:
        subj_id = seed_cat_subject(engine)
        tag_clip_frame(engine, clip_id=clip_id, subject_id=subj_id, reviewed_at=reference_now)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/?range=24h", headers=AUTH_HEADER).text

    assert _has_class(body, "clip-cat")
    assert "clip-manual" not in body, "agreement with the detector must not carry clip-manual"


def test_effective_has_cat_unreviewed_clip_uses_detector_verdict(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Unreviewed clip with ``has_cat=True`` renders as cat-positive; no ``clip-manual`` badge."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    reference_now = datetime.now(UTC).replace(microsecond=0)
    with db_session_factory(internal_root) as session:
        session.add(Clip(**_state_clip_kwargs(cam_id, reference_now, "unreviewed-detector", has_cat=True)))

    with alembic_web_test_client(config) as client:
        response = client.get("/?range=24h", headers=AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    assert _has_class(body, "clip-cat"), "unreviewed clip with has_cat=True must render as cat-positive"
    assert "clip-manual" not in body, "unreviewed clip must not carry clip-manual badge"


def test_effective_has_cat_reviewed_clip_without_memberships_uses_detector(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Reviewed clip with no frame memberships falls back to detector verdict; no ``clip-manual``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    reference_now = datetime.now(UTC).replace(microsecond=0)
    # Reviewed but no frame subjects → has_manual_cat=FALSE in the view → effective=has_cat.
    kwargs = _state_clip_kwargs(cam_id, reference_now, "reviewed-no-frames")
    kwargs["has_cat"] = False
    kwargs["reviewed_at"] = reference_now
    with db_session_factory(internal_root) as session:
        session.add(Clip(**kwargs))

    with alembic_web_test_client(config) as client:
        response = client.get("/?range=24h&lanes=boxes&clips=all", headers=AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    assert "clip-no-cat" in body, "reviewed clip with no memberships and has_cat=False must render as no-cat"
    assert "clip-manual" not in body, "clip without cat-tagged frames must not carry clip-manual badge"


_TRIPWIRES = ("{_fmt", "{self.", "{cam.", "{cfg.", "NoneType")


def _named_clip(engine: Engine, cam_id: int, *, start_ts: datetime, name: str, subject_ids: tuple[int, ...] = ()) -> int:
    """Insert a cat clip and tag one frame per subject in ``subject_ids``. A tag names the cat with no review."""
    clip_id = add_clip(engine, cam_id, start_ts=start_ts, name=name)
    for ordinal, subject_id in enumerate(subject_ids):
        tag_frame(engine, add_clip_frame(engine, clip_id, ordinal), subject_id)
    return clip_id


def _archived(engine: Engine, subject_id: int) -> int:
    """Archive the subject and return its id."""
    with get_session(engine) as session:
        subject = session.get(Subject, subject_id)
        assert subject is not None
        subject.archived_at = datetime.now(UTC) - timedelta(days=30)
    return subject_id


def _lane_blocks(body: str) -> dict[str, str]:
    """Map each lane label to the markup of its lane. The dict keeps the lane order."""
    pairs: list[tuple[str, str]] = re.findall(r'<g class="lane"[^>]*>\s*<text[^>]*>([^<]*)</text>(.*?)</g>', body, flags=re.DOTALL)
    return dict(pairs)


def _lane_markers(body: str) -> dict[str, list[int]]:
    """Map each lane label to the clip ids of its markers. The dict keeps the lane order."""
    lanes: dict[str, list[int]] = {}
    for label, block in _lane_blocks(body).items():
        clip_ids: list[str] = re.findall(r'data-clip-id="(\d+)"', block)
        lanes[label] = [int(clip_id) for clip_id in clip_ids]
    return lanes


def _strip_clip_ids(body: str) -> list[int]:
    """Return the clip ids of the strip cards, in page order."""
    strip = re.search(r'<ul class="thumb-strip".*?</ul>', body, flags=re.DOTALL)
    if strip is None:
        return []
    clip_ids: list[str] = re.findall(r'<a href="[^"]*/clips/(\d+)"', strip.group(0))
    return [int(clip_id) for clip_id in clip_ids]


def _nav_queries(body: str, label: str) -> list[dict[str, list[str]]]:
    """Return the parsed query of each link in the ``<nav>`` with ``aria-label`` ``label``.

    Each link must send its HTMX request to the same URL as its ``href``.
    """
    nav = re.search(rf'<nav[^>]*aria-label="{label}"[^>]*>(.*?)</nav>', body, flags=re.DOTALL)
    assert nav is not None, label
    links: list[tuple[str, str]] = re.findall(r'<a\s+href="([^"]+)"\s+hx-get="([^"]+)"', nav.group(1))
    assert links, label
    queries: list[dict[str, list[str]]] = []
    for href, hx_get in links:
        assert href == hx_get
        url = urlsplit(html.unescape(href))
        assert url.path == "/timeline"
        queries.append(parse_qs(url.query))
    return queries


def _assert_no_tripwire(body: str) -> None:
    for tripwire in _TRIPWIRES:
        assert tripwire not in body, tripwire


def test_cat_view_lane_order_and_archived_and_unknown_lanes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The default view gives the active cats, an archived cat with a clip, then "Unknown cat".

    An archived cat with no clip in the window gets no lane.
    """
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        # Rufus goes in first, so the lane order comes from display_order and not from the row order.
        _ = seed_cat_subject(engine, slug="rufus", display_name="Rufus", display_order=3)
        marcel = seed_cat_subject(engine, slug="marcel", display_name="Marcel", display_order=2)
        # The archived cat has the lowest display_order, so its lane after the active cats proves active-first order.
        old = _archived(engine, seed_cat_subject(engine, slug="old", display_name="Old Cat", display_order=1))
        _ = _archived(engine, seed_cat_subject(engine, slug="gone", display_name="Gone Cat", display_order=4))
        expected = {
            "Marcel": [_named_clip(engine, cam_id, start_ts=now - timedelta(hours=3), name="marcel.mp4", subject_ids=(marcel,))],
            "Rufus": [],
            "Old Cat": [_named_clip(engine, cam_id, start_ts=now - timedelta(hours=2), name="old.mp4", subject_ids=(old,))],
            "Unknown cat": [_named_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="unknown.mp4")],
        }
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/", headers=AUTH_HEADER).text

    assert _lane_markers(body) == expected
    _assert_no_tripwire(body)


def test_cat_view_has_no_unknown_lane_without_an_unknown_cat_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The "Unknown cat" lane is absent when every cat clip in the window names a cat."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        marcel = seed_cat_subject(engine)
        _ = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="marcel.mp4", subject_ids=(marcel,))
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/timeline", headers=AUTH_HEADER).text

    assert list(_lane_markers(body)) == ["Marcel"]
    assert "Unknown cat" not in body


def test_clip_that_names_two_cats_has_a_marker_in_both_lanes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A clip that names two cats shows in the lane of each cat, and once in the strip."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        marcel = seed_cat_subject(engine, slug="marcel", display_name="Marcel", display_order=1)
        rufus = seed_cat_subject(engine, slug="rufus", display_name="Rufus", display_order=2)
        both = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="both.mp4", subject_ids=(marcel, rufus))
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/timeline", headers=AUTH_HEADER).text

    assert _lane_markers(body) == {"Marcel": [both], "Rufus": [both]}
    assert _strip_clip_ids(body) == [both]


def test_non_cat_clip_shows_in_the_box_view_only(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A clip that is not a cat clip has no marker in the cat view and has a marker in the box view."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root, display_name="Pantry Box")
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        marcel = seed_cat_subject(engine)
        cat_clip = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=2), name="cat.mp4", subject_ids=(marcel,))
        no_cat_clip = add_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="empty.mp4", has_cat=False)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        cat_body = client.get("/timeline", headers=AUTH_HEADER).text
        box_body = client.get("/timeline?lanes=boxes", headers=AUTH_HEADER).text

    assert _lane_markers(cat_body) == {"Marcel": [cat_clip]}
    assert _lane_markers(box_body) == {"Pantry Box": [cat_clip, no_cat_clip]}


def test_cat_lane_markers_carry_the_cat_color(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A marker carries its cat color. A cat with no color renders, and its markers carry no color."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        marcel = seed_cat_subject(engine, slug="marcel", display_name="Marcel", display_order=1, color="#c0ffee")
        rufus = seed_cat_subject(engine, slug="rufus", display_name="Rufus", display_order=2)
        _ = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=2), name="marcel.mp4", subject_ids=(marcel,))
        _ = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="rufus.mp4", subject_ids=(rufus,))
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        response = client.get("/timeline", headers=AUTH_HEADER)

    assert response.status_code == 200
    lane_blocks = _lane_blocks(response.text)
    assert re.search(r'<rect [^>]*style="--lane-color: #c0ffee;"', lane_blocks["Marcel"])
    assert "<rect " in lane_blocks["Rufus"]
    assert "--lane-color" not in lane_blocks["Rufus"]
    _assert_no_tripwire(response.text)


def test_strip_shows_cat_clips_by_default_and_every_clip_with_clips_all(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """The strip is newest first. ``clips=cat`` keeps the cat clips. ``clips=all`` keeps every clip."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        older_cat = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=3), name="older.mp4")
        no_cat = add_clip(engine, cam_id, start_ts=now - timedelta(hours=2), name="empty.mp4", has_cat=False)
        newer_cat = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="newer.mp4")
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        strips = {
            query: _strip_clip_ids(client.get(f"/timeline{query}", headers=AUTH_HEADER).text)
            for query in ("", "?lanes=boxes", "?clips=all")
        }

    assert strips == {
        "": [newer_cat, older_cat],
        "?lanes=boxes": [newer_cat, older_cat],
        "?clips=all": [newer_cat, no_cat, older_cat],
    }


def test_strip_card_names_its_cats_or_unknown_cat(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A card names the cats of its clip. An unknown-cat clip card says "Unknown cat"."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        marcel = seed_cat_subject(engine)
        _ = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=2), name="marcel.mp4", subject_ids=(marcel,))
        _ = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="unknown.mp4")
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/timeline", headers=AUTH_HEADER).text

    strip = re.search(r'<ul class="thumb-strip".*?</ul>', body, flags=re.DOTALL)
    assert strip is not None
    cards: list[str] = re.findall(r"<li .*?</li>", strip.group(0), flags=re.DOTALL)
    assert len(cards) == 2
    assert '<span class="identity-fallback">Unknown cat</span>' in cards[0]
    assert re.search(r'class="identity-chip identity-chip-operator"[^>]*>Marcel</span>', cards[1])
    _assert_no_tripwire(body)


def test_strip_holds_48_cards_per_page(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """With 49 strip clips, page 1 holds 48 cards and page 2 holds the oldest clip."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[timedelta(minutes=5 * i + 1) for i in range(49)],
        reference_now=datetime.now(UTC),
    )

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        first = client.get("/timeline?clips=all", headers=AUTH_HEADER).text
        second = client.get("/timeline?clips=all&page=2", headers=AUTH_HEADER).text
        past_end = client.get("/timeline?clips=all&page=9", headers=AUTH_HEADER).text

    assert len(_strip_clip_ids(first)) == 48
    assert len(_strip_clip_ids(second)) == 1
    assert _strip_clip_ids(second)[0] not in _strip_clip_ids(first)
    assert _strip_clip_ids(past_end) == _strip_clip_ids(second)
    assert "Page 2 of 2" in second


def test_every_timeline_link_carries_the_view_state(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Range, view, filter, and page links keep the other parameters. Only a page link has ``page``.

    A value at its default is left out of the URL. ``range`` is always in the URL.
    """
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[timedelta(minutes=5 * i + 1) for i in range(49)],
        reference_now=datetime.now(UTC),
    )

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        first = client.get("/timeline?range=6h&lanes=boxes&clips=all", headers=AUTH_HEADER).text
        second = client.get("/timeline?range=6h&lanes=boxes&clips=all&page=2", headers=AUTH_HEADER).text

    state = {"range": ["6h"], "lanes": ["boxes"], "clips": ["all"]}
    assert _nav_queries(second, "Timeline range") == [{**state, "range": [key]} for key in ("6h", "24h", "3d")]
    assert _nav_queries(second, "Lanes") == [{"range": ["6h"], "clips": ["all"]}, state]
    assert _nav_queries(second, "Strip clips") == [{"range": ["6h"], "lanes": ["boxes"]}, state]
    assert _nav_queries(first, "Strip pages") == [{**state, "page": ["2"]}]
    assert _nav_queries(second, "Strip pages") == [state]
    _assert_no_tripwire(first)
    _assert_no_tripwire(second)


def test_unknown_lanes_and_clips_values_snap_to_the_defaults(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """An unknown ``lanes`` or ``clips`` value gives the cat view and the cat-clip strip."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        cat_clip = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=2), name="cat.mp4")
        _ = add_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="empty.mp4", has_cat=False)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/timeline?lanes=bogus&clips=bogus&page=bogus", headers=AUTH_HEADER).text

    assert _lane_markers(body) == {"Unknown cat": [cat_clip]}
    assert _strip_clip_ids(body) == [cat_clip]
    assert "bogus" not in body


def test_cat_view_empty_state_links_to_the_box_view(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """If the window has clips but no cat clip, the cat view says so and links to the box view."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        _ = seed_cat_subject(engine)
        _ = add_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="empty.mp4", has_cat=False)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/timeline?range=6h&clips=all", headers=AUTH_HEADER).text

    assert "No cat visited in this range" in body
    assert 'class="timeline-svg"' not in body
    link = re.search(r'class="empty-cta-box-view"\s+href="([^"]+)"', body)
    assert link is not None
    assert parse_qs(urlsplit(html.unescape(link.group(1))).query) == {"range": ["6h"], "lanes": ["boxes"], "clips": ["all"]}
    _assert_no_tripwire(body)


def test_box_view_with_no_cat_clip_says_so_and_links_to_all_clips(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """If the strip shows cat clips and the window holds none, the strip says so and links to ``clips=all``."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        _ = add_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="empty.mp4", has_cat=False)
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get("/timeline?range=6h&lanes=boxes", headers=AUTH_HEADER).text
        all_body = client.get("/timeline?range=6h&lanes=boxes&clips=all", headers=AUTH_HEADER).text

    assert "No cat clips in this range." in body
    assert "No cat visited in this range" not in body
    link = re.search(r'class="empty-cta-all-clips"\s+href="([^"]+)"', body)
    assert link is not None
    assert parse_qs(urlsplit(html.unescape(link.group(1))).query) == {"range": ["6h"], "lanes": ["boxes"], "clips": ["all"]}
    assert "No cat clips in this range." not in all_body
    _assert_no_tripwire(body)


def test_clip_with_a_future_start_shows_in_no_view(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A cat clip with a start after now (a fast camera clock) is in no lane and no strip, in every view."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root, display_name="Pantry Box")
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        marcel = seed_cat_subject(engine)
        past = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="past.mp4", subject_ids=(marcel,))
        _ = _named_clip(engine, cam_id, start_ts=now + timedelta(hours=1), name="future.mp4", subject_ids=(marcel,))
    finally:
        engine.dispose()

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        bodies = {query: client.get(f"/timeline{query}", headers=AUTH_HEADER).text for query in ("", "?clips=all", "?lanes=boxes")}

    assert _lane_markers(bodies[""]) == {"Marcel": [past]}
    assert _lane_markers(bodies["?lanes=boxes"]) == {"Pantry Box": [past]}
    assert {query: _strip_clip_ids(body) for query, body in bodies.items()} == {"": [past], "?clips=all": [past], "?lanes=boxes": [past]}


def test_timeline_reads_the_clips_of_its_window_once(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """One statement reads the clips of the window. The cat lanes and the strip still name the cats."""
    internal_root, storage_root = storage_dirs
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now = datetime.now(UTC)
    engine = _engine_for(internal_root)
    try:
        named = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=2), name="marcel.mp4", subject_ids=(seed_cat_subject(engine),))
        unknown = _named_clip(engine, cam_id, start_ts=now - timedelta(hours=1), name="unknown.mp4")
        _ = add_clip(engine, cam_id, start_ts=now - timedelta(minutes=30), name="empty.mp4", has_cat=False)
    finally:
        engine.dispose()
    window_reads: list[str] = []

    def _record(_conn: object, _cursor: object, statement: str, *_rest: object) -> None:
        if "clips.start_ts >=" in statement:
            window_reads.append(statement)

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        sa_event.listen(SaEngine, "before_cursor_execute", _record)
        try:
            body = client.get("/timeline", headers=AUTH_HEADER).text
        finally:
            sa_event.remove(SaEngine, "before_cursor_execute", _record)

    assert len(window_reads) == 1, window_reads
    assert _lane_markers(body) == {"Marcel": [named], "Unknown cat": [unknown]}
    assert _strip_clip_ids(body) == [unknown, named]
    _assert_no_tripwire(body)

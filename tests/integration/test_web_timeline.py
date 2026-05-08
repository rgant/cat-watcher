"""Integration tests for the cat-watcher timeline routes (Task 23).

Covers ``GET /`` and ``GET /timeline`` rendering: SVG lanes per camera, range-preset switching via
the ``range`` query parameter, density bucketing for windows past the 24h threshold (per spec
§4.7.1), and the storage-offline degradation path (banner + ``onerror`` placeholder).

The tests deliberately avoid asserting on layout numerics (lane width, x-coordinates) because those
are presentation details that the route's contract is silent on; they pin observable HTML contract —
element classes, marker counts, ``aria-label`` text — instead.
"""

import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING

from cat_watcher.db import AlertSent, AlertType, Camera, Clip

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from fastapi.testclient import TestClient
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


_AUTH_HEADER = {"Authorization": f"Basic {base64.b64encode(b'admin:pw').decode()}"}


def _seed_camera_row(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    internal_root: Path,
    *,
    name: str = "pantry",
    display_name: str = "Pantry",
) -> int:
    """Insert one camera and return its id."""
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
    """Seed N clips with ``start_ts = reference_now - offset[i]`` and unique source filenames."""
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
    manual_has_cat: bool | None = None,
    analysis_error: str | None = None,
) -> dict[str, object]:
    """Build keyword args for a ``Clip`` row exercising one detection-state variant.

    The shared ``_seed_clip_rows`` helper doesn't surface ``manual_has_cat`` or ``analysis_error``;
    callers that need those construct rows directly via this helper to keep field boilerplate out
    of test bodies.
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
        "has_cat": manual_has_cat is True,
        "manual_has_cat": manual_has_cat,
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
    """Insert one ``alerts_sent`` row at ``sent_at``."""
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
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``GET /`` returns 200 with an inline ``<svg>`` and the camera's display name in a lane label."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root, display_name="Pantry Litter Box")
    now_ish = datetime.now(UTC)
    _seed_clip_rows(db_session_factory, internal_root, camera_id=cam_id, start_offsets=[timedelta(hours=2)], reference_now=now_ish)

    with web_test_client(config) as client:
        response = client.get("/", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "<svg" in response.text
    assert "Pantry Litter Box" in response.text


def test_timeline_route_accepts_range_param(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with web_test_client(config) as client:
        response = client.get("/timeline?range=6h", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert 'class="timeline-svg"' in response.text


def test_timeline_renders_per_clip_markers_under_24h(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A 6h window with 10 seeded clips renders 10 distinct per-clip ``<rect class="clip">`` elements.

    Per spec §4.7.1 the 24h threshold gates bucketing; below it, every clip gets its own marker.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now_ish = datetime.now(UTC)
    # 10 clips spread over 4.5 hours, well inside the 6h window.
    offsets = [timedelta(hours=5) - timedelta(minutes=i * 30) for i in range(10)]
    _seed_clip_rows(db_session_factory, internal_root, camera_id=cam_id, start_offsets=offsets, reference_now=now_ish)

    with web_test_client(config) as client:
        response = client.get("/timeline?range=6h", headers=_AUTH_HEADER)

    assert response.status_code == 200
    # Per-clip SVG markers carry ``<rect class="clip ...">`` (the leading ``clip`` plus modifier
    # classes for cat/no-cat/manual/error styling). Scoping the substring to the ``<rect`` element
    # keeps the count robust against modifier-class additions and against the same ``css_classes``
    # being reused on the thumb-strip ``<li>`` cards.
    assert response.text.count('<rect class="clip ') == 10
    assert 'class="bucket"' not in response.text


def test_timeline_buckets_clips_above_24h_threshold(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A 7d window collapses 50 same-hour clips into a single heatmap cell whose ``aria-label`` includes the count.

    The clips are deliberately clustered into one hour so the bucket count (50) is unambiguous in
    the assertion. Even spreading would still bucket but bin counts of 1 don't prove the count text
    is wired through to the accessible name.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    now_ish = datetime.now(UTC)
    # 50 clips all anchored ~3 days back, spread across 60 minutes — they all fall into the same
    # per-hour bin so the bucket has count=50.
    base = timedelta(days=3)
    offsets = [base + timedelta(seconds=i * 60) for i in range(50)]
    _seed_clip_rows(db_session_factory, internal_root, camera_id=cam_id, start_offsets=offsets, reference_now=now_ish)

    with web_test_client(config) as client:
        response = client.get("/timeline?range=7d", headers=_AUTH_HEADER)

    assert response.status_code == 200
    # Bucketed rendering is on — at least one bucket cell exists, and no per-clip markers.
    assert 'class="clip"' not in response.text
    assert response.text.count('class="bucket ') >= 1
    # The bucket's aria-label must encode the count for accessibility tools.
    assert 'aria-label="50 clips' in response.text


def test_bucket_opacity_scales_per_lane_at_7d(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """At 7d, each bucket rect carries an ``opacity`` attribute scaled to the per-lane max count."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    # Use the route's actual ``now`` reference so seeded clips land in deterministic hour-bins
    # relative to ``start_window = now - 7d``.
    now_ish = datetime.now(UTC)
    # 6 clips bunched into one hour-bin and a single sparse clip 5h back. The 30-minute interior
    # anchor keeps all six clips strictly inside one hour-bin regardless of sub-second drift
    # between test seeding and the request's ``datetime.now(UTC)`` reference.
    bunched = [timedelta(hours=2, minutes=30, seconds=s) for s in range(0, 60, 10)]  # six clips at hour-2
    sparse = [timedelta(hours=5)]
    _seed_clip_rows(
        db_session_factory,
        internal_root,
        camera_id=cam_id,
        start_offsets=[*bunched, *sparse],
        reference_now=now_ish,
    )

    with web_test_client(config) as client:
        response = client.get("/timeline?range=7d", headers=_AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    # The dense bin (count=6, lane max=6) should carry opacity ~ 0.95 (0.20 + 0.75 * 6/6).
    # The sparse bin (count=1) should carry opacity ~ 0.325 (0.20 + 0.75 * 1/6).
    assert 'opacity="0.95"' in body, "expected the densest lane bucket at full opacity"
    assert 'opacity="0.325"' in body, "expected the sparse bucket at scaled opacity (0.20 + 0.75 * 1/6 = 0.325, rounded to 3 dp)"


def test_timeline_renders_offline_banner_when_storage_root_unmounted(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with web_test_client(config) as client:
        response = client.get("/", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "External storage offline" in response.text


def test_timeline_header_carries_htmx_attrs_and_banner_aria_live(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Range presets carry hx-indicator + hx-push-url; storage banner carries aria-live."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_row(db_session_factory, internal_root)
    # Drop the storage_root after config validation so the probe (`is_dir()`) returns False — same
    # trick the existing test_timeline_renders_offline_banner_when_storage_root_unmounted uses.
    storage_root.rmdir()

    with web_test_client(config) as client:
        response = client.get("/timeline?range=24h", headers=_AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    assert 'class="timeline-header"' in body
    # Each of the four presets gets both new attributes.
    assert body.count('hx-indicator="#timeline-region"') == 4
    assert body.count('hx-push-url="true"') == 4
    # Banner aria-live.
    assert 'aria-live="polite"' in body
    assert 'class="banner banner-offline"' in body


def test_timeline_omits_offline_banner_when_storage_root_mounted(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A mounted storage root produces no offline banner."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_row(db_session_factory, internal_root)

    with web_test_client(config) as client:
        response = client.get("/", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "External storage offline" not in response.text


def test_timeline_thumb_imgs_reference_placeholder_in_onerror(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with web_test_client(config) as client:
        response = client.get("/timeline?range=6h", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "clip-placeholder.svg" in response.text
    assert "onerror" in response.text


def test_root_lane_includes_alert_marker(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with web_test_client(config) as client:
        response = client.get("/", headers=_AUTH_HEADER)

    assert response.status_code == 200
    # Alert markers carry the alert type as a label so operators can read them at a glance.
    assert "FREQUENCY" in response.text


def test_timeline_renders_time_axis_with_per_range_tick_count(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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
        ("7d", 28, "tick interval 6h, 7d window -> 28 ticks"),
        ("30d", 30, "tick interval 1d, 30d window -> 30 ticks"),
    ]

    with web_test_client(config) as client:
        for range_key, expected_ticks, msg in cases:
            response = client.get(f"/timeline?range={range_key}", headers=_AUTH_HEADER)
            assert response.status_code == 200
            body = response.text
            assert '<g class="time-axis">' in body, f"{range_key}: missing time-axis group"
            tick_count = body.count('class="axis-tick"')
            assert tick_count == expected_ticks, f"{range_key}: {msg}; got {tick_count}"
            assert 'class="axis-now"' in body, f"{range_key}: missing now indicator"


def test_timeline_renders_day_boundary_markers_when_window_crosses_midnight(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with web_test_client(config) as client:
        response = client.get("/timeline?range=24h", headers=_AUTH_HEADER)

    # A 24h window in display_timezone always crosses exactly one midnight boundary.
    assert response.status_code == 200
    assert response.text.count('class="axis-day-boundary"') == 1


def test_thumb_card_renders_display_stamp_in_configured_timezone(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """At 6h/24h ranges the thumb strip renders one card per clip with HH:MM:SS in display_timezone."""
    from zoneinfo import ZoneInfo

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

    with web_test_client(config) as client:
        response = client.get("/?range=24h", headers=_AUTH_HEADER)

    assert response.status_code == 200
    expected_local = (reference_now - timedelta(hours=4)).astimezone(ZoneInfo(config.web.display_timezone))
    expected_stamp = expected_local.strftime("%H:%M:%S")
    assert f'<span class="thumb-time">{expected_stamp}</span>' in response.text


def test_timeline_empty_state_links_to_next_longer_range(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """With no clips at 24h, the empty state surfaces a 'try 7d' CTA and a /cameras link."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_row(db_session_factory, internal_root)  # camera but no clips

    with web_test_client(config) as client:
        response = client.get("/timeline?range=24h", headers=_AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    assert 'class="empty-state' in body, "expected empty-state block"
    assert "No activity in this range" in body
    assert 'class="empty-cta-next-longer-range"' in body, "expected next-longer-range CTA"
    assert 'class="empty-cta-cameras"' in body, "expected /cameras CTA"


def test_timeline_empty_state_omits_next_longer_link_at_30d(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """At the longest preset (30d), the empty state hides the next-longer-range CTA."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_row(db_session_factory, internal_root)

    with web_test_client(config) as client:
        response = client.get("/timeline?range=30d", headers=_AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    assert 'class="empty-state' in body
    # No "next" CTA at 30d (already at the longest range).
    assert "next-longer-range" not in body
    assert "/cameras" in body


def test_thumb_card_li_carries_clip_css_classes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
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

    with web_test_client(config) as client:
        response = client.get("/?range=24h", headers=_AUTH_HEADER)

    assert response.status_code == 200
    # The seeded clip has has_cat=True (i % 2 == 0 for i=0). Expect <li class="clip clip-cat">.
    assert '<li class="clip clip-cat"' in response.text


def test_thumb_card_renders_clip_manual_and_clip_error_state_classes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A manual-labeled clip emits ``clip-manual``; a detector-error clip emits ``clip-error``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    cam_id = _seed_camera_row(db_session_factory, internal_root)
    reference_now = datetime.now(UTC).replace(microsecond=0)
    # _seed_clip_rows doesn't surface manual_has_cat / analysis_error, so seed the two state
    # variants directly. _state_clip_kwargs centralises the boilerplate fields per row.
    with db_session_factory(internal_root) as session:
        session.add(Clip(**_state_clip_kwargs(cam_id, reference_now, "manual", manual_has_cat=True)))
        session.add(Clip(**_state_clip_kwargs(cam_id, reference_now, "error", analysis_error="ffmpeg returned 1: bad header")))

    with web_test_client(config) as client:
        response = client.get("/?range=24h", headers=_AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    # Manual-confirmed cat: cat + manual modifier classes both present on the same <li>.
    assert "clip-cat" in body, "expected manual-confirmed cat to carry clip-cat class"
    assert "clip-manual" in body, "expected manual-confirmed cat to carry clip-manual class"
    # Detector-error clip: clip-error modifier present on a thumb card.
    assert "clip-error" in body, "expected detector-error clip to carry clip-error class"

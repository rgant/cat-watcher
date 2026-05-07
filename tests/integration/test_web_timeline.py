"""Integration tests for the cat-watcher timeline routes (Task 23).

Covers ``GET /`` and ``GET /timeline`` rendering: SVG lanes per camera, range-preset switching
via the ``range`` query parameter, density bucketing for windows past the 24h threshold (per
spec §4.7.1), and the storage-offline degradation path (banner + ``onerror`` placeholder).

The tests deliberately avoid asserting on layout numerics (lane width, x-coordinates) because
those are presentation details that the route's contract is silent on; they pin observable HTML
contract — element classes, marker counts, ``<title>`` text — instead.
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
    _ = _seed_camera_row(db_session_factory, internal_root)

    with web_test_client(config) as client:
        response = client.get("/timeline?range=6h", headers=_AUTH_HEADER)

    assert response.status_code == 200
    assert "<svg" in response.text


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
    # Per-clip markers carry ``class="clip ..."`` (the leading ``clip`` plus modifier classes for
    # cat/no-cat/manual/error styling). Match on the leading token + trailing space so the count
    # is robust against modifier-class additions but still strict about the count.
    assert response.text.count('class="clip ') == 10
    assert 'class="bucket"' not in response.text


def test_timeline_buckets_clips_above_24h_threshold(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A 7d window collapses 50 same-hour clips into a single heatmap cell whose ``<title>`` includes the count.

    The clips are deliberately clustered into one hour so the bucket count (50) is unambiguous in
    the assertion. Even spreading would still bucket but bin counts of 1 don't prove the count
    text is wired through to the title.
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
    assert response.text.count('class="bucket"') >= 1
    # The bucket's <title> must encode the count for hover discoverability.
    assert "<title>50" in response.text or ">50 clips" in response.text or ">50</title>" in response.text


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

    Confirms the spec §4.13 fallback path: when the media route returns 503, the inline
    placeholder is shown instead of a broken-image glyph. Per-clip markers render an `<img>`
    inside the lane group, so we seed at least one clip for the assertion.
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

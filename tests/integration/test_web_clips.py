"""Integration tests for the cat-watcher clip-management routes.

Covers ``GET /clips``, ``GET /clips/{id}``, ``GET /media/clip/{id}.mp4`` (HTTP byte-Range), and
``GET /media/thumb/{id}.jpg`` — plus the storage-offline degradation path (spec §4.13). The
manual-label form is asserted on (form HTML lives on the detail page) but its POST/DELETE endpoints
land in Task 22.

Tests share the project-standard fixtures (``storage_dirs``, ``make_config``, ``web_test_client``)
from ``tests/conftest.py``. The auth path itself is exhaustively covered by ``test_web_health.py``;
this module just attaches a constant ``Authorization`` header to every request.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003  # pytest evaluates fixture annotations at collection time
from typing import TYPE_CHECKING

import pytest
from db_helpers import (
    AUTH_HEADER,
    DEFAULT_START_TS,
    build_detector_clip,
    seed_cat_subject,
    tag_clip_frame,
)  # pytest pythonpath makes this importable
from sqlalchemy import desc, select

from cat_watcher.db import Camera, Clip, ClipFrame, create_engine, get_session

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from fastapi.testclient import TestClient
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

    from cat_watcher.config import Config


def _seed_camera_and_clip(  # noqa: PLR0913  # test-fixture builder; bundling args at the call-site is noisier
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    storage_root: Path,
    clip_bytes: bytes = b"\x00" * 1024,
    thumb_bytes: bytes = b"\xff\xd8\xff\xe0",
    start_ts: datetime = DEFAULT_START_TS,
    has_cat: bool = True,
    write_files: bool = True,
    camera_name: str = "pantry",
    camera_display_name: str = "Pantry Litter Box",
) -> tuple[int, int]:
    """``write_files=False`` simulates the storage-offline degradation case: DB row exists, files don't."""
    rel_clip, rel_thumb = _relative_paths_for(camera_name, start_ts)
    if write_files:
        _materialize_clip_files(storage_root, rel_clip, clip_bytes, rel_thumb, thumb_bytes)

    with db_session_factory(internal_root) as session:
        cam = Camera(name=camera_name, display_name=camera_display_name, host="cam.example.com")
        session.add(cam)
        session.flush()
        clip = build_detector_clip(cam.id, rel_clip, rel_thumb, start_ts, len(clip_bytes), has_cat=has_cat)
        session.add(clip)
        session.flush()
        return cam.id, clip.id


def _relative_paths_for(camera_name: str, start_ts: datetime) -> tuple[str, str]:
    fname = start_ts.strftime("%H%M%S")
    date_dir = start_ts.strftime("%Y-%m-%d")
    return (
        f"clips/{camera_name}/{date_dir}/{fname}.mp4",
        f"thumbs/{camera_name}/{date_dir}/{fname}.jpg",
    )


def _materialize_clip_files(storage_root: Path, rel_clip: str, clip_bytes: bytes, rel_thumb: str, thumb_bytes: bytes) -> None:
    for rel, payload in ((rel_clip, clip_bytes), (rel_thumb, thumb_bytes)):
        full = storage_root / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        _ = full.write_bytes(payload)


def _seed_clip_frame(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    storage_root: Path,
    clip_id: int,
    frame_bytes: bytes | None = b"\xff\xd8\xff\xe0frame-bytes",
) -> int:
    """Seed a ClipFrame row + optional JPEG bytes for the per-clip thumbnail tests.

    ``frame_bytes=None`` simulates the row-without-file drift the 410 path covers. The relpath
    matches the production layout (``thumbs/<cam>/<YYYY-MM-DD>/<HHMMSS>/<NN>.jpg``) so
    filesystem-coupled regressions surface here instead of getting masked by a synthetic path.
    """
    rel_thumb = _frame_relpath_from_clip(db_session_factory, internal_root=internal_root, clip_id=clip_id)
    if frame_bytes is not None:
        full = storage_root / rel_thumb
        full.parent.mkdir(parents=True, exist_ok=True)
        _ = full.write_bytes(frame_bytes)

    with db_session_factory(internal_root) as session:
        frame = ClipFrame(clip_id=clip_id, ordinal=0, t_offset_seconds=0.0, score=0.91, thumb_path=rel_thumb)
        session.add(frame)
        session.flush()
        return frame.id


def _frame_relpath_from_clip(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    clip_id: int,
) -> str:
    with db_session_factory(internal_root) as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        camera = session.get(Camera, clip.camera_id)
        assert camera is not None
        date_dir = clip.start_ts.strftime("%Y-%m-%d")
        hhmmss = clip.start_ts.strftime("%H%M%S")
        return f"thumbs/{camera.name}/{date_dir}/{hhmmss}/00.jpg"


def _seed_clip_frame_at(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    *,
    internal_root: Path,
    clip_id: int,
    spec: tuple[int, float],
    score: float = 0.5,
) -> int:
    """Seed a ClipFrame row for the contact-sheet tests without writing a JPEG.

    The contact-sheet tests only need the ``<img>`` URL. ``thumb_path`` stays distinct per ordinal
    so a per-frame relpath regression still surfaces.
    """
    ordinal, t_offset_seconds = spec
    rel_thumb = f"thumbs/clip-{clip_id}/{ordinal:02d}.jpg"
    with db_session_factory(internal_root) as session:
        frame = ClipFrame(clip_id=clip_id, ordinal=ordinal, t_offset_seconds=t_offset_seconds, score=score, thumb_path=rel_thumb)
        session.add(frame)
        session.flush()
        return frame.id


def _seed_extra_clip(
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
    internal_root: Path,
    *,
    source_filename: str,
    start_ts: datetime,
    has_cat: bool,
) -> None:
    with db_session_factory(internal_root) as session:
        cam = session.scalar(select(Camera))
        assert cam is not None
        date_dir = start_ts.strftime("%Y-%m-%d")
        session.add(
            Clip(
                camera_id=cam.id,
                source_filename=source_filename,
                start_ts=start_ts,
                end_ts=start_ts + timedelta(seconds=30),
                duration_seconds=30.0,
                file_path=f"clips/{cam.name}/{date_dir}/{source_filename}",
                thumb_path=f"thumbs/{cam.name}/{date_dir}/{source_filename}.jpg",
                file_size_bytes=512,
                has_cat=has_cat,
                detector_version="yolov11n@deadbeef",
                ingested_at=datetime.now(UTC),
            ),
        )


def test_clips_list_returns_200_and_renders_camera_display_name(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``GET /clips`` renders the camera's display name for each row."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with alembic_web_test_client(config) as client:
        response = client.get("/clips", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "Pantry Litter Box" in response.text


def test_clips_list_renders_start_ts_in_display_timezone(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Assert each row's ``Start`` cell renders the clip start in ``web.display_timezone``.

    The displayed time matches the OSD time burned into the video. The ``<time datetime="…">``
    attribute keeps UTC ISO for HTML5 semantics; only the visible text is localized.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # 18:47:04 UTC on 2026-05-01 → 14:47:04 EDT (default display_timezone is America/New_York).
    start_ts = datetime(2026, 5, 1, 18, 47, 4, tzinfo=UTC)
    _ = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=start_ts,
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/clips", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "2026-05-01 14:47:04 EDT" in response.text
    assert 'datetime="2026-05-01T18:47:04+00:00"' in response.text


def test_clips_list_filter_by_camera_name(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``?camera=<name>`` restricts the rendered list to clips for that camera."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, pantry_clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="pantry",
        camera_display_name="Pantry",
    )
    _, garage_clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="garage",
        camera_display_name="Garage Watch",
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?camera=pantry", headers=AUTH_HEADER)

    assert response.status_code == 200
    # Both display names live in the filter <select> regardless of the active filter, so the actual
    # signal is which clip-detail links land in the table body.
    assert f"/clips/{pantry_clip_id}" in response.text
    assert f"/clips/{garage_clip_id}" not in response.text


def test_clips_list_filter_by_has_cat_true(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``?has_cat=true`` returns only cat-positive clips."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, has_cat=True)
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="070000.mp4",
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        has_cat=False,
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?has_cat=true", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "064704" in response.text
    assert "070000" not in response.text


def test_clips_list_filter_by_date_str(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``?date_str=YYYY-MM-DD`` restricts to clips whose ``start_ts`` falls on that UTC day."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 47, 4, tzinfo=UTC),
    )
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="100000.mp4",
        start_ts=datetime(2026, 5, 2, 10, 0, 0, tzinfo=UTC),
        has_cat=False,
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?date_str=2026-05-02", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "100000" in response.text
    assert "064704" not in response.text


def test_clips_list_returns_200_with_no_clips(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """An empty database renders the page (no rows) without raising."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with alembic_web_test_client(config) as client:
        response = client.get("/clips", headers=AUTH_HEADER)

    assert response.status_code == 200


def test_clip_detail_renders_video_player_targeting_media_route(
    seeded_detail_clip: tuple[Config, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """``GET /clips/{id}`` renders a ``<video>`` element pointing at the media route."""
    config, clip_id = seeded_detail_clip

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert f"/media/clip/{clip_id}.mp4" in response.text
    assert "0.92" in response.text  # max_score
    assert "yolov11n@deadbeef" in response.text  # detector_version


def test_clip_detail_heading_renders_in_display_timezone(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Assert the heading's time-of-day is rendered in ``web.display_timezone``.

    The displayed time matches the camera-OSD time burned into the video. The
    ``<time datetime="…">`` attribute keeps UTC ISO for HTML5 semantics, but the visible text uses
    the configured display zone — ``Clip.start_ts`` is stored UTC, so a raw ``isoformat()`` would
    disagree with the on-screen video timestamp by the tz offset.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # 18:47:04 UTC on 2026-05-01 → 14:47:04 EDT (default display_timezone is America/New_York,
    # which is UTC-4 in May).
    start_ts = datetime(2026, 5, 1, 18, 47, 4, tzinfo=UTC)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=start_ts,
    )

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "2026-05-01 14:47:04 EDT" in response.text
    assert 'datetime="2026-05-01T18:47:04+00:00"' in response.text


def test_clip_detail_returns_404_for_unknown_clip(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A nonexistent clip id yields ``404`` (not a 500)."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.get("/clips/9999", headers=AUTH_HEADER)

    assert response.status_code == 404


def test_clip_detail_renders_prev_next_navigation(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Detail page links to the chronologically-newer (``← Newer``) and older (``Older →``) clips.

    Three clips at distinct timestamps; visit the middle one and assert both neighbors are linked
    by id. Pin the rel attributes so a regression that swaps prev/next surfaces here.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, oldest_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="070000.mp4",
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        has_cat=True,
    )
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="080000.mp4",
        start_ts=datetime(2026, 5, 1, 8, 0, 0, tzinfo=UTC),
        has_cat=True,
    )
    with db_session_factory(internal_root) as session:
        rows = list(session.scalars(select(Clip).order_by(Clip.start_ts.asc())).all())
    middle_id = rows[1].id
    newest_id = rows[2].id

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{middle_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert f'href="http://testserver/clips/{newest_id}" rel="prev"' in response.text
    assert f'href="http://testserver/clips/{oldest_id}" rel="next"' in response.text


def test_clip_detail_disables_navigation_at_endpoints(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Assert clip-detail navigation links render as disabled at each endpoint of the timeline.

    The newest clip has ``← Newer`` rendered as a disabled span; the oldest clip's ``Older →`` is
    the disabled one. Asserts via the ``clip-nav-disabled`` CSS class so a refactor that drops the
    visual cue surfaces here.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, oldest_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="070000.mp4",
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        has_cat=True,
    )
    with db_session_factory(internal_root) as session:
        newest_id = session.scalar(select(Clip.id).order_by(desc(Clip.start_ts)).limit(1))
    assert newest_id is not None

    with alembic_web_test_client(config) as client:
        newest_response = client.get(f"/clips/{newest_id}", headers=AUTH_HEADER)
        oldest_response = client.get(f"/clips/{oldest_id}", headers=AUTH_HEADER)

    assert '<span class="clip-nav-disabled" aria-disabled="true">← Newer</span>' in newest_response.text
    assert 'rel="next"' in newest_response.text
    assert '<span class="clip-nav-disabled" aria-disabled="true">Older →</span>' in oldest_response.text
    assert 'rel="prev"' in oldest_response.text


def test_media_clip_returns_full_file_when_no_range_header(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """A request without ``Range`` returns the entire MP4 with ``200 OK``."""
    payload = b"\x00\x01\x02\x03" * 256  # 1024 bytes, distinct
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, clip_bytes=payload)

    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"].startswith("video/mp4")


def test_media_clip_honors_range_header_returns_206(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``Range: bytes=0-15`` returns ``206`` with the requested 16-byte segment + ``Content-Range``."""
    payload = bytes(range(256)) * 4  # 1024 distinct bytes
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, clip_bytes=payload)

    headers = dict(AUTH_HEADER)
    headers["Range"] = "bytes=0-15"
    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=headers)

    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 0-15/{len(payload)}"
    assert response.headers["content-length"] == "16"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.content == payload[:16]


def test_media_clip_honors_open_ended_range_header(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``Range: bytes=N-`` (no end) returns from N to EOF."""
    payload = bytes(range(256)) * 4  # 1024 distinct bytes
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, clip_bytes=payload)

    headers = dict(AUTH_HEADER)
    headers["Range"] = "bytes=512-"
    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=headers)

    assert response.status_code == 206
    assert response.headers["content-range"] == f"bytes 512-{len(payload) - 1}/{len(payload)}"
    assert response.content == payload[512:]


def test_media_clip_returns_503_when_storage_root_unmounted(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    tmp_path: Path,
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Spec §4.13: when ``storage_root`` is not accessible, ``/media/clip`` returns ``503``.

    Simulated by pointing the config at a non-existent ``storage_root`` directory — the row exists
    but the route's storage probe fails. The 503 response is what the timeline-template's
    ``onerror`` handler uses to decide between rendering the clip thumbnail and falling back to the
    bundled placeholder SVG; the same handler also drives the storage-offline banner.
    """
    internal_root, _ = storage_dirs
    missing_storage = tmp_path / "drive-not-mounted"
    config = make_config(internal_root, missing_storage)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=missing_storage,
        write_files=False,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=AUTH_HEADER)

    assert response.status_code == 503


def test_media_clip_returns_410_when_file_missing_but_storage_mounted(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """DB row exists, ``storage_root`` is mounted, but the specific file is gone — ``410 Gone``.

    This is data-integrity drift (e.g. retention sweep removed the file but the row hasn't been
    pruned yet). It's distinct from the 503 case because the drive itself is fine — only this one
    resource is unavailable. Returning 410 keeps the operator-visible signal in logs distinct.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        write_files=False,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=AUTH_HEADER)

    assert response.status_code == 410


def test_media_clip_returns_404_for_unknown_id(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """No row at all → ``404`` (distinct from 410, which means "row exists, file gone")."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.get("/media/clip/9999.mp4", headers=AUTH_HEADER)

    assert response.status_code == 404


def test_media_thumb_returns_jpeg_bytes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``/media/thumb/{id}.jpg`` returns the on-disk thumbnail with a JPEG content type."""
    payload = b"\xff\xd8\xff\xe0" + b"thumb-bytes"
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        thumb_bytes=payload,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/thumb/{clip_id}.jpg", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"].startswith("image/jpeg")


def test_media_thumb_returns_503_when_storage_unmounted(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    tmp_path: Path,
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Same 503 contract as ``/media/clip`` — drive offline → 503."""
    internal_root, _ = storage_dirs
    missing_storage = tmp_path / "drive-not-mounted"
    config = make_config(internal_root, missing_storage)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=missing_storage,
        write_files=False,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/thumb/{clip_id}.jpg", headers=AUTH_HEADER)

    assert response.status_code == 503


def test_media_thumb_returns_410_when_thumb_missing(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Same 410 contract as ``/media/clip`` — drive mounted but thumb file gone."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        write_files=False,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/thumb/{clip_id}.jpg", headers=AUTH_HEADER)

    assert response.status_code == 410


def test_media_frame_returns_jpeg_bytes(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``/media/frame/{id}.jpg`` returns the per-frame JPEG with a JPEG content type."""
    payload = b"\xff\xd8\xff\xe0" + b"per-frame-bytes"
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
    )
    frame_id = _seed_clip_frame(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        clip_id=clip_id,
        frame_bytes=payload,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/frame/{frame_id}.jpg", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert response.content == payload
    assert response.headers["content-type"].startswith("image/jpeg")


def test_media_frame_returns_404_for_unknown_id(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """No ``ClipFrame`` row → ``404`` (distinct from 410, which means "row exists, file gone")."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)

    with web_test_client(config) as client:
        response = client.get("/media/frame/9999.jpg", headers=AUTH_HEADER)

    assert response.status_code == 404


def test_media_frame_returns_503_when_storage_offline(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    tmp_path: Path,
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Same 503 contract as ``/media/thumb`` — drive offline → 503."""
    internal_root, _ = storage_dirs
    missing_storage = tmp_path / "drive-not-mounted"
    config = make_config(internal_root, missing_storage)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=missing_storage,
        write_files=False,
    )
    frame_id = _seed_clip_frame(
        db_session_factory,
        internal_root=internal_root,
        storage_root=missing_storage,
        clip_id=clip_id,
        frame_bytes=None,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/frame/{frame_id}.jpg", headers=AUTH_HEADER)

    assert response.status_code == 503


def test_media_frame_returns_410_when_file_missing(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Same 410 contract as ``/media/thumb`` — drive mounted but per-frame file gone."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        write_files=False,
    )
    frame_id = _seed_clip_frame(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        clip_id=clip_id,
        frame_bytes=None,
    )

    with web_test_client(config) as client:
        response = client.get(f"/media/frame/{frame_id}.jpg", headers=AUTH_HEADER)

    assert response.status_code == 410


@pytest.mark.parametrize(
    ("range_header", "expected_status"),
    [
        ("bytes=invalid", 400),  # malformed — RFC 7233 § 3.1
        ("0-15", 400),  # missing ``bytes=`` prefix
        ("bytes=100-50", 400),  # start > end (semantically invalid)
        ("bytes=99999-", 416),  # start past EOF — RFC 7233 § 4.4 unsatisfiable
    ],
    ids=["malformed", "missing-prefix", "inverted", "start-past-eof"],
)
def test_media_clip_returns_rfc_correct_status_for_bad_range_headers(  # noqa: PLR0913  # pylint: disable=too-many-positional-arguments  # parametrized: 4 fixtures + 2 parametrize values
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    range_header: str,
    expected_status: int,
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Bad Range headers get RFC 7233 status codes (400 / 416), not 200 or 500.

    Pins the actual behavior of the underlying ``starlette.responses.FileResponse``: malformed
    syntax gets ``400 Bad Request``; ranges that fall entirely past EOF get ``416 Range Not
    Satisfiable``. A regression that swallowed the bad header silently (returning 200 + full
    content) or that crashed (500) would break ``<video>`` clients that handle 416 by retrying
    without ``Range``.
    """
    payload = bytes(range(256)) * 4  # 1024 distinct bytes
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, clip_bytes=payload)

    headers = dict(AUTH_HEADER)
    headers["Range"] = range_header
    with web_test_client(config) as client:
        response = client.get(f"/media/clip/{clip_id}.mp4", headers=headers)

    assert response.status_code == expected_status


def test_clips_list_renders_in_start_ts_ascending_order_for_reviewed_no(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Default ``?reviewed=no`` orders unreviewed rows by ``start_ts ASC`` — oldest first.

    The review queue surfaces the oldest unreviewed clip at the top so operators work through
    footage in chronological order. Seed three unreviewed clips out-of-order; the rendered HTML
    must show their detail links oldest-first.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # Insert newest, oldest, middle — different from expected render order so the test can't
    # pass accidentally on insertion order.
    _, newest_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 3, 6, 0, 0, tzinfo=UTC),
        camera_name="garage",
        camera_display_name="Garage",
    )
    _, oldest_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _, middle_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 2, 6, 0, 0, tzinfo=UTC),
        camera_name="bedroom",
        camera_display_name="Bedroom",
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/clips", headers=AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    positions = [body.find(f"/clips/{cid}") for cid in (oldest_id, middle_id, newest_id)]
    assert all(p > 0 for p in positions), "every clip's detail link must appear in body"
    assert positions == sorted(positions), "rows must render oldest-first (start_ts ASC) for reviewed=no"


def test_clips_list_any_renders_in_start_ts_descending_order(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``?reviewed=any`` preserves the legacy ``start_ts DESC`` ordering.

    Seed three clips out-of-chronological-insert-order; the rendered HTML must show their
    detail-page links in newest-first order.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # Insert oldest, newest, middle — different order than the expected render order.
    _, oldest_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _, newest_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 3, 6, 0, 0, tzinfo=UTC),
        camera_name="garage",
        camera_display_name="Garage",
    )
    _, middle_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 2, 6, 0, 0, tzinfo=UTC),
        camera_name="bedroom",
        camera_display_name="Bedroom",
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?reviewed=any", headers=AUTH_HEADER)

    assert response.status_code == 200
    body = response.text
    positions = [body.find(f"/clips/{cid}") for cid in (newest_id, middle_id, oldest_id)]
    assert all(p > 0 for p in positions), "every clip's detail link must appear in body"
    assert positions == sorted(positions), "rows must render newest-first (start_ts DESC) for reviewed=any"


def test_clips_list_filters_compose_with_and_semantics(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """All three filters combined narrow to the intersection — pins AND semantics.

    Seed four clips covering the cross-product of (camera in {pantry, garage}) x (has_cat in
    {true, false}) on May 2; only one matches all three filters
    (``camera=pantry & has_cat=true & date_str=2026-05-02``). A regression that swaps the
    chained ``.where()`` calls for an OR-equivalent (or drops one accidentally) would
    silently broaden the result set.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    target_ts = datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC)
    # The clip that should pass all three filters.
    _, target_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="pantry",
        camera_display_name="Pantry",
        has_cat=True,
        start_ts=target_ts,
    )
    # Right camera + day, wrong has_cat.
    _seed_extra_clip(
        db_session_factory,
        internal_root,
        source_filename="120100.mp4",
        start_ts=target_ts + timedelta(minutes=1),
        has_cat=False,
    )
    # Wrong camera, right has_cat + day.
    _, garage_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="garage",
        camera_display_name="Garage",
        has_cat=True,
        start_ts=target_ts + timedelta(minutes=2),
    )
    # Right camera + has_cat, wrong day.
    _, wrong_day_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="bedroom",  # extra camera so the seed stays unique
        camera_display_name="Bedroom",
        has_cat=True,
        start_ts=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
    )

    with alembic_web_test_client(config) as client:
        response = client.get(
            "/clips?camera=pantry&has_cat=true&date_str=2026-05-02",
            headers=AUTH_HEADER,
        )

    assert response.status_code == 200
    body = response.text
    assert f"/clips/{target_id}" in body, "the only clip matching all three filters must render"
    assert f"/clips/{garage_id}" not in body, "wrong camera must be excluded"
    assert f"/clips/{wrong_day_id}" not in body, "wrong day must be excluded"
    # The right-camera-wrong-has_cat clip is added via ``_seed_extra_clip`` to the same
    # camera as the target; its ID we don't track but its filename ``120100.mp4`` is unique.
    assert "120100" not in body, "wrong has_cat must be excluded"


def _frame_button_slice(body: str, frame_id: int) -> str:
    """Return ``body`` between the frame's media URL and the next ``</button>``.

    Slicing keeps a neighbour frame's markers from leaking into a substring check on this frame's
    wrapper.
    """
    start = body.find(f"/media/frame/{frame_id}.jpg")
    return body[start : body.find("</button>", start)]


def test_clip_detail_renders_contact_sheet_in_ordinal_order(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Contact sheet renders one keyboard-accessible click-to-seek button per frame, ordinal-asc.

    Frames are inserted with shuffled ordinals so the ordering signal comes from the route's
    relationship-ordered read, not insert order. Scores are mixed above/below the default 0.35
    threshold so the same reszponse covers ordinal ordering, ``data-seek-seconds`` carriage, and the
    threshold-styling cue (``contact-sheet-score-below``) on a single page render.
    """
    internal_root, storage_root = storage_dirs
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    # Insert order 2/0/3/1 is distinct from the expected ordinal-asc render order, so a regression
    # that dropped the relationship's ``order_by`` would surface as out-of-order positions below.
    # Scores: ordinal 0 below threshold (0.10), 2 below (0.20), 1 above (0.50), 3 above (0.90).
    rows: list[tuple[tuple[int, float], float]] = [((2, 10.0), 0.20), ((0, 0.0), 0.10), ((3, 15.0), 0.90), ((1, 5.0), 0.50)]
    ids_by_ordinal: dict[int, int] = {
        spec[0]: _seed_clip_frame_at(db_session_factory, internal_root=internal_root, clip_id=clip_id, spec=spec, score=score)
        for spec, score in rows
    }

    with alembic_web_test_client(make_config(internal_root, storage_root)) as client:
        body = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER).text

    assert 'class="contact-sheet"' in body
    assert 'class="contact-sheet-button"' in body
    positions = [body.find(f"/media/frame/{ids_by_ordinal[o]}.jpg") for o in (0, 1, 2, 3)]
    assert all(p > 0 for p in positions), "every frame's media-frame URL must appear in body"
    assert positions == sorted(positions), "contact-sheet must render in ordinal-ascending order"
    for spec, _ in rows:
        assert f'data-seek-seconds="{spec[1]}"' in body
    # Sub-threshold frame (ordinal 0, score 0.10) carries the muted class; above-threshold (ordinal
    # 1, score 0.50) does not.
    assert "contact-sheet-score-below" in _frame_button_slice(body, ids_by_ordinal[0])
    assert "contact-sheet-score-below" not in _frame_button_slice(body, ids_by_ordinal[1])


def test_clip_detail_hides_contact_sheet_for_legacy_clip(
    seeded_detail_clip: tuple[Config, int],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
) -> None:
    """A clip with no ``ClipFrame`` rows must not render the contact-sheet section at all."""
    config, clip_id = seeded_detail_clip

    with alembic_web_test_client(config) as client:
        response = client.get(f"/clips/{clip_id}", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert 'class="contact-sheet"' not in response.text


# ---------------------------------------------------------------------------
# Cat? column badge rendering (effective_has_cat + manual badge)
# ---------------------------------------------------------------------------


def _clips_engine_for(internal_root: Path) -> Engine:
    """Return a short-lived engine for seeding frame/subject rows via ``tag_clip_frame``."""
    return create_engine(f"sqlite:///{internal_root / 'cat_watcher.sqlite'}")


def test_clips_list_badge_class_unreviewed_no_memberships(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``has_cat=TRUE``, unreviewed, no frame memberships → ``badge-cat`` class, no ``(manual)``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, has_cat=True)

    with alembic_web_test_client(config) as client:
        response = client.get("/clips", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "badge-cat" in response.text
    assert "badge-manual" not in response.text
    assert "(manual)" not in response.text


def test_clips_list_badge_class_reviewed_with_cat_memberships(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``has_cat=TRUE``, reviewed, cat memberships → ``badge-cat badge-manual`` class, ``(manual)`` text.

    Uses ``?reviewed=yes`` because the default ``reviewed=no`` queue excludes already-reviewed clips.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, has_cat=True)
    engine = _clips_engine_for(internal_root)
    try:
        subj_id = seed_cat_subject(engine)
        tag_clip_frame(engine, clip_id=clip_id, subject_id=subj_id, reviewed_at=DEFAULT_START_TS)
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?reviewed=yes", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "badge-cat" in response.text
    assert "badge-manual" in response.text
    assert "(manual)" in response.text


def test_clips_list_badge_class_reviewed_no_memberships_fp_correction(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``has_cat=TRUE``, reviewed, no memberships → ``badge-no-cat``, no ``(manual)`` text.

    This is the false-positive correction case: the detector said cat, the operator reviewed but
    tagged no cat frames, so ``effective_has_cat`` flips to FALSE. ``has_manual_cat`` is 0 (no
    cat frame memberships), so the manual badge does not fire — the spec requires
    ``has_manual_cat IS TRUE AND reviewed_at IS NOT NULL`` for the badge.

    Uses ``?reviewed=yes`` because the default ``reviewed=no`` queue excludes already-reviewed clips.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, has_cat=True)
    # Mark reviewed without tagging any cat frame: operator confirmed no cat.
    engine = _clips_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            clip = session.get(Clip, clip_id)
            assert clip is not None
            clip.reviewed_at = DEFAULT_START_TS
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?reviewed=yes", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "badge-no-cat" in response.text
    assert "badge-manual" not in response.text
    assert "(manual)" not in response.text


def test_clips_list_badge_class_partially_tagged_unreviewed(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``has_cat=FALSE``, cat membership present, ``reviewed_at IS NULL`` → ``badge-no-cat``, no ``(manual)``.

    Partially-tagged but unreviewed clips must not show the manual badge — the operator hasn't
    confirmed the review is complete.
    """
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root, has_cat=False)
    engine = _clips_engine_for(internal_root)
    try:
        subj_id = seed_cat_subject(engine, slug="felix", display_name="Felix")
        tag_clip_frame(engine, clip_id=clip_id, subject_id=subj_id)
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get("/clips", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "badge-no-cat" in response.text
    assert "badge-manual" not in response.text
    assert "(manual)" not in response.text


# ---------------------------------------------------------------------------
# Task 10: reviewed filter, Reviewed column, progress indicator, empty state,
# queue-context handoff
# ---------------------------------------------------------------------------


def test_clips_list_reviewed_no_excludes_reviewed_clips(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Default ``reviewed=no`` only shows clips with ``reviewed_at IS NULL``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, unreviewed_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _, reviewed_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        camera_name="garage",
        camera_display_name="Garage",
    )
    engine = _clips_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            clip = session.get(Clip, reviewed_id)
            assert clip is not None
            clip.reviewed_at = datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC)
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get("/clips", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert f"/clips/{unreviewed_id}" in response.text
    assert f"/clips/{reviewed_id}" not in response.text


def _stamp_reviewed_at(engine: Engine, clip_id: int, reviewed_at: datetime) -> None:
    """Stamp ``reviewed_at`` on a clip row for test setup."""
    with get_session(engine) as session:
        clip = session.get(Clip, clip_id)
        assert clip is not None
        clip.reviewed_at = reviewed_at


def test_clips_list_reviewed_yes_only_reviewed_clips(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """``?reviewed=yes`` shows only clips with ``reviewed_at IS NOT NULL``, ordered ``reviewed_at DESC``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, unreviewed_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _, reviewed_earlier_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        camera_name="garage",
        camera_display_name="Garage",
    )
    _, reviewed_later_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 8, 0, 0, tzinfo=UTC),
        camera_name="bedroom",
        camera_display_name="Bedroom",
    )
    engine = _clips_engine_for(internal_root)
    try:
        _stamp_reviewed_at(engine, reviewed_earlier_id, datetime(2026, 5, 2, 10, 0, 0, tzinfo=UTC))
        _stamp_reviewed_at(engine, reviewed_later_id, datetime(2026, 5, 2, 14, 0, 0, tzinfo=UTC))
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?reviewed=yes", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert f"/clips/{unreviewed_id}" not in response.text
    # reviewed_later should appear before reviewed_earlier in the page (reviewed_at DESC).
    later_pos = response.text.find(f"/clips/{reviewed_later_id}")
    earlier_pos = response.text.find(f"/clips/{reviewed_earlier_id}")
    assert later_pos < earlier_pos, "most-recently-reviewed clip must render first"


def test_clips_list_reviewed_column_shows_dash_for_unreviewed(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Reviewed column cell shows ``—`` for clips where ``reviewed_at IS NULL``."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with alembic_web_test_client(config) as client:
        response = client.get("/clips", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "—" in response.text
    assert "<th" in response.text
    assert "Reviewed" in response.text


def test_clips_list_reviewed_column_shows_date_for_reviewed(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Reviewed column cell shows short date and ``<time datetime>`` attribute for reviewed clips."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)
    reviewed_ts = datetime(2026, 5, 12, 14, 30, 0, tzinfo=UTC)
    engine = _clips_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            clip = session.get(Clip, clip_id)
            assert clip is not None
            clip.reviewed_at = reviewed_ts
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?reviewed=yes", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "2026-05-12" in response.text
    assert 'datetime="2026-05-12T14:30:00+00:00"' in response.text


def test_clips_list_progress_indicator_format(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Progress indicator renders ``{reviewed_count} / {total_count} reviewed`` above the table."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _ = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _, reviewed_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        camera_name="garage",
        camera_display_name="Garage",
    )
    engine = _clips_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            clip = session.get(Clip, reviewed_id)
            assert clip is not None
            clip.reviewed_at = datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC)
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?reviewed=any", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "1 / 2 reviewed" in response.text


def test_clips_list_progress_indicator_respects_camera_filter(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Progress counts apply only to the current camera filter scope."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # pantry: 1 unreviewed; garage: 1 reviewed
    _ = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="pantry",
        camera_display_name="Pantry",
        start_ts=datetime(2026, 5, 1, 6, 0, 0, tzinfo=UTC),
    )
    _, garage_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="garage",
        camera_display_name="Garage",
        start_ts=datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
    )
    engine = _clips_engine_for(internal_root)
    try:
        with get_session(engine) as session:
            clip = session.get(Clip, garage_id)
            assert clip is not None
            clip.reviewed_at = datetime(2026, 5, 2, 12, 0, 0, tzinfo=UTC)
    finally:
        engine.dispose()

    with alembic_web_test_client(config) as client:
        # garage filter: 1 reviewed out of 1 total
        response = client.get("/clips?reviewed=any&camera=garage", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "1 / 1 reviewed" in response.text


def test_clips_list_empty_state_when_no_matching_clips(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """When filters return zero rows, show empty-state message and reset link."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    # Seed an unreviewed clip; querying reviewed=yes should yield zero rows.
    _ = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?reviewed=yes", headers=AUTH_HEADER)

    assert response.status_code == 200
    assert "No clips match these filters." in response.text
    assert "Reset to default queue" in response.text
    assert 'href="http://testserver/clips"' in response.text


def test_clips_list_row_link_carries_filter_querystring(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """Each row link appends the current filter querystring to the detail URL."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(
        db_session_factory,
        internal_root=internal_root,
        storage_root=storage_root,
        camera_name="pantry",
        camera_display_name="Pantry",
    )

    with alembic_web_test_client(config) as client:
        response = client.get("/clips?reviewed=no&camera=pantry", headers=AUTH_HEADER)

    assert response.status_code == 200
    # HTML-encodes ``&`` as ``&amp;`` in href attributes; check for both the link segment and
    # that the full querystring carries through as expected.
    assert f"/clips/{clip_id}?reviewed=no&amp;camera=pantry" in response.text


def test_clips_list_marking_reviewed_removes_from_no_queue(
    storage_dirs: tuple[Path, Path],
    make_config: Callable[..., Config],
    alembic_web_test_client: Callable[[Config], AbstractContextManager[TestClient]],
    db_session_factory: Callable[[Path], AbstractContextManager[Session]],
) -> None:
    """After POST /clips/{id}/reviewed, reloading ``?reviewed=no`` excludes that clip."""
    internal_root, storage_root = storage_dirs
    config = make_config(internal_root, storage_root)
    _, clip_id = _seed_camera_and_clip(db_session_factory, internal_root=internal_root, storage_root=storage_root)

    with alembic_web_test_client(config) as client:
        mark_response = client.post(f"/clips/{clip_id}/reviewed", headers=AUTH_HEADER)
        assert mark_response.status_code == 204

        list_response = client.get("/clips?reviewed=no", headers=AUTH_HEADER)

    assert list_response.status_code == 200
    assert f"/clips/{clip_id}" not in list_response.text
